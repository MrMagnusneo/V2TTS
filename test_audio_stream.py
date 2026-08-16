import queue
import threading
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from audio_stream import AudioPhraseStream, FramePacket, StreamConfig


class TestAudioPhraseStream(unittest.TestCase):
    def setUp(self):
        # Use small values for fast testing
        self.config = StreamConfig(
            sample_rate=1000,
            channels=1,
            frame_ms=10,
            start_threshold=0.015,
            stop_threshold=0.01,
            min_phrase_ms=20,
            max_silence_ms=30,
        )
        self.stream = AudioPhraseStream(self.config)

    def test_callback(self):
        # Create dummy stereo input data
        # _callback should only take the first channel
        indata = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        frames = 3
        time_info = {}
        status = None

        self.stream._callback(indata, frames, time_info, status)

        self.assertEqual(self.stream._frames_q.qsize(), 1)
        packet = self.stream._frames_q.get_nowait()
        np.testing.assert_array_equal(
            packet.samples,
            np.array([1.0, 3.0, 5.0], dtype=np.float32),
        )
        self.assertGreater(packet.captured_at, 0)

    @patch('audio_stream.get_sounddevice')
    def test_iter_phrases_happy_path(self, mock_get_sounddevice):
        mock_sd = MagicMock()
        mock_get_sounddevice.return_value = mock_sd

        # Frame size: 1000 * 10 / 1000 = 10 samples
        frame_size = 10

        # Create frames that are clearly above or below threshold
        # threshold is 0.015, so 0.02 is voice, 0.005 is silence
        silence_frame = np.full(frame_size, 0.005, dtype=np.float32)
        voice_frame = np.full(frame_size, 0.02, dtype=np.float32)

        # Put frames in queue:
        # 1. Silence (kept as bounded pre-roll)
        # 2. Voice (starts collecting)
        # 3. Voice (collecting)
        # 4. Silence (collecting, silence count = 1)
        # 5. Silence (collecting, silence count = 2)
        # 6. Silence (collecting, silence count = 3 >= max_silence_ms/frame_ms (30/10=3))
        # -> Should yield phrase (pre-roll silence, Voice, Voice, Silence x3)
        queued_frames = [
            silence_frame,
            voice_frame,
            voice_frame,
            silence_frame,
            silence_frame,
            silence_frame,
        ]
        for index, frame in enumerate(queued_frames):
            self.stream._frames_q.put(
                FramePacket(samples=frame, captured_at=float(index + 1))
            )

        # Use an event to stop the iteration after queue is empty
        stop_event = threading.Event()

        # We can just run next() and see if it yields the correct numpy array
        # First we need to make queue.get() raise queue.Empty and set stop_event when empty,
        # otherwise it will wait forever
        gen = self.stream.iter_phrases(stop_event=stop_event)

        # Let's mock queue.get so it sets the stop_event when queue is empty to avoid infinite loop
        original_get = self.stream._frames_q.get
        def side_effect(timeout=None):
            try:
                return original_get(block=False)
            except queue.Empty:
                stop_event.set()
                raise queue.Empty

        with patch.object(self.stream._frames_q, 'get', side_effect=side_effect):
            try:
                result = next(gen)
            except StopIteration:
                self.fail("iter_phrases did not yield a phrase")

        # Expecting 1 pre-roll + 2 voice + 3 trailing silence frames.
        expected_audio = np.concatenate(
            [
                silence_frame,
                voice_frame,
                voice_frame,
                silence_frame,
                silence_frame,
                silence_frame,
            ]
        )
        expected_pcm16 = (expected_audio * 32767.0).astype(np.int16)

        np.testing.assert_array_equal(result.pcm16, expected_pcm16)
        self.assertEqual(result.ended_at, 6.0)

    @patch('audio_stream.get_sounddevice')
    def test_iter_phrases_stop_event(self, mock_get_sounddevice):
        stop_event = threading.Event()
        stop_event.set()

        gen = self.stream.iter_phrases(stop_event=stop_event)

        with self.assertRaises(StopIteration):
            next(gen)

    @patch('audio_stream.get_sounddevice')
    def test_iter_phrases_queue_empty(self, mock_get_sounddevice):
        mock_sd = MagicMock()
        mock_get_sounddevice.return_value = mock_sd

        stop_event = threading.Event()

        original_get = self.stream._frames_q.get

        # We want to raise queue.Empty on first call, and set stop_event on second call
        call_count = 0
        def side_effect(timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise queue.Empty
            else:
                stop_event.set()
                return original_get(block=False)

        gen = self.stream.iter_phrases(stop_event=stop_event)

        with patch.object(self.stream._frames_q, 'get', side_effect=side_effect):
            with self.assertRaises(StopIteration):
                next(gen)

        self.assertEqual(call_count, 2)

def test_phrase_includes_bounded_pre_roll() -> None:
    config = StreamConfig(
        sample_rate=1000,
        frame_ms=100,
        pre_roll_ms=200,
        min_phrase_ms=100,
        max_silence_ms=100,
        start_threshold=0.5,
        stop_threshold=0.2,
    )
    stream = AudioPhraseStream(config)
    quiet_a = np.full(100, 0.1, dtype=np.float32)
    quiet_b = np.full(100, 0.2, dtype=np.float32)
    voice = np.full(100, 0.8, dtype=np.float32)
    silence = np.zeros(100, dtype=np.float32)

    assert stream._process_frame(quiet_a) is None
    assert stream._process_frame(quiet_b) is None
    assert stream._process_frame(voice) is None
    result = stream._process_frame(silence)

    assert result is not None
    assert len(result) == 400
    np.testing.assert_array_equal(
        result[:100],
        (quiet_a * 32767).astype(np.int16),
    )
    np.testing.assert_array_equal(
        result[200:300],
        (voice * 32767).astype(np.int16),
    )


def test_pre_roll_drops_frames_older_than_configured_window() -> None:
    config = StreamConfig(
        sample_rate=1000,
        frame_ms=100,
        pre_roll_ms=200,
        start_threshold=0.5,
    )
    stream = AudioPhraseStream(config)

    for level in (0.1, 0.2, 0.3):
        stream._process_frame(np.full(100, level, dtype=np.float32))
    stream._process_frame(np.full(100, 0.8, dtype=np.float32))

    assert len(stream.phrase_frames) == 3
    assert np.allclose(stream.phrase_frames[0], 0.2)


def test_non_multiple_silence_duration_rounds_up() -> None:
    stream = AudioPhraseStream(StreamConfig(frame_ms=30, max_silence_ms=700))

    assert stream.silence_frames_limit == 24


def test_input_error_includes_device_rate_and_channels() -> None:
    sd = MagicMock()
    sd.InputStream.side_effect = RuntimeError("Invalid device")
    stream = AudioPhraseStream(
        StreamConfig(sample_rate=44100, channels=1, device=3)
    )

    with patch("audio_stream.get_sounddevice", return_value=sd):
        with pytest.raises(RuntimeError) as error:
            next(stream.iter_phrases())

    message = str(error.value)
    assert "input device 3" in message
    assert "44100 Hz" in message
    assert "1 channel" in message


if __name__ == '__main__':
    unittest.main()
