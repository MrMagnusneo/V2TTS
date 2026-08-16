import queue
import threading
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

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
        # 1. Silence (ignored because not collecting)
        # 2. Voice (starts collecting)
        # 3. Voice (collecting)
        # 4. Silence (collecting, silence count = 1)
        # 5. Silence (collecting, silence count = 2)
        # 6. Silence (collecting, silence count = 3 >= max_silence_ms/frame_ms (30/10=3))
        # -> Should yield phrase (Voice, Voice, Silence, Silence, Silence)
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

        # Expecting 2 voice frames + 3 silence frames = 5 frames of 10 samples = 50 samples
        expected_audio = np.concatenate([voice_frame, voice_frame, silence_frame, silence_frame, silence_frame])
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

if __name__ == '__main__':
    unittest.main()
