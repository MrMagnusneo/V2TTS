import contextlib
import threading

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import audio_stream
from audio_stream import AudioPhraseStream, StreamConfig


def _callback_frame(stream: AudioPhraseStream, value: float) -> None:
    samples = np.full((10, 1), value, dtype=np.float32)
    stream._callback(samples, 10, {}, None)


def test_overflow_keeps_latest_frames_and_counts_drops() -> None:
    stream = AudioPhraseStream(
        StreamConfig(sample_rate=1000, frame_ms=10, max_buffer_ms=30)
    )

    for value in range(5):
        _callback_frame(stream, float(value))

    queued = [stream._frames_q.get_nowait().samples[0] for _ in range(3)]
    assert queued == [2.0, 3.0, 4.0]
    assert stream.metrics() == audio_stream.StreamMetrics(
        queue_depth_frames=0,
        queue_ms=0,
        dropped_frames=2,
    )


def test_callback_soak_never_exceeds_default_buffer_capacity() -> None:
    stream = AudioPhraseStream(StreamConfig(sample_rate=1000, frame_ms=10))

    for value in range(50_000):
        _callback_frame(stream, float(value))

    assert stream._frames_q.maxsize == 150
    assert stream.metrics() == audio_stream.StreamMetrics(
        queue_depth_frames=150,
        queue_ms=1500,
        dropped_frames=49_850,
    )


def test_callback_records_status_without_console_io() -> None:
    stream = AudioPhraseStream(StreamConfig(sample_rate=1000, frame_ms=10))
    samples = np.zeros((10, 1), dtype=np.float32)

    with patch("builtins.print") as print_message:
        stream._callback(samples, 10, {}, "input overflow")

    print_message.assert_not_called()
    assert stream.metrics().callback_statuses == 1


def test_iter_frames_uses_production_input_stream_and_stops() -> None:
    stop = threading.Event()
    sd = MagicMock()

    @contextlib.contextmanager
    def input_stream(**kwargs):
        kwargs["callback"](
            np.ones((480, 1), dtype=np.float32) * 0.1,
            480,
            None,
            None,
        )
        yield MagicMock()

    sd.InputStream.side_effect = input_stream
    stream = AudioPhraseStream(StreamConfig(sample_rate=16000, frame_ms=30))

    with patch("audio_stream.get_sounddevice", return_value=sd):
        packets = stream.iter_frames(stop)
        packet = next(packets)
        stop.set()
        with pytest.raises(StopIteration):
            next(packets)

    assert packet.samples.shape == (480,)
    assert packet.samples.dtype == np.float32
