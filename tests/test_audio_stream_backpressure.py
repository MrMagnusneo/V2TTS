import numpy as np
from unittest.mock import patch

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
