import os
import threading

import pytest

from audio_backend import get_sounddevice
from audio_stream import AudioPhraseStream, StreamConfig
from pipeline import select_input_sample_rate


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("V2TTS_REAL_AUDIO_TEST") != "1",
    reason="set V2TTS_REAL_AUDIO_TEST=1 to exercise the default input device",
)
def test_default_input_device_delivers_bounded_audio_frames() -> None:
    sounddevice = get_sounddevice()
    sample_rate = select_input_sample_rate(sounddevice, None)
    config = StreamConfig(sample_rate=sample_rate, frame_ms=20, max_buffer_ms=200)
    stream = AudioPhraseStream(config)
    stop_event = threading.Event()
    callback_seen = threading.Event()
    iterator_errors: list[Exception] = []
    original_callback = stream._callback

    def observed_callback(*args, **kwargs) -> None:
        original_callback(*args, **kwargs)
        callback_seen.set()

    stream._callback = observed_callback

    def consume_phrases() -> None:
        try:
            for _ in stream.iter_phrases(stop_event=stop_event):
                pass
        except Exception as exc:
            iterator_errors.append(exc)

    consumer = threading.Thread(target=consume_phrases)
    consumer.start()
    callback_received = callback_seen.wait(timeout=0.75)
    stop_event.set()
    consumer.join(timeout=0.25)

    metrics = stream.metrics()
    assert not iterator_errors
    assert callback_received
    assert not consumer.is_alive()
    assert metrics.queue_depth_frames <= stream._frames_q.maxsize
