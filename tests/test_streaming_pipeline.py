import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio_stream import FramePacket, StreamMetrics
from pipeline import RunConfig
from streaming_pipeline import StreamingTTSWorker, run_streaming_pipeline
from streaming_stt import StreamingResult
from stt_profiles import STTSelection, StreamingSTTSelection


class BlockingProcessor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self.played: list[str] = []
        self.first_cancel_event = None

    def __call__(self, text: str, cancel_event) -> None:
        self.calls.append(text)
        if self.first_cancel_event is None:
            self.first_cancel_event = cancel_event
            self.started.set()
            assert self.release.wait(timeout=2)
        if not cancel_event.is_set():
            self.played.append(text)


def test_new_utterance_cancels_active_and_discards_queued_text() -> None:
    processor = BlockingProcessor()
    worker = StreamingTTSWorker(
        global_stop=threading.Event(),
        process=processor,
    )
    old_generation = worker.begin_utterance()
    worker.submit(old_generation, "старый один")
    worker.submit(old_generation, "старый два")
    assert processor.started.wait(timeout=1)

    new_generation = worker.begin_utterance()
    worker.submit(new_generation, "новый текст")
    processor.release.set()
    worker.close(drain=True)

    assert processor.first_cancel_event.is_set()
    assert processor.calls == ["старый один", "новый текст"]
    assert processor.played == ["новый текст"]


def test_stale_generation_is_rejected_and_current_text_is_coalesced() -> None:
    calls = []
    worker = StreamingTTSWorker(
        global_stop=threading.Event(),
        process=lambda text, cancel: calls.append(text),
        max_queue=2,
        autostart=False,
    )
    first = worker.begin_utterance()
    second = worker.begin_utterance()
    worker.submit(first, "stale")
    worker.submit(second, "one")
    worker.submit(second, "two")
    worker.submit(second, "three")
    worker.start()
    worker.close(drain=True)

    assert " ".join(calls).split() == ["one", "two", "three"]


def test_close_without_drain_cancels_current_work_within_bound() -> None:
    started = threading.Event()
    observed_cancel = threading.Event()

    def process(_text, cancel_event):
        started.set()
        deadline = time.monotonic() + 1
        while not cancel_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        if cancel_event.is_set():
            observed_cancel.set()

    worker = StreamingTTSWorker(threading.Event(), process)
    generation = worker.begin_utterance()
    worker.submit(generation, "cancel me")
    assert started.wait(timeout=1)

    worker.close(drain=False)

    assert observed_cancel.wait(timeout=1)


class CapturingTTSWorker:
    instances = []

    def __init__(self, *args, **kwargs) -> None:
        self.generation = 0
        self.submitted = []
        self.closed = None
        self.__class__.instances.append(self)

    def begin_utterance(self) -> int:
        self.generation += 1
        return self.generation

    def submit(self, generation: int, text: str) -> None:
        self.submitted.append((generation, text))

    def close(self, *, drain=False, **kwargs) -> None:
        self.closed = drain


def _streaming_config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=7,
        stt=STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
        stt_mode="streaming",
        streaming_stt=StreamingSTTSelection(
            "ru", "sherpa_streaming_ru_t_one"
        ),
    )


def test_streaming_pipeline_emits_partial_and_commits_each_word_once() -> None:
    CapturingTTSWorker.instances.clear()
    recognizer = MagicMock()
    recognizer.accept_audio.side_effect = [
        StreamingResult("это потоковый", False),
        StreamingResult("это потоковый тест работает", False),
        StreamingResult("это потоковый тест работает хорошо", False),
        StreamingResult("это потоковый тест работает хорошо сейчас", True),
    ]
    recognizer.finish.return_value = StreamingResult(
        "это потоковый тест работает хорошо сейчас", True
    )
    frame_stream = MagicMock()
    frame_stream.iter_frames.return_value = iter(
        [
            FramePacket(np.ones(480, np.float32) * 0.1, float(index))
            for index in range(4)
        ]
    )
    frame_stream.metrics.return_value = StreamMetrics(0, 0, 0)
    events = []

    with (
        patch("streaming_pipeline.get_sounddevice", return_value=MagicMock()),
        patch("streaming_pipeline.get_soundfile", return_value=MagicMock()),
        patch("streaming_pipeline.select_input_sample_rate", return_value=16000),
        patch("streaming_pipeline.is_streaming_model_ready", return_value=True),
        patch(
            "streaming_pipeline.ensure_streaming_model",
            return_value=Path("model"),
        ),
        patch(
            "streaming_pipeline.SherpaStreamingTranscriber",
            return_value=recognizer,
        ),
        patch(
            "streaming_pipeline.AudioPhraseStream",
            return_value=frame_stream,
        ),
        patch(
            "streaming_pipeline.StreamingTTSWorker",
            CapturingTTSWorker,
        ),
    ):
        run_streaming_pipeline(
            _streaming_config(),
            threading.Event(),
            lambda kind, payload: events.append((kind, payload)),
        )

    worker = CapturingTTSWorker.instances[0]
    assert [text for _, text in worker.submitted] == [
        "это потоковый тест работает хорошо сейчас"
    ]
    assert ("text", "это потоковый тест работает хорошо сейчас") in events
    assert any(kind == "partial" for kind, _ in events)
    metric_statuses = [
        payload
        for kind, payload in events
        if kind == "status" and payload.startswith("Streaming STT:")
    ]
    assert len(metric_statuses) == 1
    assert "partial_ms=" in metric_statuses[0]
    assert "commit_ms=" in metric_statuses[0]
    assert "queue_ms=0" in metric_statuses[0]
    assert "dropped_frames=0" in metric_statuses[0]
    assert events[-1] == ("partial", "")
    assert worker.closed is True
    recognizer.reset.assert_called_once_with()
    recognizer.close.assert_called_once_with()


def test_pause_flushes_short_text_after_350_ms() -> None:
    CapturingTTSWorker.instances.clear()
    recognizer = MagicMock()
    recognizer.accept_audio.side_effect = [
        StreamingResult("короткая фраза", False),
        StreamingResult("короткая фраза", False),
        *[StreamingResult("короткая фраза", False) for _ in range(12)],
    ]
    recognizer.finish.return_value = StreamingResult("короткая фраза", True)
    loud = FramePacket(np.ones(480, np.float32) * 0.1, 0.0)
    quiet = FramePacket(np.zeros(480, np.float32), 0.03)
    frame_stream = MagicMock()
    frame_stream.iter_frames.return_value = iter([loud, loud, *([quiet] * 12)])
    frame_stream.metrics.return_value = StreamMetrics(0, 0, 0)

    with (
        patch("streaming_pipeline.get_sounddevice", return_value=MagicMock()),
        patch("streaming_pipeline.get_soundfile", return_value=MagicMock()),
        patch("streaming_pipeline.select_input_sample_rate", return_value=16000),
        patch("streaming_pipeline.is_streaming_model_ready", return_value=True),
        patch("streaming_pipeline.ensure_streaming_model", return_value=Path("model")),
        patch("streaming_pipeline.SherpaStreamingTranscriber", return_value=recognizer),
        patch("streaming_pipeline.AudioPhraseStream", return_value=frame_stream),
        patch("streaming_pipeline.StreamingTTSWorker", CapturingTTSWorker),
    ):
        run_streaming_pipeline(_streaming_config(), threading.Event(), MagicMock())

    assert [text for _, text in CapturingTTSWorker.instances[0].submitted] == [
        "короткая фраза"
    ]
