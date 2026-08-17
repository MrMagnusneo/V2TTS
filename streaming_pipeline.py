from __future__ import annotations

import queue
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from audio_backend import get_sounddevice, get_soundfile
from audio_stream import AudioPhraseStream, StreamConfig
from streaming_models import (
    STREAMING_MODEL_PROFILES,
    ModelDownloadCancelled,
    ensure_streaming_model,
    is_streaming_model_ready,
)
from streaming_stt import SherpaStreamingTranscriber, StableTextCommitter


class CombinedCancelEvent:
    def __init__(self, global_stop, local_cancel: threading.Event) -> None:
        self._global_stop = global_stop
        self._local_cancel = local_cancel

    def is_set(self) -> bool:
        return self._global_stop.is_set() or self._local_cancel.is_set()


@dataclass(frozen=True)
class _TTSJob:
    generation: int
    text: str
    cancel_event: CombinedCancelEvent


_SENTINEL = object()


class StreamingInitializationError(RuntimeError):
    pass


class StreamingTTSWorker:
    def __init__(
        self,
        global_stop,
        process: Callable[[str, CombinedCancelEvent], None],
        max_queue: int = 4,
        *,
        autostart: bool = True,
    ) -> None:
        self._global_stop = global_stop
        self._process = process
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, max_queue))
        self._lock = threading.Lock()
        self._generation = 0
        self._local_cancel = threading.Event()
        self._closed = False
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="V2TTS-streaming-tts",
            daemon=True,
        )
        if autostart:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._thread.start()

    def begin_utterance(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("Streaming TTS worker is closed")
            self._local_cancel.set()
            self._local_cancel = threading.Event()
            self._generation += 1
            generation = self._generation
            self._drain_queue_locked()
            return generation

    def submit(self, generation: int, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if self._closed or generation != self._generation:
                return
            job = _TTSJob(
                generation=generation,
                text=text,
                cancel_event=CombinedCancelEvent(
                    self._global_stop,
                    self._local_cancel,
                ),
            )
            while True:
                try:
                    self._queue.put_nowait(job)
                    return
                except queue.Full:
                    pending_text = []
                    while True:
                        try:
                            pending = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        else:
                            self._queue.task_done()
                            if (
                                pending is not _SENTINEL
                                and pending.generation == generation
                                and not pending.cancel_event.is_set()
                            ):
                                pending_text.append(pending.text)
                    if pending_text:
                        job = _TTSJob(
                            generation=generation,
                            text=" ".join([*pending_text, job.text]),
                            cancel_event=job.cancel_event,
                        )

    def close(self, *, drain: bool = False, timeout: float = 1.3) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not drain:
                self._local_cancel.set()
                self._drain_queue_locked()
            if not self._started:
                self._started = True
                self._thread.start()
            self._queue.put(_SENTINEL)
        self._thread.join(timeout=timeout)

    def _drain_queue_locked(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is _SENTINEL:
                    return
                with self._lock:
                    current = (
                        not self._closed or not job.cancel_event.is_set()
                    ) and job.generation == self._generation
                if current and not job.cancel_event.is_set():
                    self._process(job.text, job.cancel_event)
            finally:
                self._queue.task_done()


def select_input_sample_rate(sd, device_index):
    from pipeline import select_input_sample_rate as select_rate

    return select_rate(sd, device_index)


def _rms(samples: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(samples))) + 1e-12)


def run_streaming_pipeline(config, stop_event, emit) -> None:
    from pipeline import synthesize_and_play_text

    selection = config.streaming_stt
    if selection.language != config.stt.language:
        raise StreamingInitializationError(
            "Streaming and after-phrase profiles must use the same language"
        )
    try:
        profile = STREAMING_MODEL_PROFILES[selection.profile]
    except KeyError as exc:
        raise StreamingInitializationError(
            f"Unknown streaming profile: {selection.profile}"
        ) from exc
    if profile.language != selection.language:
        raise StreamingInitializationError(
            f"Streaming profile {selection.profile} is not available for "
            f"language {selection.language}"
        )

    sd = get_sounddevice()
    sf = get_soundfile()
    sample_rate = select_input_sample_rate(sd, config.input_device)
    stream_config = StreamConfig(
        sample_rate=sample_rate,
        channels=1,
        frame_ms=30,
        min_phrase_ms=300,
        max_silence_ms=700,
        pre_roll_ms=200,
        device=config.input_device,
    )
    frame_stream = AudioPhraseStream(stream_config)

    ready = is_streaming_model_ready(selection.profile)
    emit(
        "status",
        f"{'Loading' if ready else 'Downloading'} streaming STT model "
        f"{selection.profile}...",
    )
    last_percent = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_percent
        percent = min(100, int(completed * 100 / max(1, total)))
        if percent != last_percent:
            last_percent = percent
            emit(
                "status",
                f"Downloading streaming STT model {selection.profile}: "
                f"{percent}%",
            )

    try:
        model_dir = ensure_streaming_model(
            selection.profile,
            stop_event=stop_event,
            on_progress=progress,
        )
        if stop_event.is_set():
            return
        transcriber = SherpaStreamingTranscriber(profile, model_dir)
    except ModelDownloadCancelled:
        if stop_event.is_set():
            return
        raise StreamingInitializationError("model download cancelled")
    except Exception as exc:
        raise StreamingInitializationError(str(exc)) from exc

    def process_tts(text: str, cancel_event) -> None:
        try:
            synthesize_and_play_text(
                text,
                sd,
                sf,
                config,
                cancel_event,
                on_ready=lambda metrics: emit(
                    "status",
                    f"Playing... (tts={metrics.engine}, "
                    f"tts_ms={metrics.tts_ms})",
                ),
            )
        except Exception as exc:
            if not cancel_event.is_set():
                emit("error", f"TTS/Playback failed: {exc}")

    tts_worker = StreamingTTSWorker(stop_event, process_tts)
    committer = StableTextCommitter()
    speech_active = False
    generation = 0
    silence_ms = 0
    pause_flushed = False
    last_partial = ""
    last_dropped = 0
    speech_started_at: float | None = None
    first_partial_at: float | None = None
    current_queue_ms = 0
    current_dropped_frames = 0

    def submit_chunks(chunks: list[str]) -> None:
        for chunk in chunks:
            committed_at = time.perf_counter()
            started_at = (
                speech_started_at
                if speech_started_at is not None
                else committed_at
            )
            partial_at = (
                first_partial_at
                if first_partial_at is not None
                else committed_at
            )
            emit("text", chunk)
            emit(
                "status",
                "Streaming STT: "
                f"partial_ms={max(0, int(round((partial_at - started_at) * 1000)))}, "
                f"commit_ms={max(0, int(round((committed_at - started_at) * 1000)))}, "
                f"queue_ms={current_queue_ms}, "
                f"dropped_frames={current_dropped_frames}",
            )
            tts_worker.submit(generation, chunk)

    emit("state", "listening")
    emit(
        "status",
        f"Listening... (STT: engine=sherpa-onnx, "
        f"language={selection.language}, model={selection.profile}, "
        f"sr={sample_rate})",
    )
    try:
        for packet in frame_stream.iter_frames(stop_event=stop_event):
            if stop_event.is_set():
                break
            metrics = frame_stream.metrics()
            current_queue_ms = metrics.queue_ms
            current_dropped_frames = metrics.dropped_frames
            if metrics.dropped_frames > last_dropped:
                emit(
                    "status",
                    "Audio overload: "
                    f"dropped_frames={metrics.dropped_frames}, "
                    f"queue_ms={metrics.queue_ms}",
                )
            last_dropped = metrics.dropped_frames

            level = _rms(packet.samples)
            if not speech_active and level >= stream_config.start_threshold:
                speech_active = True
                generation = tts_worker.begin_utterance()
                silence_ms = 0
                pause_flushed = False
                speech_started_at = time.perf_counter()
                first_partial_at = None
                emit("state", "recognizing")
            if speech_active:
                if level < stream_config.stop_threshold:
                    silence_ms += stream_config.frame_ms
                else:
                    silence_ms = 0
                    pause_flushed = False

            result = transcriber.accept_audio(packet.samples, sample_rate)
            if speech_active and result.text:
                if result.text != last_partial:
                    last_partial = result.text
                    if first_partial_at is None:
                        first_partial_at = time.perf_counter()
                    emit("partial", result.text)
                submit_chunks(committer.observe(result.text))

            if speech_active and silence_ms >= 350 and not pause_flushed:
                submit_chunks(committer.flush_pause())
                pause_flushed = True

            if speech_active and result.endpoint:
                final_result = transcriber.finish()
                submit_chunks(committer.finish(final_result.text or result.text))
                transcriber.reset()
                speech_active = False
                silence_ms = 0
                pause_flushed = False
                last_partial = ""
                speech_started_at = None
                first_partial_at = None
                emit("state", "listening")
                emit("partial", "")

        if speech_active and not stop_event.is_set():
            final_result = transcriber.finish()
            submit_chunks(committer.finish(final_result.text or last_partial))
            emit("partial", "")
    finally:
        tts_worker.close(drain=not stop_event.is_set())
        transcriber.close()
