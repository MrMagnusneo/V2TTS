import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

from audio_backend import get_sounddevice


@dataclass
class StreamConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    start_threshold: float = 0.015
    stop_threshold: float = 0.01
    min_phrase_ms: int = 300
    max_silence_ms: int = 700
    max_buffer_ms: int = 1500
    pre_roll_ms: int = 200
    device: Optional[int] = None


@dataclass(frozen=True)
class FramePacket:
    samples: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class CapturedPhrase:
    pcm16: np.ndarray
    ended_at: float


@dataclass(frozen=True)
class StreamMetrics:
    queue_depth_frames: int
    queue_ms: int
    dropped_frames: int
    callback_statuses: int = 0


class AudioPhraseStream:
    """Audio stream + phrase splitting by silence."""

    def __init__(self, config: StreamConfig):
        self.config = config
        max_buffer_frames = max(
            1,
            math.ceil(self.config.max_buffer_ms / self.config.frame_ms),
        )
        self._frames_q: "queue.Queue[FramePacket]" = queue.Queue(
            maxsize=max_buffer_frames
        )
        self._dropped_frames = 0
        self._callback_statuses = 0
        self.silence_frames_limit = max(
            1,
            math.ceil(self.config.max_silence_ms / self.config.frame_ms),
        )
        self.min_frames = max(1, self.config.min_phrase_ms // self.config.frame_ms)
        self.collecting = False
        self.silence_frames = 0
        self.phrase_frames = []
        self._pre_roll = deque(
            maxlen=max(
                0,
                math.ceil(self.config.pre_roll_ms / self.config.frame_ms),
            )
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            self._callback_statuses += 1
        frame = np.array(indata[:, 0], dtype=np.float32, copy=True)
        packet = FramePacket(samples=frame, captured_at=time.monotonic())

        while True:
            try:
                self._frames_q.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self._frames_q.get_nowait()
                except queue.Empty:
                    continue
                self._dropped_frames += 1

    def metrics(self) -> StreamMetrics:
        queue_depth_frames = self._frames_q.qsize()
        return StreamMetrics(
            queue_depth_frames=queue_depth_frames,
            queue_ms=queue_depth_frames * self.config.frame_ms,
            dropped_frames=self._dropped_frames,
            callback_statuses=self._callback_statuses,
        )

    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rms = math.sqrt(float(np.mean(np.square(frame))) + 1e-12)

        if not self.collecting:
            if rms >= self.config.start_threshold:
                self.collecting = True
                self.silence_frames = 0
                self.phrase_frames = [*self._pre_roll, frame]
                self._pre_roll.clear()
            else:
                self._pre_roll.append(frame)
            return None

        self.phrase_frames.append(frame)

        if rms < self.config.stop_threshold:
            self.silence_frames += 1
        else:
            self.silence_frames = 0

        enough_voice = len(self.phrase_frames) >= self.min_frames
        phrase_finished = self.silence_frames >= self.silence_frames_limit

        if enough_voice and phrase_finished:
            audio = np.concatenate(self.phrase_frames)
            audio = np.clip(audio, -1.0, 1.0)
            pcm16 = (audio * 32767.0).astype(np.int16)

            self.collecting = False
            self.silence_frames = 0
            self.phrase_frames = []
            self._pre_roll.clear()

            return pcm16

        return None

    def iter_phrases(
        self,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[CapturedPhrase, None, None]:
        sd = get_sounddevice()
        frame_samples = int(self.config.sample_rate * self.config.frame_ms / 1000)

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=frame_samples,
                device=self.config.device,
                callback=self._callback,
            ):
                while True:
                    if stop_event is not None and stop_event.is_set():
                        return

                    try:
                        packet = self._frames_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    pcm16 = self._process_frame(packet.samples)
                    if pcm16 is not None:
                        yield CapturedPhrase(
                            pcm16=pcm16,
                            ended_at=packet.captured_at,
                        )
        except Exception as exc:
            raise RuntimeError(
                f"Could not use input device {self.config.device} at "
                f"{self.config.sample_rate} Hz with "
                f"{self.config.channels} channel(s): {exc}"
            ) from exc
