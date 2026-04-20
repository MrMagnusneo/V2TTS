import math
import queue
import threading
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
    device: Optional[int] = None


class AudioPhraseStream:
    """Audio stream + phrase splitting by silence."""

    def __init__(self, config: StreamConfig):
        self.config = config
        self._frames_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        frame = np.array(indata[:, 0], dtype=np.float32, copy=True)
        self._frames_q.put(frame)

    def iter_phrases(
        self,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[np.ndarray, None, None]:
        sd = get_sounddevice()
        frame_samples = int(self.config.sample_rate * self.config.frame_ms / 1000)
        silence_frames_limit = max(1, self.config.max_silence_ms // self.config.frame_ms)
        min_frames = max(1, self.config.min_phrase_ms // self.config.frame_ms)

        collecting = False
        silence_frames = 0
        phrase_frames = []

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
                    frame = self._frames_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                rms = math.sqrt(float(np.mean(np.square(frame))) + 1e-12)

                if not collecting:
                    if rms >= self.config.start_threshold:
                        collecting = True
                        silence_frames = 0
                        phrase_frames = [frame]
                    continue

                phrase_frames.append(frame)

                if rms < self.config.stop_threshold:
                    silence_frames += 1
                else:
                    silence_frames = 0

                enough_voice = len(phrase_frames) >= min_frames
                phrase_finished = silence_frames >= silence_frames_limit

                if enough_voice and phrase_finished:
                    audio = np.concatenate(phrase_frames)
                    audio = np.clip(audio, -1.0, 1.0)
                    pcm16 = (audio * 32767.0).astype(np.int16)
                    yield pcm16
                    collecting = False
                    silence_frames = 0
                    phrase_frames = []
