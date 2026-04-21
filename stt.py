from typing import Optional
import os

import numpy as np
from faster_whisper import WhisperModel

STT_MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]
STT_DEVICES = ["cpu", "cuda"]


def default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


class WhisperTranscriber:
    """Only Whisper transcription logic."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        if compute_type is None:
            compute_type = default_compute_type(device)
        self.requested_device = device
        self.actual_device = device
        self.compute_type = compute_type

        model_kwargs = {
            "device": device,
            "compute_type": compute_type,
            "num_workers": 1,
        }
        if device == "cpu":
            # Use available cores to reduce latency on CPU mode.
            model_kwargs["cpu_threads"] = max(1, os.cpu_count() or 1)

        try:
            self.model = WhisperModel(model_size, **model_kwargs)
        except Exception:
            # Graceful CUDA fallback for packaged Windows builds without CUDA runtime.
            if device == "cuda":
                self.model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    num_workers=1,
                    cpu_threads=max(1, os.cpu_count() or 1),
                )
                self.actual_device = "cpu"
                self.compute_type = "int8"
            else:
                raise
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe_pcm16(self, pcm16: np.ndarray, sample_rate: int = 16000) -> str:
        audio = pcm16.astype(np.float32) / 32768.0
        if sample_rate != 16000:
            audio = self._resample_audio(audio, sample_rate, 16000)
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()

    @staticmethod
    def _resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("Sample rates must be positive.")
        if src_rate == dst_rate or len(audio) == 0:
            return audio

        src_len = len(audio)
        dst_len = max(1, int(round(src_len * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float32)
        return np.interp(dst_x, src_x, audio).astype(np.float32)
