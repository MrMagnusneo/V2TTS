from typing import Optional

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
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter

    def transcribe_pcm16(self, pcm16: np.ndarray, sample_rate: int = 16000) -> str:
        if sample_rate != 16000:
            raise ValueError("WhisperTranscriber expects 16kHz audio.")

        audio = pcm16.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return "".join(seg.text for seg in segments).strip()
