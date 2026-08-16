import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import pipeline
from pipeline import RunConfig, _process_phrase
from stt_profiles import STTSelection


class RestrictedOutputDevice:
    def __init__(self, supported_rate: int) -> None:
        self.supported_rate = supported_rate
        self.played: list[tuple[np.ndarray, int, int | None]] = []
        self.waited = False
        self.output_chunks: list[np.ndarray] = []
        self.output_rate: int | None = None
        self.output_device: int | None = None

    def query_devices(self, device: int | None, kind: str) -> dict:
        assert kind == "output"
        return {"default_samplerate": float(self.supported_rate)}

    def check_output_settings(
        self,
        *,
        device: int | None,
        channels: int,
        samplerate: int,
        dtype: str,
    ) -> None:
        if samplerate != self.supported_rate:
            raise RuntimeError("Invalid sample rate")

    def play(self, data: np.ndarray, samplerate: int, device: int | None) -> None:
        if samplerate != self.supported_rate:
            raise RuntimeError("Invalid sample rate")
        self.played.append((data, samplerate, device))

    def wait(self) -> None:
        self.waited = True

    def OutputStream(
        self,
        *,
        samplerate: int,
        channels: int,
        dtype: str,
        device: int | None,
    ):
        assert channels == 1
        assert dtype == "float32"
        assert samplerate == self.supported_rate
        self.output_rate = samplerate
        self.output_device = device
        owner = self

        class Stream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def write(self, data: np.ndarray) -> None:
                owner.output_chunks.append(np.array(data, copy=True))

        return Stream()


def test_prepare_output_resamples_to_device_default_rate() -> None:
    device = RestrictedOutputDevice(supported_rate=48000)
    source = np.linspace(-1.0, 1.0, 2205, dtype=np.float32)

    prepared, sample_rate = pipeline.prepare_audio_for_output(
        device,
        source,
        source_rate=22050,
        device_index=23,
    )

    assert sample_rate == 48000
    assert prepared.shape == (4800,)
    assert prepared.dtype == np.float32
    assert prepared[0] == -1.0
    assert prepared[-1] > 0.99


def test_downsampling_stereo_suppresses_frequencies_above_new_nyquist() -> None:
    device = RestrictedOutputDevice(supported_rate=16000)
    source_rate = 22050
    time_axis = np.arange(source_rate, dtype=np.float32) / source_rate
    high_tone = np.sin(2 * np.pi * 10000 * time_axis).astype(np.float32)
    stereo = np.column_stack([high_tone, high_tone])

    prepared, sample_rate = pipeline.prepare_audio_for_output(
        device,
        stereo,
        source_rate=source_rate,
        device_index=23,
    )

    assert sample_rate == 16000
    assert prepared.shape == (16000, 2)
    assert float(np.sqrt(np.mean(np.square(prepared)))) < 0.05


def test_phrase_pipeline_plays_at_supported_virtual_cable_rate() -> None:
    config = RunConfig(
        input_device=3,
        output_device=23,
        stt=STTSelection("ru", "whisper", "medium", "cuda"),
        auto_tts_model=False,
        manual_tts_model="ru_tts",
        tts_root=None,
    )
    errors: list[str] = []
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "Один"
    soundfile = MagicMock()
    soundfile.read.return_value = (
        np.linspace(-1.0, 1.0, 2205, dtype=np.float32),
        22050,
    )
    device = RestrictedOutputDevice(supported_rate=48000)

    def synthesize(**kwargs) -> str:
        Path(kwargs["out_wav"]).write_bytes(b"RIFF")
        return "ru_tts"

    def emit(kind: str, payload: str) -> None:
        if kind == "error":
            errors.append(payload)

    with patch("pipeline.synthesize_text", side_effect=synthesize):
        _process_phrase(
            np.zeros(4410, dtype=np.int16),
            transcriber,
            44100,
            "STT test",
            device,
            soundfile,
            config,
            threading.Event(),
            emit,
            queue_ms=0,
            phrase_age_ms=0,
            dropped_frames=0,
        )

    assert errors == []
    played_audio = np.concatenate(device.output_chunks, axis=0)
    assert device.output_rate == 48000
    assert device.output_device == 23
    assert played_audio.shape == (4800, 1)
