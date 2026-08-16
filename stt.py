import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from stt_profiles import STTSelection, stt_model_dir, validate_selection

WhisperModel = None

STT_DEVICES = ["cpu", "cuda"]
GIGAAM_COMPLETE_MARKER = ".v2tts-complete"


class Transcriber(Protocol):
    requested_device: str
    actual_device: str
    fallback_reason: Optional[str]

    def transcribe_pcm16(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
    ) -> str: ...


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
        model_root: Optional[Path] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ):
        global WhisperModel
        if WhisperModel is None:
            from faster_whisper import WhisperModel as whisper_model_class

            WhisperModel = whisper_model_class

        if compute_type is None:
            compute_type = default_compute_type(device)
        self.requested_device = device
        self.actual_device = device
        self.compute_type = compute_type
        self.fallback_reason: Optional[str] = None
        warning_callback = on_warning or (lambda _: None)
        if model_root is None:
            model_root = stt_model_dir("whisper", model_size)

        model_kwargs = {
            "device": device,
            "compute_type": compute_type,
            "num_workers": 1,
            "download_root": str(model_root),
        }
        if device == "cpu":
            # Use available cores to reduce latency on CPU mode.
            model_kwargs["cpu_threads"] = max(1, os.cpu_count() or 1)

        try:
            self.model = WhisperModel(model_size, **model_kwargs)
        except Exception as exc:
            # Graceful CUDA fallback for packaged Windows builds without CUDA runtime.
            if device == "cuda":
                self.fallback_reason = str(exc)
                warning_callback(f"STT CUDA unavailable; using CPU: {exc}")
                self.model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                    num_workers=1,
                    cpu_threads=max(1, os.cpu_count() or 1),
                    download_root=str(model_root),
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


def _load_onnx_asr_model(
    model_name: str,
    path: Path,
    *,
    providers: list[str],
):
    import onnx_asr

    return onnx_asr.load_model(model_name, path, providers=providers)


def _available_onnx_providers() -> set[str]:
    import onnxruntime

    return set(onnxruntime.get_available_providers())


def _iter_onnx_sessions(root: object) -> Iterator[object]:
    seen: set[int] = set()
    stack = [root]
    scalar_types = (str, bytes, int, float, bool, Path, np.ndarray)
    while stack:
        current = stack.pop()
        if isinstance(current, scalar_types) or current is None:
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)

        get_providers = getattr(current, "get_providers", None)
        disable_fallback = getattr(current, "disable_fallback", None)
        if callable(get_providers) and callable(disable_fallback):
            yield current
            continue
        if isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, Sequence):
            stack.extend(current)
        else:
            try:
                stack.extend(vars(current).values())
            except TypeError:
                continue


def _require_cuda_sessions(model: object) -> None:
    sessions = list(_iter_onnx_sessions(model))
    if not sessions:
        raise RuntimeError("Could not verify CUDA execution sessions")
    for session in sessions:
        providers = session.get_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "ONNX Runtime did not activate CUDAExecutionProvider"
            )
        session.disable_fallback()


def _create_asr_resolver(model_name: str, model_dir: Path):
    from onnx_asr.loader import create_asr_resolver

    return create_asr_resolver(model_name, model_dir, offline=False)


def _ensure_gigaam_model_files(model_name: str, model_dir: Path) -> None:
    resolver = _create_asr_resolver(model_name, model_dir)
    resolver.resolve_model()


class GigaAMTranscriber:
    def __init__(
        self,
        model_name: str,
        device: str,
        model_dir: Path,
        on_warning: Optional[Callable[[str], None]] = None,
    ):
        self.requested_device = device
        self.actual_device = device
        self.fallback_reason: Optional[str] = None
        warning_callback = on_warning or (lambda _: None)
        if (
            device == "cuda"
            and "CUDAExecutionProvider" not in _available_onnx_providers()
        ):
            self.fallback_reason = "CUDAExecutionProvider is not available"
            self.actual_device = "cpu"
            warning_callback(
                "STT CUDA unavailable; using CPU: "
                f"{self.fallback_reason}"
            )
            self.model = _load_onnx_asr_model(
                model_name,
                model_dir,
                providers=["CPUExecutionProvider"],
            )
            return

        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        try:
            self.model = _load_onnx_asr_model(
                model_name,
                model_dir,
                providers=providers,
            )
            if device == "cuda":
                _require_cuda_sessions(self.model)
        except Exception as exc:
            if device != "cuda":
                raise
            self.fallback_reason = str(exc)
            self.actual_device = "cpu"
            warning_callback(f"STT CUDA unavailable; using CPU: {exc}")
            self.model = _load_onnx_asr_model(
                model_name,
                model_dir,
                providers=["CPUExecutionProvider"],
            )

    def transcribe_pcm16(
        self,
        pcm16: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        waveform = pcm16.astype(np.float32) / 32768.0
        return str(
            self.model.recognize(waveform, sample_rate=sample_rate)
        ).strip()


def create_transcriber(
    selection: STTSelection,
    on_warning: Optional[Callable[[str], None]] = None,
) -> Transcriber:
    validate_selection(selection)
    model_dir = stt_model_dir(selection.engine, selection.model)
    try:
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        if selection.engine == "gigaam":
            marker = model_dir / GIGAAM_COMPLETE_MARKER
            if not marker.is_file():
                _ensure_gigaam_model_files(selection.model, model_dir)
                model_dir.mkdir(parents=True, exist_ok=True)
                marker.touch()
            transcriber = GigaAMTranscriber(
                selection.model,
                selection.device,
                model_dir,
                on_warning,
            )
            return transcriber
        return WhisperTranscriber(
            model_size=selection.model,
            device=selection.device,
            language=selection.language,
            model_root=model_dir,
            on_warning=on_warning,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load STT model {selection.model} in {model_dir}: {exc}"
        ) from exc


def is_stt_model_ready(selection: STTSelection) -> bool:
    model_dir = stt_model_dir(selection.engine, selection.model)
    if selection.engine == "gigaam":
        return (model_dir / GIGAAM_COMPLETE_MARKER).is_file()
    return model_dir.exists()
