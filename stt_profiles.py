import os
import sys
from dataclasses import dataclass
from pathlib import Path


LANGUAGE_LABELS = {"ru": "Russian", "en": "English"}
ENGINE_LABELS = {"gigaam": "GigaAM", "whisper": "Whisper"}
MODEL_LABELS = {
    "gigaam-v3-e2e-rnnt": "GigaAM v3 E2E RNN-T",
    "gigaam-v3-e2e-ctc": "GigaAM v3 E2E CTC",
    "small": "Whisper small",
    "medium": "Whisper medium",
    "large-v3": "Whisper large-v3",
}

_MODELS = {
    ("ru", "gigaam"): ("gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"),
    ("ru", "whisper"): ("small", "medium", "large-v3"),
    ("en", "whisper"): ("small", "medium", "large-v3"),
}


@dataclass(frozen=True)
class STTSelection:
    language: str
    engine: str
    model: str
    device: str


@dataclass(frozen=True)
class StreamingSTTSelection:
    language: str
    profile: str


_STREAMING_BY_LANGUAGE = {
    "ru": ("sherpa_streaming_ru_t_one",),
    "en": ("sherpa_streaming_en_zipformer_20m",),
}


def engines_for_language(language: str) -> tuple[str, ...]:
    if language not in LANGUAGE_LABELS:
        raise ValueError(f"Unknown STT language: {language}")
    return tuple(engine for lang, engine in _MODELS if lang == language)


def models_for(language: str, engine: str) -> tuple[str, ...]:
    try:
        return _MODELS[(language, engine)]
    except KeyError as exc:
        raise ValueError(
            f"STT engine {engine} is not available for language {language}"
        ) from exc


def validate_selection(selection: STTSelection) -> None:
    if selection.device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown STT device: {selection.device}")
    if selection.model not in models_for(selection.language, selection.engine):
        raise ValueError(
            f"STT model {selection.model} is not available for "
            f"{selection.language}/{selection.engine}"
        )


def default_selection(language: str = "ru") -> STTSelection:
    if language == "ru":
        return STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
    if language == "en":
        return STTSelection("en", "whisper", "medium", "cpu")
    raise ValueError(f"Unknown STT language: {language}")


def default_streaming_selection(
    language: str = "ru",
) -> StreamingSTTSelection:
    try:
        return StreamingSTTSelection(
            language=language,
            profile=_STREAMING_BY_LANGUAGE[language][0],
        )
    except KeyError as exc:
        raise ValueError(f"Unknown STT language: {language}") from exc


def validate_streaming_selection(selection: StreamingSTTSelection) -> None:
    if selection.profile not in _STREAMING_BY_LANGUAGE.get(
        selection.language,
        (),
    ):
        raise ValueError(
            f"Streaming profile {selection.profile} is not available for "
            f"language {selection.language}"
        )


def user_data_root() -> Path:
    if sys.platform == "win32":
        default_base = Path.home() / "AppData" / "Local"
        base = Path(os.getenv("LOCALAPPDATA", str(default_base)))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        default_base = Path.home() / ".local" / "share"
        base = Path(os.getenv("XDG_DATA_HOME", str(default_base)))
    return base / "V2TTS"


def stt_model_dir(engine: str, model: str) -> Path:
    return user_data_root() / "models" / engine / model
