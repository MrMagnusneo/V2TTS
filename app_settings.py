from __future__ import annotations

import json
import os
from pathlib import Path

from stt_profiles import (
    LANGUAGE_LABELS,
    STTSelection,
    StreamingSTTSelection,
    default_selection,
    default_streaming_selection,
    user_data_root,
    validate_selection,
    validate_streaming_selection,
)


def settings_path() -> Path:
    return user_data_root() / "settings.json"


def _defaults(language: str = "ru") -> dict:
    phrase = default_selection(language)
    streaming = default_streaming_selection(language)
    return {
        "stt_mode": "streaming",
        "streaming_language": streaming.language,
        "streaming_profile": streaming.profile,
        "stt_language": phrase.language,
        "stt_engine": phrase.engine,
        "stt_model": phrase.model,
        "stt_device": phrase.device,
        "input_device_label": "",
        "output_device_label": "",
        "auto_tts_model": True,
        "manual_tts_model": "ru_tts",
        "tts_root": None,
    }


def load_app_settings(path: Path | None = None) -> dict:
    target = Path(path) if path is not None else settings_path()
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            loaded = {}
    except (FileNotFoundError, OSError, ValueError, TypeError):
        loaded = {}

    phrase_language = loaded.get("stt_language", "ru")
    if phrase_language not in LANGUAGE_LABELS:
        phrase_language = "ru"
    streaming_language = loaded.get("streaming_language", phrase_language)
    if streaming_language not in LANGUAGE_LABELS:
        streaming_language = phrase_language

    settings = _defaults(phrase_language)
    settings.update(loaded)
    settings["stt_mode"] = (
        settings["stt_mode"]
        if settings["stt_mode"] in {"streaming", "after_phrase"}
        else "streaming"
    )

    phrase = STTSelection(
        language=phrase_language,
        engine=str(settings.get("stt_engine", "")),
        model=str(settings.get("stt_model", "")),
        device=str(settings.get("stt_device", "cpu")),
    )
    try:
        validate_selection(phrase)
    except ValueError:
        phrase = default_selection(phrase_language)
    settings.update(
        {
            "stt_language": phrase.language,
            "stt_engine": phrase.engine,
            "stt_model": phrase.model,
            "stt_device": phrase.device,
        }
    )

    streaming = StreamingSTTSelection(
        language=streaming_language,
        profile=str(settings.get("streaming_profile", "")),
    )
    try:
        validate_streaming_selection(streaming)
    except ValueError:
        streaming = default_streaming_selection(streaming_language)
    settings["streaming_language"] = streaming.language
    settings["streaming_profile"] = streaming.profile
    settings["auto_tts_model"] = bool(settings.get("auto_tts_model", True))
    return settings


def save_app_settings(settings: dict, path: Path | None = None) -> None:
    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(settings, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
