import json
from pathlib import Path

from app_settings import load_app_settings, save_app_settings


def test_missing_settings_default_to_russian_streaming(tmp_path: Path) -> None:
    settings = load_app_settings(tmp_path / "settings.json")

    assert settings["stt_mode"] == "streaming"
    assert settings["streaming_language"] == "ru"
    assert settings["streaming_profile"] == "sherpa_streaming_ru_t_one"
    assert settings["stt_language"] == "ru"
    assert settings["stt_engine"] == "gigaam"


def test_legacy_phrase_settings_migrate_without_losing_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "stt_language": "en",
                "stt_engine": "whisper",
                "stt_model": "small",
                "stt_device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    settings = load_app_settings(path)

    assert settings["stt_mode"] == "streaming"
    assert settings["streaming_language"] == "en"
    assert settings["streaming_profile"] == "sherpa_streaming_en_zipformer_20m"
    assert settings["stt_model"] == "small"


def test_invalid_profiles_fall_back_within_requested_language(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "streaming_language": "en",
                "streaming_profile": "sherpa_streaming_ru_t_one",
                "stt_language": "en",
                "stt_engine": "gigaam",
                "stt_model": "broken",
            }
        ),
        encoding="utf-8",
    )

    settings = load_app_settings(path)

    assert settings["streaming_profile"] == "sherpa_streaming_en_zipformer_20m"
    assert settings["stt_engine"] == "whisper"
    assert settings["stt_model"] == "medium"


def test_save_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = load_app_settings(path)
    settings["input_device_label"] = "Microphone"

    save_app_settings(settings, path)

    assert load_app_settings(path)["input_device_label"] == "Microphone"
    assert not (tmp_path / "settings.json.tmp").exists()
