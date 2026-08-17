from pathlib import Path

import pytest

from stt_profiles import (
    STTSelection,
    default_selection,
    engines_for_language,
    models_for,
    stt_model_dir,
    validate_selection,
)


def test_streaming_profiles_are_language_specific() -> None:
    from stt_profiles import (
        StreamingSTTSelection,
        default_streaming_selection,
        validate_streaming_selection,
    )

    assert default_streaming_selection("ru") == StreamingSTTSelection(
        "ru", "sherpa_streaming_ru_t_one"
    )
    assert default_streaming_selection("en") == StreamingSTTSelection(
        "en", "sherpa_streaming_en_zipformer_20m"
    )
    with pytest.raises(ValueError, match="not available"):
        validate_streaming_selection(
            StreamingSTTSelection("en", "sherpa_streaming_ru_t_one")
        )


def test_russian_defaults_to_gigaam_e2e_rnnt() -> None:
    assert default_selection("ru") == STTSelection(
        language="ru",
        engine="gigaam",
        model="gigaam-v3-e2e-rnnt",
        device="cpu",
    )


def test_english_exposes_only_whisper() -> None:
    assert engines_for_language("en") == ("whisper",)
    assert models_for("en", "whisper") == ("small", "medium", "large-v3")


def test_gigaam_is_rejected_for_english() -> None:
    with pytest.raises(ValueError, match="not available for language en"):
        validate_selection(
            STTSelection("en", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
        )


def test_model_is_rejected_for_the_wrong_engine() -> None:
    with pytest.raises(ValueError, match="not available for ru/whisper"):
        validate_selection(
            STTSelection("ru", "whisper", "gigaam-v3-e2e-rnnt", "cpu")
        )


def test_windows_model_path_is_outside_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("stt_profiles.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert stt_model_dir("whisper", "medium") == (
        tmp_path / "V2TTS" / "models" / "whisper" / "medium"
    )
