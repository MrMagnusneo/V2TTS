from unittest.mock import patch

import pytest

import main


def test_dependency_check_fails_with_install_instructions() -> None:
    def import_dependency(name: str) -> object:
        if name == "sounddevice":
            raise ImportError("missing sounddevice")
        return object()

    with patch("main.importlib.import_module", side_effect=import_dependency):
        with pytest.raises(RuntimeError) as error:
            main.check_runtime_dependencies()

    message = str(error.value)
    assert "sounddevice" in message
    assert "python -m pip install -r requirements.txt" in message


def test_dependency_check_reports_every_missing_package() -> None:
    def import_dependency(name: str) -> object:
        if name in {"faster_whisper", "soundfile"}:
            raise ImportError(f"missing {name}")
        return object()

    with patch("main.importlib.import_module", side_effect=import_dependency):
        with pytest.raises(RuntimeError) as error:
            main.check_runtime_dependencies()

    assert "faster_whisper" in str(error.value)
    assert "soundfile" in str(error.value)
