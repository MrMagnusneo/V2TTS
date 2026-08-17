from unittest.mock import MagicMock, patch

import pytest

import main
from main import AppController
from stt_profiles import STTSelection, StreamingSTTSelection


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


def test_dependency_check_requires_onnx_asr() -> None:
    def import_dependency(name: str) -> object:
        if name == "onnx_asr":
            raise ImportError("missing onnx_asr")
        return object()

    with patch("main.importlib.import_module", side_effect=import_dependency):
        with pytest.raises(RuntimeError, match="onnx_asr") as error:
            main.check_runtime_dependencies()

    assert "missing onnx_asr" in str(error.value)


def test_dependency_check_keeps_sherpa_optional_for_after_phrase_mode() -> None:
    def import_dependency(name: str) -> object:
        if name == "sherpa_onnx":
            raise ImportError("missing sherpa runtime")
        return object()

    with patch("main.importlib.import_module", side_effect=import_dependency):
        main.check_runtime_dependencies()


def test_controller_builds_complete_stt_selection() -> None:
    controller = object.__new__(AppController)
    controller.runner = None
    controller.input_map = {"Mic": 1}
    controller.output_map = {"Cable": 2}
    controller.root = MagicMock()
    controller.gui = MagicMock()
    settings = {
        "stt_mode": "streaming",
        "streaming_language": "ru",
        "streaming_profile": "sherpa_streaming_ru_t_one",
        "stt_language": "ru",
        "stt_engine": "gigaam",
        "stt_model": "gigaam-v3-e2e-rnnt",
        "stt_device": "cpu",
        "input_device_label": "Mic",
        "output_device_label": "Cable",
        "auto_tts_model": True,
        "manual_tts_model": "ru_tts",
        "tts_root": None,
    }

    with (
        patch("main.SpeechLoopRunner") as runner_class,
        patch("main.save_app_settings") as save_settings,
    ):
        controller.start(settings)

    config = runner_class.call_args.kwargs["config"]
    assert config.stt == STTSelection(
        "ru",
        "gigaam",
        "gigaam-v3-e2e-rnnt",
        "cpu",
    )
    assert config.stt_mode == "streaming"
    assert config.streaming_stt == StreamingSTTSelection(
        "ru", "sherpa_streaming_ru_t_one"
    )
    assert runner_class.call_args.kwargs["schedule"] is controller.root.after
    runner_class.return_value.start.assert_called_once()
    save_settings.assert_called_once_with(settings)


def test_window_close_stops_worker_before_destroying_root() -> None:
    controller = object.__new__(AppController)
    controller._closing = False
    controller.root = MagicMock()
    controller.gui = MagicMock()
    controller.runner = MagicMock()
    controller.runner.is_running.return_value = True

    controller.close()

    controller.runner.stop.assert_called_once()
    controller.root.destroy.assert_not_called()


def test_worker_stopped_destroys_window_when_close_is_pending() -> None:
    controller = object.__new__(AppController)
    controller._closing = True
    controller.root = MagicMock()
    controller.gui = MagicMock()
    controller.runner = MagicMock()

    destroyed = controller._runner_stopped()

    controller.root.destroy.assert_called_once()
    assert destroyed is True
