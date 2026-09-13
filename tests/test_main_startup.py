from pathlib import Path
import threading
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


def test_controller_exports_tts_on_background_thread(tmp_path: Path) -> None:
    controller = object.__new__(AppController)
    controller.gui = MagicMock()
    controller._tts_export_thread = None
    output = tmp_path / "speech.wav"
    main_thread = threading.get_ident()
    synthesized_on = []
    completed = threading.Event()

    def synthesize(text, out_wav, **kwargs):
        synthesized_on.append(threading.get_ident())
        Path(out_wav).write_bytes(b"RIFF-export")
        return kwargs["manual_model"]

    def enqueue(kind, message):
        completed.set()

    controller.gui.enqueue_event.side_effect = enqueue

    with patch("main.synthesize_text", side_effect=synthesize):
        controller.export_tts(
            "hello",
            str(output),
            False,
            "dectalk",
            "C:/tts",
        )
        assert completed.wait(timeout=3)

    assert output.read_bytes() == b"RIFF-export"
    assert synthesized_on and synthesized_on[0] != main_thread
    kind, message = controller.gui.enqueue_event.call_args.args
    assert kind == "tts_export_done"
    assert message == f"Saved WAV with dectalk: {output}"


def test_controller_reports_tts_export_failure_without_damaging_output(
    tmp_path: Path,
) -> None:
    controller = object.__new__(AppController)
    controller.gui = MagicMock()
    controller._tts_export_thread = None
    completed = threading.Event()
    output = tmp_path / "speech.wav"
    output.write_bytes(b"existing wav")

    def enqueue(_kind, _message):
        completed.set()

    controller.gui.enqueue_event.side_effect = enqueue

    def fail_after_writing(text, out_wav, **_kwargs):
        Path(out_wav).write_bytes(b"partial wav")
        raise RuntimeError("model failed")

    with patch("main.synthesize_text", side_effect=fail_after_writing):
        controller.export_tts(
            "hello",
            str(output),
            False,
            "coqui",
            None,
        )
        assert completed.wait(timeout=3)

    kind, message = controller.gui.enqueue_event.call_args.args
    assert kind == "tts_export_error"
    assert "model failed" in message
    assert output.read_bytes() == b"existing wav"
    assert list(tmp_path.glob(".speech.wav.*.tmp.wav")) == []


def test_window_close_waits_for_tts_export() -> None:
    controller = object.__new__(AppController)
    controller._closing = False
    controller.root = MagicMock()
    controller.gui = MagicMock()
    controller.runner = None
    controller._tts_export_thread = MagicMock()
    controller._tts_export_thread.is_alive.return_value = True

    controller.close()

    controller.gui.warn_export_in_progress.assert_called_once_with()
    controller.root.destroy.assert_not_called()
    assert controller._closing is False


def test_pipeline_close_rejects_new_tts_export_until_window_is_destroyed() -> None:
    controller = object.__new__(AppController)
    controller._closing = False
    controller.root = MagicMock()
    controller.gui = MagicMock()
    controller.runner = MagicMock()
    controller.runner.is_running.return_value = True
    controller._tts_export_thread = None

    controller.close()

    assert controller._closing is True
    controller.gui.set_export_enabled.assert_called_once_with(False)
    with pytest.raises(RuntimeError, match="closing"):
        controller.export_tts("hello", "out.wav", False, "sam", None)
