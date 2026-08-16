import queue
from unittest.mock import MagicMock

from gui import AppGUI


def bare_gui() -> AppGUI:
    gui = object.__new__(AppGUI)
    gui.language_var = MagicMock()
    gui.engine_var = MagicMock()
    gui.stt_model_var = MagicMock()
    gui.engine_combo = MagicMock()
    gui.stt_model_combo = MagicMock()
    return gui


def test_english_selection_removes_gigaam() -> None:
    gui = bare_gui()
    gui.language_var.get.return_value = "English"
    gui.engine_var.get.return_value = "GigaAM"

    gui._refresh_stt_choices()

    assert gui.engine_combo.configure.call_args.kwargs["values"] == ("Whisper",)
    gui.engine_var.set.assert_called_with("Whisper")
    gui.stt_model_var.set.assert_called_with("Whisper medium")


def test_stale_tagged_event_is_discarded_on_tk_thread() -> None:
    gui = object.__new__(AppGUI)
    gui.ui_queue = queue.Queue()
    gui.is_run_current = MagicMock(return_value=False)
    gui.status_var = MagicMock()
    gui.root = MagicMock()
    gui._append_log = MagicMock()

    gui.enqueue_event("text", "late result", run_id="old-run")
    gui._poll_ui_queue()

    gui._append_log.assert_not_called()
    gui.is_run_current.assert_called_once_with("old-run")
    gui.root.after.assert_called_once()


def test_worker_stopped_does_not_reschedule_after_window_destroy() -> None:
    gui = object.__new__(AppGUI)
    gui.ui_queue = queue.Queue()
    gui.is_run_current = MagicMock(return_value=True)
    gui.on_worker_stopped = MagicMock(return_value=True)
    gui.root = MagicMock()
    gui.ui_queue.put((None, "worker_stopped", ""))

    gui._poll_ui_queue()

    gui.on_worker_stopped.assert_called_once()
    gui.root.after.assert_not_called()


def test_collect_settings_maps_labels_to_profile_ids() -> None:
    gui = object.__new__(AppGUI)
    gui.language_var = MagicMock(get=MagicMock(return_value="Russian"))
    gui.engine_var = MagicMock(get=MagicMock(return_value="GigaAM"))
    gui.stt_model_var = MagicMock(
        get=MagicMock(return_value="GigaAM v3 E2E RNN-T")
    )
    gui.stt_device_var = MagicMock(get=MagicMock(return_value="cpu"))
    gui.input_device_var = MagicMock(get=MagicMock(return_value=" Mic "))
    gui.output_device_var = MagicMock(get=MagicMock(return_value=" Cable "))
    gui.auto_tts_var = MagicMock(get=MagicMock(return_value=True))
    gui.tts_model_var = MagicMock(get=MagicMock(return_value="ru_tts"))
    gui.tts_root_var = MagicMock(get=MagicMock(return_value=" C:/tts "))

    settings = gui._collect_settings()

    assert settings["stt_language"] == "ru"
    assert settings["stt_engine"] == "gigaam"
    assert settings["stt_model"] == "gigaam-v3-e2e-rnnt"
    assert settings["stt_device"] == "cpu"


def test_stopping_disables_start_and_stop() -> None:
    gui = object.__new__(AppGUI)
    gui.start_button = MagicMock()
    gui.stop_button = MagicMock()
    gui.status_var = MagicMock()

    gui.set_pipeline_state("stopping")

    gui.start_button.configure.assert_called_with(state="disabled")
    gui.stop_button.configure.assert_called_with(state="disabled")
    gui.status_var.set.assert_called_with("Stopping...")
