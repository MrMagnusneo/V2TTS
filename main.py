import importlib
import multiprocessing
import sys
import tkinter as tk

from app_settings import load_app_settings, save_app_settings
from audio_queue import RunConfig, SpeechLoopRunner
from devices import list_audio_devices, parse_index_from_label
from gui import AppGUI
from runtime_support import ensure_standard_streams
from stt import STT_DEVICES
from stt_profiles import (
    STTSelection,
    StreamingSTTSelection,
    default_selection,
    validate_selection,
    validate_streaming_selection,
)
from tts import TTS_MODELS, prepare_runtime_tts_root


RUNTIME_DEPENDENCIES = (
    "faster_whisper",
    "onnx_asr",
    "sounddevice",
    "soundfile",
)


def check_runtime_dependencies() -> None:
    unavailable: list[tuple[str, str]] = []
    for package in RUNTIME_DEPENDENCIES:
        try:
            importlib.import_module(package)
        except Exception as exc:
            unavailable.append(
                (package, f"{type(exc).__name__}: {exc}")
            )

    if unavailable:
        packages = "; ".join(
            f"{package} ({reason})" for package, reason in unavailable
        )
        raise RuntimeError(
            f"Missing or unusable runtime dependencies: {packages}. "
            "Install them with: python -m pip install -r requirements.txt"
        )


class AppController:
    def __init__(self):
        self.runner: SpeechLoopRunner | None = None
        self.input_map: dict[str, int] = {}
        self.output_map: dict[str, int] = {}
        self._closing = False

        runtime_tts_root = prepare_runtime_tts_root()
        initial_settings = load_app_settings()

        self.root = tk.Tk()
        self.gui = AppGUI(
            root=self.root,
            stt_devices=STT_DEVICES,
            tts_models=TTS_MODELS,
            default_tts_root=str(runtime_tts_root),
            on_refresh_devices=self.refresh_devices,
            on_start=self.start,
            on_stop=self.stop,
            on_worker_stopped=self._runner_stopped,
            is_run_current=self._is_run_current,
            initial_settings=initial_settings,
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def refresh_devices(self) -> tuple[list[str], list[str]]:
        all_devices = list_audio_devices()
        inputs = [d for d in all_devices if d.max_input_channels > 0]
        outputs = [d for d in all_devices if d.max_output_channels > 0]

        input_labels = [d.label() for d in inputs]
        output_labels = [d.label() for d in outputs]

        self.input_map = {label: d.index for label, d in zip(input_labels, inputs)}
        self.output_map = {label: d.index for label, d in zip(output_labels, outputs)}

        return input_labels, output_labels

    def _resolve_index(self, label: str, mapping: dict[str, int]) -> int | None:
        if not label:
            return None
        if label in mapping:
            return mapping[label]
        return parse_index_from_label(label)

    def start(self, settings: dict) -> None:
        if self.runner and self.runner.is_running():
            return

        input_idx = self._resolve_index(settings["input_device_label"], self.input_map)
        output_idx = self._resolve_index(
            settings["output_device_label"], self.output_map
        )

        stt_selection = STTSelection(
            language=settings["stt_language"],
            engine=settings["stt_engine"],
            model=settings["stt_model"],
            device=settings["stt_device"],
        )
        validate_selection(stt_selection)
        streaming_selection = StreamingSTTSelection(
            language=settings.get(
                "streaming_language",
                settings["stt_language"],
            ),
            profile=settings.get(
                "streaming_profile",
                "sherpa_streaming_ru_t_one",
            ),
        )
        validate_streaming_selection(streaming_selection)
        stt_mode = settings.get("stt_mode", "streaming")
        if stt_mode not in {"streaming", "after_phrase"}:
            raise ValueError(f"Unknown STT mode: {stt_mode}")
        fallback_selection = stt_selection
        if (
            stt_mode == "streaming"
            and stt_selection.language != streaming_selection.language
        ):
            fallback_selection = default_selection(streaming_selection.language)
        config = RunConfig(
            input_device=input_idx,
            output_device=output_idx,
            stt=fallback_selection,
            auto_tts_model=settings["auto_tts_model"],
            manual_tts_model=settings["manual_tts_model"],
            tts_root=settings["tts_root"],
            stt_mode=stt_mode,
            streaming_stt=streaming_selection,
        )

        self.runner = SpeechLoopRunner(
            config=config,
            on_event=lambda run_id, kind, msg: self.gui.enqueue_event(
                kind, msg, run_id
            ),
            on_stopped=lambda: self.gui.enqueue_event("worker_stopped", ""),
            schedule=self.root.after,
        )
        self.gui.set_pipeline_state("starting")
        try:
            self.runner.start()
            save_app_settings(settings)
        except Exception:
            self.runner = None
            self.gui.set_pipeline_state("idle")
            raise

    def stop(self) -> None:
        if self.runner:
            self.gui.set_pipeline_state("stopping")
            self.runner.stop()

    def _is_run_current(self, run_id: str) -> bool:
        return bool(self.runner and self.runner.accepts_events_from(run_id))

    def _runner_stopped(self) -> bool:
        self.runner = None
        if self._closing:
            self.root.destroy()
            return True
        else:
            self.gui.set_pipeline_state("idle")
            return False

    def close(self) -> None:
        self._closing = True
        if self.runner and self.runner.is_running():
            self.stop()
            return
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    ensure_standard_streams()
    multiprocessing.freeze_support()
    args = sys.argv[1:] if argv is None else argv
    check_runtime_dependencies()

    if args == ["--smoke-test"]:
        from smoke_test import run_packaged_smoke

        return run_packaged_smoke()
    if args:
        raise ValueError(f"Unknown arguments: {' '.join(args)}")

    app = AppController()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
