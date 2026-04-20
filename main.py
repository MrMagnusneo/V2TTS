import tkinter as tk

from audio_queue import RunConfig, SpeechLoopRunner
from devices import list_input_devices, list_output_devices, parse_index_from_label
from gui import AppGUI
from stt import STT_DEVICES, STT_MODEL_SIZES
from tts import TTS_MODELS, resolve_tts_paths


class AppController:
    def __init__(self):
        self.runner: SpeechLoopRunner | None = None
        self.input_map: dict[str, int] = {}
        self.output_map: dict[str, int] = {}

        default_paths = resolve_tts_paths()

        self.root = tk.Tk()
        self.gui = AppGUI(
            root=self.root,
            stt_devices=STT_DEVICES,
            stt_models=STT_MODEL_SIZES,
            tts_models=TTS_MODELS,
            default_tts_root=str(default_paths.tts_root),
            on_refresh_devices=self.refresh_devices,
            on_start=self.start,
            on_stop=self.stop,
        )

    def refresh_devices(self) -> tuple[list[str], list[str]]:
        inputs = list_input_devices()
        outputs = list_output_devices()

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
        output_idx = self._resolve_index(settings["output_device_label"], self.output_map)

        config = RunConfig(
            input_device=input_idx,
            output_device=output_idx,
            stt_device=settings["stt_device"],
            stt_model_size=settings["stt_model_size"],
            auto_tts_model=settings["auto_tts_model"],
            manual_tts_model=settings["manual_tts_model"],
            tts_root=settings["tts_root"],
        )

        self.runner = SpeechLoopRunner(
            config=config,
            on_status=self.gui.post_status,
            on_text=self.gui.post_text,
            on_error=self.gui.post_error,
        )
        self.runner.start()

    def stop(self) -> None:
        if self.runner:
            self.runner.stop()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = AppController()
    app.run()


if __name__ == "__main__":
    main()
