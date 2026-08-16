import queue
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from stt_profiles import (
    ENGINE_LABELS,
    LANGUAGE_LABELS,
    MODEL_LABELS,
    default_selection,
    engines_for_language,
    models_for,
)


_LANGUAGE_IDS = {label: key for key, label in LANGUAGE_LABELS.items()}
_ENGINE_IDS = {label: key for key, label in ENGINE_LABELS.items()}
_MODEL_IDS = {label: key for key, label in MODEL_LABELS.items()}


class AppGUI:
    def __init__(
        self,
        root: tk.Tk,
        stt_devices: list[str],
        tts_models: list[str],
        default_tts_root: str,
        on_refresh_devices: Callable[[], tuple[list[str], list[str]]],
        on_start: Callable[[dict], None],
        on_stop: Callable[[], None],
        on_worker_stopped: Callable[[], bool],
        is_run_current: Callable[[str], bool],
    ):
        self.root = root
        self.root.title("V2TTS")
        self.root.geometry("900x680")

        self.on_refresh_devices = on_refresh_devices
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_worker_stopped = on_worker_stopped
        self.is_run_current = is_run_current

        self.ui_queue: "queue.Queue[tuple[str | None, str, str]]" = queue.Queue()

        self.language_var = tk.StringVar(value=LANGUAGE_LABELS["ru"])
        self.engine_var = tk.StringVar(value=ENGINE_LABELS["gigaam"])
        self.stt_model_var = tk.StringVar(
            value=MODEL_LABELS["gigaam-v3-e2e-rnnt"]
        )
        self.stt_device_var = tk.StringVar(value="cpu")
        self.input_device_var = tk.StringVar(value="")
        self.output_device_var = tk.StringVar(value="")
        self.auto_tts_var = tk.BooleanVar(value=True)
        self.tts_model_var = tk.StringVar(value="ru_tts")
        self.tts_root_var = tk.StringVar(value=default_tts_root)
        self.status_var = tk.StringVar(value="Idle")

        self._stt_devices = stt_devices
        self._tts_models = tts_models

        self._build_ui()
        self._refresh_stt_choices()
        self.set_pipeline_state("idle")
        self.refresh_devices()
        self._poll_ui_queue()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        self._build_settings_frame(frame)
        self._build_buttons_frame(frame)
        self._build_status_frame(frame)
        self._toggle_tts_model_combo()

    def _build_settings_frame(self, frame: ttk.Frame) -> None:
        settings = ttk.LabelFrame(frame, text="Settings", padding=10)
        settings.pack(fill="x")

        ttk.Label(settings, text="STT language").grid(
            row=0,
            column=0,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.language_combo = ttk.Combobox(
            settings,
            textvariable=self.language_var,
            values=tuple(LANGUAGE_LABELS.values()),
            state="readonly",
            width=18,
        )
        self.language_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.language_combo.bind(
            "<<ComboboxSelected>>",
            self._refresh_stt_choices,
        )

        ttk.Label(settings, text="STT engine").grid(
            row=0,
            column=2,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.engine_combo = ttk.Combobox(
            settings,
            textvariable=self.engine_var,
            state="readonly",
            width=22,
        )
        self.engine_combo.grid(row=0, column=3, sticky="w", padx=4, pady=4)
        self.engine_combo.bind(
            "<<ComboboxSelected>>",
            self._refresh_stt_choices,
        )

        ttk.Label(settings, text="STT model").grid(
            row=1,
            column=0,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.stt_model_combo = ttk.Combobox(
            settings,
            textvariable=self.stt_model_var,
            state="readonly",
            width=24,
        )
        self.stt_model_combo.grid(
            row=1,
            column=1,
            sticky="w",
            padx=4,
            pady=4,
        )

        ttk.Label(settings, text="STT device").grid(
            row=1,
            column=2,
            sticky="w",
            padx=4,
            pady=4,
        )
        ttk.Combobox(
            settings,
            textvariable=self.stt_device_var,
            values=self._stt_devices,
            state="readonly",
            width=18,
        ).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(settings, text="Input device").grid(
            row=2,
            column=0,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.input_combo = ttk.Combobox(
            settings,
            textvariable=self.input_device_var,
            state="readonly",
            width=40,
        )
        self.input_combo.grid(
            row=2,
            column=1,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=4,
        )

        ttk.Label(settings, text="Output device").grid(
            row=3,
            column=0,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.output_combo = ttk.Combobox(
            settings,
            textvariable=self.output_device_var,
            state="readonly",
            width=40,
        )
        self.output_combo.grid(
            row=3,
            column=1,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=4,
        )

        ttk.Checkbutton(
            settings,
            text="Auto TTS model selection",
            variable=self.auto_tts_var,
            command=self._toggle_tts_model_combo,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=6)

        ttk.Label(settings, text="Manual TTS model").grid(
            row=4,
            column=2,
            sticky="w",
            padx=4,
            pady=4,
        )
        self.tts_combo = ttk.Combobox(
            settings,
            textvariable=self.tts_model_var,
            values=self._tts_models,
            state="readonly",
            width=18,
        )
        self.tts_combo.grid(row=4, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(settings, text="TTS root path").grid(
            row=5,
            column=0,
            sticky="w",
            padx=4,
            pady=4,
        )
        ttk.Entry(settings, textvariable=self.tts_root_var, width=62).grid(
            row=5,
            column=1,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=4,
        )

    def _build_buttons_frame(self, frame: ttk.Frame) -> None:
        button_row = ttk.Frame(frame)
        button_row.pack(fill="x", pady=(10, 8))
        ttk.Button(
            button_row,
            text="Refresh Devices",
            command=self.refresh_devices,
        ).pack(side="left")
        self.start_button = ttk.Button(
            button_row,
            text="Start",
            command=self.start,
        )
        self.start_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(
            button_row,
            text="Stop",
            command=self.stop,
        )
        self.stop_button.pack(side="left")

    def _build_status_frame(self, frame: ttk.Frame) -> None:
        status = ttk.LabelFrame(frame, text="Status", padding=10)
        status.pack(fill="both", expand=True)
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

        self.log = tk.Text(status, height=20, wrap="word")
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.log.configure(state="disabled")

    def _refresh_stt_choices(self, _event=None) -> None:
        language = _LANGUAGE_IDS[self.language_var.get()]
        engine_labels = tuple(
            ENGINE_LABELS[engine]
            for engine in engines_for_language(language)
        )
        self.engine_combo.configure(values=engine_labels)
        engine_label = self.engine_var.get()
        if engine_label not in engine_labels:
            engine_label = ENGINE_LABELS[default_selection(language).engine]
            self.engine_var.set(engine_label)

        engine = _ENGINE_IDS[engine_label]
        model_labels = tuple(
            MODEL_LABELS[model]
            for model in models_for(language, engine)
        )
        self.stt_model_combo.configure(values=model_labels)
        if self.stt_model_var.get() not in model_labels:
            default = default_selection(language)
            default_model = (
                default.model
                if default.engine == engine
                else models_for(language, engine)[0]
            )
            self.stt_model_var.set(MODEL_LABELS[default_model])

    def _toggle_tts_model_combo(self) -> None:
        state = "disabled" if self.auto_tts_var.get() else "readonly"
        self.tts_combo.configure(state=state)

    def set_pipeline_state(self, state: str) -> None:
        if state == "idle":
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Idle")
        elif state in {"starting", "listening"}:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_var.set(
                "Starting..." if state == "starting" else "Listening..."
            )
        elif state == "stopping":
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            self.status_var.set("Stopping...")
        else:
            raise ValueError(f"Unknown pipeline state: {state}")

    def refresh_devices(self) -> None:
        try:
            input_labels, output_labels = self.on_refresh_devices()
        except Exception as exc:
            self.status_var.set("Audio backend unavailable")
            messagebox.showerror("Audio error", str(exc))
            return

        self.input_combo["values"] = input_labels
        self.output_combo["values"] = output_labels
        if input_labels and not self.input_device_var.get():
            self.input_device_var.set(input_labels[0])
        if output_labels and not self.output_device_var.get():
            self.output_device_var.set(output_labels[0])

    def _collect_settings(self) -> dict:
        return {
            "stt_language": _LANGUAGE_IDS[self.language_var.get()],
            "stt_engine": _ENGINE_IDS[self.engine_var.get()],
            "stt_model": _MODEL_IDS[self.stt_model_var.get()],
            "stt_device": self.stt_device_var.get(),
            "input_device_label": self.input_device_var.get().strip(),
            "output_device_label": self.output_device_var.get().strip(),
            "auto_tts_model": self.auto_tts_var.get(),
            "manual_tts_model": self.tts_model_var.get(),
            "tts_root": self.tts_root_var.get().strip() or None,
        }

    def start(self) -> None:
        try:
            self.on_start(self._collect_settings())
        except Exception as exc:
            self.set_pipeline_state("idle")
            messagebox.showerror("Start error", str(exc))

    def stop(self) -> None:
        self.on_stop()

    def enqueue_event(
        self,
        kind: str,
        message: str,
        run_id: str | None = None,
    ) -> None:
        self.ui_queue.put((run_id, kind, message))

    def post_status(self, message: str) -> None:
        self.enqueue_event("status", message)

    def post_error(self, message: str) -> None:
        self.enqueue_event("error", message)

    def post_text(self, message: str) -> None:
        self.enqueue_event("text", message)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                run_id, kind, message = self.ui_queue.get_nowait()
                if run_id is not None and not self.is_run_current(run_id):
                    continue
                if kind == "status":
                    self.status_var.set(message)
                elif kind == "state":
                    self.set_pipeline_state(message)
                elif kind == "warning":
                    self._append_log(f"WARNING: {message}\n")
                elif kind == "error":
                    self._append_log(f"ERROR: {message}\n")
                    messagebox.showerror("Runtime error", message)
                elif kind == "text":
                    self._append_log(f"STT: {message}\n")
                elif kind == "worker_stopped":
                    if self.on_worker_stopped():
                        return
        except queue.Empty:
            pass

        self.root.after(100, self._poll_ui_queue)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")
