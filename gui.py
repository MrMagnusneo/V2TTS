import queue
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable


class AppGUI:
    def __init__(
        self,
        root: tk.Tk,
        stt_devices: list[str],
        stt_models: list[str],
        tts_models: list[str],
        default_tts_root: str,
        on_refresh_devices: Callable[[], tuple[list[str], list[str]]],
        on_start: Callable[[dict], None],
        on_stop: Callable[[], None],
    ):
        self.root = root
        self.root.title("V2TTS")
        self.root.geometry("900x620")

        self.on_refresh_devices = on_refresh_devices
        self.on_start = on_start
        self.on_stop = on_stop

        self.ui_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        self.stt_device_var = tk.StringVar(value="cpu")
        self.stt_model_var = tk.StringVar(value="medium")
        self.input_device_var = tk.StringVar(value="")
        self.output_device_var = tk.StringVar(value="")
        self.auto_tts_var = tk.BooleanVar(value=True)
        self.tts_model_var = tk.StringVar(value="ru_tts")
        self.tts_root_var = tk.StringVar(value=default_tts_root)
        self.status_var = tk.StringVar(value="Idle")

        self._stt_devices = stt_devices
        self._stt_models = stt_models
        self._tts_models = tts_models

        self._build_ui()
        self.refresh_devices()
        self._poll_ui_queue()

    def _build_ui(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        cfg = ttk.LabelFrame(frm, text="Settings", padding=10)
        cfg.pack(fill="x")

        ttk.Label(cfg, text="STT device").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            cfg,
            textvariable=self.stt_device_var,
            values=self._stt_devices,
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="STT model size").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            cfg,
            textvariable=self.stt_model_var,
            values=self._stt_models,
            state="readonly",
            width=18,
        ).grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="Input device").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.input_combo = ttk.Combobox(cfg, textvariable=self.input_device_var, state="readonly", width=40)
        self.input_combo.grid(row=1, column=1, columnspan=3, sticky="we", padx=4, pady=4)

        ttk.Label(cfg, text="Output device").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.output_combo = ttk.Combobox(cfg, textvariable=self.output_device_var, state="readonly", width=40)
        self.output_combo.grid(row=2, column=1, columnspan=3, sticky="we", padx=4, pady=4)

        ttk.Checkbutton(
            cfg,
            text="Auto TTS model selection",
            variable=self.auto_tts_var,
            command=self._toggle_tts_model_combo,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=6)

        ttk.Label(cfg, text="Manual TTS model").grid(row=3, column=2, sticky="w", padx=4, pady=4)
        self.tts_combo = ttk.Combobox(
            cfg,
            textvariable=self.tts_model_var,
            values=self._tts_models,
            state="readonly",
            width=18,
        )
        self.tts_combo.grid(row=3, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(cfg, text="TTS root path").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(cfg, textvariable=self.tts_root_var, width=62).grid(
            row=4,
            column=1,
            columnspan=3,
            sticky="we",
            padx=4,
            pady=4,
        )

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill="x", pady=(10, 8))
        ttk.Button(btn_row, text="Refresh Devices", command=self.refresh_devices).pack(side="left")
        ttk.Button(btn_row, text="Start", command=self.start).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Stop", command=self.stop).pack(side="left")

        status = ttk.LabelFrame(frm, text="Status", padding=10)
        status.pack(fill="both", expand=True)
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

        self.log = tk.Text(status, height=20, wrap="word")
        self.log.pack(fill="both", expand=True, pady=(8, 0))
        self.log.configure(state="disabled")

        self._toggle_tts_model_combo()

    def _toggle_tts_model_combo(self) -> None:
        if self.auto_tts_var.get():
            self.tts_combo.configure(state="disabled")
        else:
            self.tts_combo.configure(state="readonly")

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
            "stt_device": self.stt_device_var.get(),
            "stt_model_size": self.stt_model_var.get(),
            "input_device_label": self.input_device_var.get().strip(),
            "output_device_label": self.output_device_var.get().strip(),
            "auto_tts_model": self.auto_tts_var.get(),
            "manual_tts_model": self.tts_model_var.get(),
            "tts_root": self.tts_root_var.get().strip() or None,
        }

    def start(self) -> None:
        try:
            self.on_start(self._collect_settings())
            self.status_var.set("Starting...")
        except Exception as exc:
            messagebox.showerror("Start error", str(exc))

    def stop(self) -> None:
        self.on_stop()

    def post_status(self, msg: str) -> None:
        self.ui_queue.put(("status", msg))

    def post_text(self, msg: str) -> None:
        self.ui_queue.put(("text", msg))

    def post_error(self, msg: str) -> None:
        self.ui_queue.put(("error", msg))

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, msg = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(msg)
                elif kind == "text":
                    self._append_log(f"STT: {msg}\n")
                elif kind == "error":
                    self._append_log(f"ERROR: {msg}\n")
                    messagebox.showerror("Runtime error", msg)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_ui_queue)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")
