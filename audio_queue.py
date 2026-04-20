import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from audio_stream import AudioPhraseStream, StreamConfig
from audio_backend import get_sounddevice, get_soundfile
from stt import WhisperTranscriber
from tts import synthesize_text


@dataclass
class RunConfig:
    input_device: Optional[int]
    output_device: Optional[int]
    stt_device: str
    stt_model_size: str
    auto_tts_model: bool
    manual_tts_model: str
    tts_root: Optional[str]


class SpeechLoopRunner:
    def __init__(
        self,
        config: RunConfig,
        on_status: Callable[[str], None],
        on_text: Callable[[str], None],
        on_error: Callable[[str], None],
    ):
        self.config = config
        self.on_status = on_status
        self.on_text = on_text
        self.on_error = on_error
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            sd = get_sounddevice()
            sd.stop()
        except Exception:
            pass

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        try:
            sd = get_sounddevice()
            sf = get_soundfile()
            input_info = sd.query_devices(self.config.input_device, "input")
            raw_sr = float(input_info.get("default_samplerate") or 16000.0)
            default_sr = int(round(raw_sr)) if raw_sr > 0 else 16000
            stream_cfg = StreamConfig(sample_rate=default_sr, channels=1, device=self.config.input_device)
            phrase_stream = AudioPhraseStream(stream_cfg)
            transcriber = WhisperTranscriber(
                model_size=self.config.stt_model_size,
                device=self.config.stt_device,
                compute_type=None,
                language=None,
                beam_size=1,
                vad_filter=False,
            )

            stt_desc = f"STT: requested={transcriber.requested_device}, actual={transcriber.actual_device}, sr={stream_cfg.sample_rate}"
            self.on_status(f"Listening... ({stt_desc})")
            for phrase_pcm16 in phrase_stream.iter_phrases(stop_event=self._stop_event):
                if self._stop_event.is_set():
                    break

                try:
                    text = transcriber.transcribe_pcm16(phrase_pcm16, sample_rate=stream_cfg.sample_rate)
                except Exception as exc:
                    self.on_error(f"STT failed: {exc}")
                    continue

                if not text:
                    continue

                self.on_text(text)

                fd, out_wav = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    used_engine = synthesize_text(
                        text=text,
                        out_wav=out_wav,
                        auto_select=self.config.auto_tts_model,
                        manual_model=self.config.manual_tts_model,
                        tts_root=self.config.tts_root,
                    )
                    self.on_status(f"Listening... ({stt_desc}, tts={used_engine})")
                    data, fs = sf.read(out_wav, dtype="float32")
                    sd.play(data, fs, device=self.config.output_device)
                    sd.wait()
                except Exception as exc:
                    self.on_error(f"TTS/Playback failed: {exc}")
                finally:
                    if os.path.exists(out_wav):
                        os.remove(out_wav)

            self.on_status("Stopped")
        except Exception as exc:
            self.on_error(str(exc))
            self.on_status("Stopped (error)")
