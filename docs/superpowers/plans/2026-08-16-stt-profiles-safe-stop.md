# Safe Stop and Russian/English STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local GigaAM v3 plus explicit Russian/English Whisper profiles, store STT weights outside the executable, improve phrase capture, and make `Stop` terminate only the active pipeline while the Tkinter window remains responsive.

**Architecture:** Keep Tkinter in the parent process and move capture, model inference, TTS, and playback into one spawned child process per run. Put STT profile/catalog logic and transcriber adapters behind small interfaces; send tagged events back to the parent and reject events from an invalidated run. Escalate cancellation from a cooperative event to child-process termination without calling PortAudio or blocking `join()` in the GUI thread.

**Tech Stack:** Python 3.12, Tkinter, `multiprocessing` spawn context, NumPy, sounddevice, soundfile, faster-whisper/CTranslate2, `onnx-asr[cpu,hub]>=0.12,<0.13`, GigaAM v3 ONNX, PyInstaller, pytest.

## Global Constraints

- Support exactly two STT languages: Russian (`ru`) and English (`en`).
- Russian defaults to GigaAM `gigaam-v3-e2e-rnnt`; English defaults to Whisper `medium`.
- GigaAM is available only for Russian; Whisper must always receive explicit `ru` or `en`.
- Store STT weights under `%LOCALAPPDATA%\V2TTS\models` on Windows and the existing platform user-data root elsewhere.
- Never include downloaded STT model caches in PyInstaller `datas` or the one-file executable.
- The Tkinter process must never call `sounddevice.stop()`, own an audio stream, or synchronously join a child process.
- Stop invalidates the active run immediately; no text, error, status, or playback from that run may appear afterward.
- Cooperative shutdown grace is 300 ms; terminate grace is one additional second before `kill()`.
- Preserve current output sample-rate negotiation and anti-alias downsampling.
- Preserve the non-blocking capture callback and bounded drop-oldest frame queue.
- Use 200 ms capture pre-roll, a 300 ms minimum phrase, and 700 ms end silence.
- Run three verification passes before merge: focused tests; full tests plus compile/diff checks; Windows frozen-build workflow.

---

## File Structure

- Create `stt_profiles.py`: language/engine/model catalog, validation, defaults, and user model paths.
- Modify `stt.py`: shared transcriber protocol, Whisper cache/language improvements, GigaAM ONNX adapter, and transcriber factory.
- Modify `audio_stream.py`: rolling pre-roll and conservative phrase-boundary behavior.
- Create `pipeline.py`: child-owned capture -> STT -> TTS -> cancellable playback functional pipeline.
- Rewrite `audio_queue.py`: parent-side spawned-process lifecycle, run-ID filtering, and stop escalation; re-export `RunConfig` and `prepare_audio_for_output` for compatibility.
- Modify `gui.py`: dynamic RU/EN controls, explicit UI lifecycle states, and button state changes.
- Modify `main.py`: controller wiring, window-close lifecycle, runtime dependency check, and `freeze_support()`.
- Modify `smoke_test.py`: model-free multiprocessing frozen smoke.
- Modify `pyproject.toml`, `requirements.txt`, and `installer/V2TTS.spec`: ONNX ASR dependency and frozen imports/native libraries without model weights.
- Modify `README.md` and `installer/README.md`: profiles, model locations, first-run download, Stop semantics, and Windows build behavior.
- Create `tests/test_stt_profiles.py`, `tests/test_pipeline.py`, `tests/test_runner_lifecycle.py`, and `tests/test_gui_profiles.py`.
- Modify `tests/test_stt.py`, `test_audio_stream.py`, `tests/test_main_smoke.py`, `tests/test_audio_output.py`, `tests/test_audio_queue_pipeline.py`, `tests/test_packaged_smoke.py`, and `tests/test_windows_workflow.py`.

---

### Task 1: STT Profile Catalog and External Model Paths

**Files:**
- Create: `stt_profiles.py`
- Create: `tests/test_stt_profiles.py`

**Interfaces:**
- Produces: `STTSelection(language: str, engine: str, model: str, device: str)`.
- Produces: `LANGUAGE_LABELS`, `ENGINE_LABELS`, and `MODEL_LABELS` dictionaries.
- Produces: `engines_for_language(language: str) -> tuple[str, ...]`.
- Produces: `models_for(language: str, engine: str) -> tuple[str, ...]`.
- Produces: `default_selection(language: str = "ru") -> STTSelection`.
- Produces: `validate_selection(selection: STTSelection) -> None`.
- Produces: `user_data_root() -> Path` and `stt_model_dir(engine: str, model: str) -> Path`.

- [ ] **Step 1: Write failing catalog and path tests**

```python
# tests/test_stt_profiles.py
from pathlib import Path
import pytest

from stt_profiles import (
    STTSelection, default_selection, engines_for_language,
    models_for, stt_model_dir, validate_selection,
)


def test_russian_defaults_to_gigaam_e2e_rnnt() -> None:
    assert default_selection("ru") == STTSelection(
        language="ru", engine="gigaam",
        model="gigaam-v3-e2e-rnnt", device="cpu",
    )


def test_english_exposes_only_whisper() -> None:
    assert engines_for_language("en") == ("whisper",)
    assert models_for("en", "whisper") == ("small", "medium", "large-v3")


def test_gigaam_is_rejected_for_english() -> None:
    with pytest.raises(ValueError, match="not available for language en"):
        validate_selection(STTSelection("en", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"))


def test_windows_model_path_is_outside_executable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("stt_profiles.sys.platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert stt_model_dir("whisper", "medium") == (
        tmp_path / "V2TTS" / "models" / "whisper" / "medium"
    )
```

- [ ] **Step 2: Run the new tests and confirm the module is missing**

Run: `.venv/bin/python -m pytest tests/test_stt_profiles.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'stt_profiles'`.

- [ ] **Step 3: Implement the catalog and path functions**

```python
# stt_profiles.py
import os
import sys
from dataclasses import dataclass
from pathlib import Path

LANGUAGE_LABELS = {"ru": "Russian", "en": "English"}
ENGINE_LABELS = {"gigaam": "GigaAM", "whisper": "Whisper"}
MODEL_LABELS = {
    "gigaam-v3-e2e-rnnt": "GigaAM v3 E2E RNN-T",
    "gigaam-v3-e2e-ctc": "GigaAM v3 E2E CTC",
    "small": "Whisper small", "medium": "Whisper medium",
    "large-v3": "Whisper large-v3",
}
_MODELS = {
    ("ru", "gigaam"): ("gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"),
    ("ru", "whisper"): ("small", "medium", "large-v3"),
    ("en", "whisper"): ("small", "medium", "large-v3"),
}


@dataclass(frozen=True)
class STTSelection:
    language: str
    engine: str
    model: str
    device: str


def engines_for_language(language: str) -> tuple[str, ...]:
    if language not in LANGUAGE_LABELS:
        raise ValueError(f"Unknown STT language: {language}")
    return tuple(engine for lang, engine in _MODELS if lang == language)


def models_for(language: str, engine: str) -> tuple[str, ...]:
    try:
        return _MODELS[(language, engine)]
    except KeyError as exc:
        raise ValueError(
            f"STT engine {engine} is not available for language {language}"
        ) from exc


def validate_selection(selection: STTSelection) -> None:
    if selection.device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown STT device: {selection.device}")
    if selection.model not in models_for(selection.language, selection.engine):
        raise ValueError(
            f"STT model {selection.model} is not available for "
            f"{selection.language}/{selection.engine}"
        )


def default_selection(language: str = "ru") -> STTSelection:
    if language == "ru":
        return STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
    if language == "en":
        return STTSelection("en", "whisper", "medium", "cpu")
    raise ValueError(f"Unknown STT language: {language}")


def user_data_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "V2TTS"


def stt_model_dir(engine: str, model: str) -> Path:
    return user_data_root() / "models" / engine / model
```

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_stt_profiles.py -q`

Expected: all profile tests pass.

- [ ] **Step 5: Commit the catalog**

```bash
git add stt_profiles.py tests/test_stt_profiles.py
git commit -m "feat: add Russian and English STT profiles"
```

---

### Task 2: Whisper and GigaAM Transcriber Adapters

**Files:**
- Modify: `stt.py`
- Modify: `tests/test_stt.py`

**Interfaces:**
- Consumes: `STTSelection`, `stt_model_dir()`, and `validate_selection()` from Task 1.
- Produces: `Transcriber` protocol with `requested_device`, `actual_device`, `fallback_reason`, and `transcribe_pcm16()`.
- Produces: `GigaAMTranscriber(model_name, device, model_dir, on_warning)`.
- Produces: `create_transcriber(selection, on_warning) -> Transcriber`.

- [ ] **Step 1: Add failing adapter/factory tests**

```python
# append to tests/test_stt.py
from pathlib import Path
from stt import GigaAMTranscriber, create_transcriber
from stt_profiles import STTSelection


def test_whisper_factory_passes_explicit_russian_and_download_root(tmp_path: Path) -> None:
    with patch("stt.WhisperModel") as model_cls, patch(
        "stt.stt_model_dir", return_value=tmp_path
    ):
        transcriber = create_transcriber(
            STTSelection("ru", "whisper", "medium", "cpu")
        )
    assert transcriber.language == "ru"
    assert model_cls.call_args.kwargs["download_root"] == str(tmp_path)


def test_gigaam_uses_local_directory_and_cpu_provider(tmp_path: Path) -> None:
    load_model = MagicMock(return_value=MagicMock())
    with patch("stt._load_onnx_asr_model", load_model):
        GigaAMTranscriber("gigaam-v3-e2e-rnnt", "cpu", tmp_path)
    load_model.assert_called_once_with(
        "gigaam-v3-e2e-rnnt", tmp_path,
        providers=["CPUExecutionProvider"],
    )


def test_gigaam_converts_pcm16_and_returns_text(tmp_path: Path) -> None:
    model = MagicMock()
    model.recognize.return_value = "Привет, мир."
    with patch("stt._load_onnx_asr_model", return_value=model):
        transcriber = GigaAMTranscriber("gigaam-v3-e2e-rnnt", "cpu", tmp_path)
    pcm = np.array([0, 16384, -16384], dtype=np.int16)
    assert transcriber.transcribe_pcm16(pcm, 44100) == "Привет, мир."
    waveform = model.recognize.call_args.args[0]
    np.testing.assert_allclose(waveform, [0.0, 0.5, -0.5])
    assert model.recognize.call_args.kwargs == {"sample_rate": 44100}


def test_cuda_fallback_exposes_reason_and_warning(tmp_path: Path) -> None:
    warnings: list[str] = []
    with patch("stt.WhisperModel") as model_cls, patch(
        "stt.stt_model_dir", return_value=tmp_path
    ):
        model_cls.side_effect = [RuntimeError("cublas64_12.dll missing"), MagicMock()]
        transcriber = create_transcriber(
            STTSelection("en", "whisper", "medium", "cuda"), warnings.append
        )
    assert transcriber.actual_device == "cpu"
    assert "cublas64_12.dll missing" in transcriber.fallback_reason
    assert warnings == ["STT CUDA unavailable; using CPU: cublas64_12.dll missing"]


def test_model_load_error_names_model_and_directory(tmp_path: Path) -> None:
    with patch("stt._load_onnx_asr_model", side_effect=OSError("network down")), patch(
        "stt.stt_model_dir", return_value=tmp_path / "gigaam-model"
    ):
        with pytest.raises(RuntimeError) as error:
            create_transcriber(
                STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu")
            )
    message = str(error.value)
    assert "gigaam-v3-e2e-rnnt" in message
    assert str(tmp_path / "gigaam-model") in message
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_stt.py -q`

Expected: failures for missing `GigaAMTranscriber`, `create_transcriber`, and Whisper `download_root`.

- [ ] **Step 3: Implement the protocol, adapters, and factory**

```python
# key additions in stt.py
from pathlib import Path
from typing import Callable, Optional, Protocol
from stt_profiles import STTSelection, stt_model_dir, validate_selection


class Transcriber(Protocol):
    requested_device: str
    actual_device: str
    fallback_reason: Optional[str]
    def transcribe_pcm16(
        self, pcm16: np.ndarray, sample_rate: int = 16000
    ) -> str: ...


def _load_onnx_asr_model(model_name: str, path: Path, *, providers: list[str]):
    import onnx_asr
    return onnx_asr.load_model(model_name, path, providers=providers)


class GigaAMTranscriber:
    def __init__(self, model_name, device, model_dir, on_warning=lambda _: None):
        self.requested_device = device
        self.actual_device = device
        self.fallback_reason = None
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        try:
            self.model = _load_onnx_asr_model(
                model_name, model_dir, providers=providers
            )
        except Exception as exc:
            if device != "cuda":
                raise
            self.fallback_reason = str(exc)
            self.actual_device = "cpu"
            on_warning(f"STT CUDA unavailable; using CPU: {exc}")
            self.model = _load_onnx_asr_model(
                model_name, model_dir, providers=["CPUExecutionProvider"]
            )

    def transcribe_pcm16(self, pcm16, sample_rate=16000) -> str:
        waveform = pcm16.astype(np.float32) / 32768.0
        return str(self.model.recognize(waveform, sample_rate=sample_rate)).strip()


def create_transcriber(selection, on_warning=lambda _: None) -> Transcriber:
    validate_selection(selection)
    model_dir = stt_model_dir(selection.engine, selection.model)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if selection.engine == "gigaam":
            return GigaAMTranscriber(
                selection.model, selection.device, model_dir, on_warning
            )
        return WhisperTranscriber(
            model_size=selection.model, device=selection.device,
            language=selection.language, model_root=model_dir,
            on_warning=on_warning,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load STT model {selection.model} in {model_dir}: {exc}"
        ) from exc
```

Modify `WhisperTranscriber` to initialize `fallback_reason = None`, pass
`download_root=str(model_root)` to both GPU and CPU constructors, preserve the
original CUDA exception text, and call `on_warning` exactly once on fallback.
Do not catch CPU initialization failures.

- [ ] **Step 4: Run adapter tests**

Run: `.venv/bin/python -m pytest tests/test_stt.py tests/test_stt_profiles.py -q`

Expected: all tests pass without downloading a real model.

- [ ] **Step 5: Commit adapters**

```bash
git add stt.py tests/test_stt.py
git commit -m "feat: add local GigaAM and tuned Whisper adapters"
```

---

### Task 3: Phrase Pre-roll and Conservative Segmentation

**Files:**
- Modify: `audio_stream.py`
- Modify: `test_audio_stream.py`
- Modify: `tests/test_audio_stream_backpressure.py`

**Interfaces:**
- Produces: `StreamConfig.pre_roll_ms: int = 200`.
- Preserves: `AudioPhraseStream.iter_phrases(stop_event)` and `StreamMetrics`.
- Produces behavior: threshold-crossing audio includes only the bounded pre-roll; callback remains non-blocking.

- [ ] **Step 1: Add failing pre-roll tests**

```python
# append to test_audio_stream.py
def test_phrase_includes_bounded_pre_roll() -> None:
    cfg = StreamConfig(
        sample_rate=1000, frame_ms=100, pre_roll_ms=200,
        min_phrase_ms=100, max_silence_ms=100,
        start_threshold=0.5, stop_threshold=0.2,
    )
    stream = AudioPhraseStream(cfg)
    quiet_a = np.full(100, 0.1, dtype=np.float32)
    quiet_b = np.full(100, 0.2, dtype=np.float32)
    voice = np.full(100, 0.8, dtype=np.float32)
    silence = np.zeros(100, dtype=np.float32)

    assert stream._process_frame(quiet_a) is None
    assert stream._process_frame(quiet_b) is None
    assert stream._process_frame(voice) is None
    result = stream._process_frame(silence)

    assert result is not None
    assert len(result) == 400
    np.testing.assert_array_equal(
        result[:100], (quiet_a * 32767).astype(np.int16)
    )
    np.testing.assert_array_equal(
        result[200:300], (voice * 32767).astype(np.int16)
    )


def test_pre_roll_drops_frames_older_than_configured_window() -> None:
    cfg = StreamConfig(
        sample_rate=1000, frame_ms=100, pre_roll_ms=200,
        start_threshold=0.5,
    )
    stream = AudioPhraseStream(cfg)
    for level in (0.1, 0.2, 0.3):
        stream._process_frame(np.full(100, level, dtype=np.float32))
    stream._process_frame(np.full(100, 0.8, dtype=np.float32))
    assert len(stream.phrase_frames) == 3
    assert np.allclose(stream.phrase_frames[0], 0.2)
```

- [ ] **Step 2: Run tests and confirm `pre_roll_ms` is unsupported**

Run: `.venv/bin/python -m pytest test_audio_stream.py tests/test_audio_stream_backpressure.py -q`

Expected: failure because `StreamConfig` has no `pre_roll_ms`.

- [ ] **Step 3: Implement a bounded pre-roll deque**

Add `pre_roll_ms: int = 200` to `StreamConfig`. In `AudioPhraseStream.__init__`,
create a deque sized with `max(0, ceil(pre_roll_ms / frame_ms))`. In
`_process_frame`, append quiet frames while not collecting; when the threshold is
crossed, initialize `phrase_frames` from the deque plus the current voice frame,
then clear the deque. After emitting a phrase, clear collection state and the
pre-roll deque. Do not touch `_callback`.

```python
self._pre_roll = deque(
    maxlen=max(0, math.ceil(self.config.pre_roll_ms / self.config.frame_ms))
)

if not self.collecting:
    if rms >= self.config.start_threshold:
        self.collecting = True
        self.silence_frames = 0
        self.phrase_frames = [*self._pre_roll, frame]
        self._pre_roll.clear()
    else:
        self._pre_roll.append(frame)
    return None
```

- [ ] **Step 4: Run capture tests**

Run: `.venv/bin/python -m pytest test_audio_stream.py tests/test_audio_stream_backpressure.py tests/integration/test_audio_device_smoke.py -q`

Expected: unit tests pass; hardware smoke remains skipped unless explicitly enabled.

- [ ] **Step 5: Commit capture improvements**

```bash
git add audio_stream.py test_audio_stream.py tests/test_audio_stream_backpressure.py
git commit -m "fix: preserve phrase starts with audio pre-roll"
```

---

### Task 4: Child-owned Pipeline and Cancellable Playback

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `tests/test_audio_output.py`
- Modify: `tests/test_audio_queue_pipeline.py`

**Interfaces:**
- Consumes: `STTSelection`, `stt_model_dir()`, and `create_transcriber()`.
- Produces: `RunConfig(input_device, output_device, stt, auto_tts_model, manual_tts_model, tts_root)`.
- Produces: `prepare_audio_for_output(sd, data, source_rate, device_index)` moved unchanged from `audio_queue.py`.
- Produces: `play_audio_cancellable(sd, data, sample_rate, device_index, stop_event, chunk_ms=50) -> None`.
- Produces: `run_pipeline(config, stop_event, emit) -> None`, where `emit(kind: str, payload: str)`.
- Produces: top-level `pipeline_process_main(run_id, config, stop_event, event_queue) -> None`.

- [ ] **Step 1: Write failing cancellation and configuration tests**

```python
# tests/test_pipeline.py
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

from pipeline import RunConfig, play_audio_cancellable, run_pipeline
from stt_profiles import STTSelection


def config() -> RunConfig:
    return RunConfig(
        input_device=None, output_device=7,
        stt=STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"),
        auto_tts_model=True, manual_tts_model="ru_tts", tts_root=None,
    )


def test_playback_checks_stop_between_chunks() -> None:
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, False, True]
    stream = MagicMock()
    sd = MagicMock()
    sd.OutputStream.return_value.__enter__.return_value = stream
    audio = np.zeros(16000, dtype=np.float32)
    play_audio_cancellable(sd, audio, 16000, 7, stop_event, chunk_ms=50)
    assert stream.write.call_count == 2
    sd.play.assert_not_called()
    sd.wait.assert_not_called()


def test_run_pipeline_uses_conservative_stream_settings() -> None:
    stop_event = MagicMock()
    stop_event.is_set.return_value = True
    events: list[tuple[str, str]] = []
    sd = MagicMock()
    sd.query_devices.return_value = {"default_samplerate": 16000.0}
    with (
        patch("pipeline.get_sounddevice", return_value=sd),
        patch("pipeline.get_soundfile"),
        patch("pipeline.AudioPhraseStream") as stream_cls,
        patch("pipeline.create_transcriber") as create,
    ):
        create.return_value = SimpleNamespace(
            requested_device="cpu", actual_device="cpu"
        )
        run_pipeline(
            config(), stop_event,
            lambda kind, payload: events.append((kind, payload)),
        )
    stream_cfg = stream_cls.call_args.args[0]
    assert stream_cfg.frame_ms == 30
    assert stream_cfg.min_phrase_ms == 300
    assert stream_cfg.max_silence_ms == 700
    assert stream_cfg.pre_roll_ms == 200
```

- [ ] **Step 2: Run focused tests and confirm missing module failure**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_audio_output.py tests/test_audio_queue_pipeline.py -q`

Expected: collection fails because `pipeline.py` does not exist.

- [ ] **Step 3: Move the pipeline and add chunked playback**

Move `RunConfig`, `prepare_audio_for_output`, input sample-rate selection, phrase
processing, and the `_run` loop from `audio_queue.py` into `pipeline.py`. Replace
`stt_device`/`stt_model_size` with one `stt: STTSelection` field. Replace direct
Whisper construction with:

```python
model_dir = stt_model_dir(config.stt.engine, config.stt.model)
action = "Loading" if model_dir.exists() else "Downloading"
emit("status", f"{action} STT model {config.stt.model}...")
transcriber = create_transcriber(
    config.stt,
    on_warning=lambda message: emit("warning", message),
)
emit("state", "listening")
```

Use `StreamConfig(sample_rate=selected_rate, channels=1, frame_ms=30,
min_phrase_ms=300, max_silence_ms=700, pre_roll_ms=200,
device=config.input_device)`.

Implement playback with a child-owned stream:

```python
def play_audio_cancellable(
    sd, data, sample_rate, device_index, stop_event, chunk_ms=50
):
    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    channels = 1 if data.ndim == 1 else data.shape[1]
    with sd.OutputStream(
        samplerate=sample_rate, channels=channels,
        dtype="float32", device=device_index,
    ) as stream:
        for offset in range(0, len(data), frames_per_chunk):
            if stop_event.is_set():
                return
            stream.write(data[offset : offset + frames_per_chunk])
```

Check `stop_event.is_set()` after STT, before emitting text, after TTS, before
opening output, and between playback chunks. Wrap the process entry point so all
child events are tagged:

```python
def pipeline_process_main(run_id, config, stop_event, event_queue):
    def emit(kind: str, payload: str) -> None:
        event_queue.put((run_id, kind, payload))
    try:
        run_pipeline(config, stop_event, emit)
    except Exception as exc:
        emit("error", str(exc))
    finally:
        emit("worker_stopped", "")
```

Update old pipeline tests to import functional helpers from `pipeline`, assert
`OutputStream.write()` instead of global `play()`/`wait()`, and keep the existing
sample-rate/anti-alias assertions unchanged.

- [ ] **Step 4: Run pipeline and audio tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py tests/test_audio_output.py tests/test_audio_queue_pipeline.py -q`

Expected: all focused tests pass.

- [ ] **Step 5: Commit the child pipeline**

```bash
git add pipeline.py tests/test_pipeline.py tests/test_audio_output.py tests/test_audio_queue_pipeline.py
git commit -m "refactor: isolate cancellable speech pipeline"
```

---

### Task 5: Spawned Runner, Run-ID Filtering, and Stop Escalation

**Files:**
- Rewrite: `audio_queue.py`
- Create: `tests/test_runner_lifecycle.py`
- Modify: `tests/test_main_smoke.py`

**Interfaces:**
- Consumes: `RunConfig`, `pipeline_process_main`, and `prepare_audio_for_output` from Task 4.
- Preserves: `from audio_queue import RunConfig, SpeechLoopRunner, prepare_audio_for_output`.
- Produces: `SpeechLoopRunner(..., schedule, on_state, on_stopped, mp_context=None)`.
- Produces: `start()`, `stop()`, `is_running()`, and `active_run_id`.
- `schedule(delay_ms: int, callback: Callable[[], None]) -> object` must be non-blocking.

- [ ] **Step 1: Write failing parent lifecycle tests with fakes**

```python
# tests/test_runner_lifecycle.py
from unittest.mock import MagicMock, patch
from audio_queue import SpeechLoopRunner
from pipeline import RunConfig
from stt_profiles import STTSelection


def config() -> RunConfig:
    return RunConfig(
        input_device=None, output_device=7,
        stt=STTSelection("ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"),
        auto_tts_model=True, manual_tts_model="ru_tts", tts_root=None,
    )


class Scheduler:
    def __init__(self):
        self.calls = []
    def __call__(self, delay, callback):
        self.calls.append((delay, callback))


def build_runner(context, scheduler, events):
    return SpeechLoopRunner(
        config(),
        on_status=lambda msg: events.append(("status", msg)),
        on_text=lambda msg: events.append(("text", msg)),
        on_error=lambda msg: events.append(("error", msg)),
        on_warning=lambda msg: events.append(("warning", msg)),
        on_state=lambda state: events.append(("state", state)),
        on_stopped=lambda: events.append(("stopped", "")),
        schedule=scheduler, mp_context=context,
    )


def test_stop_never_calls_portaudio_and_schedules_escalation() -> None:
    context = MagicMock()
    process = context.Process.return_value
    process.is_alive.return_value = True
    scheduler = Scheduler()
    runner = build_runner(context, scheduler, [])
    runner.start()
    with patch("audio_backend.get_sounddevice") as get_sd:
        runner.stop()
    get_sd.assert_not_called()
    context.Event.return_value.set.assert_called_once()
    assert scheduler.calls[0][0] == 300


def test_stale_run_events_are_discarded() -> None:
    context = MagicMock()
    scheduler = Scheduler()
    events = []
    runner = build_runner(context, scheduler, events)
    runner._active_run_id = "new-run"
    runner._dispatch_event(("old-run", "text", "must not appear"))
    assert events == []


def test_escalation_terminates_then_kills_only_child() -> None:
    context = MagicMock()
    process = context.Process.return_value
    process.is_alive.return_value = True
    scheduler = Scheduler()
    runner = build_runner(context, scheduler, [])
    runner.start()
    runner.stop()
    scheduler.calls.pop(0)[1]()
    process.terminate.assert_called_once()
    assert scheduler.calls[0][0] == 1000
    scheduler.calls.pop(0)[1]()
    process.kill.assert_called_once()
```

- [ ] **Step 2: Run lifecycle tests and verify old thread runner fails**

Run: `.venv/bin/python -m pytest tests/test_runner_lifecycle.py tests/test_main_smoke.py -q`

Expected: failures because the old constructor has no scheduler/process context and `stop()` still calls sounddevice.

- [ ] **Step 3: Implement the parent process controller**

Use `multiprocessing.get_context("spawn")`, a UUID run ID, one cancellation
event, one event queue, and one process. `start()` validates no live worker,
creates primitives, starts the process, and starts a daemon event-monitor thread.
The monitor may block on the event queue and process completion; Tkinter may not.
Use `event_queue.get(timeout=0.1)` so the monitor also detects a child that exits
or is terminated without publishing `worker_stopped`.

```python
def stop(self) -> None:
    process = self._process
    if process is None:
        return
    self._active_run_id = None
    self._cancel_event.set()
    self.on_status("Stopping...")
    self._schedule(300, lambda: self._terminate_if_alive(process))


def _terminate_if_alive(self, process) -> None:
    if process is not self._process or not process.is_alive():
        return
    process.terminate()
    self._schedule(1000, lambda: self._kill_if_alive(process))


def _kill_if_alive(self, process) -> None:
    if process is self._process and process.is_alive():
        process.kill()
```

`_dispatch_event()` must compare the event run ID with `_active_run_id` before
calling any GUI callback, and map the child `state` kind to `on_state`. `worker_stopped` is handled by the monitor cleanup path,
not shown as a child status after cancellation. Cleanup closes the queue/process
handles and calls `on_stopped()` once. Do not import or call sounddevice anywhere
in `audio_queue.py`.

Re-export compatibility names:

```python
from pipeline import RunConfig, pipeline_process_main, prepare_audio_for_output
```

- [ ] **Step 4: Run lifecycle tests**

Run: `.venv/bin/python -m pytest tests/test_runner_lifecycle.py tests/test_main_smoke.py -q`

Expected: all tests pass and no test imports a real audio backend.

- [ ] **Step 5: Commit safe process lifecycle**

```bash
git add audio_queue.py tests/test_runner_lifecycle.py tests/test_main_smoke.py
git commit -m "fix: isolate Stop from Tkinter and PortAudio"
```

---

### Task 6: Dynamic GUI Profiles and Safe Window Close

**Files:**
- Modify: `gui.py`
- Modify: `main.py`
- Create: `tests/test_gui_profiles.py`
- Modify: `tests/test_main_startup.py`

**Interfaces:**
- Consumes: Task 1 catalog/labels and Task 5 runner lifecycle.
- Produces: GUI state values `idle`, `starting`, `listening`, `stopping`.
- Produces: `_collect_settings()` keys `stt_language`, `stt_engine`, `stt_model`, and `stt_device`.
- Produces: `AppGUI.set_pipeline_state(state: str) -> None`.
- Produces: `AppController.close() -> None` and `_runner_stopped() -> None`.

- [ ] **Step 1: Write failing GUI catalog/state tests without a display**

Factor selection logic into methods that accept current values and update
combobox `values`; test them using an instance created with `object.__new__` and
mock variables/widgets.

```python
# tests/test_gui_profiles.py
from unittest.mock import MagicMock
from gui import AppGUI


def bare_gui():
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


def test_stopping_disables_start_and_stop() -> None:
    gui = object.__new__(AppGUI)
    gui.start_button = MagicMock()
    gui.stop_button = MagicMock()
    gui.status_var = MagicMock()
    gui.set_pipeline_state("stopping")
    gui.start_button.configure.assert_called_with(state="disabled")
    gui.stop_button.configure.assert_called_with(state="disabled")
    gui.status_var.set.assert_called_with("Stopping...")
```

Add these controller tests to `tests/test_main_startup.py`:

```python
def test_controller_builds_complete_stt_selection() -> None:
    controller = object.__new__(AppController)
    controller.runner = None
    controller.input_map = {"Mic": 1}
    controller.output_map = {"Cable": 2}
    controller.root = MagicMock()
    controller.gui = MagicMock()
    settings = {
        "stt_language": "ru", "stt_engine": "gigaam",
        "stt_model": "gigaam-v3-e2e-rnnt", "stt_device": "cpu",
        "input_device_label": "Mic", "output_device_label": "Cable",
        "auto_tts_model": True, "manual_tts_model": "ru_tts",
        "tts_root": None,
    }
    with patch("main.SpeechLoopRunner") as runner_cls:
        controller.start(settings)
    config = runner_cls.call_args.kwargs["config"]
    assert config.stt == STTSelection(
        "ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"
    )


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
```

- [ ] **Step 2: Run GUI/controller tests and verify missing controls**

Run: `.venv/bin/python -m pytest tests/test_gui_profiles.py tests/test_main_startup.py -q`

Expected: failures for missing language/engine controls and lifecycle methods.

- [ ] **Step 3: Implement dynamic controls and state transitions**

In `AppGUI`, add language and engine readonly comboboxes before the model/device
controls. Bind `<<ComboboxSelected>>` for language and engine to
`_refresh_stt_choices()`. Display `Russian`/`English`, `GigaAM`/`Whisper`, and
the values from `MODEL_LABELS`; use reverse dictionaries in `_collect_settings()`
to return the internal IDs `ru`, `en`, `gigaam`, `whisper`, and the model ID.

Store the Start and Stop widgets as `self.start_button` and `self.stop_button`.
Implement:

```python
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
```

In `AppController.start()`, create `STTSelection` and validate it before building
`RunConfig`. Pass `schedule=self.root.after`, `on_warning` to a warning log event,
`on_state=lambda state: self.gui.enqueue_event("state", state)`, and
`on_stopped=lambda: self.gui.enqueue_event("worker_stopped", "")` to
`SpeechLoopRunner`.

In `AppController.stop()`, set GUI state to `stopping` then call runner stop. The
GUI handles `worker_stopped` inside `_poll_ui_queue` on the Tk thread and invokes
the controller's `_runner_stopped()` callback. `_runner_stopped()` sets idle state
or destroys the root if `_closing` is true.
Install `root.protocol("WM_DELETE_WINDOW", self.close)`. `close()` sets
`_closing=True`; if a runner is active it invokes the same stop path, otherwise it
destroys immediately.

Extend the GUI event queue with `warning` and `state` kinds. Warnings append to
the log without opening a fatal message box. Add an `on_worker_stopped` callback
to `AppGUI.__init__`; the `worker_stopped` queue kind invokes it from
`_poll_ui_queue`. Only Tkinter's `_poll_ui_queue` changes widgets.

- [ ] **Step 4: Run GUI and controller tests**

Run: `.venv/bin/python -m pytest tests/test_gui_profiles.py tests/test_main_startup.py tests/test_main_smoke.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit GUI/controller integration**

```bash
git add gui.py main.py tests/test_gui_profiles.py tests/test_main_startup.py
git commit -m "feat: add bilingual STT controls and safe shutdown UI"
```

---

### Task 7: Dependencies, PyInstaller, and Frozen Multiprocessing Smoke

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `installer/V2TTS.spec`
- Modify: `smoke_test.py`
- Modify: `main.py`
- Modify: `tests/test_packaged_smoke.py`
- Modify: `tests/test_windows_workflow.py`
- Modify: `tests/test_installer_build.py`

**Interfaces:**
- Produces: runtime dependency `onnx-asr[cpu,hub]>=0.12,<0.13`.
- Produces: `run_multiprocessing_smoke() -> None` invoked by packaged `--smoke-test`.
- Preserves: `V2TTS.exe --smoke-test` exit code zero without downloading STT models or opening audio devices.

- [ ] **Step 1: Write failing packaging/smoke assertions**

```python
# append to tests/test_packaged_smoke.py
def test_multiprocessing_smoke_round_trip() -> None:
    assert run_multiprocessing_smoke() is None


# append to tests/test_windows_workflow.py
def test_windows_build_installs_onnx_asr() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    assert "onnx-asr[cpu,hub]>=0.12,<0.13" in requirements


def test_pyinstaller_collects_onnx_asr_without_model_weights() -> None:
    spec = Path("installer/V2TTS.spec").read_text(encoding="utf-8")
    assert 'collect_submodules("onnx_asr")' in spec
    assert 'collect_dynamic_libs("onnxruntime")' in spec
    assert "gigaam-v3-e2e-rnnt" not in spec
```

- [ ] **Step 2: Run packaging tests and verify failure**

Run: `.venv/bin/python -m pytest tests/test_packaged_smoke.py tests/test_windows_workflow.py tests/test_installer_build.py -q`

Expected: failures for the missing multiprocessing smoke and ONNX dependency/import collection.

- [ ] **Step 3: Add dependencies and frozen runtime collection**

Add the exact requirement to both dependency files:

```text
onnx-asr[cpu,hub]>=0.12,<0.13
```

Add `onnx_asr` to `main.RUNTIME_DEPENDENCIES` so a missing or unusable runtime is
reported before the GUI starts.

In `installer/V2TTS.spec`, append `collect_submodules("onnx_asr")` to
`hiddenimports`, keep `collect_dynamic_libs("onnxruntime")`, and collect package
data required by `onnx_asr` itself. Do not add a model name, user cache path, or
Hugging Face snapshot to `datas`.

Add a top-level picklable smoke child and round trip:

```python
def _multiprocessing_smoke_child(result_queue) -> None:
    result_queue.put("ok")


def run_multiprocessing_smoke() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_multiprocessing_smoke_child, args=(result_queue,)
    )
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        raise RuntimeError("Multiprocessing smoke child did not exit")
    if process.exitcode != 0 or result_queue.get(timeout=2) != "ok":
        raise RuntimeError("Multiprocessing smoke child failed")
```

Call `multiprocessing.freeze_support()` at the first line of `main()` before
dependency checks or Tkinter construction. Invoke `run_multiprocessing_smoke()`
inside the existing `--smoke-test` path before TTS smoke cases.

- [ ] **Step 4: Run packaging-focused tests**

Run: `.venv/bin/python -m pytest tests/test_packaged_smoke.py tests/test_windows_workflow.py tests/test_installer_build.py -q`

Expected: all tests pass without model download.

- [ ] **Step 5: Commit packaging support**

```bash
git add pyproject.toml requirements.txt installer/V2TTS.spec smoke_test.py main.py tests/test_packaged_smoke.py tests/test_windows_workflow.py tests/test_installer_build.py
git commit -m "build: package GigaAM runtime and worker startup"
```

---

### Task 8: Documentation, Migration, and Complete Verification

**Files:**
- Modify: `README.md`
- Modify: `installer/README.md`
- Modify: any tests that still construct the legacy `RunConfig` fields.

**Interfaces:**
- Consumes: final UI, model locations, process lifecycle, and packaging behavior.
- Produces: user-facing Windows instructions matching actual behavior.

- [ ] **Step 1: Find and replace every legacy configuration reference**

Run:

```bash
rg -n "STT_MODEL_SIZES|stt_model_size|language=None|sd\.play|sd\.wait|sd\.stop|_MEI.*models" . --glob '!docs/superpowers/**'
```

Expected before cleanup: legacy references remain only in tests or documentation.
For each executable-code result, replace it with `STTSelection`, explicit
language, process-owned playback, or external model-path behavior. The final
search may retain `sd.stop` only inside a negative test assertion/string.

- [ ] **Step 2: Update English and Russian README sections**

Document these exact facts in both language sections:

- Russian default: local `GigaAM v3 E2E RNN-T`.
- Russian alternatives: GigaAM E2E CTC and Whisper small/medium/large-v3.
- English: explicit-English Whisper small/medium/large-v3.
- Model path: `%LOCALAPPDATA%\V2TTS\models`.
- First selection downloads weights; later recognition works offline.
- The executable contains runtimes but not STT model weights.
- `Stop` interrupts the worker and keeps the window open.
- A CUDA fallback is visible in the status/log; CPU remains supported.

Update `installer/README.md` to state that building the executable does not
download or embed GigaAM/Whisper weights and that frozen smoke starts a spawned
child but does not load STT models.

- [ ] **Step 3: Run focused feature verification (pass 1)**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_stt_profiles.py tests/test_stt.py \
  test_audio_stream.py tests/test_audio_stream_backpressure.py \
  tests/test_pipeline.py tests/test_runner_lifecycle.py \
  tests/test_gui_profiles.py tests/test_main_startup.py \
  tests/test_packaged_smoke.py -q
```

Expected: all focused tests pass; no model is downloaded and no real audio
device is opened.

- [ ] **Step 4: Run full local verification (pass 2)**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q .
git diff --check
```

Expected: full suite passes with only the opt-in hardware test skipped;
`compileall` and `git diff --check` exit zero.

- [ ] **Step 5: Commit documentation and compatibility cleanup**

```bash
git add README.md installer/README.md tests
git commit -m "docs: explain external bilingual STT models"
```

- [ ] **Step 6: Request code review before publishing**

Use `superpowers:requesting-code-review` against the implementation range. Fix
every Critical or Important finding with a failing regression test first, then
repeat Steps 3 and 4.

- [ ] **Step 7: Publish a PR and verify Windows (pass 3)**

Use `github:yeet` to confirm the diff, push a dedicated feature branch, and open
a draft PR. Wait for `.github/workflows/windows-frozen-smoke.yml` and inspect its
test, PyInstaller build, and `V2TTS.exe --smoke-test` steps using
`github:gh-fix-ci` if any check fails.

Expected: Windows frozen workflow succeeds without downloading an STT model.

- [ ] **Step 8: Merge only after all gates are green**

Mark the PR ready, merge it into `main`, and verify the merge commit and final
`main` check status through GitHub. Report the PR URL, merge SHA, test totals,
Windows workflow result, external model directory, and any first-run download
expectation to the user.
