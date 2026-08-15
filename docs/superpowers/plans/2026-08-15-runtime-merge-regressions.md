# V2TTS Runtime Merge Regressions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a compilable, single-pass speech pipeline whose Russian TTS behavior matches the documented packaged architecture.

**Architecture:** Keep `SpeechLoopRunner` as the single phrase orchestrator: `_run()` initializes resources and delegates each phrase once, while `_process_phrase()` owns STT, text notification, temporary WAV synthesis, playback, and cleanup. Restore `ru_tts`/SAM routing and remove the accidental Silero dependency so source and frozen builds agree.

**Tech Stack:** Python 3.12+, pytest, unittest.mock, tkinter, faster-whisper, sounddevice, soundfile, vendored ru_tts and SAM engines.

## Global Constraints

- A non-empty phrase performs exactly one STT, one text callback, one TTS, and one playback operation.
- Cyrillic auto-routing selects `ru_tts`; Latin auto-routing selects `sam`.
- `torch` and Silero must not become packaged runtime requirements.
- Per-phrase failures must not terminate the listening loop.
- Temporary WAV files must be removed on success and failure.

---

### Task 1: Entry-point and callback contract

**Files:**
- Create: `tests/test_main_smoke.py`
- Modify: `main.py`
- Modify: `audio_queue.py`

**Interfaces:**
- Consumes: `SpeechLoopRunner(config, on_status, on_text, on_error)`.
- Produces: stored `on_status`, `on_text`, and `on_error` callbacks and a compilable `main.py` call site.

- [ ] **Step 1: Write failing compile and constructor tests**

```python
def test_main_compiles():
    source = Path("main.py").read_text(encoding="utf-8")
    compile(source, "main.py", "exec")

def test_runner_stores_all_callbacks(run_config):
    callbacks = [lambda _: None for _ in range(3)]
    runner = SpeechLoopRunner(run_config, callbacks[0], callbacks[1], callbacks[2])
    assert (runner.on_status, runner.on_text, runner.on_error) == tuple(callbacks)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_main_smoke.py -q`

Expected: compile test fails with `SyntaxError: keyword argument repeated`; constructor test fails because `on_text` is not accepted.

- [ ] **Step 3: Implement minimal callback wiring**

Keep one `enqueue_event` callback of each type in `main.py`; add `on_text: Callable[[str], None]` between status and error in the runner constructor and assign `self.on_text = on_text`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_main_smoke.py -q`

Expected: `2 passed`.

Commit: `fix: restore controller runner callback contract`

### Task 2: Single-pass phrase processing and cleanup

**Files:**
- Create: `tests/test_audio_queue_pipeline.py`
- Modify: `audio_queue.py`

**Interfaces:**
- Consumes: `_process_phrase(phrase_pcm16, transcriber, sample_rate, stt_desc, sd, sf)`.
- Produces: one recoverable phrase-processing operation with guaranteed temporary-file cleanup.

- [ ] **Step 1: Write failing behavioral tests**

Use literal fake transcriber, sounddevice, and soundfile objects plus patched `audio_queue.synthesize_text`. Assert one STT call, one text callback, one synthesis call, `sf.read(path, dtype="float32")`, one `sd.play`, one `sd.wait`, and `not Path(path).exists()` after return. Add separate tests for empty text, STT exception, and synthesis exception.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_audio_queue_pipeline.py -q`

Expected: failures show the missing `tempfile` import and unsupported constructor callback contract until Task 1 is complete.

- [ ] **Step 3: Implement the minimal pipeline repair**

Import `tempfile`, remove `io`, and delete the duplicated STT/TTS/playback block after `_process_phrase()` in `_run()`. Keep the existing path-based `mkstemp` flow in `_process_phrase()`, close its descriptor, and remove its path in `finally`.

- [ ] **Step 4: Verify GREEN and commit**

Run: `python -m pytest tests/test_audio_queue_pipeline.py test_audio_queue.py -q`

Expected: all pipeline and sample-rate tests pass.

Commit: `fix: process each captured phrase exactly once`

### Task 3: Restore packaged TTS routing

**Files:**
- Modify: `tests/test_tts.py`
- Modify: `tts.py`
- Modify: `main.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `choose_tts_engine(text, auto_select=True, manual_model="ru_tts")`.
- Produces: `TTS_MODELS == ["ru_tts", "sam"]`, Cyrillic to `ru_tts`, Latin to `sam`, and `ru_tts` to SAM fallback.

- [ ] **Step 1: Change tests to the documented behavior and verify RED**

```python
def test_auto_select_languages():
    assert choose_tts_engine("Привет мир") == "ru_tts"
    assert choose_tts_engine("hello world") == "sam"

def test_packaged_models_only():
    assert TTS_MODELS == ["ru_tts", "sam"]
```

Run: `python -m pytest tests/test_tts.py -q`

Expected: Cyrillic and supported-model assertions fail because current code selects/includes Silero.

- [ ] **Step 2: Implement the minimal routing repair**

Remove Silero state/function/branch, set `TTS_MODELS` to `ru_tts` and `sam`, default manual model to `ru_tts`, route Cyrillic to `ru_tts`, remove `silero_tts` from the startup dependency probe, and remove `silero-tts` from `requirements.txt`.

- [ ] **Step 3: Verify GREEN and commit**

Run: `python -m pytest tests/test_tts.py -q`

Expected: all TTS tests pass.

Commit: `fix: align Russian TTS with packaged ru_tts engine`

### Task 4: Repository-wide verification

**Files:**
- Modify only if a verification failure identifies a regression caused by Tasks 1-3.

**Interfaces:**
- Consumes: all repaired runtime contracts.
- Produces: a clean compile and test result suitable for publication.

- [ ] **Step 1: Compile all Python sources**

Run: `python -m compileall -q .`

Expected: exit code 0 and no output.

- [ ] **Step 2: Run the complete suite**

Run: `python -m pytest -q`

Expected: all tests pass with no errors.

- [ ] **Step 3: Inspect the final diff and dependency consistency**

Run: `git diff main...HEAD --check && rg -n "silero_tts|silero-tts|return \"silero\"" --glob '*.py' --glob '*.txt' .`

Expected: diff check succeeds and ripgrep finds no production/dependency references.

- [ ] **Step 4: Publish branch and open a draft pull request**

Push `codex/fix-runtime-merge-regressions` and open a draft PR into `main` summarizing root causes and verification evidence.
