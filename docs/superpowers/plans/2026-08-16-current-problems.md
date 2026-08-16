# V2TTS Current Problems Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound real-time audio backlog, expose overload/latency telemetry, make startup deterministic, and add packaged Windows plus optional real-device verification.

**Architecture:** `AudioPhraseStream` owns a 1.5-second frame buffer with drop-oldest backpressure and immutable queue metrics. Captured phrases carry their last-frame timestamp into `SpeechLoopRunner`, which reports phrase age and STT/TTS durations. A headless `--smoke-test` path validates both bundled TTS engines inside the PyInstaller executable, while real-device testing remains explicitly opt-in.

**Tech Stack:** Python 3.12+, NumPy, pytest, sounddevice, soundfile, PyInstaller, GitHub Actions Windows runners.

## Global Constraints

- The capture callback must never block.
- The default frame buffer is bounded to 1500 ms of audio.
- On overflow, the oldest queued frame is discarded and the newest frame is retained.
- Existing phrase segmentation thresholds and single-pass STT/TTS behavior remain unchanged.
- Runtime startup must never invoke `pip` or modify the Python environment.
- Hardware-dependent tests are opt-in and skipped in the default suite.
- The packaged smoke test must execute both `ru_tts` and SAM without opening the GUI.

---

### Task 1: Bounded audio buffer and drop-oldest metrics

**Files:**
- Modify: `audio_stream.py`
- Create: `tests/test_audio_stream_backpressure.py`
- Modify: `test_audio_stream.py`

**Interfaces:**
- Produces: `StreamMetrics(queue_depth_frames: int, queue_ms: int, dropped_frames: int)`.
- Produces: `CapturedPhrase(pcm16: np.ndarray, ended_at: float)`.
- Produces: `AudioPhraseStream.metrics() -> StreamMetrics`.

- [ ] **Step 1: Write failing behavioral tests**

```python
def test_overflow_keeps_latest_frames_and_counts_drops():
    stream = AudioPhraseStream(StreamConfig(sample_rate=1000, frame_ms=10, max_buffer_ms=30))
    for value in range(5):
        stream._callback(np.full((10, 1), value, dtype=np.float32), 10, {}, None)
    queued = [stream._frames_q.get_nowait().samples[0] for _ in range(3)]
    assert queued == [2.0, 3.0, 4.0]
    assert stream.metrics() == StreamMetrics(3, 30, 2)
```

Add a 50,000-callback soak test that asserts queue depth never exceeds the literal capacity and `dropped_frames == 49_850` for a 150-frame buffer.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_audio_stream_backpressure.py -q`

Expected: failure because `max_buffer_ms`, packet types, and metrics do not exist.

- [ ] **Step 3: Implement minimal bounded queue**

Add frozen `FramePacket`, `CapturedPhrase`, and `StreamMetrics` dataclasses. Compute capacity with `max(1, ceil(max_buffer_ms / frame_ms))`. In `_callback`, call `put_nowait`; on `queue.Full`, remove one item with `get_nowait`, increment the drop counter, and enqueue the newest packet. Return immutable metrics from queue size and the counter.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_audio_stream_backpressure.py test_audio_stream.py -q`

Commit: `fix: bound audio capture backlog`

### Task 2: Runtime latency and overload telemetry

**Files:**
- Modify: `audio_queue.py`
- Modify: `tests/test_audio_queue_pipeline.py`

**Interfaces:**
- Consumes: `CapturedPhrase.pcm16`, `CapturedPhrase.ended_at`, and `AudioPhraseStream.metrics()`.
- Produces: status fields `stt_ms`, `tts_ms`, `queue_ms`, `phrase_age_ms`, and `dropped_frames`.

- [ ] **Step 1: Write failing timing/status tests**

Use a controlled monotonic clock sequence and a real `CapturedPhrase`. Assert that a successful phrase emits one status containing literal values for all five metrics, and that a dropped-frame count produces an `Audio overload:` warning before phrase processing.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_audio_queue_pipeline.py -q`

Expected: failures because the runner currently accepts raw arrays and reports no timing/backlog fields.

- [ ] **Step 3: Implement telemetry**

Time STT and TTS with `time.perf_counter()`. Compute phrase age from `time.monotonic() - ended_at`. Snapshot stream metrics after dequeuing each phrase, warn when the dropped counter increases, and include formatted integer milliseconds in the listening status.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_audio_queue_pipeline.py test_audio_queue.py -q`

Commit: `feat: report audio backlog and phrase latency`

### Task 3: Deterministic startup without runtime installation

**Files:**
- Modify: `main.py`
- Modify: `tests/test_main_smoke.py`

**Interfaces:**
- Produces: `check_runtime_dependencies() -> None`, which imports required packages and raises a descriptive `RuntimeError` without invoking a package manager.

- [ ] **Step 1: Write failing dependency-check tests**

Patch `builtins.__import__` to fail for `sounddevice`, call `check_runtime_dependencies()`, and assert the error names the missing package and the documented install command. Patch `subprocess.check_call` and assert it is never called.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_main_smoke.py -q`

Expected: failure because dependency checking happens as an import-time pip installation.

- [ ] **Step 3: Implement fail-fast dependency checking**

Remove `subprocess` and the import-time installer. Import packages through `importlib.import_module` inside `check_runtime_dependencies`; aggregate missing names and raise `RuntimeError("Missing runtime dependencies: ... Install with: python -m pip install -r requirements.txt")`. Call it at the start of normal GUI execution and packaged smoke execution.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_main_smoke.py -q`

Commit: `fix: remove runtime dependency installation`

### Task 4: Headless packaged TTS smoke mode

**Files:**
- Create: `smoke_test.py`
- Modify: `main.py`
- Create: `tests/test_packaged_smoke.py`

**Interfaces:**
- Produces: `run_packaged_smoke(tts_root: str | None = None) -> int`.
- Produces: CLI `V2TTS.exe --smoke-test` that exits 0 after validating `ru_tts` and SAM WAV output.

- [ ] **Step 1: Write failing smoke-mode tests**

Patch only `smoke_test.synthesize_text` to write two real WAV fixtures with `soundfile`; call `run_packaged_smoke()` and assert both engine names were requested, both WAV files were read, and the return value is 0. Patch `main.run_packaged_smoke` and assert `main(["--smoke-test"])` avoids constructing `AppController`.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_packaged_smoke.py -q`

Expected: import failure because smoke mode does not exist.

- [ ] **Step 3: Implement smoke mode**

Synthesize short Russian and English phrases into a temporary directory using manual `ru_tts` and `sam`; validate non-empty samples and positive sample rates through `get_soundfile()`. Route `--smoke-test` before GUI construction and return its exit code.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_packaged_smoke.py -q`

Commit: `feat: add packaged TTS smoke mode`

### Task 5: Windows frozen CI and opt-in real-device integration

**Files:**
- Create: `.github/workflows/windows-frozen-smoke.yml`
- Create: `pytest.ini`
- Create: `tests/integration/test_audio_device_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Produces: Windows workflow that builds with recursive submodules and runs `dist/V2TTS.exe --smoke-test`.
- Produces: opt-in test enabled by `V2TTS_REAL_AUDIO_TEST=1`.

- [ ] **Step 1: Add executable integration test and workflow validation test**

The real-device test opens `AudioPhraseStream.iter_phrases()` against the configured default input for at most one second and asserts bounded queue metrics; it skips unless the environment variable is exactly `1`. Add a unit test that parses the workflow YAML and asserts the Windows job checks out recursive submodules, runs `installer/build.py`, and executes the built smoke mode.

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/test_windows_workflow.py tests/integration/test_audio_device_smoke.py -q`

Expected: workflow test fails because the workflow is absent; hardware test is skipped by default.

- [ ] **Step 3: Add CI workflow and documentation**

Use `windows-latest`, `actions/checkout@v4` with `submodules: recursive`, `actions/setup-python@v5` with Python 3.12, MSYS2 MinGW GCC, dependency installation, `python installer/build.py --skip-install`, and `dist/V2TTS.exe --smoke-test`. Document local smoke, soak, and opt-in hardware commands.

- [ ] **Step 4: Verify GREEN and commit**

Run: `.venv/bin/python -m pytest tests/test_windows_workflow.py tests/integration/test_audio_device_smoke.py -q`

Commit: `ci: verify frozen Windows build`

### Task 6: Repository-wide verification and publication

- [ ] Run `.venv/bin/python -m compileall -q .`.
- [ ] Run `.venv/bin/python -m pytest -q` three times.
- [ ] Run `git diff --check main...HEAD` and inspect all changed files.
- [ ] Publish `codex/fix-current-problems`, open a draft PR into `main`, confirm remote diff, then merge only if explicitly requested.
