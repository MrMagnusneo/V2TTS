# Local Streaming STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local Russian and English sherpa-onnx streaming recognition that begins short, deduplicated TTS chunks before the speaker finishes, while preserving the existing phrase mode.

**Architecture:** Keep batch GigaAM/faster-whisper behind the existing `Transcriber` interface and add a separate online recognizer interface. A nonblocking audio-frame iterator feeds sherpa-onnx; a pure stable-text committer emits speakable chunks; a generation-aware worker synthesizes and plays only current chunks. The GUI selects streaming by default, persists independent streaming and phrase profiles, and falls back visibly to same-language phrase recognition when streaming initialization fails.

**Tech Stack:** Python 3.13, NumPy, sounddevice/PortAudio, sherpa-onnx 1.13.5, tkinter, multiprocessing spawn, threading, PyInstaller, pytest, GitHub Actions on Windows.

## Global Constraints

- `Streaming` is the default; `After phrase` remains selectable.
- Russian and English use explicit, separate profiles; there is no automatic language detection or mid-utterance switch.
- Models live below `%LOCALAPPDATA%\V2TTS\models\sherpa-onnx\<profile-id>\` and are never bundled into the EXE.
- The PortAudio callback performs only an in-memory copy and nonblocking bounded-queue handoff.
- Partial hypotheses are visible but are never spoken directly.
- Prefer a stable sentence/clause boundary after at least 3 words; otherwise target 6 words, force at 8, and flush a non-empty tail after 350 ms of silence.
- New user speech invalidates old playback and queued TTS results by generation.
- Stop cancels capture, STT, TTS, and playback without closing the GUI.
- Windowed Windows/spawn entry points must restore standard streams before progress-capable imports or model initialization.
- Required CI must not download model weights or require an audio device.

## File map

**Create:**

- `streaming_models.py` — immutable model catalog, path validation, verified download, safe extraction, atomic installation.
- `streaming_stt.py` — online-recognizer protocol, sherpa adapter, result type, and stable-text committer.
- `streaming_pipeline.py` — speech/endpoint state machine and generation-aware TTS scheduling.
- `app_settings.py` — atomic JSON persistence and migration defaults for UI selections.
- `tests/test_streaming_models.py` — model catalog and installation failure tests.
- `tests/test_streaming_stt.py` — committer and sherpa-adapter contract tests.
- `tests/test_streaming_pipeline.py` — fake recognizer/frame/TTS integration tests.
- `tests/test_app_settings.py` — settings migration and atomic-save tests.
- `tests/integration/test_streaming_model_smoke.py` — opt-in real-model/audio-free waveform smoke test.

**Modify:**

- `audio_stream.py` — expose production frame iteration while retaining phrase iteration.
- `pipeline.py` — split phrase runner, dispatch by mode, share cancellable synthesis/playback helper, and perform visible fallback.
- `audio_queue.py` — forward mutable partial events from the worker.
- `stt_profiles.py` — add streaming selection types and language-to-profile mapping.
- `gui.py` — mode/profile controls, partial transcript field, state locking, persisted values.
- `main.py` — load/save settings, build the expanded `RunConfig`, and validate sherpa at startup.
- `requirements.txt`, `pyproject.toml` — pin sherpa-onnx 1.13.5.
- `installer/V2TTS.spec` — collect sherpa Python metadata and native libraries without model weights.
- `smoke_test.py` — validate packaged sherpa runtime without loading a model.
- `.github/workflows/windows-frozen-smoke.yml` — retain Windows Python 3.13 build and run the expanded packaged smoke.
- `README.md` — document streaming/phrase modes, downloads, paths, and latency behavior.

---

### Task 1: Streaming profiles and verified external model installation

**Files:**
- Create: `streaming_models.py`
- Create: `tests/test_streaming_models.py`
- Modify: `stt_profiles.py`
- Modify: `tests/test_stt_profiles.py`

**Interfaces:**
- Produces: `StreamingSTTSelection(language: str, profile: str)`.
- Produces: `StreamingModelProfile`, `STREAMING_MODEL_PROFILES`, `streaming_model_dir(profile_id, root=None)`, `is_streaming_model_ready(profile_id, root=None)`, `install_streaming_model_archive(profile, source, root, on_progress) -> Path`, and `ensure_streaming_model(profile_id, stop_event, on_progress, root=None) -> Path`.
- Consumes: `stt_profiles.user_data_root()`.

- [ ] **Step 1: Write failing profile and manifest tests**

```python
def test_streaming_profiles_are_language_specific() -> None:
    assert default_streaming_selection("ru") == StreamingSTTSelection(
        "ru", "sherpa_streaming_ru_t_one"
    )
    assert default_streaming_selection("en") == StreamingSTTSelection(
        "en", "sherpa_streaming_en_zipformer_20m"
    )
    with pytest.raises(ValueError, match="not available"):
        validate_streaming_selection(
            StreamingSTTSelection("en", "sherpa_streaming_ru_t_one")
        )


def test_manifest_pins_verified_archives() -> None:
    ru = STREAMING_MODEL_PROFILES["sherpa_streaming_ru_t_one"]
    en = STREAMING_MODEL_PROFILES["sherpa_streaming_en_zipformer_20m"]
    assert (ru.archive_size, ru.sha256) == (
        128468156,
        "b9c907450e99a6e5049e279bf18368a17db0bdc5e63b7fa978943138debbe3ae",
    )
    assert (en.archive_size, en.sha256) == (
        127887156,
        "9c559283e8498d3fe95913c79ca1cb454bb26281ac2b102b41306c7d752765d9",
    )
    assert ru.required_files == ("model.onnx", "tokens.txt")
    assert en.required_files == (
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "tokens.txt",
    )
```

- [ ] **Step 2: Run the tests and confirm missing symbols fail**

Run: `.venv/bin/python -m pytest tests/test_stt_profiles.py tests/test_streaming_models.py -q`

Expected: collection fails because `StreamingSTTSelection` and `streaming_models` do not exist.

- [ ] **Step 3: Add the selection type and exact manifest**

```python
@dataclass(frozen=True)
class StreamingSTTSelection:
    language: str
    profile: str


_STREAMING_BY_LANGUAGE = {
    "ru": ("sherpa_streaming_ru_t_one",),
    "en": ("sherpa_streaming_en_zipformer_20m",),
}


def default_streaming_selection(language: str = "ru") -> StreamingSTTSelection:
    try:
        return StreamingSTTSelection(language, _STREAMING_BY_LANGUAGE[language][0])
    except KeyError as exc:
        raise ValueError(f"Unknown STT language: {language}") from exc


def validate_streaming_selection(selection: StreamingSTTSelection) -> None:
    if selection.profile not in _STREAMING_BY_LANGUAGE.get(selection.language, ()):
        raise ValueError(
            f"Streaming profile {selection.profile} is not available for "
            f"language {selection.language}"
        )
```

Define `StreamingModelProfile` with `profile_id`, `language`, `architecture`, `url`, `archive_size`, `sha256`, `archive_root`, `required_files`, and `sample_rate`. Populate the two exact archives and hashes asserted above; use architecture values `t_one_ctc` and `transducer`. Use the models' feature rates from the official factories: Russian T-One `8000`, English Zipformer `16000`. `accept_waveform` still receives the actual device rate and performs internal resampling.

Use these exact official URLs and archive roots:

```python
RU_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2"
RU_ROOT = "sherpa-onnx-streaming-t-one-russian-2025-09-08"
EN_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
EN_ROOT = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
```

- [ ] **Step 4: Add failing installation tests with injected archive bytes**

```python
def build_test_tar_bz2(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(f"fixture/{name}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def fixture_profile(archive: bytes, required_files=("model.onnx", "tokens.txt")):
    return StreamingModelProfile(
        profile_id="fixture",
        language="ru",
        architecture="t_one_ctc",
        url="https://invalid.example/fixture.tar.bz2",
        archive_size=len(archive),
        sha256=hashlib.sha256(archive).hexdigest(),
        archive_root="fixture",
        required_files=required_files,
        sample_rate=16000,
    )


def test_install_verifies_then_atomically_marks_ready(
    tmp_path: Path, monkeypatch
) -> None:
    archive = build_test_tar_bz2({"model.onnx": b"m", "tokens.txt": b"t"})
    profile = fixture_profile(archive)
    monkeypatch.setitem(STREAMING_MODEL_PROFILES, profile.profile_id, profile)
    progress = []
    installed = install_streaming_model_archive(
        profile,
        io.BytesIO(archive),
        root=tmp_path,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert (installed / "model.onnx").read_bytes() == b"m"
    assert (installed / ".v2tts-complete").is_file()
    assert is_streaming_model_ready(profile.profile_id, root=tmp_path)
    assert not list(tmp_path.rglob("*.partial"))


def test_wrong_hash_never_becomes_ready(tmp_path: Path) -> None:
    archive = build_test_tar_bz2({"model.onnx": b"m", "tokens.txt": b"t"})
    profile = dataclasses.replace(fixture_profile(archive), sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        install_streaming_model_archive(
            profile, io.BytesIO(archive), root=tmp_path, on_progress=lambda *_: None
        )
    assert not any(path.name == ".v2tts-complete" for path in tmp_path.rglob("*"))


def test_missing_required_file_never_becomes_ready(tmp_path: Path) -> None:
    archive = build_test_tar_bz2({"tokens.txt": b"t"})
    with pytest.raises(ValueError, match="model.onnx"):
        install_streaming_model_archive(
            fixture_profile(archive), io.BytesIO(archive),
            root=tmp_path, on_progress=lambda *_: None,
        )


def test_wrong_size_is_rejected_before_extraction(tmp_path: Path) -> None:
    archive = build_test_tar_bz2({"model.onnx": b"m", "tokens.txt": b"t"})
    profile = dataclasses.replace(
        fixture_profile(archive), archive_size=len(archive) + 1
    )
    with pytest.raises(ValueError, match="size"):
        install_streaming_model_archive(
            profile, io.BytesIO(archive), root=tmp_path, on_progress=lambda *_: None
        )


def test_parent_traversal_member_is_rejected(tmp_path: Path) -> None:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive_file:
        info = tarfile.TarInfo("fixture/../../escape.txt")
        info.size = 1
        archive_file.addfile(info, io.BytesIO(b"x"))
    payload = output.getvalue()
    with pytest.raises(ValueError, match="unsafe archive member"):
        install_streaming_model_archive(
            fixture_profile(payload, required_files=()), io.BytesIO(payload),
            root=tmp_path, on_progress=lambda *_: None,
        )
    assert not (tmp_path.parent / "escape.txt").exists()
```

- [ ] **Step 5: Implement safe, cancellable installation**

Implement archive download in 1 MiB chunks. Check `stop_event.is_set()` before every network read and raise `ModelDownloadCancelled`. Verify byte count and SHA-256 before extraction. For every tar member, resolve the destination and reject it unless `destination.is_relative_to(temp_root.resolve())`; reject symlinks and hard links. Validate required files, write `.v2tts-complete`, then use `os.replace(temp_model_dir, final_dir)`. Never delete an already-ready final directory.

```python
def ensure_streaming_model(
    profile_id: str,
    stop_event,
    on_progress: Callable[[int, int], None],
    root: Path | None = None,
) -> Path:
    profile = STREAMING_MODEL_PROFILES[profile_id]
    destination = streaming_model_dir(profile_id, root)
    if _validate_installed(profile, destination):
        return destination
    with urllib.request.urlopen(profile.url, timeout=30) as response:
        return _download_verify_extract(
            profile, response, destination, stop_event, on_progress
        )
```

- [ ] **Step 6: Run focused tests and commit**

Run: `.venv/bin/python -m pytest tests/test_stt_profiles.py tests/test_streaming_models.py -q`

Expected: all focused tests pass, including cancellation, path traversal, wrong hash, wrong size, missing files, and preservation of an existing verified model.

```bash
git add stt_profiles.py streaming_models.py tests/test_stt_profiles.py tests/test_streaming_models.py
git commit -m "feat: add verified streaming model profiles"
```

---

### Task 2: Stable partial-text commitment

**Files:**
- Create: `streaming_stt.py`
- Create: `tests/test_streaming_stt.py`

**Interfaces:**
- Produces: `StreamingResult(text: str, endpoint: bool)` and `OnlineTranscriber` protocol.
- Produces: `StableTextCommitter.observe(text) -> list[str]`, `.flush_pause() -> list[str]`, `.finish(text) -> list[str]`, and `.reset() -> None`.
- Has no dependency on audio, sherpa-onnx, tkinter, or TTS.

- [ ] **Step 1: Write failing Russian/English stability and deduplication tests**

```python
def test_revised_tail_is_not_spoken_and_committed_words_do_not_repeat() -> None:
    c = StableTextCommitter()
    assert c.observe("я хочу проверить потоковую") == []
    assert c.observe("я хочу проверить потоковое распознавание") == []
    assert c.observe("я хочу проверить потоковое распознавание речи") == []
    chunks = c.observe("я хочу проверить потоковое распознавание речи сейчас")
    assert chunks == ["я хочу проверить потоковое распознавание речи"]
    assert c.finish("я хочу проверить потоковое распознавание речи сейчас") == [
        "сейчас"
    ]


def test_clause_boundary_commits_after_three_words() -> None:
    c = StableTextCommitter()
    c.observe("when this works, we continue")
    assert c.observe("when this works, we continue testing") == ["when this works,"]


def test_pause_flushes_short_tail_once() -> None:
    c = StableTextCommitter()
    c.observe("короткая фраза")
    c.observe("короткая фраза")
    assert c.flush_pause() == ["короткая фраза"]
    assert c.flush_pause() == []
```

- [ ] **Step 2: Run and confirm the new module is missing**

Run: `.venv/bin/python -m pytest tests/test_streaming_stt.py -q`

Expected: collection fails with `ModuleNotFoundError: streaming_stt`.

- [ ] **Step 3: Implement token state and deterministic chunk boundaries**

Use a punctuation-aware token regex that keeps terminal punctuation attached to the preceding word. Track `previous_tokens`, `stable_counts`, `committed_count`, and `pending_tokens`. A token becomes stable when the same token exists at the same index in two consecutive observations. From the newly stable uncommitted tokens: commit the earliest comma/semicolon/colon/dash/sentence boundary after 3 words; otherwise commit 6 words when available and always cap a chunk at 8. `flush_pause()` and `finish()` emit the current non-empty uncommitted tail once.

```python
class OnlineTranscriber(Protocol):
    def accept_audio(
        self, samples: np.ndarray, sample_rate: int
    ) -> StreamingResult: ...

    def finish(self) -> StreamingResult: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class StableTextCommitter:
    def observe(self, text: str) -> list[str]:
        current = _tokenize(text)
        common = _common_prefix_len(self._previous, current)
        self._stable_len = max(self._committed_len, common)
        self._latest = current
        self._previous = current
        return self._take_ready_chunks(force=False)

    def flush_pause(self) -> list[str]:
        return self._take_ready_chunks(force=True)

    def finish(self, text: str) -> list[str]:
        self._latest = _tokenize(text)
        self._stable_len = len(self._latest)
        chunks = self._take_ready_chunks(force=True)
        self.reset()
        return chunks
```

- [ ] **Step 4: Add edge-case tests**

Cover whitespace normalization, punctuation-only input, empty revisions, repeated words (`"да да"`), Cyrillic/Latin apostrophes, an 8-word forced cap, endpoint residual after an earlier commit, and reset between utterances. Assert every spoken token appears exactly once in concatenated output.

- [ ] **Step 5: Run focused tests and commit**

Run: `.venv/bin/python -m pytest tests/test_streaming_stt.py -q`

Expected: all committer tests pass.

```bash
git add streaming_stt.py tests/test_streaming_stt.py
git commit -m "feat: stabilize streaming transcripts"
```

---

### Task 3: Production frame iterator and sherpa online adapter

**Files:**
- Modify: `audio_stream.py`
- Modify: `tests/test_audio_stream_backpressure.py`
- Modify: `streaming_stt.py`
- Modify: `tests/test_streaming_stt.py`

**Interfaces:**
- Consumes: `StreamingModelProfile` and verified model directory from Task 1.
- Produces: `AudioPhraseStream.iter_frames(stop_event) -> Generator[FramePacket, None, None]`.
- Produces: `SherpaStreamingTranscriber(profile, model_dir, num_threads=None)` implementing `OnlineTranscriber.accept_audio(samples, sample_rate)`, `.finish()`, `.reset()`, and `.close()`.

- [ ] **Step 1: Write a failing frame-iterator regression test**

```python
def test_iter_frames_uses_production_stream_and_stops() -> None:
    stop = threading.Event()
    sd = MagicMock()

    @contextlib.contextmanager
    def input_stream(**kwargs):
        kwargs["callback"](
            np.ones((480, 1), dtype=np.float32) * 0.1,
            480, None, None,
        )
        stop.set()
        yield MagicMock()

    sd.InputStream.side_effect = input_stream
    stream = AudioPhraseStream(StreamConfig(sample_rate=16000, frame_ms=30))
    with patch("audio_stream.get_sounddevice", return_value=sd):
        packets = list(stream.iter_frames(stop))
    assert len(packets) == 1
    assert packets[0].samples.shape == (480,)
```

- [ ] **Step 2: Extract frame iteration without changing phrase semantics**

Move only InputStream ownership and queue draining into `iter_frames`. Implement `iter_phrases` as a loop over `iter_frames` calling the existing `_process_frame`. Preserve callback nonblocking behavior and metrics.

```python
def iter_phrases(self, stop_event=None):
    for packet in self.iter_frames(stop_event):
        pcm16 = self._process_frame(packet.samples)
        if pcm16 is not None:
            yield CapturedPhrase(pcm16=pcm16, ended_at=packet.captured_at)
```

- [ ] **Step 3: Write failing adapter contract tests with a fake sherpa module**

```python
@pytest.mark.parametrize(
    ("profile_id", "factory"),
    [
        ("sherpa_streaming_ru_t_one", "from_t_one_ctc"),
        ("sherpa_streaming_en_zipformer_20m", "from_transducer"),
    ],
)
def test_adapter_uses_profile_specific_factory(tmp_path, profile_id, factory):
    profile = STREAMING_MODEL_PROFILES[profile_id]
    for relative_path in profile.required_files:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    online = MagicMock()
    recognizer = MagicMock()
    stream = MagicMock()
    recognizer.create_stream.return_value = stream
    recognizer.is_ready.side_effect = [True, False]
    recognizer.get_result.return_value.text = "привет"
    recognizer.is_endpoint.return_value = True
    online.from_t_one_ctc.return_value = recognizer
    online.from_transducer.return_value = recognizer
    module = SimpleNamespace(OnlineRecognizer=online)
    transcriber = SherpaStreamingTranscriber(
        profile, tmp_path, sherpa_module=module, num_threads=2
    )
    result = transcriber.accept_audio(np.zeros(480, np.float32), 16000)
    assert result == StreamingResult("привет", True)
    getattr(module.OnlineRecognizer, factory).assert_called_once()
```

- [ ] **Step 4: Implement exact sherpa factories and stream lifecycle**

For Russian call `OnlineRecognizer.from_t_one_ctc(model=..., tokens=..., num_threads=...)`. For English call `OnlineRecognizer.from_transducer(encoder=..., decoder=..., joiner=..., tokens=..., num_threads=..., sample_rate=16000, feature_dim=80, decoding_method="greedy_search", enable_endpoint_detection=True, rule1_min_trailing_silence=2.4, rule2_min_trailing_silence=1.2, rule3_min_utterance_length=300)`. Feed float32 frames, decode while ready, use `get_result(stream).text`, and query `is_endpoint(stream)`. `finish()` calls `input_finished()`, drains decode, returns the final result, and creates a fresh stream only after the caller consumes it.

```python
def accept_audio(self, samples: np.ndarray, sample_rate: int) -> StreamingResult:
    self._stream.accept_waveform(sample_rate, np.asarray(samples, np.float32))
    while self._recognizer.is_ready(self._stream):
        self._recognizer.decode_stream(self._stream)
    result = self._recognizer.get_result(self._stream)
    return StreamingResult(result.text.strip(), self._recognizer.is_endpoint(self._stream))
```

- [ ] **Step 5: Run capture, backpressure, and adapter tests and commit**

Run: `.venv/bin/python -m pytest test_audio_stream.py tests/test_audio_stream_backpressure.py tests/test_streaming_stt.py -q`

Expected: all existing phrase tests and new frame/adapter tests pass; callback tests still prove no printing or blocking queue put.

```bash
git add audio_stream.py streaming_stt.py test_audio_stream.py tests/test_audio_stream_backpressure.py tests/test_streaming_stt.py
git commit -m "feat: feed audio frames to sherpa streaming STT"
```

---

### Task 4: Generation-aware TTS scheduling

**Files:**
- Create: `streaming_pipeline.py`
- Create: `tests/test_streaming_pipeline.py`
- Modify: `pipeline.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: existing `synthesize_text`, `prepare_audio_for_output`, and `play_audio_cancellable` behavior.
- Produces: `SynthesisMetrics(engine: str, tts_ms: int)` and `synthesize_and_play_text(text, sd, sf, config, cancel_event, emit) -> SynthesisMetrics | None` shared by phrase and streaming paths.
- Produces: `StreamingTTSWorker(global_stop, process, max_queue=4)`, `.begin_utterance() -> int`, `.submit(generation, text)`, and `.close()`; `process` has signature `(text: str, cancel_event) -> None`.

- [ ] **Step 1: Extract a shared cancellable synthesis helper under tests**

Write a test proving the helper deletes its temporary WAV, skips reading/playback after cancellation, and preserves the current output sample-rate fallback. Then move the TTS half of `_process_phrase` into `synthesize_and_play_text`; keep `_process_phrase` responsible for batch STT and metrics only.

```python
def synthesize_and_play_text(text, sd, sf, config, cancel_event, emit):
    fd, output_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        emit("state", "synthesizing")
        started = time.perf_counter()
        used_engine = synthesize_text(
            text=text,
            out_wav=output_wav,
            auto_select=config.auto_tts_model,
            manual_model=config.manual_tts_model,
            tts_root=config.tts_root,
        )
        if cancel_event.is_set():
            return None
        tts_ms = int(round((time.perf_counter() - started) * 1000))
        data, source_rate = sf.read(output_wav, dtype="float32")
        data, output_rate = prepare_audio_for_output(
            sd, data, source_rate, config.output_device
        )
        if not cancel_event.is_set():
            emit("state", "playing")
            play_audio_cancellable(
                sd, data, output_rate, config.output_device, cancel_event
            )
            return SynthesisMetrics(used_engine, tts_ms)
        return None
    finally:
        Path(output_wav).unlink(missing_ok=True)
```

- [ ] **Step 2: Write failing generation and queue tests**

```python
class BlockingSynthesizer:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release_event = threading.Event()
        self.played: list[str] = []
        self.first_cancel_event = None

    def wait_until_started(self) -> None:
        assert self.started.wait(timeout=1)

    def release(self) -> None:
        self.release_event.set()

    def __call__(self, text: str, cancel_event) -> None:
        if self.first_cancel_event is None:
            self.first_cancel_event = cancel_event
        self.started.set()
        assert self.release_event.wait(timeout=1)
        if not cancel_event.is_set():
            self.played.append(text)


def test_new_utterance_cancels_playback_and_discards_old_queue() -> None:
    gate = BlockingSynthesizer()
    worker = StreamingTTSWorker(global_stop=threading.Event(), process=gate)
    old = worker.begin_utterance()
    worker.submit(old, "старый один")
    worker.submit(old, "старый два")
    gate.wait_until_started()
    new = worker.begin_utterance()
    worker.submit(new, "новый текст")
    gate.release()
    worker.close()
    assert gate.played == ["новый текст"]
    assert gate.first_cancel_event.is_set()
```

- [ ] **Step 3: Implement the bounded generation worker**

Use a local queue with maximum size 4 and one daemon worker thread. `begin_utterance` increments an integer under a lock, sets the previous local cancel event, creates a new local cancel event, and drains queued jobs. A `CombinedCancelEvent.is_set()` returns global-stop OR local-generation-cancel. Before and after synthesis, compare the job generation with the current generation. Ignore late results. `close()` sets cancellation, sends a sentinel, and joins for at most the existing bounded worker shutdown period.

- [ ] **Step 4: Run focused scheduler and legacy pipeline tests and commit**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py tests/test_pipeline.py tests/test_audio_output.py -q`

Expected: scheduler tests and all phrase playback tests pass.

```bash
git add pipeline.py streaming_pipeline.py tests/test_pipeline.py tests/test_streaming_pipeline.py
git commit -m "feat: schedule cancellable streaming TTS chunks"
```

---

### Task 5: Streaming state machine, mode dispatch, and fallback

**Files:**
- Modify: `pipeline.py`
- Modify: `streaming_pipeline.py`
- Modify: `tests/test_streaming_pipeline.py`
- Modify: `tests/test_pipeline.py`
- Modify: `audio_queue.py`
- Modify: `tests/test_runner_lifecycle.py`

**Interfaces:**
- Consumes: `StreamingSTTSelection`, `ensure_streaming_model`, `SherpaStreamingTranscriber`, `StableTextCommitter`, `AudioPhraseStream.iter_frames`, and `StreamingTTSWorker`.
- Produces: expanded `RunConfig(stt_mode, streaming_stt, stt, ...)`.
- Produces: `run_streaming_pipeline(config, stop_event, emit)` and `run_phrase_pipeline(config, stop_event, emit)`.

- [ ] **Step 1: Write a fake-engine end-to-end test**

```python
def test_streaming_pipeline_emits_partial_and_speaks_commits_once() -> None:
    recognizer = MagicMock()
    recognizer.accept_audio.side_effect = [
        StreamingResult("это потоковый", False),
        StreamingResult("это потоковый тест работает", False),
        StreamingResult("это потоковый тест работает хорошо", False),
        StreamingResult("это потоковый тест работает хорошо сейчас", True),
    ]
    recognizer.finish.return_value = StreamingResult(
        "это потоковый тест работает хорошо сейчас", True
    )
    frame_stream = MagicMock()
    frame_stream.iter_frames.return_value = iter([
        FramePacket(np.ones(480, np.float32) * 0.1, float(index))
        for index in range(4)
    ])
    events: list[tuple[str, str]] = []
    spoken: list[str] = []
    worker = MagicMock()
    worker.begin_utterance.return_value = 1
    worker.submit.side_effect = lambda generation, text: spoken.append(text)
    config = RunConfig(
        input_device=None,
        output_device=7,
        stt_mode="streaming",
        streaming_stt=StreamingSTTSelection(
            "ru", "sherpa_streaming_ru_t_one"
        ),
        stt=STTSelection(
            "ru", "gigaam", "gigaam-v3-e2e-rnnt", "cpu"
        ),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )
    with (
        patch("streaming_pipeline.ensure_streaming_model", return_value=Path("model")),
        patch("streaming_pipeline.SherpaStreamingTranscriber", return_value=recognizer),
        patch("streaming_pipeline.AudioPhraseStream", return_value=frame_stream),
        patch("streaming_pipeline.StreamingTTSWorker", return_value=worker),
    ):
        run_streaming_pipeline(
            config, threading.Event(), lambda kind, text: events.append((kind, text))
        )
    assert any(kind == "partial" for kind, _ in events)
    assert " ".join(spoken).split() == (
        "это потоковый тест работает хорошо сейчас".split()
    )
    assert len(" ".join(spoken).split()) == len(set(" ".join(spoken).split()))
```

- [ ] **Step 2: Implement speech start, pause, endpoint, and metrics**

Track RMS with the existing start/stop thresholds. On the first above-threshold frame of an utterance, call `tts_worker.begin_utterance()`, emit `state=recognizing`, and remember `speech_started_at`. Feed every frame to the recognizer. Emit `partial` only when text changes. Pass each result to the committer and submit returned chunks with the current generation. Accumulate below-threshold frame duration; at 350 ms call `flush_pause()` once. At sherpa endpoint, call `finish()`, submit residual chunks, reset recognizer/committer state, clear partial text, and emit `state=listening`.

Sample `AudioPhraseStream.metrics()` while consuming frames. Preserve overload reporting when dropped frames increase. For every committed chunk, record speech-start-to-first-partial, speech-start-to-commit, TTS, queue, and dropped-frame values; emit them in the existing status channel as `partial_ms`, `commit_ms`, `tts_ms`, `queue_ms`, and `dropped_frames`.

- [ ] **Step 3: Split the legacy runner and dispatch from `run_pipeline`**

Expand the picklable configuration explicitly and update every existing test helper that constructs it so phrase tests continue to request `after_phrase`:

```python
@dataclass(frozen=True)
class RunConfig:
    input_device: Optional[int]
    output_device: Optional[int]
    stt_mode: Literal["streaming", "after_phrase"]
    streaming_stt: StreamingSTTSelection
    stt: STTSelection
    auto_tts_model: bool
    manual_tts_model: str
    tts_root: Optional[str]
```

```python
def run_pipeline(config: RunConfig, stop_event, emit: EmitCallback) -> None:
    if config.stt_mode == "after_phrase":
        return run_phrase_pipeline(config, stop_event, emit)
    try:
        return run_streaming_pipeline(config, stop_event, emit)
    except StreamingInitializationError as exc:
        if stop_event.is_set():
            return
        emit("warning", f"Streaming STT unavailable; using after-phrase mode: {exc}")
        return run_phrase_pipeline(config, stop_event, emit)
```

Wrap only download validation, sherpa import/native load, and recognizer creation as `StreamingInitializationError`; do not restart phrase mode after an arbitrary mid-session programming error. Preserve the requested language by validating `config.streaming_stt.language == config.stt.language` before startup.

- [ ] **Step 4: Add cancellation and fallback tests**

Test 350 ms pause flush, endpoint residual, new-speech generation cancellation, stale synthesized result rejection, Stop during download, Stop during decode, Stop during TTS, corrupt model fallback, sherpa DLL-load fallback, same-language validation, and no fallback on an unrelated runtime exception. Assert Stop never emits a fatal error or closes GUI-owned state.

- [ ] **Step 5: Forward `partial` events through runner isolation**

Add `partial` to the `SpeechLoopRunner._dispatch_event` allowlist and test that current-run partials are delivered while stale-run partials are discarded exactly like text/error events.

- [ ] **Step 6: Run pipeline/lifecycle tests and commit**

Run: `.venv/bin/python -m pytest tests/test_streaming_pipeline.py tests/test_pipeline.py tests/test_runner_lifecycle.py tests/test_audio_queue_pipeline.py -q`

Expected: all streaming, phrase, stale-event, forced-stop, and fallback tests pass.

```bash
git add pipeline.py streaming_pipeline.py audio_queue.py tests/test_streaming_pipeline.py tests/test_pipeline.py tests/test_runner_lifecycle.py tests/test_audio_queue_pipeline.py
git commit -m "feat: run streaming STT with phrase fallback"
```

---

### Task 6: Persistent mode/profile UI and mutable partial transcript

**Files:**
- Create: `app_settings.py`
- Create: `tests/test_app_settings.py`
- Modify: `gui.py`
- Modify: `main.py`
- Modify: `tests/test_gui_profiles.py`
- Modify: `tests/test_main_startup.py`

**Interfaces:**
- Consumes: expanded `RunConfig`, `StreamingSTTSelection`, profile labels, and `partial` events.
- Produces: `load_app_settings(path=None) -> dict` and `save_app_settings(settings, path=None) -> None`.
- Produces GUI settings keys `stt_mode`, `streaming_language`, `streaming_profile`, and the existing phrase keys.

- [ ] **Step 1: Write failing settings migration tests**

```python
def test_missing_config_defaults_to_streaming_without_losing_phrase_defaults(tmp_path):
    settings = load_app_settings(tmp_path / "settings.json")
    assert settings["stt_mode"] == "streaming"
    assert settings["streaming_language"] == "ru"
    assert settings["streaming_profile"] == "sherpa_streaming_ru_t_one"
    assert settings["stt_engine"] == "gigaam"


def test_legacy_config_migrates_to_streaming_and_keeps_phrase_profile(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "stt_language": "en", "stt_engine": "whisper",
        "stt_model": "small", "stt_device": "cpu"
    }), encoding="utf-8")
    settings = load_app_settings(path)
    assert settings["stt_mode"] == "streaming"
    assert settings["stt_model"] == "small"
    assert settings["streaming_language"] == "en"
```

- [ ] **Step 2: Implement atomic JSON settings persistence**

Store at `user_data_root() / "settings.json"`. Merge loaded dictionaries over explicit defaults, validate enum/profile values, and replace invalid values with same-language defaults. Write UTF-8 JSON to `settings.json.tmp`, flush and `os.fsync`, then `os.replace`. Never write credentials or model data.

- [ ] **Step 3: Write failing GUI behavior tests**

Assert streaming is initially selected, mode switching enables exactly the relevant profile controls, running states disable all STT mode/language/model controls, `_collect_settings()` returns both profiles, a `partial` event replaces `partial_var` without appending to the log, a `text` event appends committed text, an empty partial clears the field, and a fallback warning remains visible after later status updates.

- [ ] **Step 4: Add controls and partial display**

Add a readonly `STT mode` combobox above language/engine controls, a `Partial:` label, and a separate persistent warning label in the status frame. Keep separate variables for streaming language/profile and phrase language/engine/model/device. `_toggle_stt_mode_controls()` disables the inactive group. `set_pipeline_state` locks both groups whenever state is not idle. Clear the warning label only when the user starts a new run; a normal status update must not overwrite it.

Extend `set_pipeline_state` to accept `recognizing`, `synthesizing`, and `playing` without changing button safety: Start stays disabled, Stop stays enabled, and the status label shows the matching capitalized state.

```python
elif kind == "partial":
    self.partial_var.set(message)
elif kind == "warning":
    self.warning_var.set(message)
    self._append_log(f"WARNING: {message}\n")
elif kind == "text":
    self.partial_var.set("")
    self._append_log(f"STT: {message}\n")
```

- [ ] **Step 5: Load at controller construction and save on successful Start**

Pass loaded values into `AppGUI`. In `AppController.start`, validate both selections, construct the expanded `RunConfig`, start the child, and only then atomically save the collected settings. A failed validation or process start must not overwrite the last working settings.

- [ ] **Step 6: Run UI/controller tests and commit**

Run: `.venv/bin/python -m pytest tests/test_app_settings.py tests/test_gui_profiles.py tests/test_main_startup.py tests/test_runner_lifecycle.py -q`

Expected: migration, atomic save, control state, partial replacement, Start failure, and Stop/window lifecycle tests pass.

```bash
git add app_settings.py gui.py main.py tests/test_app_settings.py tests/test_gui_profiles.py tests/test_main_startup.py tests/test_runner_lifecycle.py
git commit -m "feat: expose persistent streaming STT controls"
```

---

### Task 7: Windows dependency, PyInstaller, and model-free smoke coverage

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `main.py`
- Modify: `installer/V2TTS.spec`
- Modify: `smoke_test.py`
- Modify: `tests/test_windows_workflow.py`
- Modify: `tests/test_packaged_smoke.py`
- Modify: `.github/workflows/windows-frozen-smoke.yml`

**Interfaces:**
- Consumes: sherpa adapter imports but no model paths or weights.
- Produces: `run_streaming_runtime_smoke() -> None`.

- [ ] **Step 1: Write failing dependency and bundle assertions**

```python
def test_sherpa_runtime_is_pinned_and_collected_without_weights() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    project = Path("pyproject.toml").read_text(encoding="utf-8")
    spec = Path("installer/V2TTS.spec").read_text(encoding="utf-8")
    assert "sherpa-onnx==1.13.5" in requirements
    assert '"sherpa-onnx==1.13.5"' in project
    assert 'collect_dynamic_libs("sherpa_onnx")' in spec
    assert 'copy_metadata("sherpa-onnx")' in spec
    assert "sherpa-onnx-streaming-t-one" not in spec
    assert "sherpa-onnx-streaming-zipformer" not in spec
```

- [ ] **Step 2: Pin and collect sherpa runtime**

Add `sherpa-onnx==1.13.5` to both dependency files and `sherpa_onnx` to `RUNTIME_DEPENDENCIES`. In the PyInstaller spec, add package data, metadata, submodules, and dynamic libraries for `sherpa_onnx`; do not add model paths. Retain Python 3.13 and existing onnx-asr collection.

- [ ] **Step 3: Add a model-free packaged runtime smoke**

```python
def run_streaming_runtime_smoke() -> None:
    import sherpa_onnx
    if not hasattr(sherpa_onnx, "OnlineRecognizer"):
        raise RuntimeError("sherpa-onnx OnlineRecognizer is unavailable")
    from streaming_models import STREAMING_MODEL_PROFILES
    if set(STREAMING_MODEL_PROFILES) != {
        "sherpa_streaming_ru_t_one",
        "sherpa_streaming_en_zipformer_20m",
    }:
        raise RuntimeError("streaming model manifest is incomplete")
```

Call this before TTS smoke cases. Do not call `ensure_streaming_model` or create a recognizer in required CI.

- [ ] **Step 4: Extend static workflow and smoke tests**

Keep the workflow command `python installer/build.py --skip-install` followed by `.\dist\V2TTS.exe --smoke-test`. Add assertions that the build installs the pinned dependency and the packaged smoke reports sherpa success. Confirm the workflow does not cache or upload model directories.

- [ ] **Step 5: Run packaging-focused tests and commit**

Run: `.venv/bin/python -m pytest tests/test_windows_workflow.py tests/test_packaged_smoke.py tests/test_main_smoke.py tests/test_installer_build.py -q`

Expected: all dependency, spec, startup, spawn, TTS, and model-free sherpa smoke tests pass.

```bash
git add requirements.txt pyproject.toml main.py installer/V2TTS.spec smoke_test.py .github/workflows/windows-frozen-smoke.yml tests/test_windows_workflow.py tests/test_packaged_smoke.py tests/test_main_smoke.py tests/test_installer_build.py
git commit -m "build: package sherpa streaming runtime"
```

---

### Task 8: Real-model opt-in test, documentation, and final verification

**Files:**
- Create: `tests/integration/test_streaming_model_smoke.py`
- Modify: `pytest.ini`
- Modify: `README.md`

**Interfaces:**
- Consumes: public model installer and `SherpaStreamingTranscriber`.
- Produces: an opt-in, audio-device-free real model validation path.

- [ ] **Step 1: Add an opt-in waveform smoke test**

Gate with `V2TTS_STREAMING_MODEL_TEST=1`; otherwise skip. Select `V2TTS_STREAMING_PROFILE` or default to the Russian profile. Resolve an already-installed model unless `V2TTS_ALLOW_MODEL_DOWNLOAD=1`. Generate/read a deterministic 16 kHz WAV fixture supplied by the test environment, feed it in 30 ms frames, finish, and assert a non-empty transcript. Never make this test part of required CI.

```python
@pytest.mark.integration
def test_real_streaming_model_produces_text() -> None:
    if os.getenv("V2TTS_STREAMING_MODEL_TEST") != "1":
        pytest.skip("set V2TTS_STREAMING_MODEL_TEST=1")
    profile_id = os.getenv(
        "V2TTS_STREAMING_PROFILE", "sherpa_streaming_ru_t_one"
    )
    if is_streaming_model_ready(profile_id):
        model_dir = streaming_model_dir(profile_id)
    elif os.getenv("V2TTS_ALLOW_MODEL_DOWNLOAD") == "1":
        model_dir = ensure_streaming_model(
            profile_id,
            stop_event=threading.Event(),
            on_progress=lambda *_: None,
        )
    else:
        pytest.skip("install the model or set V2TTS_ALLOW_MODEL_DOWNLOAD=1")
    wav_path = os.getenv("V2TTS_STREAMING_TEST_WAV")
    if not wav_path:
        pytest.skip("set V2TTS_STREAMING_TEST_WAV to a speech WAV")
    audio, sample_rate = soundfile.read(wav_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, 0]
    transcriber = SherpaStreamingTranscriber(
        STREAMING_MODEL_PROFILES[profile_id], model_dir
    )
    frame_size = max(1, int(sample_rate * 0.03))
    result = StreamingResult("", False)
    for offset in range(0, len(audio), frame_size):
        result = transcriber.accept_audio(
            audio[offset : offset + frame_size], sample_rate
        )
    result = transcriber.finish()
    assert result.text.strip()
```

- [ ] **Step 2: Document mode behavior and operational paths**

Update README with: streaming default and after-phrase alternative; explicit Russian/English profile selection; approximate 1–2 second target and TTS dependency; first-use model sizes (128,468,156 and 127,887,156 bytes); `%LOCALAPPDATA%` path; verified/atomic downloads; fallback warning; external-model/EXE boundary; Stop behavior; and opt-in test environment variables.

- [ ] **Step 3: Run the complete local verification three times at the relevant levels**

Run focused streaming suite:

```bash
.venv/bin/python -m pytest tests/test_streaming_models.py tests/test_streaming_stt.py tests/test_streaming_pipeline.py tests/test_app_settings.py -q
```

Run full suite:

```bash
.venv/bin/python -m pytest -q
```

Run static/package-independent checks:

```bash
.venv/bin/python -m compileall -q .
git diff --check origin/main...HEAD
```

Expected: focused and full tests pass with only hardware/model opt-in skips; compileall and diff check produce no errors.

- [ ] **Step 4: Commit the opt-in test and documentation**

```bash
git add tests/integration/test_streaming_model_smoke.py pytest.ini README.md
git commit -m "docs: explain local streaming STT"
```

- [ ] **Step 5: Push and verify Windows frozen build**

Push the feature branch, open a PR to `main`, and wait for `.github/workflows/windows-frozen-smoke.yml`. The PR is not mergeable until the Windows Python 3.13 build and `.\dist\V2TTS.exe --smoke-test` succeed. Inspect failing logs before changing code; rerun the full local suite after any fix.

- [ ] **Step 6: Perform final scope and artifact checks**

```bash
git diff --name-only origin/main...HEAD
git status --short
git ls-tree -r --name-only HEAD | rg "(model\.onnx|encoder-.*\.onnx|decoder-.*\.onnx|joiner-.*\.onnx|\.tar\.bz2)$" && exit 1 || true
```

Expected: only planned source/tests/docs/build files changed, the worktree is clean, and no model archive or weight is tracked.
