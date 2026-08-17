# Local Streaming STT Design

**Date:** 2026-08-17
**Status:** Approved design; implementation not started

## Goal

Reduce the delay between the user speaking and V2TTS beginning playback. Add a
true local streaming recognizer that produces partial results while speech is
still in progress. Keep the existing phrase-based GigaAM and faster-whisper
paths as accuracy-oriented alternatives.

The balanced target is to begin TTS playback roughly 1–2 seconds after stable
speech is available on a machine fast enough to run both selected models in
real time. The UI must report measured latency instead of treating this target
as a guarantee.

## User-visible behavior

- Add an STT mode selector with `Streaming` and `After phrase` choices.
- `Streaming` is the default for new configurations. Existing settings that do
  not contain the new field migrate to `Streaming` without discarding their
  phrase-engine selection.
- `Streaming` uses local sherpa-onnx models. `After phrase` continues to use the
  current GigaAM or faster-whisper profile.
- Russian and English are explicit, separate streaming profiles. V2TTS does
  not attempt automatic language detection or switch language mid-utterance.
- The UI displays one mutable partial-transcript field. Intermediate revisions
  replace this field rather than appending noisy lines to the event log.
- Only committed text appears in the transcript log and reaches TTS.
- Beginning a new utterance cancels obsolete playback and queued TTS work so a
  backlog cannot grow behind the speaker.
- Stop cancels capture, recognition, synthesis, and playback while leaving the
  application window open and reusable.

## Model profiles and storage

The initial model catalog contains:

| Language | Profile ID | Upstream archive |
| --- | --- | --- |
| Russian | `sherpa_streaming_ru_t_one` | `sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2` |
| English | `sherpa_streaming_en_zipformer_20m` | `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2` |

The catalog is a code-owned manifest. Every entry pins the official sherpa-onnx
release URL, archive byte size, SHA-256 digest, expected extracted files,
recognizer architecture, sample rate, and endpoint parameters. Concrete hashes
must be recorded in the manifest before the implementation is merged.

Models live outside the executable under:

```text
%LOCALAPPDATA%\V2TTS\models\sherpa-onnx\<profile-id>\
```

The executable contains the sherpa-onnx Python package and required native
libraries, but no model weights. A missing model is downloaded into a sibling
temporary directory, verified, extracted there, and atomically renamed into
place. Cancellation or failure removes the temporary data. A completion marker
is written only after archive and extracted-file validation succeeds.

## Architecture

### Streaming recognizer boundary

Introduce a streaming-specific interface instead of overloading the current
whole-buffer `Transcriber` protocol. Its responsibilities are:

- create and reset one recognizer stream per utterance;
- accept contiguous normalized mono audio frames;
- decode while the engine reports that decoding is ready;
- expose the latest partial text and endpoint state;
- finalize the stream and return its last result;
- release native resources deterministically.

`SherpaStreamingTranscriber` implements this interface. The current
`WhisperTranscriber` and `GigaAMTranscriber` remain implementations of the
phrase-oriented protocol. This separation prevents batch-only engines from
pretending to offer streaming semantics.

### Capture and endpoint flow

Audio capture remains continuous and nonblocking. The PortAudio callback only
copies frames into the existing bounded queue; it never runs inference, file
I/O, logging, or blocking queue operations. A consumer feeds 20–30 ms frames to
the online recognizer.

The streaming path uses sherpa-onnx endpoint detection plus the existing energy
signal for user-speech start. A short pause of approximately 350 ms asks the
commit stage to flush safe residual text. A recognizer endpoint finalizes the
utterance, emits any remaining text, resets the recognizer stream, and returns
to listening. Exact engine rules come from the pinned profile manifest rather
than being scattered through the UI and pipeline modules.

### Stable text and deduplication

Partial hypotheses may revise their tail, so they must not be spoken directly.
A dedicated `StableTextCommitter` owns all text state:

1. Normalize whitespace without changing case, punctuation, or language.
2. Compare consecutive token sequences and track their longest common prefix.
3. Treat a token as stable only after it remains unchanged in two consecutive
   hypotheses.
4. Exclude the already committed prefix before producing new output.
5. Prefer a stable sentence or clause boundary once at least 3 new words are
   ready. Without such a boundary, target 6 words and force a chunk at 8 words.
   A 350 ms pause flushes the current non-empty tail even when it is shorter.
6. On a recognizer endpoint, commit all remaining non-empty text once.
7. Reset hypothesis state only after the endpoint residual has been handled.

Russian and English tokenization share whitespace and punctuation-aware rules;
the implementation must preserve punctuation adjacent to words. The committer
has no dependency on sherpa-onnx, audio devices, or TTS and can therefore be
tested exhaustively with deterministic text sequences.

### TTS scheduling and interruption

Committed chunks enter a bounded synthesis queue. One worker synthesizes chunks
in order, while playback remains chunk-cancellable through the existing stop
mechanism. The TTS engines continue to synthesize complete committed chunks;
this feature does not claim to make the current TTS engines sample-streaming.

User-speech start increments the pipeline generation, cancels active playback,
and discards queued artifacts from older generations. Results that complete
after cancellation carry their original generation and are ignored. Stop uses
the same cancellation path and then terminates the isolated pipeline process
using the existing bounded shutdown behavior.

## UI and configuration

Persist these independent settings:

- STT mode: streaming or after-phrase;
- streaming language/profile;
- phrase language/engine/model/device profile;
- existing TTS and audio-device settings.

Changing mode or model is disabled while the pipeline is running, matching the
current device/profile behavior. The status area distinguishes `Listening`,
`Speech detected`, `Recognizing`, `Synthesizing`, and `Playing`. It includes
partial-to-commit, STT, TTS, and queue timing so latency regressions are visible.

If a streaming model is absent, Start presents the existing model-download
flow. Download progress must work in a windowed PyInstaller process where
stdout and stderr may initially be unavailable. If download, validation, native
library loading, or recognizer initialization fails, V2TTS:

1. reports a visible actionable error without closing the application;
2. selects the saved after-phrase profile for the same language, or that
   language's existing default when none was saved;
3. displays a persistent warning that fallback occurred;
4. starts only after the fallback profile is ready.

No error silently changes the requested language.

## Packaging

- Pin sherpa-onnx to one version in runtime requirements.
- Add its Python metadata, native libraries, and required hidden imports to the
  PyInstaller analysis.
- Keep all model archives and extracted weights out of the PyInstaller bundle.
- Preserve Windows spawn safety: child-process entry points restore standard
  streams before importing or initializing code that may write progress.
- Extend the Windows GitHub Actions build with an import/startup smoke test for
  the packaged streaming runtime. The smoke test must not download a model.

## Error handling and resource limits

- The capture queue remains bounded and drops oldest audio frames rather than
  blocking the callback. Dropped-frame telemetry remains visible.
- The synthesis queue is bounded. A newer generation replaces obsolete queued
  work instead of waiting behind it.
- Empty and punctuation-only hypotheses do not reach TTS.
- Duplicate or late recognizer events are ignored by generation and utterance
  identifiers.
- A corrupt or partial model directory is never treated as ready.
- Model download cancellation, Stop, and window close clean up temporary model
  and WAV files without deleting a previously verified model.
- Streaming initialization failure cannot terminate the GUI process.

## Verification

### Unit tests

- Stable-prefix detection across revised Russian and English hypotheses.
- No repeated words across adjacent commits.
- Sentence punctuation, 3–8-word chunking, 350 ms pause flush, and endpoint
  residual flush.
- Empty results, punctuation-only results, recognizer resets, and late events.
- Generation cancellation and bounded TTS-queue replacement.
- Model manifest validation, temporary download cleanup, atomic installation,
  incomplete markers, wrong size, wrong hash, and missing extracted files.

### Integration tests without hardware

- A fake online recognizer consumes deterministic audio frames and emits
  partial and endpoint events through the production streaming pipeline.
- The production pipeline emits each committed chunk exactly once and preserves
  order.
- New speech cancels active playback and old synthesis results.
- Stop returns within the existing bounded timeout and leaves the GUI reusable.
- Streaming initialization failure performs visible same-language fallback.

### Optional and packaged tests

- An opt-in microphone/model test exercises a real sherpa-onnx stream without
  becoming a required CI test.
- The Windows job builds the windowed executable and verifies imports, startup,
  model-path resolution, and absence of model weights from the bundle.
- Before opening the implementation PR, run the full test suite, focused
  streaming tests, `compileall`, `git diff --check`, and the Windows build/smoke
  workflow.

## Non-goals

- Cloud STT, including Sber SaluteSpeech.
- Automatic bilingual detection or mid-utterance language switching.
- Bundling model weights inside the executable.
- Replacing GigaAM or faster-whisper phrase recognition.
- Converting current TTS engines into true sample-streaming synthesizers.
- Speaking every unstable partial token with no correction boundary.

## Acceptance criteria

The feature is complete when a user can select Russian or English streaming
STT, speak continuously, see revisable partial text, and hear each committed
part once without waiting for the whole utterance. Streaming is the default,
after-phrase mode remains selectable, models remain external to the EXE, new
speech removes obsolete playback backlog, and Stop never closes the window.
All required tests and the packaged Windows smoke test pass.
