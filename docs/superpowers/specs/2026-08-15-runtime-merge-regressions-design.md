# V2TTS Runtime Merge Regression Repair

## Goal

Restore the documented source and packaged runtime after incorrect merge-conflict resolution. A captured phrase must be transcribed, displayed, synthesized, and played exactly once. Russian synthesis must use the vendored `ru_tts` engine by default so source installs and PyInstaller builds have the same dependency model.

## Scope

The repair covers the confirmed runtime and dependency regressions in `main.py`, `audio_queue.py`, `tts.py`, `requirements.txt`, and their tests. It does not add bounded microphone buffering, acoustic echo cancellation, or a new VAD implementation; those are separate performance and product improvements.

## Runtime Contract

`AppController` constructs `SpeechLoopRunner` with exactly three callbacks: `on_status`, `on_text`, and `on_error`. The runner owns the phrase-processing pipeline. For every non-empty phrase it performs one STT call, emits one text event, performs one TTS call, and performs one playback operation.

`_run()` initializes audio and STT resources, iterates phrases, and delegates each phrase once to `_process_phrase()`. `_process_phrase()` handles recoverable per-phrase STT and TTS/playback failures without terminating the listening loop. Initialization failures still report an error and stop the runner.

## TTS Strategy

Automatic routing follows the repository documentation: Cyrillic text selects `ru_tts`; Latin-only text selects `sam`. Manual selection remains available for `ru_tts` and `sam`. The accidental Silero integration is removed from the entry-point dependency check, runtime router, dependency files, tests, and model list. This keeps `torch` out of the packaged application and aligns source execution with `installer/V2TTS.spec`.

The existing fallback remains deterministic: `ru_tts` falls back to `sam`. Synthesis uses a temporary WAV path because all supported engines and `soundfile.read()` already support paths. The file descriptor is closed immediately and the path is removed in a `finally` block.

## Error Handling

- STT failure: emit `STT failed: ...` and skip only the current phrase.
- Empty transcription: do not emit text, synthesize, or play audio.
- TTS or playback failure: emit `TTS/Playback failed: ...`, clean the temporary file, and continue listening.
- Runner initialization failure: emit the exception and `Stopped (error)`.

## Regression Tests

Tests must be written and observed failing before production changes.

1. Compile smoke test proves every tracked Python file compiles, catching duplicate keyword arguments in `main.py`.
2. Runner constructor test proves all three callbacks are accepted and retained.
3. Phrase pipeline test proves one phrase causes exactly one STT call, one text event, one synthesis call, and one playback call.
4. Empty and STT-error tests prove synthesis/playback is skipped while errors are reported correctly.
5. Temporary-file cleanup test proves the generated WAV path is removed after both success and TTS/playback failure.
6. TTS routing tests prove Cyrillic selects `ru_tts`, Latin selects `sam`, and the supported model list contains no unavailable packaged engine.
7. Existing test suite remains green.

## Completion Criteria

- `python -m compileall -q .` succeeds.
- `python -m pytest -q` succeeds.
- No production reference to `silero_tts` remains.
- Each phrase has one and only one STT/TTS/playback path.
- Source dependencies, README behavior, and PyInstaller exclusions agree.
