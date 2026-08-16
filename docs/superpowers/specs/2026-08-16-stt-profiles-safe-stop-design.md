# V2TTS: Safe Stop and Russian/English STT Profiles

Date: 2026-08-16

## Goal

Improve speech recognition for the two supported languages and make `Stop`
immediately interrupt the active pipeline without closing the V2TTS window.

The application will support:

- Russian-first local recognition with GigaAM v3.
- Tuned Whisper profiles for Russian and English.
- Separate on-disk STT model storage instead of embedding model weights in the
  executable.
- A process boundary around the audio/STT/TTS pipeline so a blocking native
  inference or audio call can be terminated without terminating Tkinter.

Cloud SaluteSpeech is out of scope. The Sber option in this design is the local,
offline GigaAM model and does not require an API token.

## Current Problems

### Stop can terminate the application

`SpeechLoopRunner.stop()` currently calls the global `sounddevice.stop()` from
the Tkinter thread while the background runner owns an input stream or playback.
That cross-thread PortAudio interaction can fail below Python and close the whole
process instead of only stopping the pipeline.

### Stop cannot interrupt blocking stages

Whisper transcription, TTS synthesis, and `sounddevice.wait()` are blocking.
Setting the existing thread event cannot end these calls. A result may also be
published after the user has already pressed `Stop`.

### Recognition is under-configured

- Language is always `None`, so Whisper performs language detection for every
  short phrase.
- The runner overrides the more conservative stream defaults with a 180 ms
  minimum phrase and 350 ms end silence.
- Capture starts at the threshold-crossing frame, so the first phoneme can be
  lost.
- A requested CUDA device can silently fall back to CPU without a clear warning.
- There is no Russian-specific engine or separate Russian/English profile.

## User-visible Behavior

### Settings

The STT section will contain four controls:

1. `Language`: `Russian` or `English`.
2. `STT engine`: choices valid for the selected language.
3. `STT model`: choices valid for the selected engine.
4. `STT device`: `CPU` or `CUDA` where the selected runtime supports it.

The model list changes when language or engine changes. Invalid combinations
cannot be selected.

| Language | Engine | Available models | Default |
| --- | --- | --- | --- |
| Russian | GigaAM | v3 E2E RNN-T, v3 E2E CTC | v3 E2E RNN-T |
| Russian | Whisper | small, medium, large-v3 | medium |
| English | Whisper | small, medium, large-v3 | medium |

GigaAM is deliberately restricted to the Russian profile. Whisper receives the
explicit language code `ru` or `en`; automatic language detection is not used in
these profiles.

The status area reports both requested and actual execution backends. A CUDA
fallback is shown as a warning, including the original CUDA initialization
error, rather than being silently treated as normal CPU execution.

### Model storage

Downloaded STT weights are stored under:

```text
%LOCALAPPDATA%\V2TTS\models\gigaam\
%LOCALAPPDATA%\V2TTS\models\whisper\
```

The platform-neutral fallback is the existing V2TTS user-data directory plus
`models`. Model caches are never added to PyInstaller `datas` and are never
placed in the one-file executable.

The first start of a missing model shows `Downloading model...` / `Loading
model...`. A failed or interrupted download produces a recoverable error naming
the model and target directory. The next start retries through the model hub's
cache mechanism.

The packaged SAM and `ru_tts` directories are runtime engine code, not neural
model weights, and remain part of the application runtime. No STT weights are
bundled with them.

### Stop lifecycle

The UI and pipeline have explicit states:

```text
Idle -> Starting -> Listening -> Stopping -> Idle
```

- `Start` is enabled only in `Idle`.
- `Stop` is enabled in `Starting` and `Listening`.
- Pressing `Stop` immediately changes the UI to `Stopping...` and invalidates
  the active run ID.
- Recording and playback stop immediately.
- Text, errors, or synthesized audio produced by the invalidated run are never
  delivered to the GUI.
- Once the worker is gone, the UI returns to `Idle` and `Start` becomes enabled.
- Closing the window uses the same shutdown path, then destroys Tkinter only
  after the child worker has exited.

## Architecture

### STT interface

Introduce a common transcriber protocol:

```python
class Transcriber(Protocol):
    requested_device: str
    actual_device: str

    def transcribe_pcm16(
        self,
        pcm16: np.ndarray,
        sample_rate: int,
    ) -> str: ...
```

`WhisperTranscriber` remains one implementation. A new
`GigaAMTranscriber` uses `onnx-asr` and the GigaAM v3 ONNX models. A factory
validates the selected profile, supplies the explicit language, model directory,
and device, and returns the matching implementation.

No STT engine imports or downloads at module import time. Heavy imports and model
loading occur inside the pipeline worker after it has reported its loading state.

### Pipeline process

Replace the long-lived runner thread with one spawned child process per `Start`.
The child exclusively owns:

- `sounddevice.InputStream`;
- the STT model instance;
- TTS synthesis;
- the output stream and playback;
- temporary WAV files.

The Tkinter process owns only UI state and process lifecycle. Communication uses
two multiprocessing primitives:

- a parent-to-child cancellation event;
- a child-to-parent event queue carrying `(run_id, kind, payload)` messages.

All process entry points are top-level callables and `main()` calls
`multiprocessing.freeze_support()` so the design remains compatible with a
frozen Windows build.

### Cancellation escalation

`Stop` performs the following sequence without calling PortAudio in the GUI
process:

1. Mark the current `run_id` invalid so queued or late messages are ignored.
2. Set the cooperative cancellation event.
3. The child closes its own capture/output streams and exits normally when it is
   between blocking stages.
4. A non-blocking Tkinter timer checks the child. If it remains alive after 300
   ms because native STT/TTS is blocked, the parent terminates that child.
5. A second timer verifies process death and uses the stronger process kill only
   if termination did not complete within one additional second.
6. Process handles and queues are closed, then the UI returns to `Idle`.

There is no synchronous `join()` on the Tkinter thread. This keeps the window
responsive while providing a real interruption boundary for native inference.
Only one pipeline child may exist at a time.

### Playback

Replace global `sd.play()` / `sd.wait()` with a child-owned `OutputStream` and
bounded chunk writes. The cancellation event is checked between chunks. The
existing output sample-rate negotiation and anti-alias resampling remain in use.

### Capture and phrase boundaries

Add a rolling 200 ms pre-roll while waiting for speech. When speech crosses the
start threshold, the pre-roll is prepended so initial consonants are retained.

Use the existing conservative stream defaults in the runner:

- minimum phrase: 300 ms;
- end silence: 700 ms;
- frame size: 30 ms;
- capture backlog: 1500 ms, drop-oldest on overload.

The audio callback remains non-blocking: it only copies the frame, updates
in-memory counters, and performs non-blocking queue operations.

## Error Handling

- Missing GigaAM/Whisper weights: show a model-specific download/load error and
  return to `Idle`.
- Unsupported engine/language combination: reject before a process is spawned.
- CUDA initialization failure: retry CPU only when the profile permits it and
  publish a visible warning with requested and actual devices.
- Worker crash or forced termination: parent detects the exit, closes IPC
  resources, reports a concise status, and remains usable.
- Audio-device failure: report the selected input/output device and sample rate;
  do not close the GUI.
- Stale events: discard every event whose `run_id` is not the current run.

## Packaging and Dependencies

- Add a pinned compatible `onnx-asr` dependency with the model-hub and Windows
  CPU runtime extras.
- Keep the CUDA path optional so a CPU-only Windows build still starts.
- Extend the PyInstaller spec with the runtime modules and native libraries
  needed by `onnx-asr`/ONNX Runtime, but exclude downloaded model caches.
- Preserve `--smoke-test` and add a frozen multiprocessing smoke path that starts
  and stops a stub worker without loading a real model or audio device.
- Document that the first use downloads model weights and that subsequent use is
  offline.

## Tests

### Unit tests

- Profile validation and dynamic RU/EN model lists.
- Whisper receives `language="ru"` or `language="en"` exactly.
- Both engines receive their user model directory.
- GigaAM factory/model mapping for E2E RNN-T and E2E CTC.
- CUDA fallback publishes a warning containing requested and actual devices.
- Pre-roll retains frames immediately before threshold crossing.
- Minimum phrase and end-silence settings are not overridden by shorter values.
- Playback checks cancellation between chunks.
- Events from an invalidated `run_id` are discarded.

### Lifecycle tests

- `Stop` never calls `sounddevice.stop()` in the parent process.
- Cooperative worker shutdown returns to `Idle`.
- A blocked worker is terminated after the grace interval without blocking the
  GUI callback.
- Repeated `Start -> Stop -> Start` never has two live workers.
- Pressing `Stop` during model loading, STT, TTS, and playback produces no later
  text or audio event.
- Window close stops the worker before destroying Tkinter.

### Integration and build tests

- Existing test suite remains green.
- Opt-in microphone smoke still exercises the production phrase iterator.
- A model-free GigaAM adapter smoke uses a fake ONNX session.
- PyInstaller build completes on Windows.
- Frozen `--smoke-test` verifies imports and multiprocessing startup/shutdown.

## Acceptance Criteria

1. Pressing `Stop` during recording, model loading, recognition, synthesis, or
   playback does not close the V2TTS window.
2. The visible pipeline is interrupted immediately; no text or audio from the
   stopped run appears afterward.
3. The Tkinter event loop is never blocked by process joins or PortAudio calls.
4. A stuck native worker is gone no later than 1.3 seconds after `Stop`.
5. Russian defaults to GigaAM v3 E2E RNN-T and can switch to the documented
   Whisper Russian models.
6. English exposes only explicit-English Whisper profiles.
7. All STT model weights live outside the executable in the V2TTS user model
   directory.
8. CPU-only Windows systems can run both language profiles; optional CUDA use and
   fallback are reported accurately.
9. Three verification passes succeed before merge: focused tests, full tests plus
   compile checks, and the Windows frozen-build workflow.

## Deferred Work

- Cloud SaluteSpeech integration and credential management.
- Languages other than Russian and English.
- Mixed-language recognition inside one phrase.
- Keeping a terminated model process warm across `Stop`; true interruption takes
  priority over avoiding model reload on the next `Start`.

## Technical References

- GigaAM v3 source and model descriptions:
  <https://github.com/salute-developers/GigaAM>
- Lightweight Windows-compatible ONNX runtime used by the adapter:
  <https://github.com/istupakov/onnx-asr>
- Pre-converted GigaAM v3 ONNX model set:
  <https://huggingface.co/istupakov/gigaam-v3-onnx>
