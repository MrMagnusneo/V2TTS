import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from audio_backend import get_sounddevice, get_soundfile
from audio_stream import AudioPhraseStream, StreamConfig
from stt import Transcriber, create_transcriber, is_stt_model_ready
from stt_profiles import STTSelection
from tts import synthesize_text


EmitCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class RunConfig:
    input_device: Optional[int]
    output_device: Optional[int]
    stt: STTSelection
    auto_tts_model: bool
    manual_tts_model: str
    tts_root: Optional[str]


def prepare_audio_for_output(
    sd,
    data,
    source_rate: int,
    device_index: Optional[int],
) -> tuple[np.ndarray, int]:
    audio = np.asarray(data, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]

    def supports(sample_rate: int) -> bool:
        try:
            sd.check_output_settings(
                device=device_index,
                channels=channels,
                samplerate=sample_rate,
                dtype="float32",
            )
            return True
        except Exception:
            return False

    if supports(source_rate):
        return audio, source_rate

    candidates: list[int] = []
    try:
        info = sd.query_devices(device_index, "output")
        default_rate = int(round(float(info.get("default_samplerate") or 0)))
        if default_rate > 0:
            candidates.append(default_rate)
    except Exception:
        pass

    candidates.extend([48000, 44100, 32000, 22050, 16000])
    output_rate = next(
        (
            sample_rate
            for sample_rate in dict.fromkeys(candidates)
            if sample_rate != source_rate and supports(sample_rate)
        ),
        None,
    )
    if output_rate is None:
        raise RuntimeError(
            f"Output device {device_index} does not support the TTS sample rate "
            f"{source_rate} Hz or standard fallback rates"
        )

    if len(audio) == 0:
        return audio, output_rate

    if output_rate < source_rate:
        tap_count = 127
        center = (tap_count - 1) / 2
        positions = np.arange(tap_count, dtype=np.float64) - center
        cutoff = 0.5 * output_rate / source_rate * 0.9
        kernel = 2 * cutoff * np.sinc(2 * cutoff * positions)
        kernel *= np.kaiser(tap_count, beta=8.6)
        kernel /= np.sum(kernel)
        padding = tap_count // 2

        def lowpass(channel: np.ndarray) -> np.ndarray:
            padded = np.pad(channel, (padding, padding), mode="edge")
            return np.convolve(padded, kernel, mode="valid")

        if audio.ndim == 1:
            audio = lowpass(audio)
        else:
            audio = np.column_stack(
                [lowpass(audio[:, channel]) for channel in range(channels)]
            )

    output_length = max(1, int(round(len(audio) * output_rate / source_rate)))
    source_positions = np.arange(len(audio), dtype=np.float64) / source_rate
    output_positions = np.arange(output_length, dtype=np.float64) / output_rate
    if audio.ndim == 1:
        resampled = np.interp(output_positions, source_positions, audio)
    else:
        resampled = np.column_stack(
            [
                np.interp(
                    output_positions,
                    source_positions,
                    audio[:, channel],
                )
                for channel in range(channels)
            ]
        )
    return resampled.astype(np.float32), output_rate


def play_audio_cancellable(
    sd,
    data,
    sample_rate: int,
    device_index: Optional[int],
    stop_event,
    chunk_ms: int = 50,
) -> None:
    audio = np.asarray(data, dtype=np.float32)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    output = audio[:, np.newaxis] if audio.ndim == 1 else audio
    frames_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))

    try:
        with sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            device=device_index,
        ) as stream:
            for offset in range(0, len(output), frames_per_chunk):
                if stop_event.is_set():
                    return
                stream.write(output[offset : offset + frames_per_chunk])
    except Exception as exc:
        raise RuntimeError(
            f"Could not use output device {device_index} at {sample_rate} Hz "
            f"with {channels} channel(s): {exc}"
        ) from exc


def select_input_sample_rate(sd, device_index: Optional[int]) -> int:
    try:
        info = sd.query_devices(device_index, "input")
        raw_rate = float(info.get("default_samplerate") or 16000.0)
        default_rate = int(round(raw_rate)) if raw_rate > 0 else 16000
        sd.check_input_settings(
            device=device_index,
            channels=1,
            samplerate=default_rate,
            dtype="float32",
        )
        return default_rate
    except Exception:
        default_rate = None

    for sample_rate in [16000, 32000, 44100, 48000]:
        if sample_rate == default_rate:
            continue
        try:
            sd.check_input_settings(
                device=device_index,
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
            )
            return sample_rate
        except Exception:
            continue

    return default_rate or 16000


def _process_phrase(
    phrase_pcm16: np.ndarray,
    transcriber: Transcriber,
    sample_rate: int,
    stt_description: str,
    sd,
    sf,
    config: RunConfig,
    stop_event,
    emit: EmitCallback,
    *,
    queue_ms: int,
    phrase_age_ms: int,
    dropped_frames: int,
) -> None:
    stt_started = time.perf_counter()
    try:
        text = transcriber.transcribe_pcm16(
            phrase_pcm16,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        if not stop_event.is_set():
            emit("error", f"STT failed: {exc}")
        return
    stt_ms = int(round((time.perf_counter() - stt_started) * 1000))

    if stop_event.is_set() or not text:
        return
    emit("text", text)
    if stop_event.is_set():
        return

    file_descriptor, output_wav = tempfile.mkstemp(suffix=".wav")
    os.close(file_descriptor)
    try:
        tts_started = time.perf_counter()
        used_engine = synthesize_text(
            text=text,
            out_wav=output_wav,
            auto_select=config.auto_tts_model,
            manual_model=config.manual_tts_model,
            tts_root=config.tts_root,
        )
        tts_ms = int(round((time.perf_counter() - tts_started) * 1000))
        if stop_event.is_set():
            return

        data, source_rate = sf.read(output_wav, dtype="float32")
        data, output_rate = prepare_audio_for_output(
            sd,
            data,
            source_rate=source_rate,
            device_index=config.output_device,
        )
        if stop_event.is_set():
            return

        emit(
            "status",
            f"Listening... ({stt_description}, tts={used_engine}, "
            f"stt_ms={stt_ms}, tts_ms={tts_ms}, queue_ms={queue_ms}, "
            f"phrase_age_ms={phrase_age_ms}, "
            f"dropped_frames={dropped_frames})",
        )
        play_audio_cancellable(
            sd,
            data,
            output_rate,
            config.output_device,
            stop_event,
        )
    except Exception as exc:
        if not stop_event.is_set():
            emit("error", f"TTS/Playback failed: {exc}")
    finally:
        if os.path.exists(output_wav):
            os.remove(output_wav)


def run_pipeline(
    config: RunConfig,
    stop_event,
    emit: EmitCallback,
) -> None:
    sd = get_sounddevice()
    sf = get_soundfile()
    sample_rate = select_input_sample_rate(sd, config.input_device)
    stream_config = StreamConfig(
        sample_rate=sample_rate,
        channels=1,
        frame_ms=30,
        min_phrase_ms=300,
        max_silence_ms=700,
        pre_roll_ms=200,
        device=config.input_device,
    )
    phrase_stream = AudioPhraseStream(stream_config)

    action = "Loading" if is_stt_model_ready(config.stt) else "Downloading"
    emit("status", f"{action} STT model {config.stt.model}...")
    transcriber = create_transcriber(
        config.stt,
        on_warning=lambda message: emit("warning", message),
    )
    if stop_event.is_set():
        return

    stt_description = (
        f"STT: engine={config.stt.engine}, language={config.stt.language}, "
        f"model={config.stt.model}, requested={transcriber.requested_device}, "
        f"actual={transcriber.actual_device}, sr={stream_config.sample_rate}"
    )
    emit("state", "listening")
    emit("status", f"Listening... ({stt_description})")

    last_dropped_frames = 0
    for phrase in phrase_stream.iter_phrases(stop_event=stop_event):
        if stop_event.is_set():
            break

        metrics = phrase_stream.metrics()
        if metrics.dropped_frames > last_dropped_frames:
            emit(
                "status",
                "Audio overload: "
                f"dropped_frames={metrics.dropped_frames}, "
                f"queue_ms={metrics.queue_ms}",
            )
        last_dropped_frames = metrics.dropped_frames
        phrase_age_ms = max(
            0,
            int(round((time.monotonic() - phrase.ended_at) * 1000)),
        )
        _process_phrase(
            phrase.pcm16,
            transcriber,
            stream_config.sample_rate,
            stt_description,
            sd,
            sf,
            config,
            stop_event,
            emit,
            queue_ms=metrics.queue_ms,
            phrase_age_ms=phrase_age_ms,
            dropped_frames=metrics.dropped_frames,
        )

    if not stop_event.is_set():
        emit("status", "Stopped")


def pipeline_process_main(
    run_id: str,
    config: RunConfig,
    stop_event,
    event_queue,
) -> None:
    def emit(kind: str, payload: str) -> None:
        event_queue.put((run_id, kind, payload))

    try:
        run_pipeline(config, stop_event, emit)
    except Exception as exc:
        emit("error", str(exc))
    finally:
        emit("worker_stopped", "")
