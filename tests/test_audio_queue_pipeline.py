import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from audio_stream import CapturedPhrase, StreamMetrics
from pipeline import RunConfig, _process_phrase, run_pipeline
from stt_profiles import STTSelection


def _config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=7,
        stt=STTSelection("en", "whisper", "small", "cpu"),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )


def _audio_fakes():
    sd = MagicMock()
    sf = MagicMock()
    sf.read.return_value = (np.array([0.25], dtype=np.float32), 16000)
    return sd, sf


def _emit_lists():
    statuses = []
    texts = []
    errors = []

    def emit(kind: str, payload: str) -> None:
        if kind == "status":
            statuses.append(payload)
        elif kind == "text":
            texts.append(payload)
        elif kind == "error":
            errors.append(payload)

    return statuses, texts, errors, emit


def _process(
    transcriber,
    sd,
    sf,
    emit,
    *,
    queue_ms: int = 0,
    phrase_age_ms: int = 0,
    dropped_frames: int = 0,
) -> None:
    _process_phrase(
        np.zeros(1600, dtype=np.int16),
        transcriber,
        16000,
        "STT test",
        sd,
        sf,
        _config(),
        threading.Event(),
        emit,
        queue_ms=queue_ms,
        phrase_age_ms=phrase_age_ms,
        dropped_frames=dropped_frames,
    )


def test_process_phrase_runs_pipeline_once_and_removes_temporary_wav() -> None:
    statuses, texts, errors, emit = _emit_lists()
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "hello"
    sd, sf = _audio_fakes()
    generated_paths = []

    def synthesize(**kwargs) -> str:
        generated_paths.append(kwargs["out_wav"])
        Path(kwargs["out_wav"]).write_bytes(b"RIFF")
        return "sam"

    with (
        patch("pipeline.synthesize_text", side_effect=synthesize) as synth,
        patch("pipeline.play_audio_cancellable") as play,
        patch(
            "pipeline.time.perf_counter",
            side_effect=[1.0, 1.125, 2.0, 2.25],
        ),
    ):
        _process(
            transcriber,
            sd,
            sf,
            emit,
            queue_ms=80,
            phrase_age_ms=350,
            dropped_frames=2,
        )

    transcriber.transcribe_pcm16.assert_called_once()
    assert texts == ["hello"]
    synth.assert_called_once()
    sf.read.assert_called_once_with(generated_paths[0], dtype="float32")
    play.assert_called_once()
    assert statuses == [
        "Listening... (STT test, tts=sam, stt_ms=125, tts_ms=250, "
        "queue_ms=80, phrase_age_ms=350, dropped_frames=2)"
    ]
    assert errors == []
    assert not Path(generated_paths[0]).exists()


def test_process_phrase_skips_synthesis_for_empty_transcription() -> None:
    _, texts, errors, emit = _emit_lists()
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = ""
    sd, sf = _audio_fakes()

    with patch("pipeline.synthesize_text") as synth:
        _process(transcriber, sd, sf, emit)

    assert texts == []
    assert errors == []
    synth.assert_not_called()
    sf.read.assert_not_called()


def test_process_phrase_reports_stt_error_without_synthesis() -> None:
    _, texts, errors, emit = _emit_lists()
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.side_effect = RuntimeError("decoder unavailable")
    sd, sf = _audio_fakes()

    with patch("pipeline.synthesize_text") as synth:
        _process(transcriber, sd, sf, emit)

    assert texts == []
    assert errors == ["STT failed: decoder unavailable"]
    synth.assert_not_called()
    sf.read.assert_not_called()


def test_process_phrase_removes_temporary_wav_after_tts_error() -> None:
    _, texts, errors, emit = _emit_lists()
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "hello"
    sd, sf = _audio_fakes()
    generated_paths = []

    def fail_synthesis(**kwargs):
        generated_paths.append(kwargs["out_wav"])
        raise RuntimeError("engine unavailable")

    with patch("pipeline.synthesize_text", side_effect=fail_synthesis):
        _process(transcriber, sd, sf, emit)

    assert texts == ["hello"]
    assert errors == ["TTS/Playback failed: engine unavailable"]
    sf.read.assert_not_called()
    assert not Path(generated_paths[0]).exists()


def test_run_processes_each_phrase_only_once(tmp_path: Path) -> None:
    statuses, texts, errors, emit = _emit_lists()
    stop_event = threading.Event()
    sd, sf = _audio_fakes()
    sd.query_devices.return_value = {"default_samplerate": 16000.0}
    transcriber = MagicMock()
    transcriber.requested_device = "cpu"
    transcriber.actual_device = "cpu"
    transcriber.transcribe_pcm16.return_value = "hello"
    phrase_stream = MagicMock()
    phrase_stream.iter_phrases.return_value = iter(
        [CapturedPhrase(pcm16=np.zeros(1600, dtype=np.int16), ended_at=100.0)]
    )
    phrase_stream.metrics.return_value = StreamMetrics(
        queue_depth_frames=4,
        queue_ms=80,
        dropped_frames=3,
    )

    def synthesize(**kwargs) -> str:
        Path(kwargs["out_wav"]).write_bytes(b"RIFF")
        return "sam"

    with (
        patch("pipeline.get_sounddevice", return_value=sd),
        patch("pipeline.get_soundfile", return_value=sf),
        patch("pipeline.AudioPhraseStream", return_value=phrase_stream),
        patch("pipeline.create_transcriber", return_value=transcriber),
        patch("pipeline.is_stt_model_ready", return_value=False),
        patch("pipeline.synthesize_text", side_effect=synthesize) as synth,
        patch("pipeline.play_audio_cancellable") as play,
        patch("pipeline.time.monotonic", return_value=100.35),
        patch(
            "pipeline.time.perf_counter",
            side_effect=[1.0, 1.125, 2.0, 2.25],
        ),
    ):
        run_pipeline(_config(), stop_event, emit)

    transcriber.transcribe_pcm16.assert_called_once()
    assert texts == ["hello"]
    synth.assert_called_once()
    play.assert_called_once()
    assert errors == []
    assert "Audio overload: dropped_frames=3, queue_ms=80" in statuses
    assert any(
        "stt_ms=125, tts_ms=250, queue_ms=80, phrase_age_ms=350, "
        "dropped_frames=3" in status
        for status in statuses
    )
