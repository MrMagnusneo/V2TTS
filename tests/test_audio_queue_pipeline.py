from pathlib import Path
from unittest.mock import MagicMock, patch

from audio_queue import RunConfig, SpeechLoopRunner
from audio_stream import CapturedPhrase, StreamMetrics


def _runner(statuses: list[str], texts: list[str], errors: list[str]) -> SpeechLoopRunner:
    return SpeechLoopRunner(
        RunConfig(
            input_device=None,
            output_device=7,
            stt_device="cpu",
            stt_model_size="tiny",
            auto_tts_model=True,
            manual_tts_model="ru_tts",
            tts_root=None,
        ),
        statuses.append,
        texts.append,
        errors.append,
    )


def _audio_fakes():
    sd = MagicMock()
    sf = MagicMock()
    sf.read.return_value = ([0.25], 16000)
    return sd, sf


def test_process_phrase_runs_pipeline_once_and_removes_temporary_wav() -> None:
    statuses: list[str] = []
    texts: list[str] = []
    errors: list[str] = []
    runner = _runner(statuses, texts, errors)
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "hello"
    sd, sf = _audio_fakes()
    generated_paths: list[str] = []

    def synthesize(**kwargs) -> str:
        generated_paths.append(kwargs["out_wav"])
        Path(kwargs["out_wav"]).write_bytes(b"RIFF")
        return "sam"

    with (
        patch("audio_queue.synthesize_text", side_effect=synthesize) as synth,
        patch("time.perf_counter", side_effect=[1.0, 1.125, 2.0, 2.25]),
    ):
        runner._process_phrase(
            object(),
            transcriber,
            16000,
            "STT test",
            sd,
            sf,
            queue_ms=80,
            phrase_age_ms=350,
            dropped_frames=2,
        )

    transcriber.transcribe_pcm16.assert_called_once()
    assert texts == ["hello"]
    synth.assert_called_once()
    sf.read.assert_called_once_with(generated_paths[0], dtype="float32")
    sd.play.assert_called_once_with([0.25], 16000, device=7)
    sd.wait.assert_called_once_with()
    assert statuses == [
        "Listening... (STT test, tts=sam, stt_ms=125, tts_ms=250, "
        "queue_ms=80, phrase_age_ms=350, dropped_frames=2)"
    ]
    assert errors == []
    assert not Path(generated_paths[0]).exists()


def test_process_phrase_skips_synthesis_for_empty_transcription() -> None:
    statuses: list[str] = []
    texts: list[str] = []
    errors: list[str] = []
    runner = _runner(statuses, texts, errors)
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = ""
    sd, sf = _audio_fakes()

    with patch("audio_queue.synthesize_text") as synth:
        runner._process_phrase(object(), transcriber, 16000, "STT test", sd, sf)

    assert texts == []
    assert errors == []
    synth.assert_not_called()
    sf.read.assert_not_called()
    sd.play.assert_not_called()


def test_process_phrase_reports_stt_error_without_synthesis() -> None:
    statuses: list[str] = []
    texts: list[str] = []
    errors: list[str] = []
    runner = _runner(statuses, texts, errors)
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.side_effect = RuntimeError("decoder unavailable")
    sd, sf = _audio_fakes()

    with patch("audio_queue.synthesize_text") as synth:
        runner._process_phrase(object(), transcriber, 16000, "STT test", sd, sf)

    assert texts == []
    assert errors == ["STT failed: decoder unavailable"]
    synth.assert_not_called()
    sf.read.assert_not_called()
    sd.play.assert_not_called()


def test_process_phrase_removes_temporary_wav_after_tts_error() -> None:
    statuses: list[str] = []
    texts: list[str] = []
    errors: list[str] = []
    runner = _runner(statuses, texts, errors)
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "hello"
    sd, sf = _audio_fakes()
    generated_paths: list[str] = []

    def fail_synthesis(**kwargs):
        generated_paths.append(kwargs["out_wav"])
        raise RuntimeError("engine unavailable")

    with patch("audio_queue.synthesize_text", side_effect=fail_synthesis):
        runner._process_phrase(object(), transcriber, 16000, "STT test", sd, sf)

    assert texts == ["hello"]
    assert errors == ["TTS/Playback failed: engine unavailable"]
    sf.read.assert_not_called()
    sd.play.assert_not_called()
    assert not Path(generated_paths[0]).exists()


def test_run_processes_each_phrase_only_once() -> None:
    statuses: list[str] = []
    texts: list[str] = []
    errors: list[str] = []
    runner = _runner(statuses, texts, errors)
    sd, sf = _audio_fakes()
    sd.query_devices.return_value = {"default_samplerate": 16000.0}
    transcriber = MagicMock()
    transcriber.requested_device = "cpu"
    transcriber.actual_device = "cpu"
    transcriber.transcribe_pcm16.return_value = "hello"
    phrase_stream = MagicMock()
    phrase_stream.iter_phrases.return_value = iter(
        [CapturedPhrase(pcm16=object(), ended_at=100.0)]
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
        patch("audio_queue.get_sounddevice", return_value=sd),
        patch("audio_queue.get_soundfile", return_value=sf),
        patch("audio_queue.AudioPhraseStream", return_value=phrase_stream),
        patch("audio_queue.WhisperTranscriber", return_value=transcriber),
        patch("audio_queue.synthesize_text", side_effect=synthesize) as synth,
        patch("time.monotonic", return_value=100.35),
        patch("time.perf_counter", side_effect=[1.0, 1.125, 2.0, 2.25]),
    ):
        runner._run()

    transcriber.transcribe_pcm16.assert_called_once()
    assert texts == ["hello"]
    synth.assert_called_once()
    sd.play.assert_called_once()
    sd.wait.assert_called_once()
    assert errors == []
    assert "Audio overload: dropped_frames=3, queue_ms=80" in statuses
    assert (
        "Listening... (STT: requested=cpu, actual=cpu, sr=16000, tts=sam, "
        "stt_ms=125, tts_ms=250, queue_ms=80, phrase_age_ms=350, "
        "dropped_frames=3)"
    ) in statuses
