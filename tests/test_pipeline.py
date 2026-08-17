import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tqdm import tqdm

from audio_stream import CapturedPhrase, StreamMetrics
from pipeline import (
    RunConfig,
    _process_phrase,
    pipeline_process_main,
    play_audio_cancellable,
    run_pipeline,
)
from stt_profiles import STTSelection


class StopEvent:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        self.stopped = True


def run_config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=7,
        stt=STTSelection(
            "ru",
            "gigaam",
            "gigaam-v3-e2e-rnnt",
            "cpu",
        ),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )


def test_playback_checks_stop_between_chunks() -> None:
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False, False, True]
    stream = MagicMock()
    sd = MagicMock()
    sd.OutputStream.return_value.__enter__.return_value = stream
    audio = np.zeros(16000, dtype=np.float32)

    play_audio_cancellable(
        sd,
        audio,
        sample_rate=16000,
        device_index=7,
        stop_event=stop_event,
        chunk_ms=50,
    )

    assert stream.write.call_count == 2
    assert stream.write.call_args_list[0].args[0].shape == (800, 1)
    sd.play.assert_not_called()
    sd.wait.assert_not_called()


def test_run_pipeline_uses_conservative_stream_settings(tmp_path: Path) -> None:
    stop_event = StopEvent()
    events = []
    sd = MagicMock()
    sd.query_devices.return_value = {"default_samplerate": 16000.0}
    transcriber = SimpleNamespace(
        requested_device="cpu",
        actual_device="cpu",
    )
    phrase_stream = MagicMock()
    phrase_stream.iter_phrases.return_value = iter(())

    with (
        patch("pipeline.get_sounddevice", return_value=sd),
        patch("pipeline.get_soundfile"),
        patch("pipeline.AudioPhraseStream", return_value=phrase_stream) as stream_cls,
        patch("pipeline.create_transcriber", return_value=transcriber),
        patch("pipeline.is_stt_model_ready", return_value=False),
    ):
        run_pipeline(
            run_config(),
            stop_event,
            lambda kind, payload: events.append((kind, payload)),
        )

    stream_config = stream_cls.call_args.args[0]
    assert stream_config.frame_ms == 30
    assert stream_config.min_phrase_ms == 300
    assert stream_config.max_silence_ms == 700
    assert stream_config.pre_roll_ms == 200
    assert ("state", "listening") in events


def test_stop_during_stt_discards_late_text_and_skips_tts(tmp_path: Path) -> None:
    stop_event = StopEvent()
    events = []
    sd = MagicMock()
    sd.query_devices.return_value = {"default_samplerate": 16000.0}
    transcriber = MagicMock()
    transcriber.requested_device = "cpu"
    transcriber.actual_device = "cpu"

    def transcribe(*args, **kwargs):
        stop_event.set()
        return "late result"

    transcriber.transcribe_pcm16.side_effect = transcribe
    phrase_stream = MagicMock()
    phrase_stream.iter_phrases.return_value = iter(
        [CapturedPhrase(np.zeros(1600, dtype=np.int16), ended_at=10.0)]
    )
    phrase_stream.metrics.return_value = StreamMetrics(0, 0, 0)

    with (
        patch("pipeline.get_sounddevice", return_value=sd),
        patch("pipeline.get_soundfile"),
        patch("pipeline.AudioPhraseStream", return_value=phrase_stream),
        patch("pipeline.create_transcriber", return_value=transcriber),
        patch("pipeline.is_stt_model_ready", return_value=False),
        patch("pipeline.synthesize_text") as synthesize,
    ):
        run_pipeline(
            run_config(),
            stop_event,
            lambda kind, payload: events.append((kind, payload)),
        )

    assert ("text", "late result") not in events
    synthesize.assert_not_called()


def test_stop_during_model_loading_never_enters_listening() -> None:
    stop_event = StopEvent()
    events = []
    transcriber = MagicMock(requested_device="cpu", actual_device="cpu")

    def load_model(*args, **kwargs):
        stop_event.set()
        return transcriber

    with (
        patch("pipeline.get_sounddevice", return_value=MagicMock()),
        patch("pipeline.get_soundfile"),
        patch("pipeline.select_input_sample_rate", return_value=16000),
        patch("pipeline.is_stt_model_ready", return_value=False),
        patch("pipeline.create_transcriber", side_effect=load_model),
        patch("pipeline.AudioPhraseStream") as stream_cls,
    ):
        run_pipeline(
            run_config(),
            stop_event,
            lambda kind, payload: events.append((kind, payload)),
        )

    assert ("state", "listening") not in events
    stream_cls.return_value.iter_phrases.assert_not_called()


def test_output_error_includes_device_rate_and_channels() -> None:
    sd = MagicMock()
    sd.OutputStream.side_effect = RuntimeError("Invalid sample rate")

    with pytest.raises(RuntimeError) as error:
        play_audio_cancellable(
            sd,
            np.zeros(100, dtype=np.float32),
            sample_rate=44100,
            device_index=23,
            stop_event=StopEvent(),
        )

    message = str(error.value)
    assert "output device 23" in message
    assert "44100 Hz" in message
    assert "1 channel" in message


def test_stop_during_tts_skips_file_read_and_playback() -> None:
    stop_event = StopEvent()
    transcriber = MagicMock()
    transcriber.transcribe_pcm16.return_value = "hello"
    sf = MagicMock()

    def synthesize(**kwargs):
        stop_event.set()
        return "ru_tts"

    with (
        patch("pipeline.synthesize_text", side_effect=synthesize),
        patch("pipeline.play_audio_cancellable") as play,
    ):
        _process_phrase(
            np.zeros(1600, dtype=np.int16),
            transcriber,
            16000,
            "STT test",
            MagicMock(),
            sf,
            run_config(),
            stop_event,
            MagicMock(),
            queue_ms=0,
            phrase_age_ms=0,
            dropped_frames=0,
        )

    sf.read.assert_not_called()
    play.assert_not_called()


def test_pipeline_child_restores_stderr_before_model_download(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "stderr", None)
    events = queue.Queue()

    def download_with_progress(*args, **kwargs) -> None:
        list(tqdm(range(1)))

    with patch("pipeline.run_pipeline", side_effect=download_with_progress):
        pipeline_process_main(
            "run-1",
            run_config(),
            StopEvent(),
            events,
        )

    queued = []
    while not events.empty():
        queued.append(events.get_nowait())
    assert not [event for event in queued if event[1] == "error"]
