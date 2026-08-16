from pathlib import Path

from audio_queue import RunConfig, SpeechLoopRunner
from stt_profiles import STTSelection


def _run_config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=None,
        stt=STTSelection("en", "whisper", "small", "cpu"),
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )


def test_main_compiles() -> None:
    source = Path("main.py").read_text(encoding="utf-8")

    compile(source, "main.py", "exec")


def test_runner_stores_all_callbacks() -> None:
    on_event = lambda _run_id, _kind, _message: None

    runner = SpeechLoopRunner(_run_config(), on_event)

    assert runner.on_event is on_event
