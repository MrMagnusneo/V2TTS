from pathlib import Path

from audio_queue import RunConfig, SpeechLoopRunner


def _run_config() -> RunConfig:
    return RunConfig(
        input_device=None,
        output_device=None,
        stt_device="cpu",
        stt_model_size="tiny",
        auto_tts_model=True,
        manual_tts_model="ru_tts",
        tts_root=None,
    )


def test_main_compiles() -> None:
    source = Path("main.py").read_text(encoding="utf-8")

    compile(source, "main.py", "exec")


def test_runner_stores_all_callbacks() -> None:
    on_status = lambda _: None
    on_text = lambda _: None
    on_error = lambda _: None

    runner = SpeechLoopRunner(_run_config(), on_status, on_text, on_error)

    assert runner.on_status is on_status
    assert runner.on_text is on_text
    assert runner.on_error is on_error
