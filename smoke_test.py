from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

from audio_backend import get_soundfile
from tts import synthesize_text


SMOKE_CASES = (
    ("ru_tts", "Проверка синтеза речи"),
    ("sam", "Packaged speech test"),
)


def _multiprocessing_smoke_child(result_queue) -> None:
    result_queue.put("ok")


def run_multiprocessing_smoke() -> None:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_multiprocessing_smoke_child,
        args=(result_queue,),
        name="V2TTS-smoke-child",
    )
    try:
        process.start()
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
            raise RuntimeError("Multiprocessing smoke child did not exit")
        if process.exitcode != 0:
            raise RuntimeError(
                f"Multiprocessing smoke child exited with {process.exitcode}"
            )
        if result_queue.get(timeout=2) != "ok":
            raise RuntimeError("Multiprocessing smoke child returned bad data")
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        result_queue.close()
        result_queue.join_thread()
        try:
            process.close()
        except ValueError:
            pass


def run_packaged_smoke(tts_root: Path | None = None) -> int:
    run_multiprocessing_smoke()
    soundfile = get_soundfile()

    with tempfile.TemporaryDirectory(prefix="v2tts-smoke-") as temp_dir:
        for model, text in SMOKE_CASES:
            output = Path(temp_dir) / f"{model}.wav"
            selected = synthesize_text(
                text,
                str(output),
                auto_select=False,
                manual_model=model,
                tts_root=str(tts_root) if tts_root is not None else None,
            )
            if selected != model:
                raise RuntimeError(
                    f"TTS smoke test requested {model}, but used {selected}"
                )

            samples, sample_rate = soundfile.read(str(output))
            if sample_rate <= 0 or len(samples) == 0:
                raise RuntimeError(f"TTS smoke test produced empty audio for {model}")

    print("V2TTS packaged smoke test passed: ru_tts, sam")
    return 0
