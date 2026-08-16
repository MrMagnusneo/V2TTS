from __future__ import annotations

import tempfile
from pathlib import Path

from audio_backend import get_soundfile
from tts import synthesize_text


SMOKE_CASES = (
    ("ru_tts", "Проверка синтеза речи"),
    ("sam", "Packaged speech test"),
)


def run_packaged_smoke(tts_root: Path | None = None) -> int:
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
