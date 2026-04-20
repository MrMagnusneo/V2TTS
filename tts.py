import os
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")
TTS_MODELS = ["ru_tts", "sam"]


@dataclass(frozen=True)
class TTSPaths:
    tts_root: Path
    sam_js: Path
    ru_tts_bin: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _default_tts_root() -> Path:
    return _project_root() / "tts"


def resolve_tts_paths(tts_root: Optional[str] = None) -> TTSPaths:
    env_root = os.getenv("V2TTS_TTS_ROOT")
    root = Path(tts_root or env_root or _default_tts_root()).expanduser().resolve()

    env_sam = os.getenv("V2TTS_SAM_JS")
    sam_js = Path(env_sam).expanduser().resolve() if env_sam else (root / "sam" / "dist" / "samjs.min.js")

    env_ru = os.getenv("V2TTS_RU_TTS_BIN")
    if env_ru:
        ru_bin = Path(env_ru).expanduser().resolve()
    else:
        candidates = [
            root / "ru_tts" / "src" / ".libs" / "ru_tts",
            root / "ru_tts" / "src" / ".libs" / "ru_tts.exe",
            root / "ru_tts" / "src" / "ru_tts",
            root / "ru_tts" / "src" / "ru_tts.exe",
        ]
        ru_bin = next((c for c in candidates if c.exists()), None)
        if ru_bin is None:
            which_path = shutil.which("ru_tts")
            ru_bin = Path(which_path).resolve() if which_path else candidates[0]

    return TTSPaths(tts_root=root, sam_js=sam_js, ru_tts_bin=ru_bin)


def choose_tts_engine(text: str, auto_select: bool = True, manual_model: str = "ru_tts") -> str:
    if not auto_select:
        return manual_model

    has_cyr = bool(CYR.search(text))
    has_lat = bool(LAT.search(text))
    if has_cyr:
        return "ru_tts"
    if has_lat:
        return "sam"
    return "ru_tts"


def tts_sam(text: str, out_wav: str, paths: TTSPaths):
    js = (
        "const SamJs=require(process.argv[1]);"
        "const s=new SamJs({speed:72,pitch:64,throat:128,mouth:128});"
        "const wav=s.wav(process.argv[2]);"
        "process.stdout.write(Buffer.from(wav));"
    )
    p = subprocess.run(
        ["node", "-e", js, str(paths.sam_js), text],
        check=True,
        stdout=subprocess.PIPE,
    )
    Path(out_wav).write_bytes(p.stdout)


def tts_ru_tts(text: str, out_wav: str, paths: TTSPaths):
    p = subprocess.run(
        [str(paths.ru_tts_bin), "-r", "1.0"],
        input=text.encode("koi8-r", errors="ignore"),
        stdout=subprocess.PIPE,
        check=True,
    )
    raw_signed8 = p.stdout
    wav_unsigned8 = bytes(b ^ 0x80 for b in raw_signed8)

    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(10000)
        wf.writeframes(wav_unsigned8)


def synthesize_text(
    text: str,
    out_wav: str,
    auto_select: bool = True,
    manual_model: str = "ru_tts",
    tts_root: Optional[str] = None,
):
    paths = resolve_tts_paths(tts_root)
    engine = choose_tts_engine(text, auto_select=auto_select, manual_model=manual_model)

    if engine == "sam":
        tts_sam(text, out_wav, paths)
    elif engine == "ru_tts":
        tts_ru_tts(text, out_wav, paths)
    else:
        raise ValueError(f"Unknown TTS model: {engine}")
