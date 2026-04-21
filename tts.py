import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")
TTS_MODELS = ["ru_tts", "sam"]

RU_BIN_REL_CANDIDATES = [
    Path("ru_tts/bin/ru_tts.exe"),
    Path("ru_tts/bin/ru_tts"),
    Path("ru_tts/src/.libs/ru_tts"),
    Path("ru_tts/src/.libs/ru_tts.exe"),
    Path("ru_tts/src/ru_tts"),
    Path("ru_tts/src/ru_tts.exe"),
    Path("ru_tts/ru_tts.exe"),
    Path("ru_tts/ru_tts"),
]


@dataclass(frozen=True)
class TTSPaths:
    tts_root: Path
    sam_js: Path
    ru_tts_bin: Path


def _exists_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _meipass_dir() -> Optional[Path]:
    p = getattr(sys, "_MEIPASS", None)
    if not p:
        return None
    return Path(p).resolve()


def _source_root() -> Path:
    return Path(__file__).resolve().parent


def _exe_dir() -> Path:
    return Path(sys.executable).resolve().parent


def _default_tts_root() -> Path:
    # Prefer installed runtime folder for frozen builds.
    if _is_frozen() and (_exe_dir() / "runtime" / "tts").exists():
        return _exe_dir() / "runtime" / "tts"
    if _is_frozen() and (_user_data_root() / "tts").exists():
        return _user_data_root() / "tts"
    return _source_root() / "tts"


def _user_data_root() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return (base / "V2TTS").resolve()


def _candidate_roots(user_tts_root: Optional[str]) -> list[Path]:
    roots: list[Path] = []

    if user_tts_root:
        roots.append(Path(user_tts_root).expanduser().resolve())

    env_root = os.getenv("V2TTS_TTS_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())

    if _is_frozen():
        roots.append(_exe_dir() / "runtime" / "tts")
        roots.append(_exe_dir() / "tts")
        mei = _meipass_dir()
        if mei is not None:
            roots.append(mei / "runtime" / "tts")
            roots.append(mei / "tts")
    else:
        roots.append(_source_root() / "tts")

    roots.append(_user_data_root() / "tts")
    roots.append(Path.cwd() / "tts")

    dedup: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(root)
    return dedup


def _find_first_existing_root(user_tts_root: Optional[str]) -> Optional[Path]:
    for root in _candidate_roots(user_tts_root):
        if not root.exists():
            continue
        sam_js = root / "sam" / "dist" / "samjs.min.js"
        has_ru = any(_exists_file(root / rel) for rel in RU_BIN_REL_CANDIDATES)
        if _exists_file(sam_js) or has_ru:
            return root
    return None


def prepare_runtime_tts_root(tts_root: Optional[str] = None) -> Path:
    # If user set path explicitly, honor it.
    if tts_root:
        return Path(tts_root).expanduser().resolve()
    found = _find_first_existing_root(None)
    if found is not None:
        return found
    return _default_tts_root()


def _find_ru_bin(root: Path) -> Path:
    env_ru = os.getenv("V2TTS_RU_TTS_BIN")
    if env_ru:
        return Path(env_ru).expanduser().resolve()

    for rel in RU_BIN_REL_CANDIDATES:
        candidate = root / rel
        if _exists_file(candidate):
            return candidate

    which_path = shutil.which("ru_tts")
    if which_path:
        return Path(which_path).resolve()

    return root / RU_BIN_REL_CANDIDATES[0]


def resolve_tts_paths(tts_root: Optional[str] = None) -> TTSPaths:
    root = prepare_runtime_tts_root(tts_root)

    env_sam = os.getenv("V2TTS_SAM_JS")
    sam_js = Path(env_sam).expanduser().resolve() if env_sam else (root / "sam" / "dist" / "samjs.min.js")

    ru_bin = _find_ru_bin(root)

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
    return "sam"


def _node_cmd() -> Optional[str]:
    env_node = os.getenv("V2TTS_NODE")
    if env_node:
        node_env = Path(env_node).expanduser().resolve()
        if _exists_file(node_env):
            return str(node_env)

    runtime_candidates = []
    if _is_frozen():
        runtime_candidates.extend(
            [
                _exe_dir() / "runtime" / "node" / "node.exe",
                _exe_dir() / "runtime" / "node" / "node",
            ]
        )
        mei = _meipass_dir()
        if mei is not None:
            runtime_candidates.extend(
                [
                    mei / "runtime" / "node" / "node.exe",
                    mei / "runtime" / "node" / "node",
                ]
            )

    for candidate in runtime_candidates:
        if _exists_file(candidate):
            return str(candidate)

    for name in ("node", "node.exe"):
        path = shutil.which(name)
        if path:
            return path
    return None


def tts_sam(text: str, out_wav: str, paths: TTSPaths):
    if not _exists_file(paths.sam_js):
        raise FileNotFoundError(f"SAM JS file not found: {paths.sam_js}")

    node = _node_cmd()
    if node is None:
        raise FileNotFoundError("Node.js executable not found (node/node.exe).")

    js = (
        "const SamJs=require(process.argv[1]);"
        "const s=new SamJs({speed:72,pitch:64,throat:128,mouth:128});"
        "const wav=s.wav(process.argv[2]);"
        "process.stdout.write(Buffer.from(wav));"
    )
    p = subprocess.run(
        [node, "-e", js, str(paths.sam_js), text],
        check=True,
        stdout=subprocess.PIPE,
    )
    Path(out_wav).write_bytes(p.stdout)


def tts_ru_tts(text: str, out_wav: str, paths: TTSPaths):
    if not _exists_file(paths.ru_tts_bin):
        raise FileNotFoundError(f"ru_tts binary not found: {paths.ru_tts_bin}")

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
) -> str:
    paths = resolve_tts_paths(tts_root)
    requested = choose_tts_engine(text, auto_select=auto_select, manual_model=manual_model)

    # Ordered fallback chain to keep runtime alive but only for ru_tts/sam.
    chain = [requested]
    if requested == "ru_tts":
        chain.extend(["sam"])
    elif requested == "sam":
        chain.extend([])

    last_error: Optional[Exception] = None
    tried: list[str] = []

    for engine in chain:
        if engine in tried:
            continue
        tried.append(engine)

        try:
            if engine == "ru_tts":
                tts_ru_tts(text, out_wav, paths)
            elif engine == "sam":
                tts_sam(text, out_wav, paths)
            else:
                raise ValueError(f"Unknown TTS model: {engine}")
            return engine
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise RuntimeError(
        f"TTS failed ({', '.join(tried)}). "
        f"root={paths.tts_root}; sam={paths.sam_js}; ru_tts={paths.ru_tts_bin}; error={last_error}"
    )
