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

RU_DLL_REL_CANDIDATES = [
    Path("ru_tts/bin/ru_tts.dll"),
    Path("ru_tts/lib/x64/ru_tts.dll"),
    Path("ru_tts/lib/ru_tts.dll"),
    Path("ru_tts/synthDrivers/ru_tts/lib/x64/ru_tts.dll"),
    Path("ru_tts/ru_tts.dll"),
]

RULEX_DLL_REL_CANDIDATES = [
    Path("ru_tts/bin/rulex.dll"),
    Path("ru_tts/lib/x64/rulex.dll"),
    Path("ru_tts/lib/rulex.dll"),
    Path("ru_tts/synthDrivers/ru_tts/lib/x64/rulex.dll"),
    Path("ru_tts/rulex.dll"),
]


@dataclass(frozen=True)
class TTSPaths:
    tts_root: Path
    sam_js: Path
    ru_tts_bin: Path
    ru_tts_dll: Path
    rulex_dll: Path


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
        roots.append(_source_root() / "installer" / "runtime" / "tts")

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
        has_ru = any(_exists_file(root / rel) for rel in RU_BIN_REL_CANDIDATES) or any(
            _exists_file(root / rel) for rel in RU_DLL_REL_CANDIDATES
        )
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


def _find_ru_dll(root: Path) -> Path:
    env_ru_dll = os.getenv("V2TTS_RU_TTS_DLL")
    if env_ru_dll:
        return Path(env_ru_dll).expanduser().resolve()

    for rel in RU_DLL_REL_CANDIDATES:
        candidate = root / rel
        if _exists_file(candidate):
            return candidate

    return root / RU_DLL_REL_CANDIDATES[0]


def _find_rulex_dll(root: Path) -> Path:
    env_rulex = os.getenv("V2TTS_RULEX_DLL")
    if env_rulex:
        return Path(env_rulex).expanduser().resolve()

    for rel in RULEX_DLL_REL_CANDIDATES:
        candidate = root / rel
        if _exists_file(candidate):
            return candidate

    return root / RULEX_DLL_REL_CANDIDATES[0]


def resolve_tts_paths(tts_root: Optional[str] = None) -> TTSPaths:
    root = prepare_runtime_tts_root(tts_root)

    env_sam = os.getenv("V2TTS_SAM_JS")
    sam_js = Path(env_sam).expanduser().resolve() if env_sam else (root / "sam" / "dist" / "samjs.min.js")

    ru_bin = _find_ru_bin(root)
    ru_dll = _find_ru_dll(root)
    rulex_dll = _find_rulex_dll(root)

    return TTSPaths(tts_root=root, sam_js=sam_js, ru_tts_bin=ru_bin, ru_tts_dll=ru_dll, rulex_dll=rulex_dll)


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
    else:
        runtime_candidates.extend(
            [
                _source_root() / "installer" / "runtime" / "node" / "node.exe",
                _source_root() / "installer" / "runtime" / "node" / "node",
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


def _write_ru_tts_raw_to_wav(raw_signed8: bytes, out_wav: str) -> None:
    wav_unsigned8 = bytes(b ^ 0x80 for b in raw_signed8)

    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(10000)
        wf.writeframes(wav_unsigned8)


def _tts_ru_tts_exe(text: str, paths: TTSPaths) -> bytes:
    p = subprocess.run(
        [str(paths.ru_tts_bin), "-r", "1.0"],
        input=text.encode("koi8-r", errors="ignore"),
        stdout=subprocess.PIPE,
        check=True,
    )
    return p.stdout


def _tts_ru_tts_dll(text: str, paths: TTSPaths) -> bytes:
    import ctypes
    from ctypes import (
        CFUNCTYPE,
        POINTER,
        Structure,
        byref,
        c_char_p,
        c_int,
        c_size_t,
        c_void_p,
        create_string_buffer,
        string_at,
    )

    class RuTtsConf(Structure):
        _fields_ = [
            ("speech_rate", c_int),
            ("voice_pitch", c_int),
            ("intonation", c_int),
            ("general_gap_factor", c_int),
            ("comma_gap_factor", c_int),
            ("dot_gap_factor", c_int),
            ("semicolon_gap_factor", c_int),
            ("colon_gap_factor", c_int),
            ("question_gap_factor", c_int),
            ("exclamation_gap_factor", c_int),
            ("intonational_gap_factor", c_int),
            ("flags", c_int),
        ]

    callback_t = CFUNCTYPE(c_int, c_void_p, c_size_t, c_void_p)

    dll_dirs = [paths.ru_tts_dll.parent]
    if _exists_file(paths.rulex_dll):
        dll_dirs.append(paths.rulex_dll.parent)

    added_dirs = []
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_dir in dll_dirs:
            try:
                added_dirs.append(os.add_dll_directory(str(dll_dir)))
            except Exception:
                pass

    try:
        if _exists_file(paths.rulex_dll):
            if os.name == "nt":
                ctypes.WinDLL(str(paths.rulex_dll))
            else:
                ctypes.CDLL(str(paths.rulex_dll))

        if os.name == "nt":
            ru_lib = ctypes.WinDLL(str(paths.ru_tts_dll))
        else:
            ru_lib = ctypes.CDLL(str(paths.ru_tts_dll))

        ru_lib.ru_tts_config_init.argtypes = [POINTER(RuTtsConf)]
        ru_lib.ru_tts_config_init.restype = None
        ru_lib.ru_tts_transfer.argtypes = [POINTER(RuTtsConf), c_char_p, c_void_p, c_size_t, callback_t, c_void_p]
        ru_lib.ru_tts_transfer.restype = None

        conf = RuTtsConf()
        ru_lib.ru_tts_config_init(byref(conf))

        raw = bytearray()

        @callback_t
        def on_audio(buffer, size, _user_data):
            if size:
                raw.extend(string_at(buffer, size))
            return 0

        scratch = create_string_buffer(8192)
        text_koi8 = text.encode("koi8-r", errors="ignore")
        ru_lib.ru_tts_transfer(byref(conf), text_koi8, scratch, len(scratch), on_audio, None)

        if not raw:
            raise RuntimeError("ru_tts.dll returned empty audio stream")

        return bytes(raw)
    finally:
        for token in added_dirs:
            try:
                token.close()
            except Exception:
                pass


def tts_ru_tts(text: str, out_wav: str, paths: TTSPaths):
    if _exists_file(paths.ru_tts_bin):
        raw_signed8 = _tts_ru_tts_exe(text, paths)
        _write_ru_tts_raw_to_wav(raw_signed8, out_wav)
        return

    if _exists_file(paths.ru_tts_dll):
        raw_signed8 = _tts_ru_tts_dll(text, paths)
        _write_ru_tts_raw_to_wav(raw_signed8, out_wav)
        return

    raise FileNotFoundError(f"ru_tts binary not found: {paths.ru_tts_bin}; ru_tts dll not found: {paths.ru_tts_dll}")


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
        f"root={paths.tts_root}; sam={paths.sam_js}; ru_tts={paths.ru_tts_bin}; "
        f"ru_tts_dll={paths.ru_tts_dll}; rulex_dll={paths.rulex_dll}; error={last_error}"
    )
