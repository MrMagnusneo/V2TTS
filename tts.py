import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")

TTS_MODELS = ["ru_tts", "sam", "dectalk", "silero", "coqui"]


@dataclass(frozen=True)
class TTSPaths:
    tts_root: Path
    sam_python_root: Path
    ru_tts_python_root: Path
    dectalk_python_root: Path
    silero_tts_root: Path
    coqui_tts_root: Path


_RU_TTS_ENGINES: dict[Path, object] = {}
_SILERO_TTS_ENGINES: dict[Path, object] = {}
_COQUI_TTS_ENGINES: dict[Path, object] = {}


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


def _user_data_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return (base / "V2TTS").resolve()


def _root_has_python_tts(root: Path) -> bool:
    return (
        (root / "sam-python" / "sam_python").exists()
        and (root / "ru_tts-python" / "ru_tts_python").exists()
        and (root / "dectalk-python" / "dectalk_python").exists()
        and (root / "silero-tts-wrapper" / "silero_tts").exists()
        and (root / "coqui-tts-wrapper" / "coqui_tts").exists()
    )


def _candidate_roots(user_tts_root: Optional[str]) -> list[Path]:
    roots: list[Path] = []

    if user_tts_root:
        roots.append(Path(user_tts_root).expanduser().resolve())

    env_root = os.getenv("V2TTS_TTS_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())

    if _is_frozen():
        mei = _meipass_dir()
        if mei is not None:
            roots.append(mei / "tts")
        roots.append(_exe_dir() / "tts")

    roots.extend(
        [
            _source_root() / "tts",
            _user_data_root() / "tts",
            Path.cwd() / "tts",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def prepare_runtime_tts_root(tts_root: Optional[str] = None) -> Path:
    if tts_root:
        return Path(tts_root).expanduser().resolve()

    for root in _candidate_roots(None):
        if _root_has_python_tts(root):
            return root

    return (_source_root() / "tts").resolve()


def resolve_tts_paths(tts_root: Optional[str] = None) -> TTSPaths:
    root = prepare_runtime_tts_root(tts_root)
    return TTSPaths(
        tts_root=root,
        sam_python_root=root / "sam-python",
        ru_tts_python_root=root / "ru_tts-python",
        dectalk_python_root=root / "dectalk-python",
        silero_tts_root=root / "silero-tts-wrapper",
        coqui_tts_root=root / "coqui-tts-wrapper",
    )


def _add_vendor_paths(paths: TTSPaths) -> None:
    for package_root in (
        paths.sam_python_root,
        paths.ru_tts_python_root,
        paths.dectalk_python_root,
        paths.silero_tts_root,
        paths.coqui_tts_root,
    ):
        if package_root.exists():
            package_root_str = str(package_root)
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)


def choose_tts_engine(text: str, auto_select: bool = True, manual_model: str = "ru_tts") -> str:
    if not auto_select:
        return manual_model

    if CYR.search(text):
        return "ru_tts"
    if LAT.search(text):
        return "sam"
    return "sam"


def tts_sam(text: str, out_wav: Union[str, BinaryIO], paths: TTSPaths) -> None:
    _add_vendor_paths(paths)
    from sam_python.engine import SamPythonEngine

    engine = SamPythonEngine(speed=72, pitch=64, throat=128, mouth=128)
    wav_bytes = engine.synthesize_wav(text)
    if isinstance(out_wav, str):
        Path(out_wav).write_bytes(wav_bytes)
    else:
        out_wav.write(wav_bytes)


def _ru_tts_engine(paths: TTSPaths):
    cache_key = paths.ru_tts_python_root.resolve()
    if cache_key in _RU_TTS_ENGINES:
        return _RU_TTS_ENGINES[cache_key]

    _add_vendor_paths(paths)
    from ru_tts_python.engine import RuTTSPythonEngine

    # Source runs may build the native library once. Frozen apps must ship it.
    engine = RuTTSPythonEngine(backend="nvda", auto_build=not _is_frozen())
    _RU_TTS_ENGINES[cache_key] = engine
    return engine


def tts_ru_tts(text: str, out_wav: Union[str, BinaryIO], paths: TTSPaths) -> None:
    engine = _ru_tts_engine(paths)
    wav_bytes = engine.synthesize_wav(text, args=["-r", "1.0"])
    if isinstance(out_wav, str):
        Path(out_wav).write_bytes(wav_bytes)
    else:
        out_wav.write(wav_bytes)


def tts_dectalk(text: str, out_wav: Union[str, BinaryIO], paths: TTSPaths) -> None:
    _add_vendor_paths(paths)
    from dectalk_python import DectalkSynthesizer

    wav_bytes = DectalkSynthesizer().synthesize(text).to_wav_bytes()
    if isinstance(out_wav, str):
        Path(out_wav).write_bytes(wav_bytes)
    else:
        out_wav.write(wav_bytes)


def _silero_engine(paths: TTSPaths):
    cache_key = paths.silero_tts_root.resolve()
    if cache_key in _SILERO_TTS_ENGINES:
        return _SILERO_TTS_ENGINES[cache_key]

    _add_vendor_paths(paths)
    from silero_tts import SileroTTSEngine

    engine = SileroTTSEngine()
    _SILERO_TTS_ENGINES[cache_key] = engine
    return engine


def _coqui_engine(paths: TTSPaths):
    cache_key = paths.coqui_tts_root.resolve()
    if cache_key in _COQUI_TTS_ENGINES:
        return _COQUI_TTS_ENGINES[cache_key]

    _add_vendor_paths(paths)
    from coqui_tts import CoquiTTSEngine

    engine = CoquiTTSEngine(progress_bar=False)
    _COQUI_TTS_ENGINES[cache_key] = engine
    return engine


def _synthesize_path_only_engine(
    engine,
    text: str,
    out_wav: Union[str, BinaryIO],
) -> None:
    if isinstance(out_wav, str):
        engine.synthesize_to_file(text, out_wav)
        return

    file_descriptor, temporary_path = tempfile.mkstemp(suffix=".wav")
    os.close(file_descriptor)
    temporary = Path(temporary_path)
    try:
        engine.synthesize_to_file(text, temporary)
        out_wav.write(temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def tts_silero(text: str, out_wav: Union[str, BinaryIO], paths: TTSPaths) -> None:
    _synthesize_path_only_engine(_silero_engine(paths), text, out_wav)


def tts_coqui(text: str, out_wav: Union[str, BinaryIO], paths: TTSPaths) -> None:
    _synthesize_path_only_engine(_coqui_engine(paths), text, out_wav)


def synthesize_text(
    text: str,
    out_wav: Union[str, BinaryIO],
    auto_select: bool = True,
    manual_model: str = "ru_tts",
    tts_root: Optional[str] = None,
) -> str:
    paths = resolve_tts_paths(tts_root)
    requested = choose_tts_engine(text, auto_select=auto_select, manual_model=manual_model)
    chain = [requested]
    if requested == "ru_tts":
        chain.append("sam")

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
            elif engine == "dectalk":
                tts_dectalk(text, out_wav, paths)
            elif engine == "silero":
                tts_silero(text, out_wav, paths)
            elif engine == "coqui":
                tts_coqui(text, out_wav, paths)
            else:
                raise ValueError(f"Unknown TTS model: {engine}")
            return engine
        except Exception as exc:
            last_error = exc

    assert last_error is not None
    raise RuntimeError(
        f"TTS failed ({', '.join(tried)}). "
        f"root={paths.tts_root}; error={last_error}"
    )
