from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


INSTALLER_DIR = Path(__file__).resolve().parent
ROOT = INSTALLER_DIR.parent
SAM_ROOT = ROOT / "tts" / "sam-python"
RU_TTS_ROOT = ROOT / "tts" / "ru_tts-python"


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_local_packages_on_path() -> None:
    for path in (SAM_ROOT, RU_TTS_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_submodules() -> None:
    required = [
        SAM_ROOT / "sam_python",
        RU_TTS_ROOT / "ru_tts_python",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Required TTS submodules are missing:\n"
            f"{missing_list}\n"
            "Run: git submodule update --init --recursive"
        )


def ensure_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "pyinstaller"])


def prepare_tts_native(skip_native: bool) -> None:
    if skip_native:
        return

    ensure_local_packages_on_path()

    from ru_tts_python.build_nvda_backend import build_nvda_backend

    built = build_nvda_backend()
    print(f"Prepared ru_tts native backend: {built}")

    from sam_python.engine import SamPythonEngine

    SamPythonEngine().synthesize_wav("test")
    print("Prepared SAM Python backend")


def build_exe(clean: bool) -> None:
    spec = INSTALLER_DIR / "V2TTS.spec"
    cmd = [sys.executable, "-m", "PyInstaller"]
    if clean:
        cmd.append("--clean")
    cmd.extend(["--noconfirm", str(spec)])
    run(cmd)

    exe_name = "V2TTS.exe" if sys.platform == "win32" else "V2TTS"
    output = ROOT / "dist" / exe_name
    if not output.exists():
        raise FileNotFoundError(f"PyInstaller finished but {output} was not created")
    print(f"Build complete: {output}")


def find_inno_compiler() -> str | None:
    iscc = shutil.which("iscc") or shutil.which("ISCC.exe")
    if iscc:
        return iscc

    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_windows_installer() -> None:
    if sys.platform != "win32":
        raise RuntimeError("The Inno Setup installer can only be built on Windows.")

    compiler = find_inno_compiler()
    if not compiler:
        raise FileNotFoundError("Inno Setup compiler was not found. Install Inno Setup 6 or add ISCC.exe to PATH.")

    run([compiler, str(INSTALLER_DIR / "V2TTS.iss")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build V2TTS for the current operating system.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install Python build dependencies.")
    parser.add_argument("--skip-tts-native", action="store_true", help="Do not rebuild the ru_tts native backend.")
    parser.add_argument("--no-clean", action="store_true", help="Do not pass --clean to PyInstaller.")
    parser.add_argument("--installer", action="store_true", help="Also build the Windows Inno Setup installer.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_submodules()
    ensure_dependencies(skip_install=args.skip_install)
    prepare_tts_native(skip_native=args.skip_tts_native)
    build_exe(clean=not args.no_clean)
    if args.installer:
        build_windows_installer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
