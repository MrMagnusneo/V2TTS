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


def _default_gcc_candidates(platform: str) -> list[Path]:
    if platform != "win32":
        return []

    roots = [
        Path(os.environ.get("MSYS2_ROOT", r"C:\msys64")),
        Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "chocolatey",
        Path.home() / "scoop" / "apps" / "msys2" / "current",
    ]
    return [
        roots[0] / "ucrt64" / "bin" / "gcc.exe",
        roots[0] / "mingw64" / "bin" / "gcc.exe",
        roots[1] / "bin" / "gcc.exe",
        roots[2] / "ucrt64" / "bin" / "gcc.exe",
        roots[2] / "mingw64" / "bin" / "gcc.exe",
    ]


def find_gcc(
    path_env: str | None = None,
    candidates: list[Path] | None = None,
    platform: str = sys.platform,
) -> Path | None:
    compiler = shutil.which("gcc", path=path_env)
    if compiler:
        return Path(compiler)

    search_candidates = (
        _default_gcc_candidates(platform) if candidates is None else candidates
    )
    return next((candidate for candidate in search_candidates if candidate.is_file()), None)


def ensure_gcc_available(
    path_env: str | None = None,
    candidates: list[Path] | None = None,
    platform: str = sys.platform,
) -> Path:
    compiler = find_gcc(
        path_env=path_env,
        candidates=candidates,
        platform=platform,
    )
    if compiler is None:
        if platform == "win32":
            raise RuntimeError(
                "MinGW-w64 GCC is required to build the ru_tts native backend.\n"
                "Install MSYS2 and GCC in PowerShell:\n"
                "  winget install --id MSYS2.MSYS2 -e\n"
                '  C:\\msys64\\usr\\bin\\bash.exe -lc "pacman -S --needed '
                '--noconfirm mingw-w64-ucrt-x86_64-gcc"\n'
                "Then rerun: python installer/build.py"
            )
        raise RuntimeError(
            "GCC is required to build the ru_tts native backend. "
            "Install gcc, then rerun: python installer/build.py"
        )

    compiler_dir = str(compiler.parent)
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    if compiler_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([compiler_dir, *path_parts])
    return compiler


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

    compiler = ensure_gcc_available()
    print(f"Using GCC: {compiler}")
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
