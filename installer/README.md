# Packaging

V2TTS uses one cross-platform Python build script.

## Build EXE/Binary

Run this on the target OS. PyInstaller does not cross-compile.

```bash
python installer/build.py
```

Results:

- Windows: `dist/V2TTS.exe`
- Linux: `dist/V2TTS`

The script installs Python build dependencies, builds the native `ru_tts` backend, and runs PyInstaller.

## Windows Installer

Install Inno Setup 6, then run on Windows:

```bash
python installer/build.py --installer
```

Result:

- `dist-installer/V2TTS-Setup.exe`

## Useful Flags

- `--skip-install` keeps the current Python environment unchanged.
- `--skip-tts-native` skips rebuilding the native `ru_tts` backend.
- `--no-clean` skips PyInstaller cache cleanup.

## Requirements

- Python 3.12+
- C compiler available as `gcc` for building the `ru_tts` native backend
- Linux runtime: PortAudio system package for `sounddevice`
- Windows installer: Inno Setup 6
