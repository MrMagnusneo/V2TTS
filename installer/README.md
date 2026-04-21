# Windows packaging

## 1) Build onefile EXE

From PowerShell:

```powershell
cd <project_root>\installer
.\prepare_runtime.ps1
.\build_windows.ps1
```

`prepare_runtime.ps1` is strict and fails if required runtime files are missing.

Result:
- `..\dist\V2TTS.exe`

## 2) Build installer (Inno Setup)

Install Inno Setup 6, then run:

```powershell
cd <project_root>\installer
.\build_installer.ps1
```

Result:
- `..\dist-installer\V2TTS-Setup.exe`

## Notes

- The PyInstaller spec includes:
  - all `tts` assets,
  - `faster_whisper` data,
  - `onnxruntime` binaries,
  - `installer/runtime` assets (node.exe, ru_tts.exe, samjs).
- Target machine should not require separate Node.js installation if `installer/runtime/node/node.exe` is present.
