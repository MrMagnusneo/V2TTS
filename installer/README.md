# Windows packaging

## 1) Build onefile EXE

From PowerShell:

```powershell
cd <project_root>\installer
.\build_windows.ps1
```

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
  - `pyttsx3` modules.
- Runtime TTS assets are also copied to `%LOCALAPPDATA%\V2TTS\tts` when possible.
