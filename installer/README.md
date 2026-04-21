# Windows packaging

## 1) Build onefile EXE

From PowerShell:

```powershell
cd <project_root>\installer
.\build_windows.ps1
```

`build_windows.ps1` now runs full prep automatically:
- downloads `sam` and `ru_tts` sources if missing;
- prepares runtime assets;
- tries to build `ru_tts.exe` from source when binary is missing.

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
- If `ru_tts` build toolchain is not installed, set `V2TTS_RU_TTS_EXE_URL` to a direct download URL of prebuilt `ru_tts.exe`.
