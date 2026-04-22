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
- for `ru_tts`:
  - uses local `ru_tts.exe` / `ru_tts.dll` when available;
  - otherwise downloads prebuilt package and extracts runtime binaries;
  - falls back to source build only if no prebuilt package is available;
- for Node.js:
  - uses local `node.exe` when available;
  - otherwise downloads official Node.js ZIP and extracts `node.exe`.

Result:
- `..\dist\V2TTS.exe`

## 2) Build installer (Inno Setup)

Run:

```powershell
cd <project_root>\installer
.\build_installer.ps1
```

If `..\dist\V2TTS.exe` is missing, `build_installer.ps1` automatically runs `build_windows.ps1` first.
If Inno Setup compiler is missing, the script tries to install it locally into `<project_root>\.tools\InnoSetup6`.
If you already have a custom installation, you can set `V2TTS_ISCC_PATH` to `ISCC.exe`.

Result:
- `..\dist-installer\V2TTS-Setup.exe`

## Notes

- The PyInstaller spec includes:
  - all `tts` assets,
  - `faster_whisper` data,
  - `onnxruntime` binaries,
  - `installer/runtime` assets (`node.exe`, `ru_tts.exe` or `ru_tts.dll`+`rulex.dll`, `samjs`).
- Target machine should not require separate Node.js installation if `installer/runtime/node/node.exe` is present.
- Runtime preparation env vars:
  - `V2TTS_RU_TTS_EXE_URL` - direct URL to `ru_tts.exe`.
  - `V2TTS_RU_TTS_PACKAGE_URL` - URL to ZIP/NVDA add-on containing `ru_tts.dll`/`rulex.dll` (or `ru_tts.exe`).
  - `V2TTS_NODE_EXE_URL` - direct URL to `node.exe`.
  - `V2TTS_NODE_ZIP_URL` - URL to ZIP with `node.exe`.
  - `V2TTS_NODE_VERSION` - preferred Node version(s), e.g. `22.11.0` or `22.11.0,20.18.0`.
