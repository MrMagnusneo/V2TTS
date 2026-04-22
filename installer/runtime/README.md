# Runtime assets for standalone Windows install

This folder is packaged into the app and installer so target machines can run without separately installing Node.js/ru_tts.

Required files:

- `runtime/node/node.exe`
- `runtime/tts/sam/dist/samjs.min.js`
- one of:
  - `runtime/tts/ru_tts/bin/ru_tts.exe`
  - `runtime/tts/ru_tts/bin/ru_tts.dll` (+ `runtime/tts/ru_tts/bin/rulex.dll`)

Use:

```powershell
cd installer
.\prepare_runtime.ps1
```

Or run the full build (recommended), it calls preparation automatically:

```powershell
.\build_windows.ps1
```
