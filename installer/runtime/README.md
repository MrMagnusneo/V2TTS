# Runtime assets for standalone Windows install

This folder is packaged into the app and installer so target machines can run without separately installing Node.js/ru_tts.

Required files:

- `runtime/node/node.exe`
- `runtime/tts/sam/dist/samjs.min.js`
- `runtime/tts/ru_tts/bin/ru_tts.exe`

Use:

```powershell
cd installer
.\prepare_runtime.ps1
```

Then build with:

```powershell
.\build_windows.ps1
```
