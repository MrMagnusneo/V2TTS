$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")

Set-Location $Root

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
  python -m venv .venv
}

$Py = ".venv\\Scripts\\python.exe"

if (Test-Path ".\\dist\\V2TTS.exe") {
  Remove-Item ".\\dist\\V2TTS.exe" -Force
}

& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }

& $Py -m pip install -r requirements.txt pyinstaller
if ($LASTEXITCODE -ne 0) { throw "dependency install failed with exit code $LASTEXITCODE" }

# Prepare runtime files (auto-download TTS sources, then stage node.exe/samjs/ru_tts.exe).
& (Join-Path $InstallerDir "prepare_runtime.ps1")
if ($LASTEXITCODE -ne 0) { throw "runtime preparation failed with exit code $LASTEXITCODE" }

# Build onefile EXE from deterministic spec.
& $Py -m PyInstaller --clean --noconfirm installer\\V2TTS.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE" }

if (-not (Test-Path ".\\dist\\V2TTS.exe")) {
  throw "Build finished without dist\\V2TTS.exe"
}

Write-Host "Build complete: $(Resolve-Path .\\dist\\V2TTS.exe)"
