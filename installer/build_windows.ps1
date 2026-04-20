$ErrorActionPreference = "Stop"

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")

Set-Location $Root

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
  python -m venv .venv
}

$Py = ".venv\\Scripts\\python.exe"
$Pip = ".venv\\Scripts\\pip.exe"
$PyInstaller = ".venv\\Scripts\\pyinstaller.exe"

& $Pip install --upgrade pip
& $Pip install -r requirements.txt pyinstaller

# Build onefile EXE from deterministic spec.
& $PyInstaller --clean --noconfirm installer\\V2TTS.spec

Write-Host "Build complete: $(Resolve-Path .\\dist\\V2TTS.exe)"
