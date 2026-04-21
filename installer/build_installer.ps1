$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Iscc = "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe"

if (-not (Test-Path $Iscc)) {
  throw "Inno Setup 6 not found. Install it from https://jrsoftware.org/isdl.php"
}

& $Iscc (Join-Path $InstallerDir "V2TTS.iss")
if ($LASTEXITCODE -ne 0) { throw "Installer build failed with exit code $LASTEXITCODE" }
