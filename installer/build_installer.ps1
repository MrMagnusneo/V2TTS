$ErrorActionPreference = "Stop"

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Iscc = "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe"

if (-not (Test-Path $Iscc)) {
  throw "Inno Setup 6 not found. Install it from https://jrsoftware.org/isdl.php"
}

& $Iscc (Join-Path $InstallerDir "V2TTS.iss")
