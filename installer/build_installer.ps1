$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")
$AppExe = Join-Path $Root "dist\\V2TTS.exe"

function Resolve-IsccPath {
  $candidates = @()

  if (-not [string]::IsNullOrWhiteSpace($env:V2TTS_ISCC_PATH)) {
    $candidates += $env:V2TTS_ISCC_PATH
  }

  $candidates += @(
    (Join-Path $Root ".tools\\InnoSetup6\\ISCC.exe"),
    "${env:ProgramFiles(x86)}\\Inno Setup 6\\ISCC.exe",
    "${env:ProgramFiles}\\Inno Setup 6\\ISCC.exe",
    "$env:LOCALAPPDATA\\Programs\\Inno Setup 6\\ISCC.exe"
  )

  foreach ($path in $candidates) {
    if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path $path)) {
      return $path
    }
  }

  $isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
  if ($isccCmd -and (Test-Path $isccCmd.Source)) {
    return $isccCmd.Source
  }

  return $null
}

function Install-LocalInnoSetup {
  $targetDir = Join-Path $Root ".tools\\InnoSetup6"
  $installer = Join-Path $env:TEMP "v2tts-inno-setup-$([guid]::NewGuid().ToString()).exe"
  $url = "https://jrsoftware.org/download.php/is.exe"

  try {
    Write-Host "Inno Setup not found. Downloading and installing local compiler..."
    Invoke-WebRequest -Uri $url -OutFile $installer
    & $installer `
      "/VERYSILENT" `
      "/SUPPRESSMSGBOXES" `
      "/NORESTART" `
      "/SP-" `
      "/CURRENTUSER" `
      "/DIR=$targetDir"

    $localIscc = Join-Path $targetDir "ISCC.exe"
    if (Test-Path $localIscc) {
      return $localIscc
    }
  } catch {
    Write-Warning "Automatic Inno Setup install failed: $($_.Exception.Message)"
  } finally {
    Remove-Item $installer -Force -ErrorAction SilentlyContinue
  }

  return $null
}

if (-not (Test-Path $AppExe)) {
  Write-Host "dist\\V2TTS.exe not found. Running build_windows.ps1 first..."
  & (Join-Path $InstallerDir "build_windows.ps1")
  if (-not $?) { throw "build_windows.ps1 failed" }
}

$Iscc = Resolve-IsccPath
if (-not $Iscc) {
  $Iscc = Install-LocalInnoSetup
}

if (-not (Test-Path $Iscc)) {
  throw "Inno Setup 6 not found. Install it from https://jrsoftware.org/isdl.php or set V2TTS_ISCC_PATH to ISCC.exe."
}

& $Iscc (Join-Path $InstallerDir "V2TTS.iss")
if ($LASTEXITCODE -ne 0) { throw "Installer build failed with exit code $LASTEXITCODE" }
