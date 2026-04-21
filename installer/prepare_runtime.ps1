$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")
$Runtime = Join-Path $InstallerDir "runtime"

$RuntimeSamDir = Join-Path $Runtime "tts\\sam\\dist"
$RuntimeRuDir = Join-Path $Runtime "tts\\ru_tts\\bin"
$RuntimeNodeDir = Join-Path $Runtime "node"

New-Item -ItemType Directory -Force -Path $RuntimeSamDir | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeRuDir | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeNodeDir | Out-Null

# 0) Ensure TTS sources are present (download automatically when missing).
$FetchSourcesScript = Join-Path $InstallerDir "fetch_tts_sources.ps1"
if (-not (Test-Path $FetchSourcesScript)) {
  throw "fetch_tts_sources.ps1 not found at $FetchSourcesScript"
}
& $FetchSourcesScript
if ($LASTEXITCODE -ne 0) { throw "fetch_tts_sources.ps1 failed with exit code $LASTEXITCODE" }

# 1) SAM runtime JS
$SamSrc = Join-Path $Root "tts\\sam\\dist\\samjs.min.js"
if (Test-Path $SamSrc) {
  Copy-Item $SamSrc (Join-Path $RuntimeSamDir "samjs.min.js") -Force
  Write-Host "SAM asset copied"
} else {
  throw "SAM JS not found at $SamSrc"
}

# 2) ru_tts binary (auto-build from downloaded sources when missing)
function Find-RuTtsBinary {
  param([Parameter(Mandatory=$true)][string]$RootDir)

  $RuCandidates = @(
    (Join-Path $RootDir "src\\.libs\\ru_tts.exe"),
    (Join-Path $RootDir "src\\ru_tts.exe"),
    (Join-Path $RootDir "bin\\ru_tts.exe"),
    (Join-Path $RootDir "ru_tts.exe"),
    (Join-Path $RootDir "src\\.libs\\ru_tts"),
    (Join-Path $RootDir "src\\ru_tts"),
    (Join-Path $RootDir "bin\\ru_tts"),
    (Join-Path $RootDir "ru_tts")
  )

  foreach ($candidate in $RuCandidates) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  return $null
}

function Build-RuTts {
  param(
    [Parameter(Mandatory=$true)][string]$RuRoot,
    [Parameter(Mandatory=$true)][string]$OutExe
  )

  $srcDir = Join-Path $RuRoot "src"
  if (-not (Test-Path $srcDir)) {
    throw "ru_tts src directory not found: $srcDir"
  }

  $compiler = $null
  $compilerCandidates = @("gcc.exe", "x86_64-w64-mingw32-gcc.exe", "clang.exe", "cc.exe")
  foreach ($candidate in $compilerCandidates) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
      $compiler = $cmd.Source
      break
    }
  }
  if (-not $compiler) {
    throw "C compiler not found. Install MinGW-w64 (gcc) or set V2TTS_RU_TTS_EXE_URL to a prebuilt ru_tts.exe."
  }

  $configHeader = Join-Path $srcDir "config.h"
  @'
#ifndef V2TTS_CONFIG_H
#define V2TTS_CONFIG_H
#define PACKAGE_NAME "ru_tts"
#define PACKAGE_VERSION "6.2.3"
#define VERSION "6.2.3"
#define WITHOUT_DICTIONARY 1
#endif
'@ | Set-Content -Path $configHeader -Encoding ASCII

  $requiredSources = @(
    "ru_tts.c",
    "synth.c",
    "sink.c",
    "transcription.c",
    "text2speech.c",
    "utterance.c",
    "time_planner.c",
    "speechrate_control.c",
    "intonator.c",
    "soundproducer.c",
    "numerics.c",
    "male.c",
    "female.c"
  )

  $sourceArgs = @()
  foreach ($name in $requiredSources) {
    $full = Join-Path $srcDir $name
    if (-not (Test-Path $full)) {
      throw "ru_tts source file missing: $full"
    }
    $sourceArgs += $full
  }

  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutExe) | Out-Null

  $attempts = @(
    @("-static", "-s"),
    @("-s")
  )
  $built = $false
  foreach ($extra in $attempts) {
    try {
      & $compiler `
        "-O2" "-Wall" "-Wno-unused-result" `
        "-D_CRT_SECURE_NO_WARNINGS" "-DWITHOUT_DICTIONARY" `
        "-I$srcDir" `
        @extra `
        "-o" $OutExe `
        @sourceArgs `
        "-lm"
      if ($LASTEXITCODE -eq 0 -and (Test-Path $OutExe)) {
        $built = $true
        break
      }
    } catch {
      # Try next attempt.
    }
  }

  if (-not $built) {
    throw "Failed to build ru_tts.exe from sources using compiler: $compiler"
  }
}

function Try-DownloadRuTtsExe {
  param([Parameter(Mandatory=$true)][string]$OutExe)

  $url = $env:V2TTS_RU_TTS_EXE_URL
  if ([string]::IsNullOrWhiteSpace($url)) {
    return $false
  }

  Write-Host "Downloading prebuilt ru_tts.exe from V2TTS_RU_TTS_EXE_URL..."
  Invoke-WebRequest -Uri $url -OutFile $OutExe
  return (Test-Path $OutExe)
}

$RuRoot = Join-Path $Root "tts\\ru_tts"
$RuRuntimeExe = Join-Path $RuntimeRuDir "ru_tts.exe"
$RuFound = Find-RuTtsBinary -RootDir $RuRoot

if ($RuFound) {
  Copy-Item $RuFound $RuRuntimeExe -Force
  Write-Host "ru_tts binary copied from $RuFound"
}
else {
  $downloaded = Try-DownloadRuTtsExe -OutExe $RuRuntimeExe
  if (-not $downloaded) {
    Write-Host "ru_tts.exe not found, building from source..."
    Build-RuTts -RuRoot $RuRoot -OutExe $RuRuntimeExe
    Write-Host "ru_tts built from source"
  } else {
    Write-Host "ru_tts.exe downloaded from custom URL"
  }

  if (-not (Test-Path $RuRuntimeExe)) {
    throw "ru_tts.exe is missing after build/download stage"
  }

  $RuCacheDir = Join-Path $RuRoot "bin"
  New-Item -ItemType Directory -Force -Path $RuCacheDir | Out-Null
  Copy-Item $RuRuntimeExe (Join-Path $RuCacheDir "ru_tts.exe") -Force
}

# 3) Node.js binary (for SAM)
$NodeCmd = Get-Command node.exe -ErrorAction SilentlyContinue
if ($NodeCmd -and (Test-Path $NodeCmd.Source)) {
  Copy-Item $NodeCmd.Source (Join-Path $RuntimeNodeDir "node.exe") -Force
  Write-Host "node.exe copied from $($NodeCmd.Source)"
} else {
  throw "node.exe not found in PATH. Install Node.js or place node.exe into $RuntimeNodeDir manually."
}

Write-Host "Runtime folder prepared: $Runtime"
