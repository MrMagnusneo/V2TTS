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

function Invoke-DownloadWithRetry {
  param(
    [Parameter(Mandatory=$true)][string]$Uri,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [int]$MaxAttempts = 3
  )

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Invoke-WebRequest -Uri $Uri -OutFile $OutFile
      return
    }
    catch {
      if ($attempt -ge $MaxAttempts) {
        throw
      }
      Start-Sleep -Seconds ([Math]::Min(12, 2 * $attempt))
    }
  }
}

function Test-PortableExe {
  param([Parameter(Mandatory=$true)][string]$Path)
  if (-not (Test-Path $Path)) {
    return $false
  }

  try {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    return ($bytes.Length -ge 2 -and $bytes[0] -eq 0x4D -and $bytes[1] -eq 0x5A)
  } catch {
    return $false
  }
}

function Select-PreferredDllCandidate {
  param([Parameter(Mandatory=$true)][array]$Candidates)
  if ($Candidates.Count -eq 0) {
    return $null
  }

  return (
    $Candidates |
      Sort-Object `
        @{ Expression = { if ($_.FullName -match "[\\/](x64|amd64)[\\/]") { 0 } else { 1 } } }, `
        @{ Expression = { $_.FullName } } |
      Select-Object -First 1
  ).FullName
}

function Find-RuTtsArtifacts {
  param([Parameter(Mandatory=$true)][string]$RootDir)

  $exeCandidates = @(
    (Join-Path $RootDir "bin\\ru_tts.exe"),
    (Join-Path $RootDir "src\\.libs\\ru_tts.exe"),
    (Join-Path $RootDir "src\\ru_tts.exe"),
    (Join-Path $RootDir "ru_tts.exe"),
    (Join-Path $RootDir "bin\\ru_tts"),
    (Join-Path $RootDir "src\\.libs\\ru_tts"),
    (Join-Path $RootDir "src\\ru_tts"),
    (Join-Path $RootDir "ru_tts")
  )
  $dllCandidates = @(
    (Join-Path $RootDir "bin\\ru_tts.dll"),
    (Join-Path $RootDir "lib\\x64\\ru_tts.dll"),
    (Join-Path $RootDir "lib\\ru_tts.dll"),
    (Join-Path $RootDir "synthDrivers\\ru_tts\\lib\\x64\\ru_tts.dll"),
    (Join-Path $RootDir "ru_tts.dll")
  )
  $rulexCandidates = @(
    (Join-Path $RootDir "bin\\rulex.dll"),
    (Join-Path $RootDir "lib\\x64\\rulex.dll"),
    (Join-Path $RootDir "lib\\rulex.dll"),
    (Join-Path $RootDir "synthDrivers\\ru_tts\\lib\\x64\\rulex.dll"),
    (Join-Path $RootDir "rulex.dll")
  )

  $exe = $null
  foreach ($candidate in $exeCandidates) {
    if (Test-Path $candidate) {
      $exe = $candidate
      break
    }
  }
  if (-not $exe) {
    $exeFound = Get-ChildItem -Path $RootDir -Recurse -File -Filter "ru_tts.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($exeFound) {
      $exe = $exeFound.FullName
    }
  }

  $dll = $null
  foreach ($candidate in $dllCandidates) {
    if (Test-Path $candidate) {
      $dll = $candidate
      break
    }
  }
  if (-not $dll) {
    $dllMatches = @(Get-ChildItem -Path $RootDir -Recurse -File -Filter "ru_tts.dll" -ErrorAction SilentlyContinue)
    if ($dllMatches.Count -gt 0) {
      $dll = Select-PreferredDllCandidate -Candidates $dllMatches
    }
  }

  $rulex = $null
  foreach ($candidate in $rulexCandidates) {
    if (Test-Path $candidate) {
      $rulex = $candidate
      break
    }
  }
  if (-not $rulex) {
    $rulexMatches = @(Get-ChildItem -Path $RootDir -Recurse -File -Filter "rulex.dll" -ErrorAction SilentlyContinue)
    if ($rulexMatches.Count -gt 0) {
      $rulex = Select-PreferredDllCandidate -Candidates $rulexMatches
    }
  }

  return [PSCustomObject]@{
    Exe = $exe
    Dll = $dll
    RulexDll = $rulex
  }
}

function Copy-RuTtsArtifacts {
  param(
    [Parameter(Mandatory=$true)]$Artifacts,
    [Parameter(Mandatory=$true)][string]$OutDir
  )

  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  $copied = @()

  if ($Artifacts.Exe -and (Test-Path $Artifacts.Exe)) {
    Copy-Item $Artifacts.Exe (Join-Path $OutDir "ru_tts.exe") -Force
    $copied += "ru_tts.exe"
  }
  if ($Artifacts.Dll -and (Test-Path $Artifacts.Dll)) {
    Copy-Item $Artifacts.Dll (Join-Path $OutDir "ru_tts.dll") -Force
    $copied += "ru_tts.dll"
  }
  if ($Artifacts.RulexDll -and (Test-Path $Artifacts.RulexDll)) {
    Copy-Item $Artifacts.RulexDll (Join-Path $OutDir "rulex.dll") -Force
    $copied += "rulex.dll"
  }

  if ($copied.Count -gt 0) {
    Write-Host "ru_tts artifacts staged: $($copied -join ', ')"
    return $true
  }
  return $false
}

function Extract-RuTtsArtifactsFromArchive {
  param(
    [Parameter(Mandatory=$true)][string]$ArchivePath,
    [Parameter(Mandatory=$true)][string]$OutDir
  )

  $tmpZip = Join-Path $env:TEMP "v2tts-rutts-$([guid]::NewGuid().ToString()).zip"
  $tmpExtract = Join-Path $env:TEMP "v2tts-rutts-$([guid]::NewGuid().ToString())"

  try {
    Copy-Item $ArchivePath $tmpZip -Force
    Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force
    $artifacts = Find-RuTtsArtifacts -RootDir $tmpExtract
    if (-not $artifacts.Exe -and -not $artifacts.Dll) {
      return $false
    }
    return (Copy-RuTtsArtifacts -Artifacts $artifacts -OutDir $OutDir)
  }
  catch {
    return $false
  }
  finally {
    Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
    Remove-Item $tmpExtract -Recurse -Force -ErrorAction SilentlyContinue
  }
}

function Try-DownloadRuTtsArtifacts {
  param([Parameter(Mandatory=$true)][string]$OutDir)

  function Get-RuTtsDefaultUrls {
    $result = @()
    try {
      $catalog = Invoke-WebRequest -Uri "https://nvda-addons.ru/page.php?id=ru_tts"
      $pattern = "https://nvda\.ru/uploads/addons/RuTTS-V\.[0-9.]+\.nvda-addon"
      $matches = [System.Text.RegularExpressions.Regex]::Matches($catalog.Content, $pattern)
      foreach ($m in $matches) {
        $url = $m.Value
        if (-not $result.Contains($url)) {
          $result += $url
        }
      }
    } catch {
      # Fall back to pinned URLs below.
    }

    if ($result.Count -eq 0) {
      $result += @(
        "https://nvda.ru/uploads/addons/RuTTS-V.2025.11.12.nvda-addon",
        "https://nvda.ru/uploads/addons/RuTTS-V.2025.10.10.nvda-addon",
        "https://nvda.ru/uploads/addons/RuTTS-V.2025.06.21.nvda-addon"
      )
    }

    return $result
  }

  $urls = @()
  if (-not [string]::IsNullOrWhiteSpace($env:V2TTS_RU_TTS_EXE_URL)) {
    $urls += $env:V2TTS_RU_TTS_EXE_URL
  }
  if (-not [string]::IsNullOrWhiteSpace($env:V2TTS_RU_TTS_PACKAGE_URL)) {
    $urls += $env:V2TTS_RU_TTS_PACKAGE_URL
  }
  if ($urls.Count -eq 0) {
    $urls += Get-RuTtsDefaultUrls
  }

  foreach ($url in $urls) {
    $tmpFile = Join-Path $env:TEMP "v2tts-rutts-$([guid]::NewGuid().ToString()).bin"
    try {
      Write-Host "Downloading ru_tts package: $url"
      Invoke-DownloadWithRetry -Uri $url -OutFile $tmpFile

      if (Test-PortableExe -Path $tmpFile) {
        Copy-Item $tmpFile (Join-Path $OutDir "ru_tts.exe") -Force
        Write-Host "ru_tts.exe downloaded"
        return $true
      }

      if (Extract-RuTtsArtifactsFromArchive -ArchivePath $tmpFile -OutDir $OutDir) {
        Write-Host "ru_tts artifacts extracted from archive"
        return $true
      }

      Write-Warning "Downloaded ru_tts package is not a supported EXE/ZIP: $url"
    }
    catch {
      Write-Warning "ru_tts package download/extract failed for ${url}: $($_.Exception.Message)"
    }
    finally {
      Remove-Item $tmpFile -Force -ErrorAction SilentlyContinue
    }
  }

  return $false
}

# 0) Ensure TTS sources are present (download automatically when missing).
$FetchSourcesScript = Join-Path $InstallerDir "fetch_tts_sources.ps1"
if (-not (Test-Path $FetchSourcesScript)) {
  throw "fetch_tts_sources.ps1 not found at $FetchSourcesScript"
}
& $FetchSourcesScript
if (-not $?) { throw "fetch_tts_sources.ps1 failed" }

# 1) SAM runtime JS
$SamSrc = Join-Path $Root "tts\\sam\\dist\\samjs.min.js"
if (Test-Path $SamSrc) {
  Copy-Item $SamSrc (Join-Path $RuntimeSamDir "samjs.min.js") -Force
  Write-Host "SAM asset copied"
} else {
  throw "SAM JS not found at $SamSrc"
}

# 2) ru_tts runtime artifacts (prefer exe, fallback to dll/rulex from prebuilt package).

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
  $compilerKind = $null
  $compilerCandidates = @("gcc.exe", "x86_64-w64-mingw32-gcc.exe", "clang.exe", "zig.exe", "cc.exe")
  foreach ($candidate in $compilerCandidates) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
      $compiler = $cmd.Source
      $compilerKind = $candidate
      break
    }
  }
  if (-not $compiler) {
    throw "C compiler not found and no prebuilt ru_tts package could be downloaded. Install MinGW-w64 (gcc), or set V2TTS_RU_TTS_EXE_URL / V2TTS_RU_TTS_PACKAGE_URL."
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
      if ($compilerKind -eq "zig.exe") {
        & $compiler "cc" `
          "-O2" "-Wall" "-Wno-unused-result" `
          "-D_CRT_SECURE_NO_WARNINGS" "-DWITHOUT_DICTIONARY" `
          "-I$srcDir" `
          @extra `
          "-o" $OutExe `
          @sourceArgs `
          "-lm"
      } else {
        & $compiler `
          "-O2" "-Wall" "-Wno-unused-result" `
          "-D_CRT_SECURE_NO_WARNINGS" "-DWITHOUT_DICTIONARY" `
          "-I$srcDir" `
          @extra `
          "-o" $OutExe `
          @sourceArgs `
          "-lm"
      }
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

$RuRoot = Join-Path $Root "tts\\ru_tts"
$ruArtifacts = Find-RuTtsArtifacts -RootDir $RuRoot
$ruReady = Copy-RuTtsArtifacts -Artifacts $ruArtifacts -OutDir $RuntimeRuDir

if (-not $ruReady) {
  $downloaded = Try-DownloadRuTtsArtifacts -OutDir $RuntimeRuDir
  if (-not $downloaded) {
    $RuRuntimeExe = Join-Path $RuntimeRuDir "ru_tts.exe"
    Write-Host "ru_tts prebuilt package not available, building from source..."
    Build-RuTts -RuRoot $RuRoot -OutExe $RuRuntimeExe
    Write-Host "ru_tts built from source"
  }
}

$RuRuntimeExe = Join-Path $RuntimeRuDir "ru_tts.exe"
$RuRuntimeDll = Join-Path $RuntimeRuDir "ru_tts.dll"
$RuRuntimeRulex = Join-Path $RuntimeRuDir "rulex.dll"
if (-not (Test-Path $RuRuntimeExe) -and -not (Test-Path $RuRuntimeDll)) {
  throw "ru_tts runtime is missing after staging. Expected ru_tts.exe or ru_tts.dll in $RuntimeRuDir"
}
if ((Test-Path $RuRuntimeDll) -and (-not (Test-Path $RuRuntimeRulex))) {
  Write-Warning "ru_tts.dll is present, but rulex.dll is missing. If synthesis fails, provide rulex.dll via V2TTS_RU_TTS_PACKAGE_URL."
}

$RuCacheDir = Join-Path $RuRoot "bin"
New-Item -ItemType Directory -Force -Path $RuCacheDir | Out-Null
foreach ($name in @("ru_tts.exe", "ru_tts.dll", "rulex.dll")) {
  $src = Join-Path $RuntimeRuDir $name
  if (Test-Path $src) {
    Copy-Item $src (Join-Path $RuCacheDir $name) -Force
  }
}

function Try-DownloadNodeExeDirect {
  param([Parameter(Mandatory=$true)][string]$OutExe)

  $url = $env:V2TTS_NODE_EXE_URL
  if ([string]::IsNullOrWhiteSpace($url)) {
    return $false
  }

  $tmp = Join-Path $env:TEMP "v2tts-node-$([guid]::NewGuid().ToString()).exe"
  try {
    Write-Host "Downloading node.exe from V2TTS_NODE_EXE_URL..."
    Invoke-DownloadWithRetry -Uri $url -OutFile $tmp
    if (-not (Test-PortableExe -Path $tmp)) {
      throw "Downloaded node file is not a valid Windows executable"
    }
    Copy-Item $tmp $OutExe -Force
    return $true
  }
  finally {
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
  }
}

function Try-DownloadNodeFromZip {
  param([Parameter(Mandatory=$true)][string]$OutExe)

  $urls = @()
  if (-not [string]::IsNullOrWhiteSpace($env:V2TTS_NODE_ZIP_URL)) {
    $urls += $env:V2TTS_NODE_ZIP_URL
  }

  $versions = @()
  if (-not [string]::IsNullOrWhiteSpace($env:V2TTS_NODE_VERSION)) {
    $versions += ($env:V2TTS_NODE_VERSION -split "[,; ]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  }
  if ($versions.Count -eq 0) {
    $versions = @("22.11.0", "20.18.0")
  }
  foreach ($ver in $versions) {
    $urls += "https://nodejs.org/dist/v$ver/node-v$ver-win-x64.zip"
  }

  foreach ($url in $urls) {
    $tmpZip = Join-Path $env:TEMP "v2tts-node-$([guid]::NewGuid().ToString()).zip"
    $tmpExtract = Join-Path $env:TEMP "v2tts-node-$([guid]::NewGuid().ToString())"
    try {
      Write-Host "Downloading Node.js package: $url"
      Invoke-DownloadWithRetry -Uri $url -OutFile $tmpZip
      Expand-Archive -Path $tmpZip -DestinationPath $tmpExtract -Force
      $nodeFile = Get-ChildItem -Path $tmpExtract -Recurse -File -Filter "node.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
      if (-not $nodeFile) {
        throw "node.exe not found in archive"
      }
      Copy-Item $nodeFile.FullName $OutExe -Force
      return $true
    }
    catch {
      Write-Warning "Node.js download/extract failed for ${url}: $($_.Exception.Message)"
    }
    finally {
      Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
      Remove-Item $tmpExtract -Recurse -Force -ErrorAction SilentlyContinue
    }
  }

  return $false
}

# 3) Node.js binary (for SAM)
$NodeOut = Join-Path $RuntimeNodeDir "node.exe"
$NodeCmd = Get-Command node.exe -ErrorAction SilentlyContinue
if ($NodeCmd -and (Test-Path $NodeCmd.Source)) {
  Copy-Item $NodeCmd.Source $NodeOut -Force
  Write-Host "node.exe copied from $($NodeCmd.Source)"
} elseif (Test-Path $NodeOut) {
  Write-Host "node.exe already present in runtime"
} else {
  $nodeReady = Try-DownloadNodeExeDirect -OutExe $NodeOut
  if (-not $nodeReady) {
    $nodeReady = Try-DownloadNodeFromZip -OutExe $NodeOut
  }
  if (-not $nodeReady) {
    throw "node.exe not found and automatic download failed. Install Node.js or set V2TTS_NODE_EXE_URL / V2TTS_NODE_ZIP_URL."
  }
}

Write-Host "Runtime folder prepared: $Runtime"
