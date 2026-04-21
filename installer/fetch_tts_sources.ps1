$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")
$TtsRoot = Join-Path $Root "tts"

New-Item -ItemType Directory -Force -Path $TtsRoot | Out-Null

function Download-GitHubRepo {
  param(
    [Parameter(Mandatory=$true)][string]$Owner,
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$DestDir
  )

  $tmpZip = Join-Path $env:TEMP "$Repo-$([guid]::NewGuid().ToString()).zip"
  $tmpDir = Join-Path $env:TEMP "$Repo-$([guid]::NewGuid().ToString())"

  $urls = @(
    "https://codeload.github.com/$Owner/$Repo/zip/refs/heads/master",
    "https://codeload.github.com/$Owner/$Repo/zip/refs/heads/main"
  )

  try {
    $ok = $false
    foreach ($url in $urls) {
      try {
        Invoke-WebRequest -Uri $url -OutFile $tmpZip
        $ok = $true
        break
      } catch {
        # try next URL
      }
    }

    if (-not $ok) {
      throw "Failed to download $Owner/$Repo from GitHub"
    }

    Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force
    $src = Get-ChildItem -Path $tmpDir -Directory | Select-Object -First 1
    if (-not $src) {
      throw "Archive for $Owner/$Repo does not contain source folder"
    }

    if (Test-Path $DestDir) {
      Remove-Item $DestDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Copy-Item (Join-Path $src.FullName "*") $DestDir -Recurse -Force
  }
  finally {
    Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
    Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue
  }
}

$SamDir = Join-Path $TtsRoot "sam"
$RuDir = Join-Path $TtsRoot "ru_tts"

$SamReady = Test-Path (Join-Path $SamDir "dist\\samjs.min.js")
if (-not $SamReady) {
  Write-Host "Downloading SAM sources..."
  Download-GitHubRepo -Owner "discordier" -Repo "sam" -DestDir $SamDir
} else {
  Write-Host "SAM sources already present"
}

$RuReady = Test-Path (Join-Path $RuDir "README.md")
if (-not $RuReady) {
  Write-Host "Downloading ru_tts sources..."
  Download-GitHubRepo -Owner "poretsky" -Repo "ru_tts" -DestDir $RuDir
} else {
  Write-Host "ru_tts sources already present"
}

$SamReadyAfter = Test-Path (Join-Path $SamDir "dist\\samjs.min.js")
if (-not $SamReadyAfter) {
  throw "SAM download finished, but dist\\samjs.min.js is missing in $SamDir"
}

$RuReadyAfter = Test-Path (Join-Path $RuDir "src\\ru_tts.c")
if (-not $RuReadyAfter) {
  throw "ru_tts download finished, but src\\ru_tts.c is missing in $RuDir"
}

Write-Host "TTS sources ready in $TtsRoot"
