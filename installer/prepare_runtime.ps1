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

# 1) SAM runtime JS
$SamSrc = Join-Path $Root "tts\\sam\\dist\\samjs.min.js"
if (Test-Path $SamSrc) {
  Copy-Item $SamSrc (Join-Path $RuntimeSamDir "samjs.min.js") -Force
  Write-Host "SAM asset copied"
} else {
  throw "SAM JS not found at $SamSrc"
}

# 2) ru_tts binary (must exist as prebuilt exe)
$RuCandidates = @(
  (Join-Path $Root "tts\\ru_tts\\src\\.libs\\ru_tts.exe"),
  (Join-Path $Root "tts\\ru_tts\\src\\ru_tts.exe"),
  (Join-Path $Root "tts\\ru_tts\\ru_tts.exe")
)
$RuFound = $null
foreach ($c in $RuCandidates) {
  if (Test-Path $c) {
    $RuFound = $c
    break
  }
}

if ($RuFound) {
  Copy-Item $RuFound (Join-Path $RuntimeRuDir "ru_tts.exe") -Force
  Write-Host "ru_tts binary copied from $RuFound"
} else {
  throw "ru_tts.exe not found. Build/provide ru_tts.exe and place it in one of:`n$($RuCandidates -join "`n")"
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
