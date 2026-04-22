$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

$InstallerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $InstallerDir "..")

Set-Location $Root

function New-VenvIfMissing {
  if (Test-Path ".venv\\Scripts\\python.exe") {
    return
  }

  $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
  $python312Cmd = Get-Command python3.12.exe -ErrorAction SilentlyContinue
  $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue

  $attempts = @()
  if ($pyLauncher) {
    $attempts += [PSCustomObject]@{ Exe = $pyLauncher.Source; Args = @("-3.12", "-m", "venv", ".venv") }
  }
  if ($python312Cmd) {
    $attempts += [PSCustomObject]@{ Exe = $python312Cmd.Source; Args = @("-m", "venv", ".venv") }
  }
  if ($pythonCmd) {
    $attempts += [PSCustomObject]@{ Exe = $pythonCmd.Source; Args = @("-m", "venv", ".venv") }
  }
  if ($pyLauncher) {
    $attempts += [PSCustomObject]@{ Exe = $pyLauncher.Source; Args = @("-3", "-m", "venv", ".venv") }
  }

  foreach ($attempt in $attempts) {
    try {
      & $attempt.Exe @($attempt.Args)
      if ($LASTEXITCODE -eq 0 -and (Test-Path ".venv\\Scripts\\python.exe")) {
        return
      }
    } catch {
      # Try next Python candidate.
    }
  }

  throw "Python 3.12+ not found in PATH. Install Python and retry."
}

function Get-PythonVersionInfo {
  param([Parameter(Mandatory=$true)][string]$PythonExe)

  $versionText = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to detect Python version from $PythonExe"
  }

  $parts = $versionText.Trim().Split(".")
  if ($parts.Count -lt 2) {
    throw "Unexpected Python version format: $versionText"
  }

  $major = [int]$parts[0]
  $minor = [int]$parts[1]

  return [PSCustomObject]@{
    Major = $major
    Minor = $minor
    Text = "$major.$minor"
  }
}

function Ensure-VenvWithPython312 {
  New-VenvIfMissing

  $venvPy = ".venv\\Scripts\\python.exe"
  $ver = Get-PythonVersionInfo -PythonExe $venvPy
  if ($ver.Major -gt 3 -or ($ver.Major -eq 3 -and $ver.Minor -ge 12)) {
    return
  }

  Write-Warning "Current .venv uses Python $($ver.Text). Recreating virtual environment with Python 3.12+..."
  Remove-Item ".venv" -Recurse -Force
  New-VenvIfMissing

  $ver = Get-PythonVersionInfo -PythonExe $venvPy
  if ($ver.Major -lt 3 -or ($ver.Major -eq 3 -and $ver.Minor -lt 12)) {
    throw "Python 3.12+ is required, but the available interpreter is $($ver.Text)."
  }
}

function Remove-InvalidProxyEnv {
  $proxyVars = @("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
  foreach ($name in $proxyVars) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    $uri = $null
    $ok = [Uri]::TryCreate($value, [UriKind]::Absolute, [ref]$uri)
    if (-not $ok -or [string]::IsNullOrWhiteSpace($uri.Host)) {
      Write-Warning "Ignoring invalid proxy setting in $name"
      Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
  }
}

Ensure-VenvWithPython312

$Py = ".venv\\Scripts\\python.exe"
Remove-InvalidProxyEnv

if (Test-Path ".\\dist\\V2TTS.exe") {
  Remove-Item ".\\dist\\V2TTS.exe" -Force
}

try {
  & $Py -m pip --disable-pip-version-check install --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }
}
catch {
  Write-Warning "pip upgrade failed, continuing with current pip: $($_.Exception.Message)"
}

$depsInstalled = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
  & $Py -m pip --disable-pip-version-check install -r requirements.txt pyinstaller
  if ($LASTEXITCODE -eq 0) {
    $depsInstalled = $true
    break
  }

  if ($attempt -lt 3) {
    Write-Warning "dependency install failed (attempt $attempt/3), retrying..."
    Start-Sleep -Seconds (2 * $attempt)
  }
}
if (-not $depsInstalled) {
  throw "dependency install failed after retries. Check internet/proxy settings and Python 3.12+ compatibility."
}

# Prepare runtime files (auto-download TTS sources, then stage node.exe/samjs/ru_tts runtime).
& (Join-Path $InstallerDir "prepare_runtime.ps1")
if (-not $?) { throw "runtime preparation failed" }

# Build onefile EXE from deterministic spec.
& $Py -m PyInstaller --clean --noconfirm installer\\V2TTS.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed with exit code $LASTEXITCODE" }

if (-not (Test-Path ".\\dist\\V2TTS.exe")) {
  throw "Build finished without dist\\V2TTS.exe"
}

Write-Host "Build complete: $(Resolve-Path .\\dist\\V2TTS.exe)"
