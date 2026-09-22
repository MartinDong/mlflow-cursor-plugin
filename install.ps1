# Create the plugin virtualenv and a local config.env (Windows / PowerShell).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$MinMinor = 10

function Find-Python {
  $candidates = @()
  try {
    $py = & py -3 -c "import sys; print(sys.executable)" 2>$null
    if ($py) { $candidates += $py.Trim() }
  } catch {}
  foreach ($name in @("python", "python3")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch "WindowsApps") {
      $candidates += $cmd.Source
    }
  }
  $candidates += @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
  )
  foreach ($cand in $candidates | Select-Object -Unique) {
    if (-not (Test-Path $cand)) { continue }
    & $cand -c "import sys; raise SystemExit(0 if sys.version_info >= (3, $MinMinor) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { return $cand }
  }
  throw "install.ps1: need Python 3.$MinMinor+ on PATH (or via the py launcher)"
}

$Python = Find-Python
$Ver = & $Python -c "import sys; print(sys.version.split()[0])"
Write-Host "install.ps1: Python: $Python ($Ver)"

$Venv = Join-Path $Root ".venv"
if (Test-Path $Venv) {
  Remove-Item -Recurse -Force $Venv
}

& $Python -m venv $Venv
$Vpy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Vpy)) {
  throw "install.ps1: venv python missing at $Vpy"
}

& $Vpy -m pip install --upgrade pip
& $Vpy -m pip install -r (Join-Path $Root "requirements.txt")

$Config = Join-Path $Root "config.env"
$Example = Join-Path $Root "config.env.example"
if (-not (Test-Path $Config)) {
  Copy-Item $Example $Config
  Write-Host "Wrote $Config — set MLFLOW_TRACKING_URI and MLFLOW_TRACKING_PASSWORD."
}

Write-Host "Plugin ready: $Root"
Write-Host "Hook runner: $Vpy"
