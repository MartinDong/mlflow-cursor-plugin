# Install this plugin into the current Windows Cursor user profile:
# - ensures .venv
# - writes ~/.cursor/hooks.json (MLflow tracing hooks)
# - merges mlflow-mcp into ~/.cursor/mcp.json from config.env
$ErrorActionPreference = "Stop"

$PluginRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CursorHome = Join-Path $env:USERPROFILE ".cursor"
$HooksDir = Join-Path $CursorHome "hooks"
$HooksFile = Join-Path $CursorHome "hooks.json"
$McpFile = Join-Path $CursorHome "mcp.json"
$ConfigFile = Join-Path $PluginRoot "config.env"
$VenvPython = Join-Path $PluginRoot ".venv\Scripts\python.exe"
$HookScript = Join-Path $PluginRoot "scripts\mlflow_cursor.py"
$RunHookJs = Join-Path $PluginRoot "scripts\run_hook.js"
$InstallLocal = Join-Path $PluginRoot "install.ps1"

function Read-EnvFile([string]$Path) {
  $map = @{}
  if (-not (Test-Path $Path)) { return $map }
  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $i = $line.IndexOf("=")
    $k = $line.Substring(0, $i).Trim()
    $v = $line.Substring($i + 1).Trim().Trim("'`"")
    $map[$k] = $v
  }
  return $map
}

Write-Host "install-system.ps1: plugin = $PluginRoot"

if (-not (Test-Path $VenvPython)) {
  Write-Host "install-system.ps1: creating venv via install.ps1..."
  & powershell -ExecutionPolicy Bypass -File $InstallLocal
}

if (-not (Test-Path $VenvPython)) {
  throw "missing $VenvPython"
}
if (-not (Test-Path $HookScript)) {
  throw "missing $HookScript"
}

$envMap = Read-EnvFile $ConfigFile
foreach ($req in @("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")) {
  if (-not $envMap.ContainsKey($req) -or [string]::IsNullOrWhiteSpace($envMap[$req])) {
    throw "config.env missing $req - copy config.env.example and fill values"
  }
}

New-Item -ItemType Directory -Force -Path $CursorHome | Out-Null
New-Item -ItemType Directory -Force -Path $HooksDir | Out-Null

# IMPORTANT (Windows): do NOT quote paths and do NOT use node under "Program Files".
# Quoted commands / .cmd go through a shell that re-encodes stdin and corrupts
# non-ASCII JSON (Chinese thought/tool payloads). Prefer space-free python.exe path.
if ($VenvPython -match '\s' -or $HookScript -match '\s') {
  throw "plugin path contains spaces ($PluginRoot); move the repo to a path without spaces so hooks can avoid the Windows shell"
}

$hookCmd = "$VenvPython $HookScript"
$Wrapper = Join-Path $HooksDir "mlflow-cursor-tracing.cmd"
$wrapperBody = @"
@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set MLFLOW_DISABLE_AGENT_HINT=1
$VenvPython $HookScript
exit /b %ERRORLEVEL%
"@
Set-Content -Encoding ASCII -Path $Wrapper -Value $wrapperBody

$hooksObj = [ordered]@{
  version = 1
  hooks   = [ordered]@{
    beforeSubmitPrompt  = @(@{ command = $hookCmd; timeout = 8 })
    afterAgentThought   = @(@{ command = $hookCmd; timeout = 8 })
    afterAgentResponse  = @(@{ command = $hookCmd; timeout = 8 })
    postToolUse         = @(@{ command = $hookCmd; timeout = 8 })
    postToolUseFailure  = @(@{ command = $hookCmd; timeout = 8 })
    subagentStart       = @(@{ command = $hookCmd; timeout = 8 })
    subagentStop        = @(@{ command = $hookCmd; timeout = 8 })
    preCompact          = @(@{ command = $hookCmd; timeout = 8 })
    stop                = @(@{ command = $hookCmd; timeout = 90 })
    sessionEnd          = @(@{ command = $hookCmd; timeout = 90 })
  }
}

# Preserve non-mlflow hooks if the user already has a hooks.json
if (Test-Path $HooksFile) {
  try {
    $existing = Get-Content $HooksFile -Raw | ConvertFrom-Json
    if ($existing.hooks) {
      foreach ($prop in $existing.hooks.PSObject.Properties) {
        $name = $prop.Name
        if ($hooksObj.hooks.Contains($name)) { continue }
        $hooksObj.hooks[$name] = $prop.Value
      }
    }
  } catch {
    Write-Host "install-system.ps1: existing hooks.json not merged ($($_.Exception.Message))"
  }
}

$hooksJson = $hooksObj | ConvertTo-Json -Depth 8
# PowerShell ConvertTo-Json may use different formatting; write UTF-8 without BOM
[System.IO.File]::WriteAllText($HooksFile, $hooksJson)

# MCP: merge mlflow-mcp from config.env
$uri = $envMap["MLFLOW_TRACKING_URI"]
# Prefer URI without embedded password if separate auth fields exist
if ($uri -match '://[^:]+:[^@]+@') {
  # keep as-is; MLflow MCP accepts credentials in URI
} elseif ($envMap["MLFLOW_TRACKING_USERNAME"] -and $envMap["MLFLOW_TRACKING_PASSWORD"]) {
  $u = [Uri]$uri
  $uri = "{0}://{1}:{2}@{3}:{4}" -f $u.Scheme, $envMap["MLFLOW_TRACKING_USERNAME"], $envMap["MLFLOW_TRACKING_PASSWORD"], $u.Host, $u.Port
}

$mlflowMcp = [ordered]@{
  command = "uv"
  args    = @("run", "--with", "mlflow[mcp]>=3.16.1", "mlflow", "mcp", "run")
  env     = [ordered]@{
    UV_DEFAULT_INDEX       = "https://pypi.tuna.tsinghua.edu.cn/simple"
    MLFLOW_TRACKING_URI    = $uri
    MLFLOW_EXPERIMENT_ID   = $envMap["MLFLOW_EXPERIMENT_ID"]
    MLFLOW_MCP_TOOLS       = "genai"
  }
}

$mcpRoot = [ordered]@{ mcpServers = [ordered]@{} }
if (Test-Path $McpFile) {
  try {
    $mcpExisting = Get-Content $McpFile -Raw | ConvertFrom-Json
    if ($mcpExisting.mcpServers) {
      foreach ($prop in $mcpExisting.mcpServers.PSObject.Properties) {
        if ($prop.Name -eq "mlflow-mcp") { continue }
        $mcpRoot.mcpServers[$prop.Name] = $prop.Value
      }
    }
  } catch {
    Write-Host "install-system.ps1: existing mcp.json not fully merged ($($_.Exception.Message))"
  }
}
$mcpRoot.mcpServers["mlflow-mcp"] = $mlflowMcp
[System.IO.File]::WriteAllText($McpFile, ($mcpRoot | ConvertTo-Json -Depth 8))

# Do NOT also write project .cursor/hooks.json with the same commands — Cursor runs
# both user and project hooks, which double-fires every event on this repo.
$ProjectCursorDir = Join-Path $PluginRoot ".cursor"
New-Item -ItemType Directory -Force -Path $ProjectCursorDir | Out-Null
$ProjectHooksPath = Join-Path $ProjectCursorDir "hooks.json"
if (Test-Path $ProjectHooksPath) {
  Remove-Item -Force $ProjectHooksPath
  Write-Host "install-system.ps1: removed $ProjectHooksPath (avoid duplicate with user hooks)"
}

$pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
if (-not $pwsh) {
  Write-Host ""
  Write-Host "WARNING: pwsh (PowerShell 7) not on PATH."
  Write-Host "  Cursor feeds hook stdin through Windows PowerShell 5.1, which mis-decodes"
  Write-Host "  UTF-8 on CP936 and corrupts Chinese text. Install PS7, then Reload Window:"
  Write-Host "    winget install --id Microsoft.PowerShell -e"
  Write-Host "  The plugin also reverses lossless ACP mojibake, but lossy payloads still drop."
} else {
  Write-Host "install-system.ps1: pwsh found at $($pwsh.Source) (preferred for UTF-8 hook stdin)"
}

Write-Host "install-system.ps1: wrote $HooksFile"
Write-Host "install-system.ps1: hook command = $hookCmd"
Write-Host "install-system.ps1: wrote $McpFile (mlflow-mcp)"
Write-Host "Done. Reload Cursor window (Developer: Reload Window) so hooks/MCP pick up."
