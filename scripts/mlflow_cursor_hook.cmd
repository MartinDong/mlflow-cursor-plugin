@echo off
REM Direct Windows hook entry: venv python + mlflow_cursor.py (no node required).
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "MLFLOW_DISABLE_AGENT_HINT=1"
set "ROOT=%~dp0.."
if exist "%ROOT%\.venv\Scripts\python.exe" (
  "%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\mlflow_cursor.py"
  exit /b %ERRORLEVEL%
)
echo mlflow_cursor_hook.cmd: missing .venv; run install.ps1 >&2
echo {}
exit /b 0
