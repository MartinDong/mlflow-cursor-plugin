@echo off
setlocal
node "%~dp0run_hook.js"
exit /b %ERRORLEVEL%
