@echo off
REM Double-click this to put an Ai4Me shortcut (with icon) on your Desktop.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0create-desktop-shortcut.ps1"
echo.
pause
