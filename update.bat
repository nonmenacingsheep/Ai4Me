@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo    Updating Aitha
echo ============================================
echo.

echo [1/4] Stopping any running Aitha instance...
REM Kill ONLY this app's processes: the Electron exe living under this folder, and
REM whatever owns the backend port (7823). Other Electron apps (VS Code, etc.) run
REM their own electron.exe from their own folders, so they are left untouched.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path '%~dp0').Path.TrimEnd('\').ToLower(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'electron.exe' -and $_.ExecutablePath -and $_.ExecutablePath.ToLower().StartsWith($root) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; try { Get-NetTCPConnection -LocalPort 7823 -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } } catch {}"
echo.

echo [2/4] Pulling latest code from GitHub...
git pull
if errorlevel 1 (
  echo.
  echo  !! git pull failed - see the message above.
  echo     If it mentions local changes, you edited tracked files; sort that out, then re-run.
  pause
  exit /b 1
)
echo.

echo [3/4] Checking dependencies...
git diff --quiet "HEAD@{1}" HEAD -- backend/requirements.txt 2>nul
if errorlevel 1 (
  echo    requirements.txt changed - installing Python deps...
  pip install -r backend\requirements.txt
) else (
  echo    Python deps unchanged.
)
git diff --quiet "HEAD@{1}" HEAD -- package.json package-lock.json 2>nul
if errorlevel 1 (
  echo    package.json changed - installing Node deps...
  call npm install
) else (
  echo    Node deps unchanged.
)
echo.

echo [4/4] Launching Aitha...
start "" "%~dp0run.bat"
echo.
echo Done. Aitha is launching in a new window.
echo (If a feature still looks old, fully close the app window once and relaunch.)
endlocal
