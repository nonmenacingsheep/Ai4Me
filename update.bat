@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo    Updating Aitha
echo ============================================
echo.

echo [1/4] Stopping any running Aitha instance...
REM Kill ONLY this app's processes, three ways so an orphaned backend can't survive:
REM   (a) electron.exe whose path is under this folder (leaves VS Code etc. alone),
REM   (b) any python whose COMMAND LINE points at this folder's backend (dev mode runs
REM       the SYSTEM python, so its exe path is NOT under here — match the script path),
REM   (c) whatever still owns backend port 7823.
REM Then WAIT until 7823 is actually free, so the fresh backend can bind and run the
REM new code (otherwise an old in-memory backend keeps serving stale API responses).
powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path '%~dp0').Path.TrimEnd('\').ToLower(); Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.ToLower().StartsWith($root) -and $_.Name -eq 'electron.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -and $_.CommandLine.ToLower().Contains($root) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; try { Get-NetTCPConnection -LocalPort 7823 -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue } } catch {}; $free=$false; for($i=0;$i -lt 25;$i++){ try { $busy=[bool](Get-NetTCPConnection -LocalPort 7823 -State Listen -ErrorAction Stop) } catch { $busy=$false }; if(-not $busy){ $free=$true; break }; Start-Sleep -Milliseconds 400 }; if($free){ Write-Host '    backend stopped; port 7823 is free.' } else { Write-Host '    WARNING: port 7823 still in use - close Aitha fully and retry.' }"
echo.

echo [2/4] Pulling latest code from GitHub...
for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "BEFORE=%%i"
git pull
if errorlevel 1 (
  echo.
  echo  !! git pull failed - see the message above.
  echo     If it mentions local changes, you edited tracked files; sort that out, then re-run.
  echo.
  pause
  exit /b 1
)
for /f %%i in ('git rev-parse --short HEAD 2^>nul') do set "AFTER=%%i"
echo     code: %BEFORE%  -^>  %AFTER%
if "%BEFORE%"=="%AFTER%" (
  echo     ^(already on the latest commit - nothing new to pull^)
) else (
  echo     updated. latest commit:
  git log -1 --oneline
)
echo.

echo [3/4] Checking dependencies...
git diff --quiet %BEFORE% %AFTER% -- backend/requirements.txt 2>nul
if errorlevel 1 (
  echo    requirements.txt changed - installing Python deps...
  pip install -r backend\requirements.txt
) else (
  echo    Python deps unchanged.
)
git diff --quiet %BEFORE% %AFTER% -- package.json package-lock.json 2>nul
if errorlevel 1 (
  echo    package.json changed - installing Node deps...
  call npm install
) else (
  echo    Node deps unchanged.
)
echo.

REM Mark setup complete so start.bat skips its first-run install next time.
echo done> "backend\.setup-done"

echo [4/4] Launching Aitha...
start "" "%~dp0run.bat"
echo.
echo ============================================
echo  Done. Aitha is launching in a new window.
echo  If a feature still looks old: fully close the
echo  app window, then run this again.
echo ============================================
echo.
pause
endlocal
