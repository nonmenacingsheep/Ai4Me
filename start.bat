@echo off
setlocal
cd /d "%~dp0"

REM First-time setup installs dependencies ONCE, then later launches skip straight to
REM starting the app so it opens fast. To force a reinstall (e.g. after editing
REM requirements), delete backend\.setup-done or run update.bat.
if not exist "backend\.setup-done" (
  echo [Ai4Me] First-time setup - installing Python dependencies...
  echo         This downloads a few hundred MB and can take several minutes. Please wait.
  pip install -r backend\requirements.txt
  echo [Ai4Me] Installing Node dependencies...
  call npm install
  echo done> "backend\.setup-done"
) else (
  if not exist "node_modules" (
    echo [Ai4Me] Restoring Node dependencies...
    call npm install
  )
)

echo [Ai4Me] Launching...
npm start
endlocal
