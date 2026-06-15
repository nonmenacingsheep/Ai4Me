@echo off
cd /d "%~dp0"

echo [Ai4Me] Installing Python dependencies...
pip install -r backend\requirements.txt --quiet

echo [Ai4Me] Installing Node dependencies...
npm install --silent

echo [Ai4Me] Launching...
npm start
