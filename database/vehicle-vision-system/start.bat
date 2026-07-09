@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements.txt -q
echo Starting server...
python run.py
pause
