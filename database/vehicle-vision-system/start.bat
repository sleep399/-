@echo off
cd /d "%~dp0"
set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\ctpgr\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python313\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set PORT_PID=
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8001 .*LISTENING"') do set PORT_PID=%%a
if defined PORT_PID (
  echo Server is already running on http://localhost:8001
  echo PID: %PORT_PID%
  pause
  exit /b 0
)
echo Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt -q
echo Starting server...
"%PYTHON_EXE%" run.py
pause
