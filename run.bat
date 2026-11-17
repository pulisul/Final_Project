@echo off
rem IntellectAgent run (Windows).  Usage:  run.bat <command>   (doctor ^| app ^| test ^| redteam ^| docs [folder] ^| qa [folder] ^| eval [folder] ^| setup-model)
setlocal
cd /d "%~dp0"
set "PYX="
if exist ".venv-real\Scripts\python.exe" set "PYX=.venv-real\Scripts\python.exe"
if not defined PYX py -3.11 -c "import sys" >nul 2>&1 && set "PYX=py -3.11"
if not defined PYX python --version >nul 2>&1 && set "PYX=python"
if not defined PYX (
  echo [FAIL] Python was not found. Run install.bat first.
  exit /b 2
)
%PYX% manage.py %*
exit /b %ERRORLEVEL%
