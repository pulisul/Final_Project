@echo off
rem IntellectAgent install (Windows). Double-click or run:  install.bat [options]
rem Options are the same as scripts\setup_env.py:  --yes --dry-run --skip-model --skip-embedding --skip-tests --recreate --install-ollama --model NAME
setlocal
cd /d "%~dp0"
set "PYX="
py -3.11 -c "import sys" >nul 2>&1 && set "PYX=py -3.11"
if not defined PYX (
  python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1 && set "PYX=python"
)
if not defined PYX (
  echo [FAIL] Python 3.11 was not found. This project supports Python 3.11 only.
  echo   Install:  winget install -e --id Python.Python.3.11   ^(or python.org, check "Add to PATH"^)
  echo   Then run install.bat again.
  pause
  exit /b 2
)
echo Python: %PYX%
%PYX% scripts\setup_env.py %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
