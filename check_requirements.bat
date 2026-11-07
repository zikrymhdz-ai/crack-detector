@echo off
title CrackScan - Requirement Checker
cd /d "%~dp0"

echo ============================================
echo   CrackScan - Checking Python requirements
echo ============================================
echo.

REM ── Check Python installed ──
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM ── Create requirements.txt if it doesn't exist yet ──
if not exist "requirements.txt" (
    echo Creating requirements.txt ...
    (
        echo flask^>=2.3.0
        echo opencv-python-headless^>=4.8.0
        echo numpy^>=1.24.0
        echo ultralytics^>=8.0.200
        echo gunicorn^>=21.2.0
        echo Pillow^>=10.0.0
        echo gdown^>=4.7.1
    ) > requirements.txt
    echo.
)

echo Checking installed packages...
echo.

set MISSING=0

call :check flask flask
call :check cv2 opencv-python-headless
call :check numpy numpy
call :check ultralytics ultralytics
call :check gunicorn gunicorn
call :check PIL Pillow
call :check gdown gdown

echo.

if "%MISSING%"=="0" (
    echo ============================================
    echo   All requirements already installed.
    echo   You can run your app now.
    echo ============================================
    pause
    exit /b 0
)

echo ============================================
echo   Missing packages detected. Installing now...
echo   ^(this may take a few minutes^)
echo ============================================
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ============================================
echo   Installation complete!
echo   You can run your app now.
echo ============================================
pause
exit /b 0

REM ── Subroutine: check a single package ──
:check
set IMPORT_NAME=%~1
set DISPLAY_NAME=%~2
python -c "import %IMPORT_NAME%" >nul 2>nul
if errorlevel 1 (
    echo [MISSING] %DISPLAY_NAME%
    set MISSING=1
) else (
    echo [OK]      %DISPLAY_NAME%
)
exit /b 0
