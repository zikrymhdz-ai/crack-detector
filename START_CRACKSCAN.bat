@echo off
REM ────────────────────────────────────────────────────────────────
REM  One-click startup for the CrackScan backend.
REM
REM  1. Kills any server already holding port 5000
REM  2. Starts app.py (auto-detected camera) + marker size
REM  3. Verifies /live_stats came up and prints which camera is live
REM
REM  The live camera is AUTO-DETECTED: the first local camera that returns a frame,
REM  falling back to the HiMicro RTSP device configured in app.py. Pin a specific
REM  source by setting CAMERA_SOURCE before running this file, e.g.
REM      set CAMERA_SOURCE=0            (local device index 0)
REM      set CAMERA_SOURCE=himicro      (HiMicro RTSP, as before)
REM      set CAMERA_SOURCE=rtsp://...   (any stream URL)
REM ────────────────────────────────────────────────────────────────
setlocal

cd /d "%~dp0"

REM ── 1. stop anything already on port 5000 ─────────────────────
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5000" ^| findstr LISTENING') do (
    echo [setup] stopping existing server on port 5000, PID %%p
    taskkill /F /PID %%p >nul 2>&1
)
REM ping is used instead of "timeout" because timeout fails when stdin is redirected
ping -n 3 127.0.0.1 >nul

REM ── 2. environment ────────────────────────────────────────────
if "%ARUCO_MARKER_SIZE_CM%"=="" (
    if exist "aruco_size.txt" (
        set /p ARUCO_MARKER_SIZE_CM=<"aruco_size.txt"
    ) else (
        set "ARUCO_MARKER_SIZE_CM=10.0"
    )
)

echo [setup] camera              = auto-detect (CAMERA_SOURCE pins it)
echo [setup] ARUCO_MARKER_SIZE_CM = %ARUCO_MARKER_SIZE_CM%
echo [setup] starting app.py ...
echo.

start "CrackScan Server" python app.py

REM ── 3. wait for it to answer, then confirm which camera is live ─
echo [setup] waiting for the server to come up...
set /a tries=0
:wait
ping -n 3 127.0.0.1 >nul
set /a tries+=1
curl -s -m 3 http://127.0.0.1:5000/live_stats >nul 2>&1
if not errorlevel 1 goto ready
if %tries% lss 45 goto failed
goto wait

:ready
echo.
echo [setup] server is up.
curl -s -m 5 http://127.0.0.1:5000/live_stats
echo.
echo.
echo   Web page : http://localhost:5000/
echo   Live view: http://localhost:5000/live
echo   Cracks   : curl http://127.0.0.1:5000/api/cracks
echo.
if not defined CRACKSCAN_NO_PAUSE pause
exit /b 0

:failed
echo.
echo [setup] ERROR - server did not answer within 90 seconds.
echo         Check the server window for a traceback.
echo         The model load alone can take ~30s on first start.
echo.
if not defined CRACKSCAN_NO_PAUSE pause
exit /b 1
