@echo off
setlocal

set "ROOT=%~dp0"

echo ================================================================
echo   atom-mem memory service
echo ================================================================
echo.
echo   API:   http://127.0.0.1:8003
echo   Docs:  http://127.0.0.1:8003/docs
echo   WebUI: http://127.0.0.1:8003/
echo.

if not exist "%ROOT%app\memos\memos_system\api\graphiti_server.py" (
    echo [ERROR] application files not found under %ROOT%app\memos
    pause
    exit /b 1
)

powershell -NoProfile -Command "try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:8003/health' -TimeoutSec 2; if ($r.service -eq 'atom-mem') { exit 0 } else { exit 2 } } catch { exit 1 }"
if %errorlevel%==0 (
    echo [OK] atom-mem is already running at http://127.0.0.1:8003/
    start "" "http://127.0.0.1:8003/"
    pause
    exit /b 0
)
if %errorlevel%==2 (
    echo [ERROR] Port 8003 is answering, but it is not atom-mem.
    echo         Stop the other service first, then run this file again.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%start_memos.ps1"
pause
endlocal
