@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"

echo ================================================================
echo   atom-mem WebUI
echo ================================================================
echo.
echo   Console: http://127.0.0.1:8003/
echo   API:     http://127.0.0.1:8003/graphiti
echo.

powershell -NoProfile -Command "try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:8003/health' -TimeoutSec 2; if ($r.service -eq 'atom-mem') { exit 0 } else { exit 2 } } catch { exit 1 }"
if %errorlevel%==0 goto open

if %errorlevel%==2 (
    echo [ERROR] Port 8003 is used by another service.
    pause
    exit /b 1
)

echo [INFO] Service is not running; starting 4.MEMOS-API.bat in a new window...
start "atom-mem service" "%ROOT%4.MEMOS-API.bat"

echo [INFO] Waiting for http://127.0.0.1:8003/health ...
powershell -NoProfile -Command "$ok=$false; for($i=0;$i -lt 60;$i++){ try { $r=Invoke-RestMethod -Uri 'http://127.0.0.1:8003/health' -TimeoutSec 1; if($r.service -eq 'atom-mem'){ $ok=$true; break } } catch {}; Start-Sleep -Milliseconds 500 }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
    echo [ERROR] Service did not become ready. Check the service window for details.
    pause
    exit /b 1
)

:open
start "" "http://127.0.0.1:8003/"
echo [OK] Opened http://127.0.0.1:8003/
endlocal
