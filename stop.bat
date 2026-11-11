@echo off
setlocal

echo Stopping the process listening on port 8003...
powershell -NoProfile -Command "$c=Get-NetTCPConnection -LocalPort 8003 -State Listen -ErrorAction SilentlyContinue; if(-not $c){ Write-Host '[OK] Nothing is listening on 8003.'; exit 0 }; $c | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; Write-Host ('[OK] Stopped PID ' + $_) }"
pause
endlocal
