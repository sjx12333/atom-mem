param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  atom-mem memory service (8003)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$python = $PythonExe
if (-not $python) { $python = $env:ATOM_MEM_PYTHON }
if (-not $python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $python = $command.Source }
}
if (-not $python) {
    Write-Host "[ERROR] Python not found. Install Python 3.11+ and add it to PATH," -ForegroundColor Red
    Write-Host "        or set ATOM_MEM_PYTHON to python.exe." -ForegroundColor Red
    exit 1
}

& $python -c "import fastapi, uvicorn, neo4j, openai, pydantic" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Required packages are missing for: $python" -ForegroundColor Red
    Write-Host "        Run: $python -m pip install -r `"$Root\requirements.txt`"" -ForegroundColor Yellow
    exit 1
}

# The service must see the package root so data/ and config/ stay outside the
# code tree.  Do not use ``python -E`` here: it would ignore these variables.
$env:ATOM_MEM_HOME = $Root
$env:PYTHONPATH = Join-Path $Root "app\memos"
$env:PYTHONNOUSERSITE = "1"

Set-Location (Join-Path $Root "app\memos")
Write-Host "[INFO] Python: $python" -ForegroundColor Green
Write-Host "[INFO] Data:   $Root\data" -ForegroundColor Green
Write-Host "[INFO] Config: $Root\config\graphiti_config.json" -ForegroundColor Green
Write-Host "[INFO] API:    http://127.0.0.1:8003" -ForegroundColor Green
Write-Host "[INFO] WebUI:  http://127.0.0.1:8003/" -ForegroundColor Green
Write-Host ""

& $python -s -m memos_system.api.graphiti_server
Write-Host "`nService stopped." -ForegroundColor Red
