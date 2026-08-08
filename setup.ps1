param(
    [switch]$InitDb
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Invoke-Step($Name, $Action) {
    Write-Host "`n[setup] $Name..." -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name" }
}

if (-not (Test-Path $Py)) {
    Write-Host "[setup] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $Venv
    & $Py -m pip install --upgrade pip
}

Invoke-Step "Installing core requirements" {
    & $Py -m pip install -r (Join-Path $Root "backend\requirements.txt")
}

$EnableChroma = Read-Host "Install ChromaDB RAG? [y/N]"
if ($EnableChroma -match '^[Yy]') {
    Invoke-Step "Installing chroma-hnswlib 0.7.5 (prebuilt Windows wheel)" {
        & $Py -m pip install "chroma-hnswlib==0.7.5"
    }
    Invoke-Step "Installing chromadb 0.6.3 (no deps)" {
        & $Py -m pip install "chromadb==0.6.3" --no-deps
    }
    Invoke-Step "Installing chromadb dependencies" {
        & $Py -m pip install -r (Join-Path $Root "backend\requirements-chromadb.txt")
    }
    Invoke-Step "Installing build backend" {
        & $Py -m pip install "build>=1.0.3"
    }
} else {
    Write-Host "`nSkipping ChromaDB. RAG will use keyword fallback. Enable later with: wrapper.ps1 -Chroma" -ForegroundColor DarkYellow
}

if ($InitDb) {
    Write-Host "`n[setup] Initializing database..." -ForegroundColor Cyan
    $env:PYTHONPATH = Join-Path $Root "backend"
    & $Py -c "from app.core.db import init_db; init_db(); print('Database created')"
}

Write-Host "`n[setup] Done. Run with: .\start.ps1" -ForegroundColor Green