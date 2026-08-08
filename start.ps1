param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$BackendDir = Join-Path $Root "backend"
$LogDir = Join-Path $Root "logs"
$BackendLog = Join-Path $LogDir "backend.log"
$Health = "http://127.0.0.1:8765/api/health"
$Port = 8765

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-BackendRunning {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Start-Backend {
    if (Get-BackendRunning) {
        Write-Host "[Meruem] Backend already listening on port $Port (reusing)." -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $VenvPy)) {
        Write-Error "Virtual env not found at $VenvPy. Run .\setup.ps1 first."
        exit 1
    }
    Write-Host "[Meruem] Starting FastAPI backend on http://127.0.0.1:$Port (own window, log: $BackendLog)" -ForegroundColor Cyan
    $env:PYTHONPATH = $BackendDir
    # Fresh console window so startup errors are visible and the server survives
    # independently of this script. Logs also mirrored to file.
    $cmd = "`$env:PYTHONPATH='$BackendDir'; `$ErrorActionPreference='Continue'; " +
           "& '$VenvPy' -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload " +
           "2>&1 | Tee-Object -FilePath '$BackendLog'; " +
           "Read-Host '`nBackend exited. Press Enter to close...'"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd -WindowStyle Normal

    if (-not $NoWait) {
        Write-Host "[Meruem] Waiting for backend healthcheck..." -ForegroundColor Gray
        $ok = $false
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 800
            if (Get-BackendRunning) {
                try {
                    $r = Invoke-RestMethod -Uri $Health -TimeoutSec 3
                    if ($r.status -eq "ok") { $ok = $true; break }
                } catch {
                    # listening but not healthy yet; keep waiting
                }
            }
        }
        if (-not $ok) {
            Write-Host "`n[Meruem] BACKEND FAILED TO START. Last log lines:" -ForegroundColor Red
            if (Test-Path $BackendLog) {
                Get-Content $BackendLog -Tail 25 | ForEach-Object { Write-Host "  $_" }
            }
            Write-Host "`nCheck the backend window/log. Common fixes:" -ForegroundColor Yellow
            Write-Host "  - Run .\setup.ps1 to install deps"
            Write-Host "  - Ensure port 8765 is free: netstat -ano | findstr 8765"
            exit 1
        }
        Write-Host "[Meruem] Backend healthy: $Health" -ForegroundColor Green
    }
}

function Start-Frontend {
    Push-Location (Join-Path $Root "frontend")
    try {
        Write-Host "[Meruem] Starting Vite dev server on http://localhost:1420" -ForegroundColor Cyan
        & npm run dev
    } finally {
        Pop-Location
    }
}

if ($BackendOnly) { Start-Backend; exit }

if (-not $FrontendOnly) {
    Start-Backend
    if (-not $NoWait -and -not (Get-BackendRunning)) {
        Write-Host "[Meruem] Backend not running; aborting frontend launch." -ForegroundColor Red
        exit 1
    }
} else {
    if (-not (Get-BackendRunning)) {
        Write-Host "[Meruem] WARNING: Frontend-only mode but no backend on port $Port." -ForegroundColor Yellow
        Write-Host "          Run .\start.ps1 -BackendOnly in another terminal first." -ForegroundColor Yellow
    }
}

Start-Frontend