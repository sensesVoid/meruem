param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$RepoRoot = Split-Path $Root -Parent
$VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
$BackendLog = Join-Path $LogDir "qimera-backend.log"
$Health = "http://127.0.0.1:8766/api/health"
$ApiPort = 8766
$UiPort = 1421

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Get-QiMeraApiRunning {
    return [bool](Get-NetTCPConnection -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue)
}

function Start-QiMeraApi {
    if (Get-QiMeraApiRunning) {
        Write-Host "[QiMera] API already listening on port $ApiPort (reusing)." -ForegroundColor Yellow
        return
    }
    if (-not (Test-Path $VenvPy)) {
        Write-Error "Virtual env not found at $VenvPy. Run Meruem's .\setup.ps1 first."
        exit 1
    }
    Write-Host "[QiMera] Starting API on http://127.0.0.1:$ApiPort (own window, log: $BackendLog)" -ForegroundColor Cyan
    # Put Julia (juliaup) on PATH so the juliacall bridge can find it, even in a
    # terminal opened before Julia was installed.
    $JuliaBin = Join-Path $env:USERPROFILE ".juliaup\bin"
    $cmd = "Set-Location '$Root'; `$env:PYTHONPATH='$RepoRoot;$Root'; " +
           "if (Test-Path '$JuliaBin') { `$env:PATH = '$JuliaBin;' + `$env:PATH }; " +
           "& '$VenvPy' -m uvicorn app.main:app --host 127.0.0.1 --port $ApiPort " +
           "2>&1 | Tee-Object -FilePath '$BackendLog'; " +
           "Read-Host '`nQiMera API exited. Press Enter to close...'"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd -WindowStyle Normal

    if (-not $NoWait) {
        Write-Host "[QiMera] Waiting for API healthcheck..." -ForegroundColor Gray
        $ok = $false
        for ($i = 0; $i -lt 30; $i++) {
            try { if ((Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) { $ok = $true; break } } catch { }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ok) { Write-Host "[QiMera] API never became healthy. See $BackendLog" -ForegroundColor Red }
    }
}

function Start-QiMeraUi {
    if ([bool](Get-NetTCPConnection -LocalPort $UiPort -State Listen -ErrorAction SilentlyContinue)) {
        Write-Host "[QiMera] UI already listening on port $UiPort (reusing)." -ForegroundColor Yellow
        return
    }
    $UiDir = Join-Path $Root "frontend"
    if (-not (Test-Path (Join-Path $UiDir "node_modules"))) {
        Write-Host "[QiMera] Installing frontend deps (first run)..." -ForegroundColor Yellow
        Push-Location $UiDir
        try { npm install } finally { Pop-Location }
    }
    Write-Host "[QiMera] Starting UI on http://127.0.0.1:$UiPort (own window)" -ForegroundColor Cyan
    $cmd = "Set-Location '$UiDir'; npm run dev | Tee-Object -FilePath '$LogDir\qimera-ui.log'; " +
           "Read-Host '`nQiMera UI exited. Press Enter to close...'"
    Start-Process powershell -ArgumentList "-NoProfile", "-Command", $cmd -WindowStyle Normal
    Write-Host "[QiMera] Open http://127.0.0.1:$UiPort (or use the 'QiMera' link in Meruem's TopNav)" -ForegroundColor Green
}

if (-not $BackendOnly) { Start-QiMeraUi }
if (-not $FrontendOnly) { Start-QiMeraApi }