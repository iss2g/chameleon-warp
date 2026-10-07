<#
.SYNOPSIS
  Run Chameleon Warp natively on Windows: one process serves the UI and the API.

.DESCRIPTION
  Settings come from the environment or a .env file in the repo root (see
  .env.example and docs\CONFIGURATION.md). Run scripts\setup.ps1 once first.

.PARAMETER BindHost
  Interface to listen on. 127.0.0.1 (default) = this machine only;
  0.0.0.0 = reachable from your LAN (there is no authentication).

.PARAMETER Port
  Port to listen on (default 8000, or PORT from .env).

.PARAMETER Open
  Open the app in the default browser once it is up.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -Open
#>
param(
  [string]$BindHost = "",
  [int]$Port = 0,
  [switch]$Open
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$py = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "backend\.venv not found — run scripts\setup.ps1 first."
}

# Load .env (KEY=VALUE lines; comments and blanks ignored) without overriding
# variables already set in this shell.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
  foreach ($line in Get-Content $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
      $name = $Matches[1]; $value = $Matches[2].Trim('"').Trim("'")
      if (-not [Environment]::GetEnvironmentVariable($name)) {
        [Environment]::SetEnvironmentVariable($name, $value)
      }
    }
  }
}

if (-not $BindHost) { $BindHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" } }
if (-not $Port) { $Port = if ($env:PORT) { [int]$env:PORT } else { 8000 } }
if (-not $env:TQDM_DISABLE) { $env:TQDM_DISABLE = "1" }

if (-not (Test-Path (Join-Path $frontend "dist\index.html"))) {
  Write-Host "[start] UI not built yet — building it now" -ForegroundColor Yellow
  Push-Location $frontend
  try {
    npm ci --no-audit --no-fund
    if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
  } finally {
    Pop-Location
  }
}

$url = "http://${BindHost}:$Port"
Write-Host "[start] Chameleon Warp -> $url   (Ctrl+C to stop)" -ForegroundColor Green
if ($Open) {
  $browseUrl = if ($BindHost -eq "0.0.0.0") { "http://127.0.0.1:$Port" } else { $url }
  Start-Job -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 60; $i++) {
      try { Invoke-WebRequest "$u/api/health" -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process $u; return } catch { Start-Sleep -Milliseconds 500 }
    }
  } -ArgumentList $browseUrl | Out-Null
}

Push-Location $backend
try {
  & $py -m uvicorn main:app --host $BindHost --port $Port
} finally {
  Pop-Location
}
