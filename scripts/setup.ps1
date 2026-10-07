<#
.SYNOPSIS
  One-time setup for running Chameleon Warp natively on Windows.

.DESCRIPTION
  Creates backend\.venv with Python 3.9, installs the backend dependencies,
  installs the frontend dependencies and builds the UI. Then run
  scripts\start.ps1.

  Needs Python 3.9 (the `py` launcher, `uv`, or -Python <path>) and Node.js 18+.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Python C:\Python39\python.exe
#>
param(
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$venv = Join-Path $backend ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"

function Say($msg) { Write-Host "[setup] $msg" -ForegroundColor Cyan }

function Test-Py39($exe, [string[]]$pre = @()) {
  try {
    & $exe @pre -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 9) else 1)" 2>$null
    return $LASTEXITCODE -eq 0
  } catch { return $false }
}

# ---- 1. Python virtualenv ----
if ((Test-Path $venvPy) -and (Test-Py39 $venvPy)) {
  Say "reusing existing venv at backend\.venv"
} elseif ($Python) {
  if (-not (Test-Py39 $Python)) { throw "$Python is not Python 3.9" }
  Say "creating venv with $Python"
  & $Python -m venv $venv
} elseif ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Py39 "py" @("-3.9"))) {
  Say "creating venv with py -3.9"
  & py -3.9 -m venv $venv
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
  Say "creating venv with uv (downloads Python 3.9 if needed)"
  & uv venv --seed -p 3.9 $venv
} else {
  throw "Python 3.9 not found. Install it from python.org (or install uv) and re-run, or pass -Python <path>."
}
if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }

Say "installing backend dependencies"
& $venvPy -m pip install -q --upgrade pip wheel
$req = Join-Path $backend "requirements.txt"
$core = Join-Path $venv "req-core.txt"
Get-Content $req | Where-Object { $_ -notmatch '^\s*pypardiso' } | Set-Content $core
& $venvPy -m pip install -q -r $core
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
# Intel MKL Pardiso: 2-4x faster solves. Optional — SuperLU fallback.
$pardiso = (Get-Content $req | Where-Object { $_ -match '^\s*pypardiso' } | Select-Object -First 1)
& $venvPy -m pip install -q $pardiso
if ($LASTEXITCODE -ne 0) { Say "pypardiso unavailable — using SciPy SuperLU (slower, same results)" }

# ---- 2. Frontend ----
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  throw "npm not found. Install Node.js 18+ (https://nodejs.org) and re-run."
}
Say "installing frontend dependencies and building the UI"
Push-Location $frontend
try {
  npm ci --no-audit --no-fund
  if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
  npm run build
  if ($LASTEXITCODE -ne 0) { throw "frontend build failed" }
} finally {
  Pop-Location
}

# ---- 3. Blender (optional, for FBX) ----
Push-Location $backend
try {
  & $venvPy -c "import sys, fbx_io; sys.exit(0 if fbx_io.blender_available() else 1)" 2>$null
  if ($LASTEXITCODE -eq 0) { Say "Blender found — FBX import/export enabled" }
  else { Say "Blender not found — .obj works; for FBX install Blender 4.x or set BLENDER_PATH" }
} finally {
  Pop-Location
}

Say "done. Start the app with: powershell -ExecutionPolicy Bypass -File scripts\start.ps1"
