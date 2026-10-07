#!/usr/bin/env bash
# One-time setup for running Chameleon Warp natively (Linux / macOS).
#
#   ./scripts/setup.sh        # creates backend/.venv, installs deps, builds the UI
#   ./scripts/start.sh        # then run it
#
# Needs Python 3.9 (or `uv`, which fetches 3.9 for you) and Node.js 18+.
# Set PYTHON=/path/to/python3.9 to pick a specific interpreter.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/backend/.venv"

say() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -z "${PYTHON:-}" ]]; then
  say "Apple Silicon: the pinned numpy/scipy have no native arm64 wheels."
  say "Use Docker instead (docker compose up -d --build), or point PYTHON at an"
  say "x86_64 Python 3.9 running under Rosetta and re-run this script."
fi

# ---- 1. Python virtualenv ----------------------------------------------------
py_is_39() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 9) else 1)' 2>/dev/null; }

if [[ -x "$VENV/bin/python" ]] && py_is_39 "$VENV/bin/python"; then
  say "reusing existing venv at backend/.venv"
elif [[ -n "${PYTHON:-}" ]]; then
  py_is_39 "$PYTHON" || die "PYTHON=$PYTHON is not Python 3.9"
  say "creating venv with $PYTHON"
  "$PYTHON" -m venv "$VENV"
elif command -v python3.9 >/dev/null 2>&1; then
  say "creating venv with python3.9"
  python3.9 -m venv "$VENV"
elif command -v uv >/dev/null 2>&1; then
  say "creating venv with uv (downloads Python 3.9 if needed)"
  uv venv --seed -p 3.9 "$VENV"
else
  die "Python 3.9 not found. Install it (or uv: https://docs.astral.sh/uv/) and re-run — or use Docker."
fi

say "installing backend dependencies"
"$VENV/bin/python" -m pip install -q --upgrade pip wheel
grep -viE '^\s*pypardiso' "$ROOT/backend/requirements.txt" > "$VENV/req-core.txt"
"$VENV/bin/python" -m pip install -q -r "$VENV/req-core.txt"
# Intel MKL Pardiso: 2-4x faster solves on x86_64. Optional — SuperLU fallback.
"$VENV/bin/python" -m pip install -q "$(grep -iE '^\s*pypardiso' "$ROOT/backend/requirements.txt")" \
  || say "pypardiso unavailable here — using SciPy SuperLU (slower, same results)"

# ---- 2. Frontend -------------------------------------------------------------
command -v npm >/dev/null 2>&1 || die "npm not found. Install Node.js 18+ (https://nodejs.org) and re-run."
say "installing frontend dependencies and building the UI"
( cd "$ROOT/frontend" && npm ci --no-audit --no-fund && npm run build )

# ---- 3. Blender (optional, for FBX) ------------------------------------------
if ( cd "$ROOT/backend" && "$VENV/bin/python" -c 'import sys, fbx_io; sys.exit(0 if fbx_io.blender_available() else 1)' ) 2>/dev/null; then
  say "Blender found — FBX import/export enabled"
else
  say "Blender not found — .obj works; for FBX install Blender 4.x or set BLENDER_PATH"
fi

say "done. Start the app with: ./scripts/start.sh"
