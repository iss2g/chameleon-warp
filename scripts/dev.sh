#!/usr/bin/env bash
# Development mode: backend with auto-reload on :8000 plus the Vite dev server
# with hot module reload on :5173 (it proxies /api to the backend).
# Run ./scripts/setup.sh once first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/backend/.venv/bin/python"
[[ -x "$PY" ]] || { echo "backend/.venv not found — run ./scripts/setup.sh first" >&2; exit 1; }

( cd "$ROOT/backend" && exec "$PY" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000 ) &
BACKEND_PID=$!
trap 'kill "$BACKEND_PID" 2>/dev/null || true' EXIT INT TERM

cd "$ROOT/frontend"
[[ -d node_modules ]] || npm ci --no-audit --no-fund
npm run dev
