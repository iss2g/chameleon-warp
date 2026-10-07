#!/usr/bin/env bash
# Run Chameleon Warp natively: one process serves the UI and the API.
#
#   ./scripts/start.sh                 # http://127.0.0.1:8000
#   PORT=9000 ./scripts/start.sh
#   HOST=0.0.0.0 ./scripts/start.sh    # reachable from your LAN (no auth!)
#
# Settings come from the environment or a .env file in the repo root
# (see .env.example and docs/CONFIGURATION.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/backend/.venv/bin/python"

if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

[[ -x "$PY" ]] || { echo "backend/.venv not found — run ./scripts/setup.sh first" >&2; exit 1; }
if [[ ! -f "$ROOT/frontend/dist/index.html" ]]; then
  echo "[start] UI not built yet — building it now"
  ( cd "$ROOT/frontend" && npm ci --no-audit --no-fund && npm run build )
fi

echo "[start] Chameleon Warp → http://$HOST:$PORT   (Ctrl+C to stop)"
cd "$ROOT/backend"
exec "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT"
