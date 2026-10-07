#!/usr/bin/env bash
# Run the backend test suite (smoke tests in backend/tests/).
#
#   ./scripts/test.sh                 # everything
#   ./scripts/test.sh wrap refit      # only tests whose file name contains a pattern
#
# Most tests talk to a backend on 127.0.0.1:8000, so this script starts one on
# a throwaway workspace (port 8000 must be free) and stops it afterwards.
# limits_smoke.py needs deliberately low mesh caps, so it gets its own backend.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# PYTHON overrides the interpreter (CI uses the runner's Python directly).
PY="$(command -v "${PYTHON:-$ROOT/backend/.venv/bin/python}" || true)"
[[ -n "$PY" ]] || { echo "backend/.venv not found — run ./scripts/setup.sh first" >&2; exit 1; }
cd "$ROOT/backend"

WORK="$(mktemp -d)"
LOG="$WORK/backend.log"
PID=""
export TQDM_DISABLE=1

start_backend() {   # extra VAR=value assignments as arguments
  env DT_WORKSPACE_ROOT="$WORK/ws" "$@" "$PY" -m uvicorn main:app --host 127.0.0.1 --port 8000 >>"$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do
    "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=1)" 2>/dev/null && return 0
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
  done
  echo "backend failed to start — log:" >&2; cat "$LOG" >&2; exit 1
}
stop_backend() {
  if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null; fi
  PID=""
}
trap 'stop_backend; rm -rf "$WORK"' EXIT

PATTERNS=("$@")
selected() {
  [[ ${#PATTERNS[@]} -eq 0 ]] && return 0
  for p in "${PATTERNS[@]}"; do [[ "$1" == *"$p"* ]] && return 0; done
  return 1
}

passed=0; failed=()
run_test() {
  local t="$1" start=$SECONDS
  if "$PY" "$t" >"$WORK/out.txt" 2>&1; then
    passed=$((passed + 1)); printf '  \033[32mPASS\033[0m %-36s %4ss\n' "$(basename "$t")" $((SECONDS - start))
  else
    failed+=("$t"); printf '  \033[31mFAIL\033[0m %-36s %4ss\n' "$(basename "$t")" $((SECONDS - start))
    tail -20 "$WORK/out.txt" | sed 's/^/       /'
  fi
}

echo "[test] starting backend on :8000"
start_backend
for t in tests/*_smoke.py; do
  [[ "$t" == tests/limits_smoke.py ]] && continue
  selected "$t" && run_test "$t"
done
stop_backend

if selected tests/limits_smoke.py; then
  echo "[test] restarting backend with low mesh caps for limits_smoke.py"
  start_backend DT_MAX_SOURCE_TRIANGLES=500 DT_MAX_TARGET_TRIANGLES_DT=100 DT_MAX_TARGET_TRIANGLES=100000
  run_test tests/limits_smoke.py
  stop_backend
fi

echo "[test] $passed passed, ${#failed[@]} failed"
[[ ${#failed[@]} -eq 0 ]]
