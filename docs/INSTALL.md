# Installation

Chameleon Warp is one process: a FastAPI backend that also serves the built web
UI. You can run it in Docker or natively. Either way you end up with the app at
**<http://localhost:8000>**.

- [Docker](#docker)
- [Native: Linux / macOS](#native-linux--macos)
- [Native: Windows](#native-windows)
- [FBX support (Blender)](#fbx-support-blender)
- [Updating](#updating)
- [Using it from another machine](#using-it-from-another-machine)
- [Uninstalling](#uninstalling)
- [Troubleshooting](#troubleshooting)

## Docker

The simplest way on every platform, including Apple Silicon.

```bash
git clone https://github.com/iss2g/chameleon-warp.git
cd chameleon-warp
docker compose up -d --build
```

Then open <http://localhost:8000>.

| Command | |
|---|---|
| `docker compose up -d --build` | build (first time / after an update) and start in the background |
| `docker compose logs -f` | follow the logs |
| `docker compose stop` / `start` | pause / resume |
| `docker compose down` | stop and remove the container (sessions are kept) |
| `docker compose down -v` | …and delete all sessions |

Options, set in a `.env` file next to `docker-compose.yml` (copy
[`.env.example`](../.env.example)) or on the command line:

```bash
PORT=9000 docker compose up -d            # different port
WITH_BLENDER=0 docker compose up -d --build   # smaller image, .obj only
```

Notes:

- The image bundles **Blender 4.2** for FBX import/export (x86_64 only; on
  arm64 it is skipped automatically and FBX is unavailable). `WITH_BLENDER=0`
  builds a much smaller image without it.
- **Intel MKL Pardiso** is installed on x86_64 for 2–4× faster sparse solves;
  elsewhere the solver falls back to SciPy's SuperLU automatically.
- Sessions live in the `chameleon-data` Docker volume, so they survive
  rebuilds and restarts.
- Any [configuration](CONFIGURATION.md) variable in `.env` is passed to the app.

## Native: Linux / macOS

Requirements:

- **Python 3.9** exactly — the algorithm core pins numpy 1.20 / SciPy 1.6,
  which have no wheels for newer Pythons. If you have
  [uv](https://docs.astral.sh/uv/), the setup script uses it to fetch 3.9
  automatically.
- **Node.js 18+** (only to build the UI).
- Optional: **Blender 4.x** for FBX.

```bash
git clone https://github.com/iss2g/chameleon-warp.git
cd chameleon-warp
./scripts/setup.sh
./scripts/start.sh
```

`setup.sh` creates `backend/.venv`, installs the Python dependencies (trying
`pypardiso` and falling back silently if it is unavailable), runs `npm ci` and
builds the UI into `frontend/dist`. It is safe to re-run.

`start.sh` serves the UI and API on `127.0.0.1:8000`. Change it with
environment variables or a `.env` file in the repo root:

```bash
PORT=9000 ./scripts/start.sh
```

To use a specific interpreter: `PYTHON=/opt/python3.9/bin/python3.9 ./scripts/setup.sh`.

**Apple Silicon:** numpy 1.20 / SciPy 1.6 have no native arm64 macOS wheels.
Use Docker, or install an x86_64 Python 3.9 and run it under Rosetta:
`PYTHON=/path/to/x86_64/python3.9 ./scripts/setup.sh`.

## Native: Windows

Requirements: **Python 3.9** from [python.org](https://www.python.org/downloads/release/python-3913/)
(the `py` launcher is used to find it) or [uv](https://docs.astral.sh/uv/),
**Node.js 18+**, and optionally **Blender 4.x**.

```powershell
git clone https://github.com/iss2g/chameleon-warp.git
cd chameleon-warp
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -Open
```

`setup.ps1 -Python C:\path\to\python.exe` picks a specific interpreter.
`start.ps1` accepts `-Port 9000`, `-BindHost 0.0.0.0` and `-Open` (opens the
browser when the backend is up), and also reads `.env` from the repo root.

## FBX support (Blender)

FBX files are read and written by running Blender headless in the background.
Without Blender everything works with `.obj`; FBX uploads and the
"Download .fbx" button report that Blender is missing.

The backend looks for Blender in this order:

1. the `BLENDER_PATH` environment variable (full path to the executable);
2. `blender` on `PATH`;
3. the default install locations (`C:\Program Files\Blender Foundation\…`,
   `/Applications/Blender.app`).

Blender 4.x is required (the import/export scripts use the 4.x OBJ operators).

## Updating

```bash
git pull
docker compose up -d --build     # Docker
./scripts/setup.sh               # native (then restart start.sh)
```

After an update the app shows a "What's new" popup once.

## Using it from another machine

By default the app only listens on `127.0.0.1`. To reach it from your LAN:

- Docker: change the port mapping in `docker-compose.yml` from
  `"127.0.0.1:${PORT:-8000}:8000"` to `"${PORT:-8000}:8000"`.
- Native: `HOST=0.0.0.0 ./scripts/start.sh` (or `start.ps1 -BindHost 0.0.0.0`).

There is **no authentication**: anyone who can reach the port can use the app
and, knowing a session id, see that session's meshes. Only do this on a
network you trust, or put it behind a reverse proxy with auth.

## Uninstalling

- Docker: `docker compose down -v --rmi local`, then delete the folder.
- Native: delete the folder. Sessions live in `backend/workspace/` unless you
  set `DT_WORKSPACE_ROOT`.

## Troubleshooting

**`pip` fails building numpy / SciPy.** You are not on Python 3.9 (or on
Apple Silicon natively). Check `backend/.venv/bin/python --version`; delete
`backend/.venv` and re-run the setup with a 3.9 interpreter, or use Docker.

**`pypardiso unavailable` during setup.** Harmless: solves use SciPy SuperLU
instead (slower, identical results). Pardiso is x86_64-only.

**Port 8000 is already in use.** Use `PORT=9000`.

**A job is slow / the machine is busy.** The solver uses all cores through
BLAS. Cap it with `MKL_NUM_THREADS` / `OMP_NUM_THREADS` (see
[CONFIGURATION.md](CONFIGURATION.md)).

**"Source mesh too dense".** The source reference is limited to 100k
triangles by default — decimate it, or raise `DT_MAX_SOURCE_TRIANGLES` (solve
time and memory grow quickly).

**My session disappeared.** Idle sessions are deleted after 7 days, and the
oldest ones earlier if all sessions together exceed 50 GB. Both are
configurable.

**UI shows a JSON message instead of the app.** The UI has not been built:
run `npm run build` in `frontend/` (the start scripts do this automatically).
