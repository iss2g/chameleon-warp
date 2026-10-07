# Configuration

Everything works with the defaults. To change a setting, set an environment
variable, or put it in a `.env` file in the repository root (copy
[`.env.example`](../.env.example)). The `.env` file is read by
`docker compose` and by `scripts/start.sh` / `scripts/start.ps1`.

Backend settings are read once at startup (`backend/infra/config.py`), so
restart the app after a change.

## Launcher

| Variable | Default | |
|---|---|---|
| `PORT` | `8000` | Port the app listens on (Docker: the host port). |
| `HOST` | `127.0.0.1` | Native start scripts only. `0.0.0.0` makes it reachable from other machines — there is no authentication. |
| `WITH_BLENDER` | `1` | Docker build only. `0` builds a smaller image without Blender (no FBX). |

## Storage and sessions

| Variable | Default | |
|---|---|---|
| `DT_WORKSPACE_ROOT` | `backend/workspace` (Docker: `/data/workspace`) | Where sessions (uploads, results, caches) are stored. |
| `DT_SESSION_TTL_SECONDS` | `604800` (7 days) | A session untouched for this long is deleted. Opening it in the browser counts as activity. |
| `DT_SESSION_SWEEP_INTERVAL_SECONDS` | `1800` | How often expired sessions are cleaned up. |
| `DT_MAX_TOTAL_STORAGE_BYTES` | `53687091200` (50 GB) | Disk budget for all sessions together. Above it, the least recently used sessions are deleted early. `0` = no limit. |
| `DT_MAX_SESSION_STORAGE_BYTES` | `1073741824` (1 GB) | Disk cap for one session. |

## Mesh and upload limits

| Variable | Default | |
|---|---|---|
| `DT_MAX_UPLOAD_BYTES` | `209715200` (200 MB) | Largest single upload. |
| `DT_MAX_SOURCE_TRIANGLES` | `100000` | Source reference cap. Solve time and memory grow quickly with the source size. |
| `DT_MAX_TARGET_TRIANGLES` | `1500000` | Target reference cap. Wrap handles dense targets well (they are only searched, not solved for). |
| `DT_MAX_TARGET_TRIANGLES_DT` | `100000` | Above this, the target is still accepted for Wrap, but DT Transfer is disabled for it — the DT result inherits the target's topology. |
| `DT_MAX_POSES_PER_SESSION` | `250` | Maximum number of poses (DT Transfer). |

## Performance

| Variable | Default | |
|---|---|---|
| `DT_COMPUTE_CONCURRENCY` | `1` | Heavy jobs that may run at the same time. The math is CPU-bound and already multi-threaded; two parallel jobs each run about half as fast. |
| `MKL_NUM_THREADS`, `OMP_NUM_THREADS` | all cores | Cap the threads used by the linear algebra, e.g. to keep the machine responsive while a job runs. |
| `DT_LAYER_WORKERS` | `min(cores, 8)` | Processes used by Refit's automatic layer detection. |

## Integration

| Variable | Default | |
|---|---|---|
| `BLENDER_PATH` | auto-detected | Full path to the Blender 4.x executable, if it is not on `PATH` or in the default install location. |
| `DT_FRONTEND_DIST` | `frontend/dist` | Built UI to serve. If the folder doesn't exist, the backend serves only the API. |
| `DT_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated origins allowed to call the API from a browser. Only needed if you serve the UI from a different origin than the API. |

## Logging

| Variable | Default | |
|---|---|---|
| `DT_LOG_JSON` | `0` | `1` = one JSON object per log line. |
| `DT_LOG_FILE` | unset | Also write logs to this file, rotated by size. |
| `DT_LOG_MAX_BYTES`, `DT_LOG_BACKUPS` | `20971520`, `5` | Rotation size and number of kept files for `DT_LOG_FILE`. |
| `TQDM_DISABLE` | `1` (start scripts, Docker) | Hides the solver's progress bars in the log. |
