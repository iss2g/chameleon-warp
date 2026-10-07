# HTTP API

The web UI is a client of a plain JSON/HTTP API, so you can script the backend
directly — batch-wrap a folder of heads, run a transfer from a build pipeline,
and so on.

While the app is running, interactive OpenAPI docs with every field and
default are at **<http://localhost:8000/docs>**. This page is the overview.

All paths are under `/api`. Errors are JSON `{"detail": "…"}` with a 4xx/5xx
status. There is no authentication; the session id is the only handle on a
session's data.

## Workflow

```
POST /api/sessions                          → session_id
POST /api/sessions/{id}/upload/source_ref   (multipart "file")
POST /api/sessions/{id}/upload/target_ref   (multipart "file")
PUT  /api/sessions/{id}/markers             {"markers": [{"source": i, "target": j}, …]}
POST /api/sessions/{id}/wrap                → {"job_id", "state"}
GET  /api/sessions/{id}/job                 poll until state is "done" or "error"
GET  /api/sessions/{id}/wrap.obj            download the result
```

Heavy operations (`run`, `wrap`, `refit`, `refit_via_proxy`) return
immediately with a job id; the work happens in the background. A session has
at most one active job (a second submit returns `409`).

## Example (Python, standard library only)

```python
import json, time, urllib.request, uuid

BASE = "http://127.0.0.1:8000/api"

def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def upload(sid, role, path):
    boundary = uuid.uuid4().hex
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{path.split('/')[-1]}\"\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{BASE}/sessions/{sid}/upload/{role}", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))

sid = call("POST", "/sessions")["session_id"]
upload(sid, "source_ref", "basemesh.obj")
upload(sid, "target_ref", "scan.obj")
call("PUT", f"/sessions/{sid}/markers",
     {"markers": [{"source": 120, "target": 5310}, {"source": 873, "target": 9921},
                  {"source": 2048, "target": 14002}]})   # pick indices in the UI, or export them as JSON

call("POST", f"/sessions/{sid}/wrap", {"iterations": 8, "smooth_result": 2})
while (job := call("GET", f"/sessions/{sid}/job"))["state"] not in ("done", "error"):
    time.sleep(1)
print(job["state"], job.get("error") or job["result"]["residual"])

urllib.request.urlretrieve(f"{BASE}/sessions/{sid}/wrap.obj", "wrapped.obj")
```

The UI's **Export** button writes markers as `{"markers": [{"source", "target"}, …]}`,
which you can send to `PUT /markers` as-is.

## Endpoints

### Sessions

| Method & path | |
|---|---|
| `POST /api/sessions` | Create a session. Returns the session summary (below). |
| `GET /api/sessions/{id}` | Session summary. Any request on a session refreshes its expiry. |
| `DELETE /api/sessions/{id}` | Delete the session and its files. |
| `GET /api/queue` | `{waiting, running, concurrency}` of the compute queue. |
| `GET /api/health` | `{"name": "Chameleon Warp", "status": "ok"}`. |

Session summary fields: `session_id`, `source_ref` / `target_ref` / `proxy_ref`
(`{name}` or null), `source_uploaded_as_fbx`, `target_uploaded_as_fbx`,
`target_dt_available`, `poses` (`[{pose_id, name}]`), `marker_count`,
`result_pose_ids`, `has_results`, `has_wrap`, `wrap_mode`
(`wrap|region|fit|refit`), `refit_state`, `last_seen`, `ttl_seconds`,
`expires_at`, `cache` (which pipeline stages are cached).

### Uploads and meshes

| Method & path | |
|---|---|
| `POST /api/sessions/{id}/upload/{role}` | Multipart field `file`. `role` = `source_ref`, `target_ref` (`.obj` or `.fbx`) or `proxy_ref` (`.obj`). An FBX source also extracts its shape keys as poses. Returns the summary plus upload info (warnings, extracted poses). |
| `POST /api/sessions/{id}/upload/poses` | Multipart field `files` (repeatable), `.obj` only, must match the source's vertex/face count. |
| `DELETE /api/sessions/{id}/poses/{pose_id}` | Remove a pose. |
| `GET /api/sessions/{id}/mesh/{role}` | Mesh as JSON (`vertices`, `faces`, counts) for `source_ref`, `target_ref`, `proxy_ref` or `wrap` (the wrap/fit/refit result, with per-vertex `distances_frac`). |
| `GET /api/sessions/{id}/poses/{pose_id}/mesh` | A pose as mesh JSON. |
| `GET /api/sessions/{id}/results/{pose_id}/mesh` | A DT result as mesh JSON. |

### Markers

| Method & path | |
|---|---|
| `PUT /api/sessions/{id}/markers` | `{"markers": [{"source": int, "target": int}, …]}` — vertex indices into the source and target references. Replaces the list. |
| `GET /api/sessions/{id}/markers` | The current list. |

### Jobs

| Method & path | Body |
|---|---|
| `POST /api/sessions/{id}/wrap` | `WrapPayload` — Wrap, Wrap Region (with `region`) or Fit (`use_closest_point: false`). |
| `POST /api/sessions/{id}/run` | `{iterations=8, smoothness=1.0, identity_weight=0.001}` — DT Transfer of all poses. |
| `POST /api/sessions/{id}/refit` | `RefitPayload` — garment onto body. |
| `POST /api/sessions/{id}/refit_via_proxy` | `RefitPayload` — needs a wrap result of the proxy onto the body and an uploaded `proxy_ref`. |
| `GET /api/sessions/{id}/job` | `{job_id, kind, state: queued|running|done|error, queue_position, progress: {stage, iter, total}, error, result}`. `result` is the endpoint's response once done. 404 if the session never ran a job. |

**WrapPayload** (all optional):

| Field | Default | |
|---|---|---|
| `iterations` | 8 | 1–20 |
| `smoothness` | 1.0 | |
| `identity_weight` | 0.001 | |
| `use_closest_point` | true | `false` = Fit mode (markers only, no surface snapping) |
| `smooth_result` | 0 | Taubin passes after the solve (0–50) |
| `align_to_source` | true | Similarity-align the target to the source via the markers first |
| `match_max_angle_deg` | 60 | Normal compatibility for matches (10–90) |
| `match_distance_frac` | 0.02 | Floor of the adaptive match-distance cut, fraction of the target diagonal |
| `project_result` | true | Snap onto the target surface at the end |
| `project_distance_frac` | 0.02 | Max snap distance |
| `soft_markers` / `soft_markers_weight` | true / 10 | Relax marker pins late in the solve |
| `region` | null | `{waypoints: [int ≥3], seed?, invert=false, feather_width=0, feather_exponent=2}` |

**RefitPayload** (all optional): `preset` (`accessory|cloth|armor|skintight`),
`iterations` (10), `pre_transform` (row-major 4×4, 16 floats), `preserve`
and `frozen` (source vertex indices), `layer_mode` (`legacy|auto`),
`auto_polish`, `growth_steps`, `collision` (`none|push_out|hide_body`), and
optional overrides `grip_width`, `seam_depth`, `pin_radius`, `offset`,
`contact_tight`, `contact_free`, `thickness`, `stiffness`, `smoothness`.
Distances are in the mesh's world units.

### Previews (synchronous, no solve)

| Method & path | |
|---|---|
| `POST /api/sessions/{id}/region/preview` | `RegionPayload` → `{loop, interior, interior_count, frozen_count, feather_count, total}`. |
| `POST /api/sessions/{id}/refit_field` | `RefitPayload` → per-vertex `grip_frac` (and `layer` in auto mode) plus field stats. |

### Results

| Method & path | |
|---|---|
| `GET /api/sessions/{id}/wrap.obj` | Wrap / Wrap Region / Fit / Refit result, in the source's original polygon layout. |
| `GET /api/sessions/{id}/results/{pose_id}.obj` | One DT result. |
| `GET /api/sessions/{id}/results.zip` | All DT results. |
| `GET /api/sessions/{id}/results.fbx` | All DT results as shape keys on the target (needs Blender; `503` otherwise). |

### Refit body mask and UI state

| Method & path | |
|---|---|
| `GET /api/sessions/{id}/body_mask` | `{hidden_vertices, body_vertex_count}` for the last refit. |
| `PUT /api/sessions/{id}/body_mask` | `{"hidden_vertices": [int, …]}` — replace the mask. |
| `POST /api/sessions/{id}/body_mask/dilate` | `{"rings": n}` — grow (n > 0) or shrink (n < 0) by vertex rings, −20…20. |
| `GET /api/sessions/{id}/body_mask.json` | Download the mask as a file. |
| `PUT /api/sessions/{id}/refit_state` | `{"refit_state": {…}}` — opaque UI state (placement, paint) persisted with the session. |
