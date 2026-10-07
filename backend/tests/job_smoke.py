"""
End-to-end test for the async job model (/wrap + GET /job), against a RUNNING
backend on 127.0.0.1:8000. urllib only.

Verifies:
  • POST /wrap returns a job_id immediately (not the result)
  • a second submit while the first is active is rejected 409
  • GET /job reports progress (stage advances) and queue_position
  • the job reaches state=done and carries the endpoint result payload
  • GET /job after completion still returns the (persisted) terminal snapshot

Run:  .venv\\Scripts\\python.exe tests\\job_smoke.py
"""
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from jobclient import wait_for_job  # noqa: E402

BASE = "http://127.0.0.1:8000"


def http(method, path, body=None, content_type="application/json", raw=False):
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = content_type
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    resp = urllib.request.urlopen(req)
    payload = resp.read()
    if raw:
        return resp, payload
    return json.loads(payload) if payload else None


def upload(session_id, role, filename, obj_text):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + obj_text.encode() + f"\r\n--{boundary}--\r\n".encode()
    return http("POST", f"/api/sessions/{session_id}/upload/{role}", body,
                content_type=f"multipart/form-data; boundary={boundary}")


def uv_sphere(segments, rings, radii=(1.0, 1.0, 1.0)):
    rx, ry, rz = radii
    verts = [(0.0, 0.0, rz)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z, r = math.cos(theta), math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((rx * r * math.cos(phi), ry * r * math.sin(phi), rz * z))
    verts.append((0.0, 0.0, -rz))
    top, bottom = 1, len(verts)

    def rv(i, j):
        return 2 + (i - 1) * segments + (j % segments)

    faces = []
    for j in range(segments):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append((rv(i, j), rv(i + 1, j), rv(i + 1, j + 1)))
            faces.append((rv(i, j), rv(i + 1, j + 1), rv(i, j + 1)))
    for j in range(segments):
        faces.append((bottom, rv(rings - 1, j + 1), rv(rings - 1, j)))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i) for i in f) for f in faces]
    return "\n".join(lines) + "\n", verts


def nearest(verts, point):
    best, best_d = 0, 1e30
    for idx, v in enumerate(verts):
        d = sum((v[k] - point[k]) ** 2 for k in range(3))
        if d < best_d:
            best, best_d = idx, d
    return best


failures = []


def check(name, cond, detail=""):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


src_text, src_verts = uv_sphere(20, 14)
tgt_text, tgt_verts = uv_sphere(16, 11, radii=(1.2, 0.9, 1.05))

s = http("POST", "/api/sessions")["session_id"]
upload(s, "source_ref", "src.obj", src_text)
upload(s, "target_ref", "tgt.obj", tgt_text)

dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
markers = [{"source": nearest(src_verts, d),
            "target": nearest(tgt_verts, (d[0] * 1.2, d[1] * 0.9, d[2] * 1.05))}
           for d in dirs]
http("PUT", f"/api/sessions/{s}/markers", {"markers": markers})

# ---- 1. submit returns a job_id, not the result ----
sub = http("POST", f"/api/sessions/{s}/wrap",
           {"iterations": 8, "smoothness": 1.0, "identity_weight": 0.001})
check("submit returns job_id", isinstance(sub, dict) and "job_id" in sub, f"{sub}")
check("submit has no result yet", "vertex_count" not in sub, f"keys={list(sub or {})}")
check("state is queued/running", sub.get("state") in ("queued", "running"), sub.get("state"))

# ---- 2. concurrent submit is 409 ----
try:
    http("POST", f"/api/sessions/{s}/wrap",
         {"iterations": 8, "smoothness": 1.0, "identity_weight": 0.001})
    check("concurrent submit -> 409", False, "second submit unexpectedly accepted")
except urllib.error.HTTPError as e:
    check("concurrent submit -> 409", e.code == 409, str(e.code))

# ---- 3. poll: watch progress + queue_position ----
stages_seen = set()
saw_position = False
deadline = time.time() + 600
while True:
    job = http("GET", f"/api/sessions/{s}/job")
    stages_seen.add(job.get("progress", {}).get("stage"))
    if isinstance(job.get("queue_position"), int) and job["queue_position"] >= 0:
        saw_position = True
    if job.get("state") in ("done", "error"):
        break
    if time.time() > deadline:
        break
    time.sleep(0.2)

check("job finished done", job.get("state") == "done", f"state={job.get('state')} err={job.get('error')}")
check("reported a queue_position", saw_position, f"stages={stages_seen}")
check("progressed through a compute stage",
      bool(stages_seen & {"precompute", "iterations", "projection", "export"}),
      f"stages={sorted(str(x) for x in stages_seen)}")
check("done carries result payload",
      isinstance(job.get("result"), dict) and job["result"].get("vertex_count") == len(src_verts),
      f"result_verts={job.get('result', {}).get('vertex_count')} (expected {len(src_verts)})")

# ---- 4. terminal snapshot persists on re-poll ----
again = http("GET", f"/api/sessions/{s}/job")
check("re-poll still done", again.get("state") == "done", again.get("state"))

# ---- 5. after completion a new submit is accepted ----
sub2 = http("POST", f"/api/sessions/{s}/wrap",
            {"iterations": 6, "smoothness": 1.0, "identity_weight": 0.001})
check("resubmit after done accepted", "job_id" in (sub2 or {}), f"{sub2}")
wait_for_job(http, s)

http("DELETE", f"/api/sessions/{s}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
