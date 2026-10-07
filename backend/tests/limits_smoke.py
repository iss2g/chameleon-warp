"""
End-to-end test for upload triangle-count limits + session-expiry fields,
against a RUNNING backend on 127.0.0.1:8000.

MUST be run against a backend started with LOW caps so small test spheres trip
them, e.g.:

  DT_MAX_SOURCE_TRIANGLES=500 DT_MAX_TARGET_TRIANGLES_DT=100 \
  DT_MAX_TARGET_TRIANGLES=100000 .venv\\Scripts\\python.exe -m uvicorn main:app ...

Then:  .venv\\Scripts\\python.exe tests\\limits_smoke.py
"""
import json
import math
import sys
import urllib.error
import urllib.request
import uuid

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


def upload_pose(session_id, filename, obj_text):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + obj_text.encode() + f"\r\n--{boundary}--\r\n".encode()
    return http("POST", f"/api/sessions/{session_id}/upload/poses", body,
                content_type=f"multipart/form-data; boundary={boundary}")


def uv_sphere(seg, rings, radii=(1.0, 1.0, 1.0)):
    rx, ry, rz = radii
    V = [(0.0, 0.0, rz)]
    for i in range(1, rings):
        th = math.pi * i / rings
        z, r = math.cos(th), math.sin(th)
        for j in range(seg):
            p = 2 * math.pi * j / seg
            V.append((rx * r * math.cos(p), ry * r * math.sin(p), rz * z))
    V.append((0.0, 0.0, -rz))
    top, bot = 1, len(V)

    def rv(i, j):
        return 2 + (i - 1) * seg + (j % seg)

    F = []
    for j in range(seg):
        F.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(seg):
            F.append((rv(i, j), rv(i + 1, j), rv(i + 1, j + 1)))
            F.append((rv(i, j), rv(i + 1, j + 1), rv(i, j + 1)))
    for j in range(seg):
        F.append((bot, rv(rings - 1, j + 1), rv(rings - 1, j)))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += ["f " + " ".join(str(i) for i in f) for f in F]
    return "\n".join(lines) + "\n", len(F)


failures = []


def check(name, cond, detail=""):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


small_src, n_small = uv_sphere(12, 8)      # 168 tris
dense_src, n_dense = uv_sphere(24, 16)     # 720 tris

s = http("POST", "/api/sessions")["session_id"]

# ---- 1. dense source rejected 400 ----
try:
    upload(s, "source_ref", "dense.obj", dense_src)
    check("dense source -> 400", False, f"accepted {n_dense} tris")
except urllib.error.HTTPError as e:
    detail = e.read().decode()
    check("dense source -> 400", e.code == 400 and "dense" in detail.lower(), f"{e.code}")

# ---- 2. small source accepted ----
r = upload(s, "source_ref", "src.obj", small_src)
check("small source accepted", r.get("source_ref") is not None, f"{n_small} tris")

# ---- 3. target above DT cap: accepted + flagged + warning ----
r = upload(s, "target_ref", "tgt.obj", small_src)  # 168 tris > DT cap(100)
check("dense-for-DT target accepted", r.get("target_ref") is not None, "")
check("target_dt_available=false", r.get("target_dt_available") is False,
      f"target_dt_available={r.get('target_dt_available')}")
check("target_warning present", isinstance(r.get("target_warning"), str),
      f"warning={r.get('target_warning')}")

# ---- 4. summary carries expiry fields ----
summ = http("GET", f"/api/sessions/{s}")
check("summary has ttl_seconds", isinstance(summ.get("ttl_seconds"), (int, float)),
      f"ttl_seconds={summ.get('ttl_seconds')}")
check("summary has expires_at in the future",
      isinstance(summ.get("expires_at"), (int, float)) and summ["expires_at"] > summ.get("last_seen", 0),
      f"expires_at={summ.get('expires_at')}")

# ---- 5. /run refused when DT unavailable ----
upload_pose(s, "pose.obj", small_src)
try:
    http("POST", f"/api/sessions/{s}/run", {})
    check("run refused (DT unavailable) -> 400", False, "unexpectedly accepted")
except urllib.error.HTTPError as e:
    detail = e.read().decode()
    check("run refused (DT unavailable) -> 400",
          e.code == 400 and "dense" in detail.lower(), f"{e.code}: {detail[:80]}")

http("DELETE", f"/api/sessions/{s}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
