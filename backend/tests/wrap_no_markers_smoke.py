"""
Fix: Wrap and Region need NO markers when source and target share topology
(same rationale as same-topology /run). Fit still requires markers.

Against a running backend on 127.0.0.1:8000.
Run:  .venv\\Scripts\\python.exe tests\\wrap_no_markers_smoke.py
"""
import json
import math
import os
import sys
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from jobclient import submit_and_wait  # noqa: E402

BASE = "http://127.0.0.1:8000"


def http(method, path, body=None, content_type="application/json"):
    data = None
    headers = {}
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers["Content-Type"] = content_type
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    resp = urllib.request.urlopen(req)
    payload = resp.read()
    return json.loads(payload) if payload else None


def upload(session_id, role, filename, obj_text, field="file", sub="upload/"):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + obj_text.encode() + f"\r\n--{boundary}--\r\n".encode()
    path = f"/api/sessions/{session_id}/{sub}{role}" if role else f"/api/sessions/{session_id}/{sub}"
    return http("POST", path, body, content_type=f"multipart/form-data; boundary={boundary}")


SEG, RINGS = 20, 14


def quad_sphere_obj(scale_z=1.0):
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, RINGS):
        theta = math.pi * i / RINGS
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(SEG):
            phi = 2.0 * math.pi * j / SEG
            verts.append((r * math.cos(phi), r * math.sin(phi), scale_z * z))
    verts.append((0.0, 0.0, -scale_z))
    top, bottom = 1, len(verts)

    def rv(i, j):
        return 2 + (i - 1) * SEG + (j % SEG)

    faces = []
    for j in range(SEG):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, RINGS - 1):
        for j in range(SEG):
            faces.append((rv(i, j), rv(i + 1, j), rv(i + 1, j + 1), rv(i, j + 1)))
    for j in range(SEG):
        faces.append((bottom, rv(RINGS - 1, j + 1), rv(RINGS - 1, j)))
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(ix) for ix in f) for f in faces]
    return "\n".join(lines) + "\n"


failures = []


def check(name, cond, detail=""):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def new_session_same_topo():
    src = quad_sphere_obj(1.0)          # sphere
    tgt = quad_sphere_obj(0.6)          # same topology, squashed on Z
    s = http("POST", "/api/sessions")["session_id"]
    upload(s, "source_ref", "src.obj", src)
    upload(s, "target_ref", "tgt.obj", tgt)
    return s


# 1) Plain Wrap, identical topology, ZERO markers -> should succeed.
s = new_session_same_topo()
r = submit_and_wait(http, s, "wrap", {"iterations": 4})
check("wrap 0 markers (same topo) succeeds",
      r.get("vertex_count", 0) > 0, f"verts={r.get('vertex_count')}")
http("DELETE", f"/api/sessions/{s}")

# 2) Region wrap, identical topology, ZERO markers -> should succeed.
# Waypoints = a ring of vertices around the equator (0-indexed = OBJ index - 1).
# ring i=7: OBJ rv = 2 + 6*SEG + j = 122 + j  -> 0-indexed = 121 + j
s = new_session_same_topo()
ring = [121 + (j % SEG) for j in (0, 5, 10, 15)]
r = submit_and_wait(http, s, "wrap",
                    {"iterations": 4, "region": {"waypoints": ring, "feather_width": 0.0}})
check("region 0 markers (same topo) succeeds",
      r.get("vertex_count", 0) > 0, f"verts={r.get('vertex_count')}")
http("DELETE", f"/api/sessions/{s}")

# 3) Fit with 0 markers must STILL fail (needs markers to anchor islands).
s = new_session_same_topo()
try:
    submit_and_wait(http, s, "wrap", {"iterations": 1, "use_closest_point": False})
    check("fit 0 markers still rejected", False, "expected an error, got success")
except RuntimeError as e:
    check("fit 0 markers still rejected", "marker" in str(e).lower(), str(e)[:80])
http("DELETE", f"/api/sessions/{s}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
