"""
End-to-end: /run (DT transfer) restores the TARGET's original quads/UVs in the
downloaded result, the same way /wrap does for the source. Against a RUNNING
backend on 127.0.0.1:8000.

Uses same-topology mode (source == target quad sphere → identity mapping, no
markers) so we isolate the polygon-preserving export path.

Run:  .venv\\Scripts\\python.exe tests\\run_quad_smoke.py
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


def upload(session_id, role, filename, obj_text, field="file", sub="upload/"):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + obj_text.encode() + f"\r\n--{boundary}--\r\n".encode()
    path = f"/api/sessions/{session_id}/{sub}{role}" if role else f"/api/sessions/{session_id}/{sub}"
    return http("POST", path, body, content_type=f"multipart/form-data; boundary={boundary}")


def quad_sphere_obj(segments, rings, scale=1.0):
    """UV sphere with quad rows + triangle fans at the poles, per-vertex UVs."""
    verts = [(0.0, 0.0, scale)]
    uvs = [(0.5, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((scale * r * math.cos(phi), scale * r * math.sin(phi), scale * z))
            uvs.append((j / segments, 1.0 - i / rings))
    verts.append((0.0, 0.0, -scale))
    uvs.append((0.5, 0.0))
    top, bottom = 1, len(verts)

    def rv(i, j):
        return 2 + (i - 1) * segments + (j % segments)

    faces = []
    for j in range(segments):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append((rv(i, j), rv(i + 1, j), rv(i + 1, j + 1), rv(i, j + 1)))
    for j in range(segments):
        faces.append((bottom, rv(rings - 1, j + 1), rv(rings - 1, j)))

    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uvs]
    lines += ["f " + " ".join(f"{ix}/{ix}" for ix in f) for f in faces]
    n_quads = sum(1 for f in faces if len(f) == 4)
    return "\n".join(lines) + "\n", len(verts), n_quads


failures = []


def check(name, cond, detail=""):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# Source == target (identical quad sphere) → same-topology identity transfer.
ref_text, n_verts, n_quads = quad_sphere_obj(20, 14)
# A pose: the same topology, squashed on Z (a deformation to transfer).
pose_text, _, _ = quad_sphere_obj(20, 14)
pose_text = "\n".join(
    (f"v {ln.split()[1]} {ln.split()[2]} {float(ln.split()[3]) * 0.6:.6f}"
     if ln.startswith("v ") else ln)
    for ln in pose_text.splitlines()
) + "\n"

s = http("POST", "/api/sessions")["session_id"]
upload(s, "source_ref", "src.obj", ref_text)
upload(s, "target_ref", "tgt.obj", ref_text)
upload(s, "", "pose.obj", pose_text, field="files", sub="upload/poses")

r = submit_and_wait(http, s, "run", {})
check("run identity", r.get("identity_mapping") is True, f"identity={r.get('identity_mapping')}")
check("polygons_preserved flag", r.get("polygons_preserved") is True,
      f"polygons_preserved={r.get('polygons_preserved')}")

pose_id = r["results"][0]["pose_id"]
_, body = http("GET", f"/api/sessions/{s}/results/{pose_id}.obj", raw=True)
text = body.decode()
f_sizes = [len(ln.split()) - 1 for ln in text.splitlines() if ln.startswith("f ")]
got_quads = sum(1 for k in f_sizes if k == 4)
n_vt = sum(1 for ln in text.splitlines() if ln.startswith("vt "))

check("downloaded result keeps quads", got_quads == n_quads,
      f"{got_quads} quads (expected {n_quads})")
check("UVs survive in result", n_vt == n_verts, f"{n_vt} vt lines (expected {n_verts})")

http("DELETE", f"/api/sessions/{s}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
