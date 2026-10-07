"""
End-to-end test for polygon-preserving wrap export, against a RUNNING
backend on 127.0.0.1:8000. urllib only, same style as fit_smoke.py.

Source = QUAD UV sphere (quads between rings, triangle fans at poles) with
per-vertex UVs. Target = triangulated ellipsoid, different resolution.
After /wrap, the downloaded OBJ must keep the author's quads and UVs.

Run:  .venv\\Scripts\\python.exe tests\\quad_http_smoke.py
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


def upload(session_id, role, filename, obj_text):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + obj_text.encode() + f"\r\n--{boundary}--\r\n".encode()
    return http("POST", f"/api/sessions/{session_id}/upload/{role}", body,
                content_type=f"multipart/form-data; boundary={boundary}")


# ---------------------------------------------------------------- meshes
def quad_sphere_obj(segments, rings, radii=(1.0, 1.0, 1.0)):
    """UV sphere as the artist would author it: quad rows, tri fans at poles,
    per-vertex vt, f tokens in v/vt form."""
    rx, ry, rz = radii
    verts = [(0.0, 0.0, rz)]
    uvs = [(0.5, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((rx * r * math.cos(phi), ry * r * math.sin(phi), rz * z))
            uvs.append((j / segments, 1.0 - i / rings))
    verts.append((0.0, 0.0, -rz))
    uvs.append((0.5, 0.0))
    top, bottom = 1, len(verts)   # 1-based

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
    return "\n".join(lines) + "\n", len(verts), faces


def tri_sphere_obj(segments, rings, radii=(1.0, 1.0, 1.0)):
    text, n, faces = quad_sphere_obj(segments, rings, radii)
    # brute force: let the backend triangulate by giving it pure triangles
    out = []
    vt_or_v = [ln for ln in text.splitlines() if not ln.startswith("f ")]
    out.extend(ln.split(" ", 1)[0] + " " + ln.split(" ", 1)[1]
               for ln in vt_or_v if ln.startswith("v "))
    for f in faces:
        ix = [str(i) for i in f]
        for k in range(1, len(ix) - 1):
            out.append(f"f {ix[0]} {ix[k]} {ix[k + 1]}")
    return "\n".join(out) + "\n"


def nearest_1based(verts_text, point):
    best, best_d = 0, 1e30
    idx = 0
    for ln in verts_text.splitlines():
        if not ln.startswith("v "):
            continue
        _, x, y, z = ln.split()[:4]
        d = (float(x) - point[0]) ** 2 + (float(y) - point[1]) ** 2 + (float(z) - point[2]) ** 2
        if d < best_d:
            best, best_d = idx, d
        idx += 1
    return best  # 0-based vertex index as the API expects


failures = []


def check(name, cond, detail):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


src_text, n_src_verts, src_faces = quad_sphere_obj(24, 16)
n_src_quads = sum(1 for f in src_faces if len(f) == 4)
tgt_text = tri_sphere_obj(18, 12, radii=(1.2, 0.9, 1.05))

s = http("POST", "/api/sessions")["session_id"]
upload(s, "source_ref", "quad_head.obj", src_text)
upload(s, "target_ref", "scan.obj", tgt_text)

dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
markers = [{"source": nearest_1based(src_text, d),
            "target": nearest_1based(tgt_text, (d[0] * 1.2, d[1] * 0.9, d[2] * 1.05))}
           for d in dirs]
http("PUT", f"/api/sessions/{s}/markers", {"markers": markers})

r = submit_and_wait(http, s, "wrap",
         {"iterations": 8, "smoothness": 1.0, "identity_weight": 0.001,
          "smooth_result": 3})

check("wrap ok", r["vertex_count"] == n_src_verts,
      f"{r['vertex_count']} verts (expected {n_src_verts})")
check("polygons_preserved flag", r.get("polygons_preserved") is True,
      f"polygons_preserved={r.get('polygons_preserved')}")
check("residual reported", isinstance(r.get("residual"), dict) and "p95" in r["residual"],
      f"residual={r.get('residual')}")

_, body = http("GET", f"/api/sessions/{s}/wrap.obj", raw=True)
text = body.decode()
f_sizes = [len(ln.split()) - 1 for ln in text.splitlines() if ln.startswith("f ")]
n_quads = sum(1 for k in f_sizes if k == 4)
n_vt = sum(1 for ln in text.splitlines() if ln.startswith("vt "))
n_vn = sum(1 for ln in text.splitlines() if ln.startswith("vn "))

check("downloaded OBJ keeps quads", n_quads == n_src_quads,
      f"{n_quads} quads (expected {n_src_quads}), {len(f_sizes)} faces total")
check("UVs survive", n_vt == n_src_verts, f"{n_vt} vt lines")
check("stale normals dropped", n_vn == 0, f"{n_vn} vn lines")

vp = http("GET", f"/api/sessions/{s}/mesh/wrap")
check("viewport still gets triangles", vp["face_count"] == len(f_sizes) + n_src_quads,
      f"viewport {vp['face_count']} tris from {len(f_sizes)} polygons")
check("heatmap distances attached",
      len(vp.get("distances_frac", [])) == vp["vertex_count"]
      and isinstance(vp.get("residual"), dict),
      f"{len(vp.get('distances_frac', []))} per-vertex distances "
      f"(expected {vp['vertex_count']}), residual={'yes' if vp.get('residual') else 'no'}")
check("heatmap distances sane",
      all(-1.0 <= d <= 1.0 for d in vp.get("distances_frac", [2.0])),
      "all fractions within [-1, 1]")

http("DELETE", f"/api/sessions/{s}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
