"""
End-to-end test for the same-topology /run mode, against a RUNNING backend
on 127.0.0.1:8000. urllib only, same style as fit_smoke.py.

When source_ref and target_ref share identical connectivity (the
wrap -> blendshape-retarget workflow), /run must skip correspondence,
use the identity triangle mapping, and require ZERO markers.

Scenario: source = unit tri-sphere, pose = same sphere with a "hat" bump
(top pushed out radially), target = SAME topology ellipsoid. Transfer must
put an adapted bump on the ellipsoid: bump-region vertices move away from
the ellipsoid, the rest barely move.

Run:  .venv\\Scripts\\python.exe tests\\same_topology_smoke.py
"""
import json
import math
import os
import sys
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from jobclient import submit_and_wait, wait_for_job  # noqa: E402

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


def multipart(field, filename, content):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + content.encode() + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def upload_ref(session_id, role, filename, obj_text):
    body, ctype = multipart("file", filename, obj_text)
    return http("POST", f"/api/sessions/{session_id}/upload/{role}", body, content_type=ctype)


def upload_pose(session_id, filename, obj_text):
    body, ctype = multipart("files", filename, obj_text)
    return http("POST", f"/api/sessions/{session_id}/upload/poses", body, content_type=ctype)


# ---------------------------------------------------------------- meshes
def tri_sphere(segments, rings):
    """Unit UV sphere, triangles only. Returns (verts list, faces list 1-based)."""
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
    verts.append((0.0, 0.0, -1.0))
    top, bottom = 1, len(verts)

    def rv(i, j):
        return 2 + (i - 1) * segments + (j % segments)

    faces = []
    for j in range(segments):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            a, b, c, d = rv(i, j), rv(i + 1, j), rv(i + 1, j + 1), rv(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(segments):
        faces.append((bottom, rv(rings - 1, j + 1), rv(rings - 1, j)))
    return verts, faces


def to_obj(verts, faces):
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += ["f " + " ".join(str(i) for i in f) for f in faces]
    return "\n".join(lines) + "\n"


def parse_obj_verts(text):
    out = []
    for ln in text.splitlines():
        if ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            out.append((float(x), float(y), float(z)))
    return out


failures = []


def check(name, cond, detail):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


SEG, RING = 20, 14
verts, faces = tri_sphere(SEG, RING)
n_tris = len(faces)

BUMP = 1.25
bump_idx = [i for i, (x, y, z) in enumerate(verts) if z > 0.6]
pose_verts = [(x * BUMP, y * BUMP, z * BUMP) if i in set(bump_idx) else (x, y, z)
              for i, (x, y, z) in enumerate(verts)]
ELL = (1.4, 0.8, 1.0)
tgt_verts = [(x * ELL[0], y * ELL[1], z * ELL[2]) for x, y, z in verts]

# ---------------------------------------------------------------- happy path
s = http("POST", "/api/sessions")["session_id"]
upload_ref(s, "source_ref", "neutral.obj", to_obj(verts, faces))
upload_ref(s, "target_ref", "ellipsoid_same_topo.obj", to_obj(tgt_verts, faces))
upload_pose(s, "hat.obj", to_obj(pose_verts, faces))
# NO markers at all.

r = submit_and_wait(http, s, "run", {})
check("run with zero markers succeeds", r is not None and len(r["results"]) == 1,
      f"results={r and r['results']}")
check("identity_mapping flag", r.get("identity_mapping") is True,
      f"identity_mapping={r.get('identity_mapping')}")
check("mapping is per-triangle identity", r["mapping_size"] == n_tris,
      f"mapping_size={r['mapping_size']} (expected {n_tris})")

pose_id = r["results"][0]["pose_id"]
_, body = http("GET", f"/api/sessions/{s}/results/{pose_id}.obj", raw=True)
out_verts = parse_obj_verts(body.decode())
check("result vertex count = target's", len(out_verts) == len(verts),
      f"{len(out_verts)} verts")
check("result finite", all(all(math.isfinite(c) for c in v) for v in out_verts),
      "all coordinates finite")

# Bump region must move away from the ellipsoid; the far side must not.
# The DT solve is global (smooth falloff + a translation-ish drift), so:
# estimate the drift as the mean displacement of the SOUTH hemisphere
# (far from the bump), subtract it, then compare magnitudes.
bump_set = set(bump_idx)
south_idx = [i for i, (x, y, z) in enumerate(verts) if z < -0.2]
disp = [(o[0] - t[0], o[1] - t[1], o[2] - t[2]) for o, t in zip(out_verts, tgt_verts)]
drift = tuple(sum(d[k] for i, d in enumerate(disp) if i in set(south_idx)) / len(south_idx)
              for k in range(3))

def mag_nodrift(i):
    return math.sqrt(sum((disp[i][k] - drift[k]) ** 2 for k in range(3)))

mean_bump = sum(mag_nodrift(i) for i in bump_idx) / len(bump_idx)
mean_south = sum(mag_nodrift(i) for i in south_idx) / len(south_idx)
check("bump transferred onto ellipsoid",
      mean_bump > 0.05 and mean_bump > 4 * mean_south,
      f"bump mean offset {mean_bump:.3f} vs south {mean_south:.3f} "
      f"(drift {tuple(round(d, 3) for d in drift)})")

# ---------------------------------------------------------------- cache path
r2 = submit_and_wait(http, s, "run", {})
check("second run hits mapping cache", "mapping" in r2.get("cache_hits", []),
      f"hits={r2.get('cache_hits')}")
check("second run still identity", r2.get("identity_mapping") is True,
      f"identity_mapping={r2.get('identity_mapping')}")
http("DELETE", f"/api/sessions/{s}")

# ------------------------------------------------- different topology guard
verts2, faces2 = tri_sphere(SEG - 2, RING)  # different resolution
s2 = http("POST", "/api/sessions")["session_id"]
upload_ref(s2, "source_ref", "neutral.obj", to_obj(verts, faces))
upload_ref(s2, "target_ref", "other_topo.obj", to_obj(verts2, faces2))
upload_pose(s2, "hat.obj", to_obj(pose_verts, faces))
# The markers<3 check for non-identical topology now runs inside the worker
# (it needs the meshes loaded), so it surfaces as a failed job rather than an
# immediate 400.
try:
    submit_and_wait(http, s2, "run", {})
    check("different topology + 0 markers -> job error", False, "no error raised")
except RuntimeError as e:
    check("different topology + 0 markers -> job error", "marker" in str(e).lower(),
          str(e)[:100])
http("DELETE", f"/api/sessions/{s2}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
