"""
End-to-end HTTP test for regional wrap against a RUNNING backend (:8000).
Exercises /region/preview and /wrap with a region payload, plus validations.
urllib only — no third-party deps.

Run:  .venv\\Scripts\\python.exe tests\\region_http_smoke.py
"""
import json
import math
import os
import sys
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from jobclient import submit_and_wait  # noqa: E402

BASE = "http://127.0.0.1:8000"
S, R = 24, 16
BOUNDARY_RING = 6

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def http(method, path, body=None, raw=False):
    data, headers = None, {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    resp = urllib.request.urlopen(req)
    payload = resp.read()
    if raw:
        return resp, payload
    return json.loads(payload) if payload else None


def upload(sid, role, filename, text):
    b = uuid.uuid4().hex
    body = (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + text.encode() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        BASE + f"/api/sessions/{sid}/upload/{role}", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"}, method="POST")
    return json.loads(urllib.request.urlopen(req).read())


def svert(i, j, radius=1.0):
    t, p = math.pi * i / R, 2 * math.pi * j / S
    return (radius * math.sin(t) * math.cos(p), radius * math.cos(t), radius * math.sin(t) * math.sin(p))


def sidx(i, j):
    return (i - 1) * S + j


def obj(verts, faces):
    return "\n".join([f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
                     + [f"f {a+1} {b+1} {c+1}" for a, b, c in faces]) + "\n"


def parse_verts(text):
    out = []
    for ln in text.splitlines():
        if ln.startswith("v "):
            _, x, y, z = ln.split()[:4]
            out.append((float(x), float(y), float(z)))
    return out


def band(rings, seg):
    f = []
    for i in range(rings - 1):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            c, d = (i + 1) * seg + j, (i + 1) * seg + (j + 1) % seg
            f += [(a, b, c), (b, d, c)]
    return f


# source sphere
src_verts = [svert(i, j) for i in range(1, R) for j in range(S)]
src_faces = band(R - 1, S)
# target: bump sphere, different resolution
S2, R2, BUMP = 30, 20, 0.35
tgt_verts = []
for i in range(1, R2):
    for j in range(S2):
        x = math.sin(math.pi * i / R2) * math.cos(2 * math.pi * j / S2)
        y = math.cos(math.pi * i / R2)
        z = math.sin(math.pi * i / R2) * math.sin(2 * math.pi * j / S2)
        r = 1.0 + BUMP * max(0.0, y) ** 2
        tgt_verts.append((x * r, y * r, z * r))
tgt_faces = band(R2 - 1, S2)

waypoints = [sidx(BOUNDARY_RING, j) for j in range(0, S, 4)]
seed = sidx(1, 0)
# Mix markers INSIDE the cap (editable) with markers OUTSIDE (frozen rings).
# The frozen-side markers used to collide with the freeze pins -> 500; they
# must now be dropped silently and the wrap must still succeed.
markers = (
    [{"source": sidx(1, j), "target": 0} for j in range(0, S, 8)]      # cap (editable)
    + [{"source": sidx(10, j), "target": 5} for j in range(0, S, 8)]   # frozen (outside)
)

sid = http("POST", "/api/sessions")["session_id"]
upload(sid, "source_ref", "sphere.obj", obj(src_verts, src_faces))
upload(sid, "target_ref", "bump.obj", obj(tgt_verts, tgt_faces))
http("PUT", f"/api/sessions/{sid}/markers", {"markers": markers})

# ---- 1. preview ----
print("[1] /region/preview")
pv = http("POST", f"/api/sessions/{sid}/region/preview",
          {"waypoints": waypoints, "seed": seed, "feather_width": 0.8})
check("loop is the full ring", len(pv["loop"]) == S, f"{len(pv['loop'])} vs {S}")
check("interior is the cap", pv["interior_count"] == (BOUNDARY_RING - 1) * S,
      f"{pv['interior_count']} vs {(BOUNDARY_RING-1)*S}")
check("frozen+interior account for all verts",
      pv["interior_count"] + pv["frozen_count"] == pv["total"],
      f"{pv['interior_count']}+{pv['frozen_count']} vs {pv['total']}")
check("feather band nonempty", pv["feather_count"] > 0, str(pv["feather_count"]))

# ---- 2. regional wrap ----
print("[2] /wrap with region")
submit_and_wait(http, sid, "wrap",
     {"iterations": 8, "smoothness": 1.0, "identity_weight": 0.001,
      "use_closest_point": True, "align_to_source": False,  # isolate region mechanics
      "region": {"waypoints": waypoints, "seed": seed, "feather_width": 0.8}})
resp, raw = http("GET", f"/api/sessions/{sid}/wrap.obj", raw=True)
res = parse_verts(raw.decode())
check("vertex count preserved", len(res) == len(src_verts))

frozen_ids = [sidx(i, j) for i in range(BOUNDARY_RING, R) for j in range(S)]
frozen_err = max(
    max(abs(res[v][k] - src_verts[v][k]) for k in range(3)) for v in frozen_ids)
# OBJ serialization writes %.6f, so a round-tripped frozen vertex can differ
# by up to ~5e-7 from the in-memory source. The solver pins it exactly; this
# bound just accounts for the 6-decimal OBJ rounding.
check("frozen region identical (within OBJ precision)", frozen_err < 2e-6,
      f"max diff {frozen_err:.2e}")

cap_ids = [sidx(1, j) for j in range(S)]
src_r = sum(math.dist(src_verts[v], (0, 0, 0)) for v in cap_ids) / S
res_r = sum(math.dist(res[v], (0, 0, 0)) for v in cap_ids) / S
check("cap moved outward toward bump", res_r > src_r + 0.05, f"{src_r:.3f} -> {res_r:.3f}")
check("filename says _regional_to_",
      "_regional_to_" in resp.headers.get("Content-Disposition", ""),
      resp.headers.get("Content-Disposition", ""))

# ---- 3. validations ----
print("[3] validations")
try:
    http("POST", f"/api/sessions/{sid}/region/preview",
         {"waypoints": waypoints, "seed": waypoints[0], "feather_width": 0.5})
    check("seed-on-boundary -> 400", False, "unexpected success")
except urllib.error.HTTPError as e:
    check("seed-on-boundary -> 400", e.code == 400, str(e.code))

try:
    # open 'loop' of 3 far-apart collinear-ish verts won't separate from this seed deep inside
    http("POST", f"/api/sessions/{sid}/region/preview",
         {"waypoints": [sidx(1, 0), sidx(1, 1), sidx(1, 2)], "seed": sidx(R - 1, 0),
          "feather_width": 0.5})
    check("non-separating loop -> 400", False, "unexpected success")
except urllib.error.HTTPError as e:
    detail = e.read().decode()
    check("non-separating loop -> 400", e.code == 400, f"{e.code}: {detail[:80]}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
