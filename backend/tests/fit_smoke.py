"""
End-to-end smoke test for Fit mode (use_closest_point=False) against a
RUNNING backend on 127.0.0.1:8000. No third-party deps — urllib only.

Scenario: "head" = open UV sphere (radius 1). "beard" = a band of the same
sphere offset radially to 1.05 and translated away by (0.3, 0.1, 0) to
simulate a misaligned accessory. Anchors pair the band's top ring with the
matching head vertices.

Checks:
  1. Fit pins anchors EXACTLY onto the head's marker vertices.
  2. The far edge of the band follows ~rigidly and is NOT shrink-wrapped:
     it keeps its radial offset (~1.05) instead of collapsing to 1.0.
  3. Classic Wrap on the same data pulls the far edge toward the surface
     (printed for contrast).
  4. Fit on a source with a marker-less disconnected island -> HTTP 400.
  5. Download filename says "_fitted_to_".

Run:  .venv\\Scripts\\python.exe tests\\fit_smoke.py
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
S = 24   # longitude segments
R = 16   # latitude divisions


# ---------------------------------------------------------------- helpers
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


def sphere_vert(i, j, radius=1.0):
    theta = math.pi * i / R
    phi = 2 * math.pi * j / S
    return (radius * math.sin(theta) * math.cos(phi),
            radius * math.cos(theta),
            radius * math.sin(theta) * math.sin(phi))


def obj_text(verts, faces):
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return "\n".join(lines) + "\n"


def band_faces(n_rings, n_seg):
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_seg):
            a = i * n_seg + j
            b = i * n_seg + (j + 1) % n_seg
            c = (i + 1) * n_seg + j
            d = (i + 1) * n_seg + (j + 1) % n_seg
            faces += [(a, b, c), (b, d, c)]
    return faces


def parse_obj_verts(text):
    out = []
    for line in text.splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            out.append((float(x), float(y), float(z)))
    return out


def dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# ---------------------------------------------------------------- meshes
# Head: open UV sphere, rings 1..R-1 (poles skipped to keep indexing flat).
head_verts = [sphere_vert(i, j) for i in range(1, R) for j in range(S)]
head_faces = band_faces(R - 1, S)


def head_idx(i, j):
    return (i - 1) * S + j


# Beard: rings 8..12, radius 1.05, translated (+0.3, +0.1, 0).
BAND_LO, BAND_HI = 8, 12
T = (0.3, 0.1, 0.0)
beard_verts = [
    tuple(c + t for c, t in zip(sphere_vert(i, j, 1.05), T))
    for i in range(BAND_LO, BAND_HI + 1) for j in range(S)
]
beard_faces = band_faces(BAND_HI - BAND_LO + 1, S)


def beard_idx(i, j):
    return (i - BAND_LO) * S + j


# Anchors: every 3rd vertex of the band's top ring -> matching head vertex.
markers = [
    {"source": beard_idx(BAND_LO, j), "target": head_idx(BAND_LO, j)}
    for j in range(0, S, 3)
]

anchor_ring = [beard_idx(BAND_LO, j) for j in range(S)]
far_ring = [beard_idx(BAND_HI, j) for j in range(S)]


def run_fit_session(extra_verts=(), extra_faces=(), payload=None):
    """Create session, upload meshes (+optional extra island), run wrap/fit."""
    sid = http("POST", "/api/sessions")["session_id"]
    verts = beard_verts + list(extra_verts)
    faces = beard_faces + list(extra_faces)
    upload(sid, "source_ref", "beard.obj", obj_text(verts, faces))
    upload(sid, "target_ref", "head.obj", obj_text(head_verts, head_faces))
    http("PUT", f"/api/sessions/{sid}/markers", {"markers": markers})
    submit_and_wait(http, sid, "wrap", payload)
    resp, raw = http("GET", f"/api/sessions/{sid}/wrap.obj", raw=True)
    return sid, resp, parse_obj_verts(raw.decode())


failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- 1+2: fit
print("[1] Fit (use_closest_point=false, rigidity=1.0)")
sid, resp, fitted = run_fit_session(payload={
    "iterations": 1, "smoothness": 1.0, "identity_weight": 1.0,
    "use_closest_point": False, "align_to_source": False,
})
check("vertex count preserved", len(fitted) == len(beard_verts),
      f"{len(fitted)} vs {len(beard_verts)}")
check("no NaNs", all(all(math.isfinite(c) for c in v) for v in fitted))

anchor_err = max(
    dist(fitted[m["source"]], head_verts[m["target"]]) for m in markers
)
check("anchors pinned exactly", anchor_err < 1e-4, f"max err {anchor_err:.2e}")

# Falloff profile: anchors are pinned at radius 1.00 (a 5% radial squeeze of
# the band), and that squeeze must DECAY with distance from the anchor ring —
# mean ring radius should grow monotonically from ~1.00 back toward 1.05.
# Shrink-wrap would flatten the whole profile to ~1.00 instead.
ring_radii = []
for i in range(BAND_LO, BAND_HI + 1):
    ring = [beard_idx(i, j) for j in range(S)]
    ring_radii.append(sum(dist(fitted[k], (0, 0, 0)) for k in ring) / len(ring))
print(f"  INFO  ring radius profile (anchor -> free edge): "
      + " ".join(f"{r:.4f}" for r in ring_radii))
monotone = all(b >= a - 2e-3 for a, b in zip(ring_radii, ring_radii[1:]))
check("squeeze decays away from anchors", monotone)
far_radius = ring_radii[-1]
check("far edge keeps offset (no shrink-wrap)", far_radius > 1.02,
      f"mean radius {far_radius:.4f} (shrink-wrapped would be ~1.00)")

cd = resp.headers.get("Content-Disposition", "")
check("filename says _fitted_to_", "_fitted_to_" in cd, cd)

# ------------------------------------------------------- 3: wrap contrast
print("[2] Classic Wrap on same data (contrast)")
_, resp_w, wrapped = run_fit_session(payload={
    "iterations": 8, "smoothness": 1.0, "identity_weight": 0.001,
    "use_closest_point": True, "align_to_source": False,
})
wrap_far_radius = sum(dist(v, (0, 0, 0)) for v in (wrapped[k] for k in far_ring)) / len(far_ring)
print(f"  INFO  wrap far-edge mean radius: {wrap_far_radius:.4f} "
      f"(fit kept {far_radius:.4f})")
check("wrap pulls far edge closer to surface than fit", wrap_far_radius < far_radius)
check("wrap filename says _wrapped_to_",
      "_wrapped_to_" in resp_w.headers.get("Content-Disposition", ""))

# ------------------------------------------------- 4: island validation
print("[3] Fit rejects marker-less island")
island_base = len(beard_verts)
island_verts = [(5.0, 5.0, 5.0), (5.1, 5.0, 5.0), (5.0, 5.1, 5.0)]
island_faces = [(island_base, island_base + 1, island_base + 2)]
# The island/marker-coverage check runs inside the worker now, so it surfaces
# as a failed job (RuntimeError from wait_for_job) instead of an HTTP 400.
try:
    run_fit_session(extra_verts=island_verts, extra_faces=island_faces, payload={
        "iterations": 1, "smoothness": 1.0, "identity_weight": 1.0,
        "use_closest_point": False, "align_to_source": False,
    })
    check("island without markers -> job error", False, "request unexpectedly succeeded")
except RuntimeError as e:
    check("island without markers -> job error", "island" in str(e), str(e)[:120])

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
