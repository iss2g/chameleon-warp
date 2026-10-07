"""
Headless smoke test for the wrap-quality rework (surface matching + robust
pruning + final projection + anti-fold). Imports dt_core directly — no
running server needed.

Scenario: source = dense unit UV sphere. Target = COARSER ellipsoid
(1.25, 0.85, 1.05) with an open hole at the top (faces above z=0.7*rz
removed) — a stand-in for eye sockets / mesh cuts. 9 markers on the equator
and lower hemisphere, none near the hole.

Checks (the three failure modes this rework targets):
  1. no NaN / inf in the result
  2. no flipped (inverted) triangles vs the source orientation
  3. no runaway stretching: max edge-length ratio result/source < 3
  4. tight fit where the target has coverage: >= 70% of vertices end up
     within 1% of the bbox diagonal from the target surface
  5. no edge suction / escape: the cap over the hole stays a dome (max z
     high) and nothing leaves the target bbox by more than 5% of the diagonal

Run:  .venv\\Scripts\\python.exe tests\\wrap_quality_smoke.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dt_core import meshlib                                        # noqa: E402
from dt_core.correspondence import (                               # noqa: E402
    compute_correspondence_with_setup, get_vertex_normals,
    precompute_source, precompute_target,
)
from dt_core.surface import (                                      # noqa: E402
    closest_points_on_surface, damp_folds, project_onto_surface,
)


# ---------------------------------------------------------------- geometry
def uv_sphere(segments: int, rings: int):
    """Unit UV sphere with poles; outward-facing CCW triangles."""
    verts = [(0.0, 0.0, 1.0)]
    for i in range(1, rings):
        theta = math.pi * i / rings
        z = math.cos(theta)
        r = math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append((r * math.cos(phi), r * math.sin(phi), z))
    verts.append((0.0, 0.0, -1.0))
    top, bottom = 0, len(verts) - 1

    def rv(i, j):  # ring vertex index, i in [1, rings-1]
        return 1 + (i - 1) * segments + (j % segments)

    faces = []
    for j in range(segments):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            a, b = rv(i, j), rv(i + 1, j)
            c, d = rv(i + 1, j + 1), rv(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(segments):
        faces.append((bottom, rv(rings - 1, j + 1), rv(rings - 1, j)))
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


def nearest_vertex(verts, point):
    return int(np.argmin(((verts - np.asarray(point)) ** 2).sum(axis=1)))


def flipped_count(v_ref, v_new, faces):
    f = faces[:, :3]

    def raw(v):
        return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])

    return int((np.einsum('ij,ij->i', raw(v_new), raw(v_ref)) < 0).sum())


# ---------------------------------------------------------------- build data
SRC_S, SRC_R = 32, 24
TGT_S, TGT_R = 20, 14
RADII = np.array([1.25, 0.85, 1.05])
HOLE_Z = 0.70  # unit-sphere z above which target faces are removed

sv, sf = uv_sphere(SRC_S, SRC_R)
source = meshlib.Mesh(vertices=sv, faces=sf)

tv, tf = uv_sphere(TGT_S, TGT_R)
tv = tv * RADII
keep = ~(tv[tf].mean(axis=1)[:, 2] > HOLE_Z * RADII[2])
tf = tf[keep]
target = meshlib.Mesh(vertices=tv, faces=tf)

marker_dirs = [
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1),
    (0.7, 0.7, -0.2), (-0.7, 0.7, -0.2), (0.7, -0.7, -0.2), (-0.7, -0.7, -0.2),
]
markers = []
for d in marker_dirs:
    d = np.asarray(d, dtype=float)
    d /= np.linalg.norm(d)
    markers.append((nearest_vertex(sv, d), nearest_vertex(tv, d * RADII)))
markers = np.array(markers, dtype=np.int64)
assert len(np.unique(markers[:, 0])) == len(markers), "duplicate source markers"

print(f"source: {len(sv)} verts / {len(sf)} tris;  "
      f"target: {len(tv)} verts / {len(tf)} tris (open hole);  "
      f"markers: {len(markers)}")

# ---------------------------------------------------------------- pipeline
src_setup = precompute_source(source)
tgt_setup = precompute_target(target)
surf = tgt_setup.surface

warped, _mapping = compute_correspondence_with_setup(
    src_setup, tgt_setup, markers,
    iterations=8, smoothness=1.0, identity_weight=1e-3,
)

# Final projection, same as /wrap does with smooth_result=0.
vn = get_vertex_normals(warped.vertices, warped.faces[:, :3])
v2, snapped = project_onto_surface(warped.vertices, vn, surf, 0.02 * surf.diag)
v2, n_flip_proj = damp_folds(warped.vertices, v2, warped.faces[:, :3])
final = meshlib.Mesh(vertices=v2, faces=warped.faces)

# ---------------------------------------------------------------- checks
failures = []


def check(name, cond, detail):
    status = "ok  " if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


check("finite", bool(np.isfinite(final.vertices).all()),
      "all coordinates finite")

n_flipped = flipped_count(sv, final.vertices, sf)
check("no fold-overs", n_flipped == 0,
      f"{n_flipped} triangles inverted vs source orientation")

edges = np.unique(np.sort(np.concatenate(
    [sf[:, [0, 1]], sf[:, [1, 2]], sf[:, [2, 0]]]), axis=1), axis=0)
l_src = np.linalg.norm(sv[edges[:, 0]] - sv[edges[:, 1]], axis=1)
l_fin = np.linalg.norm(final.vertices[edges[:, 0]] - final.vertices[edges[:, 1]], axis=1)
stretch = float((l_fin / l_src).max())
check("no runaway stretch", stretch < 3.0,
      f"max edge stretch x{stretch:.2f} (limit x3)")

_, _, dists, _ = closest_points_on_surface(final.vertices, surf)
frac_tight = float((dists <= 0.01 * surf.diag).mean())
check("tight fit on covered area", frac_tight >= 0.70,
      f"{frac_tight * 100:.1f}% of verts within 1% of diag "
      f"(snapped {int(snapped.sum())}/{len(sv)})")

cap = sv[:, 2] > 0.85  # source verts pointing at the hole
cap_max_z = float(final.vertices[cap, 2].max())
check("hole cap stays a dome", cap_max_z >= 0.85,
      f"cap max z = {cap_max_z:.3f} (rim sits at z~{HOLE_Z * RADII[2]:.2f})")

lo, hi = target.box()
margin = 0.05 * surf.diag
outside = ((final.vertices < lo - margin) | (final.vertices > hi + margin)).any(axis=1)
check("stays inside target bounds", not outside.any(),
      f"{int(outside.sum())} verts escaped bbox+5% margin")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
