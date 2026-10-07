"""
Headless test for dt_core.collision (refit stage 2).

Body = closed sphere (radius 1, outward normals). Two checks:

  push_out: a flat garment patch is placed cutting THROUGH the sphere so many
    of its vertices start inside. After resolve_push_out every vertex must sit
    at least `offset` outside the body.

  body_hide_mask: a small cap shell hovers over the north pole. Body vertices
    tucked under it (near the north pole) must be masked hidden; the south pole
    must stay visible.

Run:  .venv\\Scripts\\python.exe tests\\collision_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import collision                                  # noqa: E402
from dt_core.surface import build_surface, closest_points_on_surface  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_sphere(R=12, S=16, radius=1.0):
    """Closed UV sphere; faces re-wound so every triangle normal points out."""
    verts = [(0.0, 0.0, radius)]                       # north pole = 0
    for i in range(1, R):
        th = math.pi * i / R
        for j in range(S):
            ph = 2 * math.pi * j / S
            verts.append((radius * math.sin(th) * math.cos(ph),
                          radius * math.sin(th) * math.sin(ph),
                          radius * math.cos(th)))
    south = len(verts)
    verts.append((0.0, 0.0, -radius))
    verts = np.array(verts, dtype=float)

    def ring(i, j):                                    # i in 1..R-1
        return 1 + (i - 1) * S + (j % S)

    faces = []
    for j in range(S):                                 # north cap
        faces.append((0, ring(1, j), ring(1, j + 1)))
    for i in range(1, R - 1):                           # middle quads
        for j in range(S):
            faces += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)),
                      (ring(i, j), ring(i + 1, j + 1), ring(i, j + 1))]
    for j in range(S):                                 # south cap
        faces.append((south, ring(R - 1, j + 1), ring(R - 1, j)))
    faces = np.array(faces, dtype=np.int64)

    # Re-wind: outward normal has positive dot with the (origin-centred) centroid.
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    cen = (a + b + c) / 3.0
    flip = np.einsum('ij,ij->i', nrm, cen) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return verts, faces


sph_v, sph_f = make_sphere()
body_surf = build_surface(sph_v, sph_f)

# sanity: outward normals (centroid dot normal > 0 for all faces)
a, b, c = sph_v[sph_f[:, 0]], sph_v[sph_f[:, 1]], sph_v[sph_f[:, 2]]
nrm = np.cross(b - a, c - a)
check("sphere normals all point outward", np.all(np.einsum('ij,ij->i', nrm, (a + b + c) / 3) > 0))

# ---------------------------------------------------------------- push_out
print("[1] push_out: garment patch cutting through the body")
M = 11
xs = np.linspace(-0.8, 0.8, M)
gverts = np.array([(xs[i], xs[j], 0.3) for i in range(M) for j in range(M)], dtype=float)
_, _, s0 = collision.signed_offset(gverts, body_surf)
n_inside0 = int((s0 < 0).sum())
OFFSET = 0.05
pushed, remaining = collision.resolve_push_out(gverts, body_surf, offset=OFFSET, rounds=8)
_, _, s1 = collision.signed_offset(pushed, body_surf)
print(f"  INFO  inside before: {n_inside0}/{len(gverts)}; min signed after {s1.min():.4f} "
      f"(offset {OFFSET}); remaining inside {remaining}")
check("patch actually started intersecting the body", n_inside0 > 20, f"{n_inside0} inside")
check("no garment vertex left inside after push_out", remaining == 0)
check("all garment vertices are >= offset outside the body", s1.min() >= OFFSET - 1e-3,
      f"min signed = {s1.min():.4f}")

# ---------------------------------------------------------------- body_hide_mask
print("[2] body_hide_mask: cap over the north pole hides the skin beneath")
K = 9
cs = np.linspace(-0.5, 0.5, K)
cap_v = np.array([(cs[i], cs[j], 1.1) for i in range(K) for j in range(K)], dtype=float)
cap_f = []
cvid = lambda i, j: i * K + j
for i in range(K - 1):
    for j in range(K - 1):
        a_, b_, c_, d_ = cvid(i, j), cvid(i + 1, j), cvid(i, j + 1), cvid(i + 1, j + 1)
        cap_f += [(a_, b_, c_), (b_, d_, c_)]          # CCW from +z -> normal +z (outward/up)
cap_surf = build_surface(cap_v, np.array(cap_f, dtype=np.int64))

hidden = collision.body_hide_mask(sph_v, cap_surf, offset=0.0, max_depth=0.6)
north = 0
south = len(sph_v) - 1
n_hidden = int(hidden.sum())
print(f"  INFO  {n_hidden}/{len(sph_v)} body verts hidden; north hidden={hidden[north]} "
      f"south hidden={hidden[south]}")
check("north-pole skin is hidden under the cap", bool(hidden[north]))
check("south-pole skin stays visible", not bool(hidden[south]))
check("only a minority of the body is hidden (localized under the cap)",
      0 < n_hidden < len(sph_v) // 3, f"{n_hidden} hidden")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
