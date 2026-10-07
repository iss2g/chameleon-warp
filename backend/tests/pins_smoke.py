"""
Headless test for dt_core.pins — geodesic pin patches on a double wall.

Mesh: a single strip folded 180° (a collar cross-section extruded along x). The
INNER wall (z=0) and OUTER wall (z=0.05) are euclidean-close (0.05 apart) but
geodesically far (you must travel up to the fold at y=1 and back, ~1.0). We
verify:

  * a geodesic patch from an inner-wall vertex covers the inner wall and does
    NOT bleed onto the euclidean-close outer wall (the whole point);
  * a euclidean patch of the same radius WOULD wrongly catch the outer wall
    (documents why geodesic matters);
  * apply_pins: an outer poke RELEASES grip there, an inner poke GRIPS.

Run:  .venv\\Scripts\\python.exe tests\\pins_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import pins                                          # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# ---- folded-strip double wall -------------------------------------------
# Cross-section path (y, z): inner wall y:0->1 at z=0, fold at y=1, outer wall
# y:1->0 at z=0.05. Extruded along x = {0, 0.5, 1}.
path = [
    (0.00, 0.00), (0.25, 0.00), (0.50, 0.00), (0.75, 0.00), (1.00, 0.00),  # inner 0-4
    (1.00, 0.05),                                                          # fold   5
    (0.75, 0.05), (0.50, 0.05), (0.25, 0.05), (0.00, 0.05),                # outer 6-9
]
xs = [0.0, 0.5, 1.0]
verts = np.array([(x, y, z) for (y, z) in path for x in xs], dtype=float)
npath, nx = len(path), len(xs)
vid = lambda p, c: p * nx + c
faces = []
for p in range(npath - 1):
    for c in range(nx - 1):
        a, b, cc, d = vid(p, c), vid(p, c + 1), vid(p + 1, c), vid(p + 1, c + 1)
        faces += [(a, b, d), (a, d, cc)]
faces = np.array(faces, dtype=np.int64)

inner_mid = vid(2, 1)   # (0.5, 0.0)   middle of the inner wall, center column
outer_near = vid(7, 1)  # (0.5, 0.05)  euclidean-close (0.05) partner on outer wall
eucl = float(np.linalg.norm(verts[inner_mid] - verts[outer_near]))
print(f"inner_mid={inner_mid} outer_near={outer_near}  euclidean gap={eucl:.3f}")

# radius spans a couple of path steps (0.25 each) but is far below the ~1.05
# geodesic distance around the fold to the outer wall.
radius = 0.4

print("[1] geodesic patch stays on the inner wall")
gp = pins.geodesic_patch(verts, faces, np.array([inner_mid]), radius)
check("weight ~1 at the inner center", gp[inner_mid] > 0.95, f"{gp[inner_mid]:.3f}")
check("inner neighbour is covered", gp[vid(3, 1)] > 0.05, f"{gp[vid(3,1)]:.3f}")
check("euclidean-close OUTER vertex is NOT caught (geodesic)",
      gp[outer_near] < 1e-6, f"outer w = {gp[outer_near]:.4f}")

print("[2] euclidean patch WOULD wrongly catch the outer wall (why geodesic)")
d_eucl = np.linalg.norm(verts - verts[inner_mid], axis=1)
eucl_w = np.clip(1 - d_eucl / radius, 0, 1) ** 2
check("euclidean would grip the outer vertex (this is the bug we avoid)",
      eucl_w[outer_near] > 0.3, f"eucl outer w = {eucl_w[outer_near]:.3f}")

print("[3] apply_pins: outer releases, inner grips")
base = np.ones(len(verts), dtype=float)     # pretend everything is gripped
out = pins.apply_pins(base, verts, faces,
                      inner=np.array([inner_mid]), outer=np.array([outer_near]),
                      radius=radius)
check("outer poke released grip on the outer wall", out[outer_near] < 0.05,
      f"{out[outer_near]:.3f}")
check("inner poke keeps grip on the inner wall", out[inner_mid] > 0.95,
      f"{out[inner_mid]:.3f}")

# a zero base field: inner poke should introduce grip locally, outer stays 0
zero = np.zeros(len(verts), dtype=float)
out2 = pins.apply_pins(zero, verts, faces,
                       inner=np.array([inner_mid]), outer=np.array([outer_near]),
                       radius=radius)
check("inner poke adds grip on a bare field", out2[inner_mid] > 0.95,
      f"{out2[inner_mid]:.3f}")
check("outer wall stays free on a bare field", out2[outer_near] < 1e-6,
      f"{out2[outer_near]:.4f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
