"""
Headless test for the per-vertex CONFORM WEIGHT (refit foundation) — calls the
dt_core math directly (no HTTP).

The conform weight scales the closest-point (shrink-wrap) pull per source
vertex: 1 = stick to the target surface, 0 = no pull (carried only by the
stiffness = identity_weight and smoothness terms). That single knob is what
lets one engine do tight cloth (weight ~1 everywhere) and a beard graft
(weight 1 on the attachment seam, 0 on the free-riding bulk).

Test rig (chosen so the result can't be explained by a rigid slide, and
doesn't depend on how well a convex bump conforms):

    source = flat 11x11 patch at z=0 (normals +z)
    target = flat plane at z=-0.5 (normals +z), directly below
    the 4 corners are FROZEN at their source position (a region anchor)

With the corners pinned at z=0, a pulled interior can only DRAPE down toward
the target — a real deformation, not a rigid translation. So:

  A. conform=1 (pull ON)  -> interior drapes down toward z=-0.5;
  B. conform=0 (pull OFF) -> interior stays flat at z=0 (held by the frozen
     corners), proving weight-0 vertices receive no surface pull;
  C. conform=1 on ONE vertex only -> that vertex dips, far vertices stay flat,
     proving the pull is applied per vertex.

Run:  .venv\\Scripts\\python.exe tests\\refit_conform_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib                                    # noqa: E402
from dt_core.correspondence import (                           # noqa: E402
    RegionSpec, precompute_source, precompute_target,
    compute_correspondence_with_setup,
)

N = 11
TARGET_Z = -0.5
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def grid(z):
    xs = np.linspace(-1.0, 1.0, N)
    verts = np.array([(xs[i], xs[j], z) for i in range(N) for j in range(N)], dtype=float)
    faces = []
    vid = lambda i, j: i * N + j
    for i in range(N - 1):
        for j in range(N - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            faces += [(a, b, c), (b, d, c)]      # CCW from +z -> normal +z
    return meshlib.Mesh(vertices=verts, faces=np.array(faces, dtype=np.int64))


source = grid(0.0)
target = grid(TARGET_Z)
n = len(source.vertices)
no_markers = np.zeros((0, 2), dtype=np.int64)

corners = np.array([0, N - 1, (N - 1) * N, N * N - 1], dtype=np.int64)
editable_mask = np.ones(n, dtype=bool)
editable_mask[corners] = False
region = RegionSpec(frozen_idx=corners, editable_mask=editable_mask,
                    soft_idx=np.array([], dtype=np.int64), soft_w=np.array([], dtype=float))
interior = np.nonzero(editable_mask)[0]

src_setup = precompute_source(source)
tgt_setup = precompute_target(target)


def solve(conform_weight):
    res, _ = compute_correspondence_with_setup(
        src_setup, tgt_setup, no_markers,
        iterations=10, smoothness=1.0, identity_weight=0.001,
        use_closest_point=True, region=region, conform_weight=conform_weight,
    )
    return res.vertices


# ------------------------------------------------------- A: pull ON vs B: OFF
print("[1] conform gate: pull ON (=1) vs OFF (=0)")
on = solve(np.ones(n))
off = solve(np.zeros(n))
check("no NaNs", np.all(np.isfinite(on)) and np.all(np.isfinite(off)))
on_int = on[interior, 2].mean()
off_int = off[interior, 2].mean()
print(f"  INFO  interior mean z: ON {on_int:.4f}  OFF {off_int:.4f}  (target {TARGET_Z})")
check("pull ON: interior drapes toward the target", on_int < -0.30,
      f"interior mean z = {on_int:.4f}")
check("pull OFF: interior stays flat (weight-0 => no pull)", abs(off_int) < 0.03,
      f"interior mean z = {off_int:.4f}")
check("conform weight gates the pull (ON << OFF)", off_int - on_int > 0.30,
      f"ON {on_int:.4f} vs OFF {off_int:.4f}")

# ------------------------------------------------------- C: per-vertex pull
print("[2] per-vertex: conform=1 on the centre only")
centre = (N // 2) * N + (N // 2)
near_corner = 1                        # (0,1): weight-0 vertex adjacent to frozen corner 0
cw = np.zeros(n)
cw[centre] = 1.0
one = solve(cw)
check("no NaNs (single)", np.all(np.isfinite(one)))
print(f"  INFO  centre z {one[centre,2]:.4f}  near-corner z {one[near_corner,2]:.4f}")
check("the single weighted vertex is pulled down", one[centre, 2] < -0.10,
      f"centre z = {one[centre,2]:.4f}")
check("a weight-0 vertex held by the anchor barely moves", abs(one[near_corner, 2]) < 0.10,
      f"near-corner z = {one[near_corner,2]:.4f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
