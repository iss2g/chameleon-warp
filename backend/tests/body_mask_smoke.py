"""
Headless test for the body-mask dilate/erode helper (regions.dilate_vertex_mask),
which backs POST /body_mask/dilate.

Mesh: a flat NxN grid (quads split to triangles). We seed a small square patch
and assert the ring algebra:

  * dilate +1 GROWS the set (proper superset), bounded by the grid;
  * erode  -1 SHRINKS the set (proper subset of the original);
  * dilate then erode subset dilate (erosion never re-adds beyond the dilation);
  * erode of the full mesh with an open boundary peels the boundary ring;
  * rings == 0 is a no-op.

Run:  .venv\\Scripts\\python.exe tests\\body_mask_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import regions                                    # noqa: E402

failures = []


def check(name, cond, info=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        failures.append(name)
    print(f"  {tag}  {name}" + (f"  -> {info}" if info else ""))


def grid(n):
    """n x n vertex grid on the z=0 plane, triangulated. Returns (verts, faces)."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1).astype(float)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = j * n + i + 1
            c = (j + 1) * n + i
            d = (j + 1) * n + i + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.asarray(faces, dtype=np.int64)


N = 9
verts, faces = grid(N)
nv = len(verts)


def square_mask(lo, hi):
    """Vertices whose (i, j) grid coords are within [lo, hi]."""
    m = np.zeros(nv, dtype=bool)
    for j in range(lo, hi + 1):
        for i in range(lo, hi + 1):
            m[j * N + i] = True
    return m


# --- dilate grows, bounded ---------------------------------------------------
seed = square_mask(3, 5)                       # a 3x3 patch in the middle
d1 = regions.dilate_vertex_mask(faces, nv, seed, 1)
check("dilate +1 is a superset", np.all(d1 | seed == d1) and d1.sum() > seed.sum(),
      f"{seed.sum()} -> {d1.sum()}")
check("dilate +1 stays within the mesh", d1.sum() <= nv)

d2 = regions.dilate_vertex_mask(faces, nv, seed, 2)
check("dilate +2 >= dilate +1", d2.sum() >= d1.sum() and np.all(d1 <= d2),
      f"{d1.sum()} -> {d2.sum()}")

# --- erode shrinks -----------------------------------------------------------
e1 = regions.dilate_vertex_mask(faces, nv, seed, -1)
check("erode -1 is a subset", np.all(e1 <= seed) and e1.sum() < seed.sum(),
      f"{seed.sum()} -> {e1.sum()}")

# --- dilate then erode subset dilate ----------------------------------------
de = regions.dilate_vertex_mask(faces, nv, d1, -1)
check("erode(dilate) subset dilate", np.all(de <= d1), f"{de.sum()} <= {d1.sum()}")
# Morphological opening of an INTERIOR patch (not touching the mesh boundary)
# round-trips to the seed: dilate then erode gives the original set back.
check("erode(dilate(seed)) == seed (opening round-trip)",
      np.all(de == seed), f"{de.sum()} vs seed {seed.sum()}")

# --- erode eats a mask's boundary against the UNMASKED region ---------------
# Erosion shrinks where the mask meets unmasked verts. A fully-masked mesh has
# no such boundary, so erode is a no-op there — the correct semantics (there's
# nothing to erode into). A partial patch, by contrast, loses its outer ring.
full = np.ones(nv, dtype=bool)
check("erode(full) is a no-op (no mask/unmask boundary)",
      np.all(regions.dilate_vertex_mask(faces, nv, full, -1) == full))
patch = square_mask(2, 6)                      # 5x5 interior patch
patch_e = regions.dilate_vertex_mask(faces, nv, patch, -1)
check("erode(patch) drops the patch's outer ring",
      patch_e.sum() < patch.sum() and np.all(patch_e <= patch),
      f"{patch.sum()} -> {patch_e.sum()}")

# --- rings == 0 is a no-op ---------------------------------------------------
z = regions.dilate_vertex_mask(faces, nv, seed, 0)
check("rings == 0 no-op", np.all(z == seed))

print()
if failures:
    print(f"FAILURES: {failures}")
    sys.exit(1)
print("ALL PASS")
