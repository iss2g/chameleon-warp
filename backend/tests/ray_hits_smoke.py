"""
Headless test for surface.ray_hits_ordered — the L1 ordered ray-hits primitive.

The body-side layer signal needs ORDERED crossings per ray (1st hit = driven,
2nd+ = follower), edge-crossing dedup, a per-ray distance cap, and self-fan
exclusion. Cases below pin each of those.

Run:  .venv\\Scripts\\python.exe tests\\ray_hits_smoke.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core.surface import build_surface, ray_hits_ordered   # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def uv_sphere(radius=1.0, S=48, R=24):
    """Closed UV sphere with outward-consistent winding; returns (verts, faces)."""
    verts = []
    for i in range(R + 1):
        theta = math.pi * i / R                 # 0..pi (pole to pole)
        for j in range(S):
            phi = 2 * math.pi * j / S
            verts.append((radius * math.sin(theta) * math.cos(phi),
                          radius * math.sin(theta) * math.sin(phi),
                          radius * math.cos(theta)))
    vid = lambda i, j: i * S + (j % S)
    faces = []
    for i in range(R):
        for j in range(S):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)


def first_hits(ray_idx, t, tri, n_rays):
    """Per-ray first-hit t (nan where the ray had no hit)."""
    out_t = np.full(n_rays, np.nan)
    out_tri = np.full(n_rays, -1, dtype=np.int64)
    counts = np.zeros(n_rays, dtype=np.int64)
    if len(ray_idx):
        uniq, first = np.unique(ray_idx, return_index=True)
        out_t[uniq] = t[first]
        out_tri[uniq] = tri[first]
        np.add.at(counts, ray_idx, 1)
    return out_t, out_tri, counts


# Two concentric spheres joined into one surf (r=1 inner, r=2 outer).
vi, fi = uv_sphere(1.0)
vo, fo = uv_sphere(2.0)
verts = np.vstack([vi, vo])
faces = np.vstack([fi, fo + len(vi)])
surf = build_surface(verts, faces)

# Rays from the center outward along a spray of directions (avoid grazing
# exactly through shared vertices/edges by using generic directions).
rng = np.random.default_rng(0)
dirs = rng.normal(size=(200, 3))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
origins = np.zeros_like(dirs)
t_max = np.full(len(dirs), 5.0)

print("[1] two concentric spheres — every ray reports 2 ordered hits ~1 then ~2")
ri, t, tri = ray_hits_ordered(origins, dirs, t_max, surf)
_, _, counts = first_hits(ri, t, tri, len(dirs))
check("every ray has exactly 2 hits", bool((counts == 2).all()),
      f"count histogram {np.bincount(counts)}")
# ordering + values: group by ray, check first ~1, second ~2
ft, ftri, _ = first_hits(ri, t, tri, len(dirs))
check("first hit t ~ 1 (inner sphere)", bool(np.nanmax(np.abs(ft - 1.0)) < 0.05),
      f"max |t-1| = {np.nanmax(np.abs(ft - 1.0)):.4f}")
# second hit: the max-t per ray
second = np.full(len(dirs), np.nan)
for r in range(len(dirs)):
    m = ri == r
    if m.sum() == 2:
        second[r] = t[m].max()
check("second hit t ~ 2 (outer sphere)", bool(np.nanmax(np.abs(second - 2.0)) < 0.05),
      f"max |t-2| = {np.nanmax(np.abs(second - 2.0)):.4f}")
# tri ownership: first hit belongs to the inner sphere (tri index < len(fi))
check("first hit owned by inner sphere", bool((ftri < len(fi)).all()))

print("[2] edge dedup — a ray aimed at a shared edge counts one crossing")
# Single triangle pair sharing edge; aim a ray exactly through the shared edge
# midpoint. Two triangles both contain that point -> must dedup to 1 hit.
ev = np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0.001], [0, -1, 0.001]], dtype=float)
ef = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int64)   # share edge 0-1
esurf = build_surface(ev, ef)
eo = np.array([[0.0, 0.0, -1.0]])
ed = np.array([[0.0, 0.0, 1.0]])
eri, et, etri = ray_hits_ordered(eo, ed, np.array([5.0]), esurf)
check("shared-edge crossing dedups to 1 hit", len(eri) == 1, f"got {len(eri)} hits")

print("[3] per-ray cap — t_max=1.5 reports only the inner sphere")
ri3, t3, tri3 = ray_hits_ordered(origins, dirs, np.full(len(dirs), 1.5), surf)
_, _, counts3 = first_hits(ri3, t3, tri3, len(dirs))
check("only inner sphere within cap 1.5", bool((counts3 == 1).all()),
      f"count histogram {np.bincount(counts3)}")
check("capped hits are the inner sphere", bool((tri3 < len(fi)).all()))

print("[4] exclude_verts — a ray leaving a vertex ignores its own fan")
# Origin sits ON an inner-sphere vertex, fired outward. Without exclusion its
# own fan would register as a t~0 hit; with exclusion the first real hit is the
# outer sphere.
vsel = 200                                     # some inner-sphere vertex
p = verts[vsel]
d = p / np.linalg.norm(p)                       # radially outward
o = p[None, :] + d[None, :] * 1e-6              # nudge to avoid t<=t_eps self
excl = np.array([vsel], dtype=np.int64)
ri4, t4, tri4 = ray_hits_ordered(o, d[None, :], np.array([5.0]), surf,
                                 exclude_verts=excl)
check("excluded ray skips its own fan (hits outer sphere ~1)",
      len(ri4) >= 1 and abs(t4.min() - 1.0) < 0.1 and (tri4 >= len(fi)).all(),
      f"hits={len(ri4)} first_t={t4.min() if len(t4) else float('nan'):.3f}")

print("[5] perf — 100k SHORT rays vs ~20k-tri mesh (no assert)")
# Realistic layer-classification geometry: rays leave FROM the surface and reach
# a short distance (a cloth standoff), so the candidate ball stays small. Firing
# long rays from the center would make every ray gather the whole mesh — not the
# regime this primitive runs in.
vb, fb = uv_sphere(1.0, S=140, R=72)            # ~20k tris
bsurf = build_surface(vb, fb)
print(f"  INFO  perf mesh tris = {len(fb)}")
base = rng.normal(size=(100000, 3)); base /= np.linalg.norm(base, axis=1, keepdims=True)
po = base * 0.9                                 # just inside the sphere
pdir = base                                     # radially outward
t0 = time.time()
pri, pt, ptri = ray_hits_ordered(po, pdir, np.full(len(po), 0.3), bsurf)
dt = time.time() - t0
print(f"  INFO  100k short rays (cap 0.3): {dt:.2f}s, {len(pri)} hits")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
