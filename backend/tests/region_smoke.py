"""
Headless test for *regional wrap* — calls the dt_core math directly (no HTTP).

Scenario: source = open UV sphere (radius 1). The user draws a boundary loop
at one latitude; the cap above it is editable, everything below is frozen.
Target = a DIFFERENT-resolution sphere with a bump on the +Y pole. We wrap
only the cap onto the bump and assert:

  1. Frozen region (outside the loop) is byte-identical to the source.
  2. The editable cap actually moved outward toward the bump.
  3. The feather makes the seam smooth: interior verts adjacent to the
     boundary barely move; displacement grows toward the cap centre.
  4. regions.py helpers (loop-from-waypoints, flood-fill, feather) agree
     with the hand-computed ring partition.

Run:  .venv\\Scripts\\python.exe tests\\region_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib                                    # noqa: E402
from dt_core import regions                                    # noqa: E402
from dt_core.correspondence import (                           # noqa: E402
    RegionSpec, precompute_source, precompute_target,
    compute_correspondence_with_setup,
)

S, R = 24, 16          # source sphere resolution (open: rings 1..R-1)
BOUNDARY_RING = 6      # latitude index of the boundary loop
FEATHER_WIDTH = 0.8    # world-space feather band width

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# ----------------------------------------------------------- source sphere
def svert(i, j, radius=1.0):
    theta = math.pi * i / R
    phi = 2 * math.pi * j / S
    return (radius * math.sin(theta) * math.cos(phi),
            radius * math.cos(theta),
            radius * math.sin(theta) * math.sin(phi))


def sidx(i, j):
    return (i - 1) * S + j


src_verts = np.array([svert(i, j) for i in range(1, R) for j in range(S)], dtype=float)
src_faces = []
for i in range(R - 2):
    for j in range(S):
        a, b = i * S + j, i * S + (j + 1) % S
        c, d = (i + 1) * S + j, (i + 1) * S + (j + 1) % S
        src_faces += [(a, b, c), (b, d, c)]
src_faces = np.array(src_faces, dtype=np.int64)
source = meshlib.Mesh(vertices=src_verts.copy(), faces=src_faces)
n = len(src_verts)

# ----------------------------------------------------------- target (bump, different res)
S2, R2 = 30, 20
BUMP = 0.35
tgt_verts = []
for i in range(1, R2):
    for j in range(S2):
        x, y, z = svert_t = (
            math.sin(math.pi * i / R2) * math.cos(2 * math.pi * j / S2),
            math.cos(math.pi * i / R2),
            math.sin(math.pi * i / R2) * math.sin(2 * math.pi * j / S2),
        )
        up = max(0.0, y)               # +Y hemisphere
        r = 1.0 + BUMP * up * up       # push the +Y pole outward
        tgt_verts.append((x * r, y * r, z * r))
tgt_verts = np.array(tgt_verts, dtype=float)
tgt_faces = []
for i in range(R2 - 2):
    for j in range(S2):
        a, b = i * S2 + j, i * S2 + (j + 1) % S2
        c, d = (i + 1) * S2 + j, (i + 1) * S2 + (j + 1) % S2
        tgt_faces += [(a, b, c), (b, d, c)]
target = meshlib.Mesh(vertices=tgt_verts, faces=np.array(tgt_faces, dtype=np.int64))

# ----------------------------------------------------------- region via regions.py
print("[1] regions.py helpers")
adj = regions.vertex_adjacency(src_faces, n)
waypoints = [sidx(BOUNDARY_RING, j) for j in range(0, S, 4)]      # 6 clicks on the ring
loop = regions.loop_from_waypoints(src_verts, src_faces, waypoints)
expected_ring = set(sidx(BOUNDARY_RING, j) for j in range(S))
check("loop-from-waypoints recovers the full latitude ring",
      set(loop.tolist()) == expected_ring,
      f"{len(loop)} verts vs {S}")

seed = sidx(1, 0)
interior = regions.flood_fill_interior(adj, loop, seed)
expected_interior = set(sidx(i, j) for i in range(1, BOUNDARY_RING) for j in range(S))
check("flood-fill recovers the cap interior",
      set(interior.tolist()) == expected_interior,
      f"{len(interior)} verts vs {len(expected_interior)}")

soft_w_full = regions.feather_weights(src_verts, src_faces, interior, loop, FEATHER_WIDTH)
keep = soft_w_full > 1e-6
soft_idx = interior[keep]
soft_w = soft_w_full[keep]
check("feather is nonempty and bounded in [0,1]",
      len(soft_idx) > 0 and soft_w.min() >= 0 and soft_w.max() <= 1.0,
      f"{len(soft_idx)} soft verts, w in [{soft_w.min():.2f},{soft_w.max():.2f}]")

frozen_idx = np.setdiff1d(np.arange(n), interior)
editable_mask = np.zeros(n, dtype=bool)
editable_mask[interior] = True
region = RegionSpec(frozen_idx=frozen_idx, editable_mask=editable_mask,
                    soft_idx=soft_idx, soft_w=soft_w)

# ----------------------------------------------------------- markers (guide the cap apex)
cap_top = [sidx(1, j) for j in range(0, S, 8)]
markers = np.array([[v, int(np.argmin(np.linalg.norm(tgt_verts - src_verts[v], axis=1)))]
                    for v in cap_top], dtype=np.int64)

# ----------------------------------------------------------- run regional wrap
print("[2] regional wrap solve")
src_setup = precompute_source(source)
tgt_setup = precompute_target(target)
result, _ = compute_correspondence_with_setup(
    src_setup, tgt_setup, markers,
    iterations=8, smoothness=1.0, identity_weight=0.001,
    use_closest_point=True, region=region,
)
res = result.vertices
check("no NaNs", np.all(np.isfinite(res)))

# 1. frozen region byte-identical
frozen_err = np.abs(res[frozen_idx] - src_verts[frozen_idx]).max()
check("frozen region byte-identical to source", frozen_err < 1e-9,
      f"max abs diff {frozen_err:.2e}")

# 2. cap moved outward toward the bump
def ring_radius(verts, i):
    idx = [sidx(i, j) for j in range(S)]
    return float(np.mean(np.linalg.norm(verts[idx], axis=1)))

src_top, res_top = ring_radius(src_verts, 1), ring_radius(res, 1)
check("editable cap moved outward toward bump", res_top > src_top + 0.05,
      f"top-ring radius {src_top:.3f} -> {res_top:.3f}")

# 3. feather: displacement ramps smoothly from the seam inward to the apex.
#    The boundary ring itself is frozen (disp 0); the interior rings should
#    form a MONOTONE ramp with the seam-adjacent ring held close to source
#    (feather) and the apex free to follow the bump — no crease at the seam.
disp = {i: float(np.mean(np.linalg.norm(
            res[[sidx(i, j) for j in range(S)]] - src_verts[[sidx(i, j) for j in range(S)]],
            axis=1))) for i in range(1, BOUNDARY_RING)}
print("  INFO  per-ring displacement (apex->seam): "
      + " ".join(f"r{i}:{disp[i]:.4f}" for i in range(1, BOUNDARY_RING)))
seam_disp, apex_disp = disp[BOUNDARY_RING - 1], disp[1]
monotone = all(disp[i] >= disp[i + 1] - 1e-4 for i in range(1, BOUNDARY_RING - 1))
check("displacement is a monotone ramp apex->seam (smooth, no oscillation)", monotone)
check("seam held close to source by feather", seam_disp < 0.06,
      f"ring {BOUNDARY_RING-1} disp {seam_disp:.4f}")
check("transition is gradual (seam << apex)", seam_disp < 0.4 * apex_disp,
      f"seam {seam_disp:.4f} vs apex {apex_disp:.4f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
