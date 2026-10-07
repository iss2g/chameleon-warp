"""
Headless test for the parallelized signal-A vote path (layers._votes_parallel).

The auto layer classifier's cost is body_first_hit_votes; it is split across
processes for speed. Because rays are independent and votes accumulate additively
(+1.0 on garment-triangle verts), the multi-process result MUST be bitwise
identical to the single-process one. This pins that invariant AND exercises real
process spawn (so a Windows spawn/pickling regression is caught here, not in prod).

MUST stay guarded by `if __name__ == "__main__"`: forcing the parallel path spawns
child processes which re-import this module — without the guard that fork-bombs.

Run:  .venv\\Scripts\\python.exe tests\\layer_parallel_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import layers                                   # noqa: E402
from dt_core.surface import build_surface                    # noqa: E402


def uv_sphere(radius=1.0, S=48, R=24):
    """Closed UV sphere, outward winding; returns (verts, faces)."""
    verts = []
    for i in range(R + 1):
        theta = math.pi * i / R
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


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  - ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    # Body inside, garment shell outside - rays from the body along its outward
    # normals hit the garment (first crossing = driven). Sphere centered at the
    # origin, so the exact outward normal is the radial direction (v/|v|); using
    # that avoids get_vertex_normals' pole degeneracies and pins ray direction.
    # Enough body verts to clear the spawn threshold once we drop it below.
    bv, bf = uv_sphere(radius=1.0, S=60, R=40)
    gv, gf = uv_sphere(radius=1.15, S=48, R=24)
    body_surf = build_surface(bv, bf)
    gsurf = build_surface(gv, gf)
    bvn = bv / np.linalg.norm(bv, axis=1, keepdims=True)
    n_g = len(gv)
    keep = np.ones(len(bv), dtype=bool)
    cap = 0.6
    args = (bv, bvn, gsurf, body_surf, cap, n_g, keep, 15.0, 4)

    print(f"[setup] body verts={len(bv)}  garment verts={n_g}")

    # Serial reference.
    d1, f1 = layers.body_first_hit_votes(bv, bvn, gsurf, body_surf, cap, n_g,
                                         keep_origin=keep, cone_deg=15.0, n_cone=4)
    check("[1] serial votes are non-trivial",
          d1.sum() > 0, f"driven_sum={d1.sum():.0f} follower_sum={f1.sum():.0f}")

    # Force the parallel path on this small mesh by dropping the spawn threshold,
    # then run with 4 workers. Restore the threshold afterward.
    saved = layers._MP_MIN_ORIGINS
    layers._MP_MIN_ORIGINS = 0
    try:
        dP, fP = layers._votes_parallel(*args, n_workers=4)
    finally:
        layers._MP_MIN_ORIGINS = saved

    check("[2] parallel driven votes bitwise-identical to serial",
          np.array_equal(d1, dP),
          f"max_abs_diff={np.abs(d1 - dP).max():.3g}")
    check("[3] parallel follower votes bitwise-identical to serial",
          np.array_equal(f1, fP),
          f"max_abs_diff={np.abs(f1 - fP).max():.3g}")

    # Below-threshold call must transparently fall back to serial (identical).
    dS, fS = layers._votes_parallel(*args, n_workers=4)   # threshold restored -> serial
    check("[4] below-threshold falls back to serial (identical)",
          np.array_equal(d1, dS) and np.array_equal(f1, fS))

    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
