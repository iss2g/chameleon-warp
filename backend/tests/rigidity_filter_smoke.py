"""
Headless test for the rigidity-consistency match filter
(surface.filter_matches_rigidity, refit feature A).

The filter drops surface matches that disagree with their neighbourhood's
local rigid motion — a garment patch captured by the wrong body region, one
collar side gripping while the other hangs. Each such match is individually
plausible (passes the normal / distance tests); only the local coherence of
the displacement field exposes it.

Scenarios:
  1. Wrong-limb contamination — a compact patch of a flat grid is matched to a
     large, jittered sideways offset (a chunk captured by a different limb).
     The patch must be dropped; the rigid remainder must survive.
  2. Global rigid rotation of ALL targets -> the field is rigid everywhere,
     residual ~0 uniformly -> nothing dropped.
  3. Global 10x scale of ALL targets -> residual is large but UNIFORM
     (Kabsch fits rotation only) -> nothing dropped (near-uniform guard).

Run:  .venv\\Scripts\\python.exe tests\\rigidity_filter_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import regions                                     # noqa: E402
from dt_core.surface import filter_matches_rigidity             # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_grid(G=40, spacing=0.1):
    """Flat G x G triangulated grid in the z=0 plane. Returns (verts, faces)."""
    xs, ys = np.meshgrid(np.arange(G) * spacing, np.arange(G) * spacing)
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(G * G)], axis=1)
    vid = lambda i, j: i * G + j
    faces = []
    for i in range(G - 1):
        for j in range(G - 1):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    return verts.astype(float), np.array(faces, dtype=np.int64)


def main():
    G, spacing = 50, 0.1
    verts, faces = make_grid(G, spacing)
    N = len(verts)
    adj = regions.weighted_adjacency(verts, faces)
    m_idx = np.arange(N)                       # every vertex has a match

    # ---------------------------------------------------------------
    # 1. Wrong-limb contamination.
    # ---------------------------------------------------------------
    print("[1] wrong-limb patch — contaminated matches must be dropped")
    rng = np.random.default_rng(1)
    ij = np.arange(N)
    row, col = ij // G, ij % G
    # Compact interior blob (does not touch the grid border).
    patch = (row >= 16) & (row < 24) & (col >= 16) & (col < 24)
    m_cp = verts.copy()                        # identity match everywhere (clean)
    # The wrong limb is a different surface: a big sideways translation plus
    # per-vertex jitter on the order of an edge length (not a clean rigid copy).
    m_cp[patch, 0] += 5.0
    m_cp[patch] += rng.normal(0.0, 0.5 * spacing, size=(int(patch.sum()), 3))

    keep, info = filter_matches_rigidity(verts, adj, m_idx, m_cp)
    dropped = ~keep
    patch_drop = dropped[patch].mean()
    clean_drop = dropped[~patch].mean()
    check(">=90% of the contaminated patch dropped", patch_drop >= 0.90,
          f"{100 * patch_drop:.1f}%  (passes {info['pass_drops']})")
    check("<=2% of clean matches dropped", clean_drop <= 0.02,
          f"{100 * clean_drop:.2f}%")

    # ---------------------------------------------------------------
    # 2. Global rigid rotation — nothing should drop.
    # ---------------------------------------------------------------
    print("[2] global rotation of all targets — field stays rigid")
    theta = 0.7
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                   [np.sin(theta), np.cos(theta), 0.0],
                   [0.0, 0.0, 1.0]])
    m_cp_rot = (verts + np.array([3.0, -2.0, 1.0])) @ Rz.T
    keep_r, info_r = filter_matches_rigidity(verts, adj, m_idx, m_cp_rot)
    check("rotation drops nothing", (~keep_r).sum() == 0,
          f"dropped {int((~keep_r).sum())}  (passes {info_r['pass_drops']})")

    # ---------------------------------------------------------------
    # 3. Global 10x scale — uniform residual, nothing should drop.
    # ---------------------------------------------------------------
    print("[3] global 10x scale of all targets — residual uniform")
    m_cp_scale = verts * 10.0
    keep_s, info_s = filter_matches_rigidity(verts, adj, m_idx, m_cp_scale)
    check("scale drops nothing", (~keep_s).sum() == 0,
          f"dropped {int((~keep_s).sum())}  (passes {info_s['pass_drops']})")

    print()
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
