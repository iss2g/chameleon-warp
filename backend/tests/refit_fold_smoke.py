"""
Headless regression test for the double-wall ("collar") case — the one the
original smoke tests could not see.

Body = vertical cylinder (a neck). Garment = a band around it whose top part
folds outward and back DOWN over itself: inner wall near the skin, crease at
the top, outer wall hanging outside — a collar. Placed with a small standoff.

Assertions:
  * layer classification: outer-wall verts are auto-released (occluded), inner
    wall grips — no hand-poked pins needed (improvement 3);
  * the fold SURVIVES the fit: the outer wall keeps (approximately) its rest
    standoff from the inner wall instead of collapsing flat (improvement 1);
  * the inner wall actually grips the body (contact field works, improvement 2);
  * a hanging skirt below the contact band stays hanging (no shrink-wrap).

Also unit-checks arap_bind on a folded strip: rotating the handles must carry
the free fold rigidly (fold angle preserved).

Run:  .venv\\Scripts\\python.exe tests\\refit_fold_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, refit                             # noqa: E402
from dt_core.arap import arap_bind                             # noqa: E402
from dt_core.collision import signed_offset                    # noqa: E402
from dt_core.surface import build_surface                      # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_cylinder(radius=1.0, z0=-2.0, z1=2.0, S=32, R=24) -> meshlib.Mesh:
    """Closed vertical cylinder (capped) with outward normals."""
    zs = np.linspace(z0, z1, R)
    verts = [(radius * math.cos(2 * math.pi * j / S),
              radius * math.sin(2 * math.pi * j / S), z)
             for z in zs for j in range(S)]
    vid = lambda i, j: i * S + (j % S)
    faces = []
    for i in range(R - 1):
        for j in range(S):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    bot, top = len(verts), len(verts) + 1
    verts += [(0.0, 0.0, z0), (0.0, 0.0, z1)]
    for j in range(S):
        faces += [(bot, vid(0, j + 1), vid(0, j)),
                  (top, vid(R - 1, j), vid(R - 1, j + 1))]
    return meshlib.Mesh(vertices=np.array(verts, dtype=float),
                        faces=np.array(faces, dtype=np.int64))


def make_collar(r_in=1.06, gap=0.14, z_bot=-1.0, z_crease=0.6, z_out_bot=0.0,
                S=32, rows_in=9, rows_out=4):
    """Open band around the cylinder with a fold at the top.

    Inner wall: radius r_in, z from z_bot up to z_crease (open boundary at the
    bottom = the "neckline" seam that should grip). Crease at the top, then the
    outer wall comes back down at radius r_in+gap to z_out_bot (its bottom rim
    is the free-hanging open boundary).
    Returns (mesh, inner_idx, outer_idx, crease_idx).
    """
    ring = lambda r, z: [(r * math.cos(2 * math.pi * j / S),
                          r * math.sin(2 * math.pi * j / S), z) for j in range(S)]
    verts, ring_rows = [], []
    zs_in = np.linspace(z_bot, z_crease, rows_in)
    for z in zs_in:                                   # inner wall, bottom -> top
        ring_rows.append(len(verts)); verts += ring(r_in, z)
    # crease ring (halfway out, slightly above)
    ring_rows.append(len(verts)); verts += ring(r_in + gap / 2, z_crease + gap / 2)
    zs_out = np.linspace(z_crease, z_out_bot, rows_out)
    for z in zs_out:                                  # outer wall, top -> bottom
        ring_rows.append(len(verts)); verts += ring(r_in + gap, z)
    faces = []
    for a, b in zip(ring_rows[:-1], ring_rows[1:]):
        for j in range(S):
            p, q = a + j, a + (j + 1) % S
            r_, s_ = b + j, b + (j + 1) % S
            faces += [(p, q, s_), (p, s_, r_)]
    n_in = rows_in * S
    inner_idx = np.arange(0, n_in)
    crease_idx = np.arange(n_in, n_in + S)
    outer_idx = np.arange(n_in + S, len(verts))
    mesh = meshlib.Mesh(vertices=np.array(verts, dtype=float),
                        faces=np.array(faces, dtype=np.int64))
    return mesh, inner_idx, outer_idx, crease_idx


body = make_cylinder()
collar, inner_idx, outer_idx, crease_idx = make_collar()
body_surf = build_surface(body.vertices, body.faces)
GAP = 0.14

print("[1] layer classification — outer wall auto-released, inner grips")
res = refit.run_refit(collar, body, "cloth", iterations=8)
st = res.stats
print(f"  INFO  stats: contact={st['contact_verts']} occluded={st['occluded_verts']} "
      f"seam_grip={st['seam_grip_verts']} bind={st['bind']}")
f_inner = res.conform_field[inner_idx]
f_outer = res.conform_field[outer_idx]
check("inner wall grips (field > 0 on most of it)",
      (f_inner > 0.2).mean() > 0.5, f"gripping frac {(f_inner > 0.2).mean():.2f}")
check("outer wall released (field ~ 0)",
      (f_outer < 0.05).mean() > 0.9, f"released frac {(f_outer < 0.05).mean():.2f}")

print("[2] the fold survives — outer wall keeps its standoff from the body")
_, _, s_after = signed_offset(res.verts, body_surf)
inner_after = s_after[inner_idx].mean()
outer_after = s_after[outer_idx].mean()
print(f"  INFO  mean standoff after fit: inner {inner_after:.3f}, outer {outer_after:.3f} "
      f"(rest gap {GAP})")
check("inner wall pulled to the body (small standoff)",
      0.0 <= inner_after < 0.05, f"{inner_after:.3f}")
check("outer wall stands off by ~ the rest gap (fold not flattened)",
      outer_after > 0.6 * GAP, f"{outer_after:.3f} vs gap {GAP}")
# radial spread of the outer wall bottom rim should stay a ring, not collapse
r_outer = np.linalg.norm(res.verts[outer_idx][:, :2], axis=1)
check("outer wall still a coherent ring (low radial spread)",
      r_outer.std() < 0.05, f"std {r_outer.std():.3f}")

print("[3] hanging cloth stays hanging (contact field, no shrink-wrap)")
# skirt: open cone hanging BELOW the collar, far from the cylinder wall
S2, RR = 32, 6
sk_v, sk_rows = [], []
for i, (r, z) in enumerate(zip(np.linspace(1.05, 1.8, RR), np.linspace(0.5, -2.2, RR))):
    sk_rows.append(len(sk_v))
    sk_v += [(r * math.cos(2 * math.pi * j / S2), r * math.sin(2 * math.pi * j / S2), z)
             for j in range(S2)]
sk_f = []
for a, b in zip(sk_rows[:-1], sk_rows[1:]):
    for j in range(S2):
        p, q, r_, s_ = a + j, a + (j + 1) % S2, b + j, b + (j + 1) % S2
        sk_f += [(p, q, s_), (p, s_, r_)]
skirt = meshlib.Mesh(vertices=np.array(sk_v, dtype=float),
                     faces=np.array(sk_f, dtype=np.int64))
hem = np.arange(sk_rows[-1], len(sk_v))            # bottom rim, far from body
res2 = refit.run_refit(skirt, body, "cloth", iterations=8)
_, _, s_hem_before = signed_offset(skirt.vertices[hem], body_surf)
_, _, s_hem_after = signed_offset(res2.verts[hem], body_surf)
print(f"  INFO  hem standoff: before {s_hem_before.mean():.3f}, after {s_hem_after.mean():.3f}")
check("hem still hangs (kept most of its standoff)",
      s_hem_after.mean() > 0.5 * s_hem_before.mean(),
      f"{s_hem_after.mean():.3f} vs {s_hem_before.mean():.3f}")
top_ring = np.arange(0, S2)                        # near the body -> should grip
_, _, s_top_after = signed_offset(res2.verts[top_ring], body_surf)
check("waist gripped the body", s_top_after.mean() < 0.06, f"{s_top_after.mean():.3f}")

print("[4] FLARED collar — outer wall sees the body past the inner wall")
# Outer wall flares outward and reaches BELOW the inner wall's bottom, so its
# verts have line-of-sight to the cylinder (no occlusion). Only the normal-
# opposition release protects it: the folded sheet's normals face the body.
# Regression for: high-field outer wall, matches pruned by the angle filter,
# smoothness quietly unfolds it, and the bind pass pins the flattened result.
def make_flared_collar(r_in=1.06, z_bot=-0.4, z_crease=0.7, S=32, rows_in=8, rows_out=8):
    ring = lambda r, z: [(r * math.cos(2 * math.pi * j / S),
                          r * math.sin(2 * math.pi * j / S), z) for j in range(S)]
    verts, ring_rows = [], []
    for z in np.linspace(z_bot, z_crease, rows_in):     # inner wall up
        ring_rows.append(len(verts)); verts += ring(r_in, z)
    # flare: radius grows as it comes down past the inner wall's bottom
    for r, z in zip(np.linspace(r_in + 0.06, 1.75, rows_out),
                    np.linspace(z_crease, -0.9, rows_out)):
        ring_rows.append(len(verts)); verts += ring(r, z)
    faces = []
    for a, b in zip(ring_rows[:-1], ring_rows[1:]):
        for j in range(S):
            p, q = a + j, a + (j + 1) % S
            r_, s_ = b + j, b + (j + 1) % S
            faces += [(p, q, s_), (p, s_, r_)]
    inner = np.arange(0, rows_in * S)
    outer = np.arange(rows_in * S, len(verts))
    return (meshlib.Mesh(vertices=np.array(verts, dtype=float),
                         faces=np.array(faces, dtype=np.int64)), inner, outer)

fl, fl_in, fl_out = make_flared_collar()
# distance of the flared rim from the cylinder wall at rest
_, _, s_fl0 = signed_offset(fl.vertices, body_surf)
for preset in ("cloth", "skintight"):
    resf = refit.run_refit(fl, body, preset, iterations=8)
    _, _, s_fl = signed_offset(resf.verts, body_surf)
    keep = s_fl[fl_out].mean() / max(s_fl0[fl_out].mean(), 1e-9)
    print(f"  INFO  [{preset}] outer standoff kept {100*keep:.0f}% "
          f"(inner after {s_fl[fl_in].mean():.3f}); "
          f"opposed={resf.stats['opposed_verts']} occluded={resf.stats['occluded_verts']}")
    check(f"[{preset}] flared outer wall not sucked onto the body",
          keep > 0.55, f"kept {100*keep:.0f}%")
    check(f"[{preset}] inner wall grips", s_fl[fl_in].mean() < 0.06,
          f"{s_fl[fl_in].mean():.3f}")
    # Feature A: the rigidity-consistency filter ran (refit ships it ON) and
    # reported a drop count in the refit stats.
    check(f"[{preset}] rigidity filter reported", "rigidity_dropped" in resf.stats,
          f"dropped={resf.stats.get('rigidity_dropped')}")

print("[5] frozen paint — painted patch keeps its exact shape and rides along")
# Freeze the ENTIRE collar fold (inner top rows + crease + outer wall) of the
# parallel-wall collar; only the un-frozen lower band grips. The frozen patch
# must keep its internal edge lengths verbatim (near-rigid) and no grip.
frz = np.concatenate([inner_idx[-2 * 32:], crease_idx, outer_idx])  # top of collar
resz = refit.run_refit(collar, body, "cloth", iterations=8, frozen=frz)
check("frozen verts have zero grip", float(resz.conform_field[frz].max()) == 0.0,
      f"max field {resz.conform_field[frz].max():.3f}")
# internal edge-length preservation over frozen-frozen edges
fset = np.zeros(len(collar.vertices), bool); fset[frz] = True
e = np.concatenate([collar.faces[:, [0, 1]], collar.faces[:, [1, 2]], collar.faces[:, [2, 0]]])
e = e[fset[e[:, 0]] & fset[e[:, 1]]]
l0 = np.linalg.norm(collar.vertices[e[:, 0]] - collar.vertices[e[:, 1]], axis=1)
l1 = np.linalg.norm(resz.verts[e[:, 0]] - resz.verts[e[:, 1]], axis=1)
stretch = np.abs(l1 / np.maximum(l0, 1e-12) - 1.0)
print(f"  INFO  frozen={len(frz)} verts, {len(e)} internal edges, "
      f"max stretch {100*stretch.max():.2f}%  stats={resz.stats['frozen_verts']}")
check("frozen patch is near-rigid (max edge stretch < 2%)", stretch.max() < 0.02,
      f"{100*stretch.max():.2f}%")
check("un-frozen band still grips the body",
      float(np.abs(signed_offset(resz.verts[inner_idx[:2*32]], body_surf)[2]).mean()) < 0.06)

print("[6] arap_bind — rigid ride-along preserves a fold under handle rotation")
# folded strip in the XZ plane: base [0,1]x{0}, fold up at 45deg
M = 11
xs = np.linspace(0.0, 1.0, M)
strip_v = [(x, y, 0.0) for x in xs[:6] for y in (0.0, 0.2)]
# fold: continues at 45 degrees upward
for i, x in enumerate(xs[6:], start=1):
    d = i * (xs[1] - xs[0])
    strip_v += [(xs[5] + d / math.sqrt(2), y, d / math.sqrt(2)) for y in (0.0, 0.2)]
strip_v = np.array(strip_v, dtype=float)
strip_f = []
for i in range(M - 1):
    a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
    strip_f += [(a, b, d), (a, d, c)]
strip_f = np.array(strip_f, dtype=np.int64)
handles = np.arange(0, 8)                          # first 4 columns = base
th = math.radians(30)
Rz = np.array([[math.cos(th), -math.sin(th), 0],
               [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
hpos = strip_v[handles] @ Rz.T
out, stats = arap_bind(strip_v, strip_f, handles, hpos, iterations=10)
# fold angle between last base segment and first fold segment must survive
def fold_angle(v):
    a = v[10] - v[8]      # along base near the crease
    b = v[14] - v[12]     # along the fold
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(np.clip(np.dot(a, b), -1, 1)))
ang0, ang1 = fold_angle(strip_v), fold_angle(out)
print(f"  INFO  fold angle rest {ang0:.1f} deg -> bound {ang1:.1f} deg; {stats}")
check("output finite", bool(np.all(np.isfinite(out))))
check("handles honored", float(np.abs(out[handles] - hpos).max()) < 1e-9)
check("fold angle preserved within 10 deg", abs(ang1 - ang0) < 10.0,
      f"{ang0:.1f} -> {ang1:.1f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
