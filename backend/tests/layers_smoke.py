"""
Headless test for dt_core.layers — the L2/L3 layer auto-segmentation.

Cases 1–4 (L2): the score signals are orientation-independent w.r.t. the
garment and robust to deep penetration. Cases 5–8 (L3): the released mask adds
smoothing, hysteresis and a component policy.

Synthetic rigs are built in-test (like refit_fold_smoke). The core assertion
throughout: inner (driven) wall scores high, outer (follower) wall scores low,
and NOTHING about the result reads garment normals.

Run:  .venv\\Scripts\\python.exe tests\\layers_smoke.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, layers                           # noqa: E402
from dt_core.collision import signed_offset                   # noqa: E402
from dt_core.correspondence import get_vertex_normals         # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_neck(radius=1.0, z0=-2.0, z1=2.0, S=40, R=28) -> meshlib.Mesh:
    """Closed capped cylinder = a neck, outward normals."""
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


def make_double_collar(r_in=1.06, gap=0.14, z_bot=-1.0, z_crease=0.6,
                       z_out_bot=0.0, S=40, rows_in=9, rows_out=6, flare=0.0):
    """Watertight two-layer tube around the neck: inner wall at r_in, crease at
    the top, outer wall coming back down at r_in+gap (flared outward by `flare`
    at its bottom rim). Closed at the crease AND at the two bottom rims via a
    connecting band, so it is a real double wall (not two loose sheets).

    Returns (mesh, inner_idx, outer_idx, crease_idx).
    """
    ring = lambda r, z: [(r * math.cos(2 * math.pi * j / S),
                          r * math.sin(2 * math.pi * j / S), z) for j in range(S)]
    verts, rows = [], []
    for z in np.linspace(z_bot, z_crease, rows_in):        # inner wall up
        rows.append(len(verts)); verts += ring(r_in, z)
    rows.append(len(verts)); verts += ring(r_in + gap / 2, z_crease + gap / 2)  # crease
    zs_out = np.linspace(z_crease, z_out_bot, rows_out)
    rs_out = np.linspace(r_in + gap, r_in + gap + flare, rows_out)
    for r, z in zip(rs_out, zs_out):                       # outer wall down
        rows.append(len(verts)); verts += ring(r, z)
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
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


def score_collar(body, collar, active_frac=None, active=None):
    """Run layer_scores on a placed collar against the body. Active = all verts
    unless a mask is given."""
    pv = collar.vertices
    _cp, _n, bs = signed_offset(pv, body_surf(body))
    body_diag = float(np.linalg.norm(body.vertices.max(0) - body.vertices.min(0)))
    bvn = get_vertex_normals(body.vertices, body.faces)
    if active is None:
        active = np.ones(len(pv), dtype=bool)
    sc = layers.layer_scores(
        pv, collar.faces, body, bs, bvn, active,
        contact_free=0.06 * body_diag, thickness=0.004 * body_diag,
        body_diag=body_diag)
    return sc


_surf_cache = {}


def body_surf(body):
    key = id(body)
    if key not in _surf_cache:
        from dt_core.surface import build_surface
        _surf_cache[key] = build_surface(body.vertices, body.faces)
    return _surf_cache[key]


body = make_neck()

print("[1] double-wall collar, closed shell — inner driven, outer follower")
collar, inner, outer, crease = make_double_collar()
t0 = time.time()
sc = score_collar(body, collar)
print(f"  INFO  scored in {time.time()-t0:.2f}s; e_cls={sc.e_cls:.4f} "
      f"coverage={sc.coverage.mean():.2f}")
print(f"  INFO  inner score mean {sc.score[inner].mean():.3f}, "
      f"outer score mean {sc.score[outer].mean():.3f}")
check("inner wall scores high (> 0.7)", sc.score[inner].mean() > 0.7,
      f"{sc.score[inner].mean():.3f}")
check("outer wall scores low (< 0.3)", sc.score[outer].mean() < 0.3,
      f"{sc.score[outer].mean():.3f}")

print("[2] flipped garment normals — identical scores (never read them)")
flipped = meshlib.Mesh(vertices=collar.vertices.copy(),
                       faces=collar.faces[:, ::-1].copy())
sc_flip = score_collar(body, flipped)
same = np.allclose(sc.score, sc_flip.score, atol=1e-9)
check("scores identical under face inversion", same,
      f"max delta {np.abs(sc.score - sc_flip.score).max():.2e}")
# document the legacy opposed-normal test's failure here
vn = get_vertex_normals(collar.vertices, collar.faces)
vn_flip = get_vertex_normals(flipped.vertices, flipped.faces)
_cp, bn, _s = signed_offset(collar.vertices, body_surf(body))
opp = np.einsum('ij,ij->i', vn, bn) < 0
opp_flip = np.einsum('ij,ij->i', vn_flip, bn) < 0
print(f"  INFO  legacy 'opposed' flips {int((opp != opp_flip).sum())} verts "
      f"under inversion (our scores flip 0)")

print("[3] flared collar — outer wall still follower (zero first-hit votes)")
fcollar, finner, fouter, _ = make_double_collar(flare=0.7)
scf = score_collar(body, fcollar)
print(f"  INFO  flared outer score mean {scf.score[fouter].mean():.3f}")
check("flared inner wall high", scf.score[finner].mean() > 0.7,
      f"{scf.score[finner].mean():.3f}")
check("flared outer wall low (< 0.3)", scf.score[fouter].mean() < 0.3,
      f"{scf.score[fouter].mean():.3f}")

print("[4] submerged placement — erosion restores the ordering")
# Scale the collar down so the inner wall sits INSIDE the neck (deep penetration)
sub = meshlib.Mesh(vertices=collar.vertices * 0.82, faces=collar.faces.copy())
_cp, _n, bs_sub = signed_offset(sub.vertices, body_surf(body))
print(f"  INFO  penetrating frac {(bs_sub < 0).mean():.2f}")
scs = score_collar(body, sub)
print(f"  INFO  e_cls={scs.e_cls:.4f}; inner {scs.score[inner].mean():.3f} "
      f"outer {scs.score[outer].mean():.3f}")
check("erosion engaged (e_cls > 0)", scs.e_cls > 0.0, f"{scs.e_cls:.4f}")
check("submerged inner wall still driven (> 0.6)", scs.score[inner].mean() > 0.6,
      f"{scs.score[inner].mean():.3f}")
check("submerged outer wall still follower (< 0.4)", scs.score[outer].mean() < 0.4,
      f"{scs.score[outer].mean():.3f}")

# =====================================================================
#  L3 — released_mask_auto
# =====================================================================

print("[5] sailor collar — cloth over the shirt-back follows, bare-skin patch driven")
# A flat draped collar needs upward-facing SKIN beneath it to stack against, so
# this case uses a dedicated "shoulders" body: a disk facing +z. The collar is a
# flat disk just above it; a SHIRT sheet sits between skin and collar under the
# BACK half only. Body rays fire +z: under the back they hit the shirt first,
# then the collar (collar there = follower); under the front they hit bare
# collar first (driven).
def make_disk(z, r0, r1, S=48, rings=6, up=True):
    """Flat annular/solid disk at height z, radius r0->r1, normals +z (up)."""
    verts, rows = [], []
    for r in np.linspace(r0, r1, rings):
        rows.append(len(verts))
        verts += [(r * math.cos(2 * math.pi * j / S),
                   r * math.sin(2 * math.pi * j / S), z) for j in range(S)]
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
        for j in range(S):
            p, q = a + j, a + (j + 1) % S
            r_, s_ = b + j, b + (j + 1) % S
            # wind so the normal points +z
            faces += ([(p, r_, s_), (p, s_, q)] if up else [(p, q, s_), (p, s_, r_)])
    return (np.array(verts, dtype=float), np.array(faces, dtype=np.int64),
            np.arange(rows[0], len(verts)))


def make_back_sheet(z, r0, r1, S=48, rings=5):
    """Half (angular [0, pi]) flat annulus at height z, clean (no stray verts)."""
    half = S // 2 + 1
    verts, rows = [], []
    for r in np.linspace(r0, r1, rings):
        rows.append(len(verts))
        verts += [(r * math.cos(math.pi * j / (half - 1)),
                   r * math.sin(math.pi * j / (half - 1)), z) for j in range(half)]
    faces = []
    for a, b in zip(rows[:-1], rows[1:]):
        for j in range(half - 1):
            p, q = a + j, a + (j + 1)
            r_, s_ = b + j, b + (j + 1)
            faces += [(p, q, s_), (p, s_, r_)]
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int64)

sh_v, sh_f, _ = make_disk(0.0, 0.2, 2.4, up=True)
shoulders = meshlib.Mesh(vertices=sh_v, faces=sh_f)
# garment = collar disk (z=0.12, full) + shirt back sheet (z=0.06)
col_v, col_f, col_ids = make_disk(0.12, 0.3, 2.0, up=True)
n_col = len(col_v)
sh_sheet_v, sh_sheet_f = make_back_sheet(0.06, 0.32, 1.95)
g_v = np.vstack([col_v, sh_sheet_v])
g_f = np.vstack([col_f, sh_sheet_f + n_col])
sailor = meshlib.Mesh(vertices=g_v, faces=g_f)
collar_ids = np.arange(0, n_col)
ang = np.arctan2(g_v[collar_ids, 1], g_v[collar_ids, 0])
back_ids = collar_ids[(ang >= 0.3) & (ang <= math.pi - 0.3)]   # over the shirt
front_ids = collar_ids[ang < -0.3]                             # over bare skin

sc5 = score_collar(shoulders, sailor)
active5 = np.ones(len(sailor.vertices), dtype=bool)
rel5, layer5, st5 = layers.released_mask_auto(sc5, sailor.vertices, sailor.faces, active5)
print(f"  INFO  stats {st5}")
print(f"  INFO  back(over-shirt) score {sc5.score[back_ids].mean():.2f} released "
      f"{rel5[back_ids].mean():.2f}; front(bare) score {sc5.score[front_ids].mean():.2f} "
      f"released {rel5[front_ids].mean():.2f}")
check("collar-over-shirt patch mostly follower", rel5[back_ids].mean() > 0.5,
      f"{rel5[back_ids].mean():.2f}")
check("collar-over-bare-skin patch mostly driven", rel5[front_ids].mean() < 0.5,
      f"{rel5[front_ids].mean():.2f}")

print("[6] pouch component — whole disconnected box demoted to layer 2")
pcollar, p_inner, p_outer, _ = make_double_collar()
# a small disconnected box near the body next to the band
bs_box = []
cx, cy, cz = 1.15, 0.0, -0.5
d = 0.06                                         # a small attachment (button/pouch)
corners = [(cx + sx * d, cy + sy * d, cz + sz * d)
           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
base = len(pcollar.vertices)
box_faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
             (2, 3, 7), (2, 7, 6), (0, 4, 5), (0, 5, 1)]
pv = np.vstack([pcollar.vertices, np.array(corners, dtype=float)])
pf = np.vstack([pcollar.faces, np.array(box_faces, dtype=np.int64) + base])
pouch = meshlib.Mesh(vertices=pv, faces=pf)
box_ids = np.arange(base, base + 8)
sc6 = score_collar(body, pouch)
active6 = np.ones(len(pv), dtype=bool)
rel6, layer6, st6 = layers.released_mask_auto(sc6, pv, pf, active6)
print(f"  INFO  stats {st6}; box layers {np.bincount(layer6[box_ids].astype(int) + 0)}")
check("whole pouch box is layer 2", bool((layer6[box_ids] == 2).all()),
      f"box layers set {set(layer6[box_ids].tolist())}")

print("[7] speckle robustness — 5% random score flips wash out under smoothing")
sc7 = score_collar(body, collar)
rel_clean, layer_clean, _ = layers.released_mask_auto(sc7, collar.vertices, collar.faces,
                                                      np.ones(len(collar.vertices), bool))
rng = np.random.default_rng(7)
noisy = sc7.score.copy()
flip = rng.random(len(noisy)) < 0.05
noisy[flip] = 1.0 - noisy[flip]
import dataclasses
sc7n = dataclasses.replace(sc7, score=noisy)
rel_noisy, layer_noisy, _ = layers.released_mask_auto(sc7n, collar.vertices, collar.faces,
                                                      np.ones(len(collar.vertices), bool))
agree = (rel_clean == rel_noisy).mean()
print(f"  INFO  {int(flip.sum())} verts flipped; mask agreement {agree:.3f}")
check("noisy mask ~ identical to clean (>= 0.98)", agree >= 0.98, f"{agree:.3f}")

print("[8] no-double-wall regression — single-wall shirt releases nothing")
# open single-wall cylinder around the neck (no outer layer at all)
def make_single(S=40, rows=10, r=1.06):
    ring = lambda z: [(r * math.cos(2 * math.pi * j / S),
                       r * math.sin(2 * math.pi * j / S), z) for j in range(S)]
    verts, rr = [], []
    for z in np.linspace(-1.0, 1.0, rows):
        rr.append(len(verts)); verts += ring(z)
    faces = []
    for a, b in zip(rr[:-1], rr[1:]):
        for j in range(S):
            p, q = a + j, a + (j + 1) % S
            r_, s_ = b + j, b + (j + 1) % S
            faces += [(p, q, s_), (p, s_, r_)]
    return meshlib.Mesh(vertices=np.array(verts, float), faces=np.array(faces, np.int64))

single = make_single()
sc8 = score_collar(body, single)
rel8, layer8, st8 = layers.released_mask_auto(sc8, single.vertices, single.faces,
                                              np.ones(len(single.vertices), bool))
print(f"  INFO  released frac {rel8.mean():.3f}; stats {st8}")
check("single-wall shirt releases (almost) nothing", rel8.mean() < 0.05,
      f"released frac {rel8.mean():.3f}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
