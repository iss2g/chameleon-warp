"""
Headless test for refit.build_conform_field / open_boundary_vertices.

A garment's grip interface = the open boundaries of its mesh. We verify the
auto-built conform field:

  * a flat patch's open boundary is exactly its perimeter ring;
  * the "accessory" (seam) preset -> 1 on that perimeter, feathering to 0 into
    the interior (grip the seam, free the bulk);
  * the "skintight" (full) preset -> 1 everywhere;
  * a closed mesh (tetrahedron) has no open boundary -> the seam field is all
    zeros (nothing grips; caller falls back to pins).

Run:  .venv\\Scripts\\python.exe tests\\refit_field_smoke.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import refit                                       # noqa: E402

N = 11
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


# ---- flat patch -----------------------------------------------------------
xs = np.linspace(-1.0, 1.0, N)
verts = np.array([(xs[i], xs[j], 0.0) for i in range(N) for j in range(N)], dtype=float)
faces = []
vid = lambda i, j: i * N + j
for i in range(N - 1):
    for j in range(N - 1):
        a, b, c, d = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
        faces += [(a, b, c), (b, d, c)]
faces = np.array(faces, dtype=np.int64)
n = len(verts)

print("[1] open boundary of a patch = its perimeter")
bnd = set(refit.open_boundary_vertices(faces, n).tolist())
perimeter = set(vid(i, j) for i in range(N) for j in range(N)
                if i in (0, N - 1) or j in (0, N - 1))
check("open_boundary_vertices == perimeter ring", bnd == perimeter,
      f"{len(bnd)} vs {len(perimeter)}")

print("[2] accessory (seam) preset field")
field = refit.build_conform_field(verts, faces, refit.PRESETS["accessory"])
per = np.array(sorted(perimeter))
interior = np.setdiff1d(np.arange(n), per)
centre = vid(N // 2, N // 2)
near = vid(1, N // 2)          # interior vertex one ring in from the top edge
check("field in [0,1]", field.min() >= 0.0 and field.max() <= 1.0,
      f"[{field.min():.3f},{field.max():.3f}]")
check("seam (perimeter) weights are 1", np.allclose(field[per], 1.0),
      f"min perim {field[per].min():.3f}")
check("deep interior (centre) has ~0 weight", field[centre] < 0.05,
      f"centre w = {field[centre]:.3f}")
check("near-seam interior > deep interior (feather decays inward)",
      field[near] > field[centre] + 0.1,
      f"near {field[near]:.3f} vs centre {field[centre]:.3f}")

print("[3] skintight (full) preset field")
full = refit.build_conform_field(verts, faces, refit.PRESETS["skintight"])
check("skintight field is all ones", np.allclose(full, 1.0))

# ---- closed mesh (tetrahedron) -------------------------------------------
print("[4] closed mesh has no open boundary -> zero seam field")
tverts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float)
tfaces = np.array([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], dtype=np.int64)
tbnd = refit.open_boundary_vertices(tfaces, 4)
check("tetrahedron has no open boundary", len(tbnd) == 0, f"{len(tbnd)} boundary verts")
tfield = refit.build_conform_field(tverts, tfaces, refit.PRESETS["accessory"])
check("seam field on a closed mesh is all zeros", np.allclose(tfield, 0.0))

# ---- proximity gate (double-wall standoff) -------------------------------
# Two disjoint quads standing in for a double-wall collar: an INNER wall near
# the body and an OUTER wall standing off from it. Both are open boundaries, so
# the ungated field grips both — collapsing the outer wall onto the body. The
# proximity gate should grip ONLY the near wall and leave the far one free.
print("[5] proximity gate drops the far (outer) wall's grip")
dv = np.array([
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),        # near wall  (verts 0-3)
    (0, 0, 5), (1, 0, 5), (1, 1, 5), (0, 1, 5),        # far wall   (verts 4-7)
], dtype=float)
dfaces = np.array([(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)], dtype=np.int64)
# distance-to-body: near wall ~0.1, far wall ~5.0.
dbody = np.array([0.1, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0, 5.0], dtype=float)
preset = refit.PRESETS["accessory"]

ungated = refit.build_conform_field(dv, dfaces, preset)
check("ungated: both walls grip (all 8 verts = 1)", np.allclose(ungated, 1.0),
      f"sum {ungated.sum():.1f}")

gated = refit.build_conform_field(dv, dfaces, preset, body_dist=dbody, seam_depth=1.0)
check("gated: near wall grips (verts 0-3 = 1)", np.allclose(gated[:4], 1.0),
      f"near {gated[:4]}")
check("gated: far wall free (verts 4-7 ~ 0)", gated[4:].max() < 0.05,
      f"far max {gated[4:].max():.3f}")

# Fallback: if the gate would drop everything (garment placed far from the
# body), keep the full boundary so the fit still has something to grip.
allfar = np.full(8, 9.0)
fb = refit.build_conform_field(dv, dfaces, preset, body_dist=allfar, seam_depth=1.0)
check("gate empties -> fall back to full boundary", np.allclose(fb, 1.0),
      f"sum {fb.sum():.1f}")

# ---- L3: auto layer mode round-trip --------------------------------------
# build_refit_field(layer_mode="auto") must attach a per-vertex layer id
# (len == n_verts, values in {0,1,2}) and layer_stats; legacy mode leaves both
# None. This is exactly what /refit_field serializes into the response.
print("[6] layer_mode='auto' attaches a per-vertex layer id + stats")
import math                                                    # noqa: E402
from dt_core import meshlib                                    # noqa: E402


def _cyl(radius=1.0, z0=-1.5, z1=1.5, S=20, R=16):
    zs = np.linspace(z0, z1, R)
    v = [(radius * math.cos(2 * math.pi * j / S), radius * math.sin(2 * math.pi * j / S), z)
         for z in zs for j in range(S)]
    vid_ = lambda i, j: i * S + (j % S)
    f = []
    for i in range(R - 1):
        for j in range(S):
            a, b, c, d = vid_(i, j), vid_(i, j + 1), vid_(i + 1, j), vid_(i + 1, j + 1)
            f += [(a, b, d), (a, d, c)]
    bot, top = len(v), len(v) + 1
    v += [(0, 0, z0), (0, 0, z1)]
    for j in range(S):
        f += [(bot, vid_(0, j + 1), vid_(0, j)), (top, vid_(R - 1, j), vid_(R - 1, j + 1))]
    return meshlib.Mesh(vertices=np.array(v, float), faces=np.array(f, np.int64))


def _double_collar(r_in=1.06, gap=0.14, S=20, rows_in=6, rows_out=4):
    ring = lambda r, z: [(r * math.cos(2 * math.pi * j / S), r * math.sin(2 * math.pi * j / S), z)
                         for j in range(S)]
    v, rows = [], []
    for z in np.linspace(-0.8, 0.5, rows_in):
        rows.append(len(v)); v += ring(r_in, z)
    rows.append(len(v)); v += ring(r_in + gap / 2, 0.5 + gap / 2)
    for z in np.linspace(0.5, 0.0, rows_out):
        rows.append(len(v)); v += ring(r_in + gap, z)
    f = []
    for a, b in zip(rows[:-1], rows[1:]):
        for j in range(S):
            p, q = a + j, a + (j + 1) % S
            r_, s_ = b + j, b + (j + 1) % S
            f += [(p, q, s_), (p, s_, r_)]
    inner = np.arange(0, rows_in * S)
    outer = np.arange((rows_in + 1) * S, len(v))
    return (meshlib.Mesh(vertices=np.array(v, float), faces=np.array(f, np.int64)),
            inner, outer)

_body = _cyl()
_collar, _inner, _outer = _double_collar()
rf_auto = refit.build_refit_field(_collar, _body, "cloth", layer_mode="auto")
nv = len(_collar.vertices)
check("auto: layer present", rf_auto.layer is not None)
check("auto: layer length == n_verts",
      rf_auto.layer is not None and len(rf_auto.layer) == nv,
      f"{None if rf_auto.layer is None else len(rf_auto.layer)} vs {nv}")
check("auto: layer values in {0,1,2}",
      rf_auto.layer is not None and set(np.unique(rf_auto.layer).tolist()) <= {0, 1, 2},
      f"{None if rf_auto.layer is None else set(np.unique(rf_auto.layer).tolist())}")
check("auto: layer_stats mode == auto",
      (rf_auto.layer_stats or {}).get("mode") == "auto", f"{rf_auto.layer_stats}")
check("auto: outer wall classified follower (layer >= 1)",
      rf_auto.layer is not None and (rf_auto.layer[_outer] >= 1).mean() > 0.8,
      f"{(rf_auto.layer[_outer] >= 1).mean():.2f}" if rf_auto.layer is not None else "n/a")
check("auto: inner wall stays driven (layer 0)",
      rf_auto.layer is not None and (rf_auto.layer[_inner] == 0).mean() > 0.8,
      f"{(rf_auto.layer[_inner] == 0).mean():.2f}" if rf_auto.layer is not None else "n/a")

rf_legacy = refit.build_refit_field(_collar, _body, "cloth", layer_mode="legacy")
check("legacy: layer is None (no auto classification)", rf_legacy.layer is None)
check("legacy: layer_stats reports the mode only",
      (rf_legacy.layer_stats or {}).get("mode") == "legacy"
      and "driven" not in (rf_legacy.layer_stats or {}),
      f"{rf_legacy.layer_stats}")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
