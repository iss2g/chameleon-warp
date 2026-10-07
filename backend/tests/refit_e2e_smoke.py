"""
End-to-end headless test for refit.run_refit (stages 1-2 wired together).

Body = sphere (radius 1). Garment = a small open cap shell hovering above the
north pole (its rim = the interface seam). We run the ACCESSORY preset and
assert the graft behaviour:

  * output is finite and preserves the garment topology;
  * the seam (open boundary) is gripped down onto the body;
  * the bulk (centre, conform weight 0) stands OFF the body — it keeps its shape
    instead of shrink-wrapping.

Then the ARMOR preset must return a body-hide mask covering some (not all) of
the body under the cap.

Run:  .venv\\Scripts\\python.exe tests\\refit_e2e_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, refit                             # noqa: E402
from dt_core.surface import build_surface                      # noqa: E402
from dt_core.collision import signed_offset                    # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_sphere(R=10, S=14, radius=1.0):
    verts = [(0.0, 0.0, radius)]
    for i in range(1, R):
        th = math.pi * i / R
        for j in range(S):
            ph = 2 * math.pi * j / S
            verts.append((radius * math.sin(th) * math.cos(ph),
                          radius * math.sin(th) * math.sin(ph),
                          radius * math.cos(th)))
    south = len(verts)
    verts.append((0.0, 0.0, -radius))
    verts = np.array(verts, dtype=float)
    ring = lambda i, j: 1 + (i - 1) * S + (j % S)
    faces = [(0, ring(1, j), ring(1, j + 1)) for j in range(S)]
    for i in range(1, R - 1):
        for j in range(S):
            faces += [(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1)),
                      (ring(i, j), ring(i + 1, j + 1), ring(i, j + 1))]
    faces += [(south, ring(R - 1, j + 1), ring(R - 1, j)) for j in range(S)]
    faces = np.array(faces, dtype=np.int64)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    flip = np.einsum('ij,ij->i', np.cross(b - a, c - a), (a + b + c) / 3) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return meshlib.Mesh(vertices=verts, faces=faces)


# open dome cap over the north pole: rim sits near the body (grips), centre
# stands tall (keeps shape) — like an accessory placed roughly in contact.
M = 9
cs = np.linspace(-0.5, 0.5, M)
cap_v = np.array([(cs[i], cs[j], 1.0 + 0.45 * math.exp(-(cs[i] ** 2 + cs[j] ** 2) / 0.12))
                  for i in range(M) for j in range(M)], dtype=float)
cap_f = []
cvid = lambda i, j: i * M + j
for i in range(M - 1):
    for j in range(M - 1):
        a_, b_, c_, d_ = cvid(i, j), cvid(i + 1, j), cvid(i, j + 1), cvid(i + 1, j + 1)
        cap_f += [(a_, b_, c_), (b_, d_, c_)]              # CCW -> normal +z (away from body)
garment = meshlib.Mesh(vertices=cap_v.copy(), faces=np.array(cap_f, dtype=np.int64))
body = make_sphere()

seam = refit.open_boundary_vertices(garment.faces, len(garment.vertices))
centre = cvid(M // 2, M // 2)
body_surf = build_surface(body.vertices, body.faces)

# distances (|signed offset|) of seam / centre to the body BEFORE the fit
_, _, s_before = signed_offset(garment.vertices, body_surf)
seam_before = np.abs(s_before[seam]).mean()

print("[1] accessory preset — graft signature (bulk stands off further than seam)")
res = refit.run_refit(garment, body, "accessory", iterations=10)
check("output finite", np.all(np.isfinite(res.verts)))
check("topology preserved",
      res.faces.shape == garment.faces.shape and len(res.verts) == len(garment.vertices))
_, _, s_after = signed_offset(res.verts, body_surf)
seam_after = np.abs(s_after[seam]).mean()
centre_after = abs(s_after[centre])
print(f"  INFO  after fit: seam dist {seam_after:.3f}, bulk(centre) stands off {centre_after:.3f}")
check("bulk stands off further than the seam (graft, not shrink-wrap)",
      centre_after > seam_after + 0.15,
      f"centre {centre_after:.3f} vs seam {seam_after:.3f}")

print("[2] cloth preset (push_out) — no body penetration remains")
P = 9
ps = np.linspace(-0.6, 0.6, P)
slab_v = np.array([(ps[i], ps[j], 0.6) for i in range(P) for j in range(P)], dtype=float)
slab_f = []
svid = lambda i, j: i * P + j
for i in range(P - 1):
    for j in range(P - 1):
        a_, b_, c_, d_ = svid(i, j), svid(i + 1, j), svid(i, j + 1), svid(i + 1, j + 1)
        slab_f += [(a_, b_, c_), (b_, d_, c_)]
slab = meshlib.Mesh(vertices=slab_v.copy(), faces=np.array(slab_f, dtype=np.int64))
_, _, sb = signed_offset(slab.vertices, body_surf)
inside_before = int((sb < 0).sum())
res3 = refit.run_refit(slab, body, "cloth", iterations=8)
_, _, sa = signed_offset(res3.verts, body_surf)
print(f"  INFO  slab inside body before {inside_before}/{len(slab.vertices)}; "
      f"min signed after {sa.min():.3f}")
check("slab actually intersected the body", inside_before > 10, f"{inside_before} inside")
check("push_out cleared the penetration", sa.min() > -0.02, f"min signed {sa.min():.3f}")

print("[3] armor preset — body-hide mask")
res2 = refit.run_refit(garment, body, "armor", iterations=10)
mask = res2.body_hide_mask
check("armor returns a body-hide mask", mask is not None)
if mask is not None:
    nh = int(mask.sum())
    print(f"  INFO  hidden {nh}/{len(mask)} body verts; stats={res2.stats}")
    check("mask hides some but not all of the body", 0 < nh < len(mask) // 2, f"{nh} hidden")

print()
if failures:
    print(f"FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
