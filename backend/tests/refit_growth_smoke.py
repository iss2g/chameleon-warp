"""
Headless test for growth continuation (refit feature B).

Growth solves refit against a SEQUENCE of eroded bodies growing to the true
shape, so a deeply-penetrating placement (a garment starting mostly INSIDE the
body) tracks the body outward with local, side-correct closest points instead
of tearing.

Scenarios:
  1. Deep penetration — an open cylinder "shirt" placed inside a larger sphere
     (~every vertex starts inside). growth_steps=3 must end penetration-free
     and coherent (p95 edge stretch vs placed within bound), engage 3 stages,
     and be no worse than the single-body solve (K=1, printed for the record).
  2. No-penetration shortcut — the same shirt placed OUTSIDE the body with
     growth_steps=4 must auto-collapse to 1 stage (nothing to grow through).
  3. Real asset (optional) — if testfit/ is present, scale the garment down
     until it penetrates and assert K=3 seats it penetration-free.

Run:  .venv\\Scripts\\python.exe tests\\refit_growth_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, refit                             # noqa: E402
from dt_core.collision import signed_offset                    # noqa: E402
from dt_core.surface import build_surface                      # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_uv_sphere(radius=1.3, S=40, R=24):
    verts, faces = [], []
    for i in range(1, R):
        phi = math.pi * i / R
        for j in range(S):
            th = 2 * math.pi * j / S
            verts.append((radius * math.sin(phi) * math.cos(th),
                          radius * math.sin(phi) * math.sin(th),
                          radius * math.cos(phi)))
    north = len(verts); verts.append((0, 0, radius))
    south = len(verts); verts.append((0, 0, -radius))
    vid = lambda i, j: (i - 1) * S + (j % S)
    for i in range(1, R - 1):
        for j in range(S):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    for j in range(S):
        faces += [(north, vid(1, j + 1), vid(1, j))]
        faces += [(south, vid(R - 1, j), vid(R - 1, j + 1))]
    v = np.array(verts, float)
    f = np.array(faces, np.int64)
    # Orient every face outward (star-shaped about the origin): flip any face
    # whose normal points back toward the centre, so signed_offset's sign is
    # correct (outside > 0, inside < 0).
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    cent = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    inward = np.einsum('ij,ij->i', n, cent) < 0
    f[inward] = f[inward][:, ::-1]
    return meshlib.Mesh(vertices=v, faces=f)


def make_open_cylinder(radius=1.0, z0=-0.8, z1=0.8, S=40, R=12):
    """Open (uncapped) cylinder around the z-axis — a tube 'shirt'."""
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
    return meshlib.Mesh(vertices=np.array(verts, float), faces=np.array(faces, np.int64))


def edge_stretch(placed, result, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    l0 = np.linalg.norm(placed[e[:, 0]] - placed[e[:, 1]], axis=1)
    l1 = np.linalg.norm(result[e[:, 0]] - result[e[:, 1]], axis=1)
    good = l0 > 1e-9
    return np.abs(l1[good] - l0[good]) / l0[good]


def main():
    body = make_uv_sphere(1.3)
    surf = build_surface(body.vertices, body.faces)

    # ---------------------------------------------------------------
    # 1. Deep penetration — shirt inside the sphere.
    # ---------------------------------------------------------------
    print("[1] deep penetration — cylinder shirt inside a larger sphere")
    shirt = make_open_cylinder(1.0)          # radius 1.0 < sphere 1.3 -> inside
    _, _, s0 = signed_offset(shirt.vertices, surf)
    frac_in = float((s0 < 0).mean())
    print(f"  INFO  placed: {100 * frac_in:.0f}% of shirt verts start inside the body")

    res1 = refit.run_refit(shirt, body, "skintight", iterations=9, growth_steps=1)
    res3 = refit.run_refit(shirt, body, "skintight", iterations=9, growth_steps=3)
    _, _, s1 = signed_offset(res1.verts, surf)
    _, _, s3 = signed_offset(res3.verts, surf)
    pen1 = int((s1 < -0.005).sum())
    pen3 = int((s3 < -0.005).sum())
    st1 = np.percentile(edge_stretch(shirt.vertices, res1.verts, shirt.faces), 95)
    st3 = np.percentile(edge_stretch(shirt.vertices, res3.verts, shirt.faces), 95)
    print(f"  INFO  K=1: penetrating={pen1}  p95 stretch={100 * st1:.0f}%")
    print(f"  INFO  K=3: penetrating={pen3}  p95 stretch={100 * st3:.0f}%")

    check("K=3 engaged 3 growth stages", res3.stats["growth"]["steps"] == 3,
          f"steps={res3.stats['growth']['steps']} e0={res3.stats['growth']['e0']:.3f}")
    check("K=3 finite", bool(np.isfinite(res3.verts).all()))
    check("K=3 ends penetration-free (s<-0.005 == 0)", pen3 == 0, f"{pen3} verts")
    check("K=3 stays coherent (p95 edge stretch < 60%)", st3 < 0.60,
          f"{100 * st3:.0f}%")
    check("K=3 no worse than K=1 on penetration", pen3 <= pen1,
          f"K3={pen3} K1={pen1}")

    # ---------------------------------------------------------------
    # 2. No-penetration shortcut — placed outside, K auto-collapses to 1.
    # ---------------------------------------------------------------
    print("[2] no penetration — growth auto-collapses to a single stage")
    outside = make_open_cylinder(1.5)        # radius 1.5 > sphere 1.3 -> outside
    res_o = refit.run_refit(outside, body, "skintight", iterations=8, growth_steps=4)
    check("collapsed to 1 stage", res_o.stats["growth"]["steps"] == 1,
          f"steps={res_o.stats['growth']['steps']} "
          f"pen_at_start={res_o.stats['growth']['penetrating_at_start']}")
    check("output finite", bool(np.isfinite(res_o.verts).all()))

    # ---------------------------------------------------------------
    # 3. Real asset (optional).
    # ---------------------------------------------------------------
    tf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "..", "testfit")
    tg, bc = os.path.join(tf, "tg.obj"), os.path.join(tf, "body_clean.obj")
    if os.path.exists(tg) and os.path.exists(bc):
        print("[3] real asset — scale the garment down until it penetrates")
        g = meshlib.Mesh.load(tg); b = meshlib.Mesh.load(bc)
        bsurf = build_surface(b.vertices, b.faces)
        # Shrink the garment about its centroid until most of it penetrates.
        c = g.vertices.mean(0)
        pre = np.eye(4); pre[:3, :3] *= 0.6; pre[:3, 3] = c - 0.6 * c
        res = refit.run_refit(g, b, "cloth", iterations=8, growth_steps=3,
                              pre_transform=pre)
        _, _, sr = signed_offset(res.verts, bsurf)
        frac = float((sr < -0.005).mean())
        check("real asset K=3 mostly seated (< 2% penetrating)", frac < 0.02,
              f"{100 * frac:.2f}% penetrating, steps={res.stats['growth']['steps']}")
    else:
        print("[3] real asset — SKIPPED (testfit/ not present)")

    print()
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
