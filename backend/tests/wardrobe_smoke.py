"""
Headless test for the proxy wardrobe (refit feature D, dt_core.wardrobe).

A garment fitted to a basemesh (proxy) is bound to the proxy surface once, then
transported through the basemesh->body deformation (a wrap result) and cleaned
up with a short refit.

Scenarios:
  1. Binding round-trip — bind a band to the undeformed proxy, apply the
     binding back on the SAME proxy: the garment is reproduced to < 1e-9.
  2. Transport — deform the proxy sphere into an ellipsoid; the transported
     band sits on the ellipsoid at the same latitude with the standoff
     preserved (offset is not stretch-aware, so within 30%).
  3. End-to-end — transported band + a short refit against the ellipsoid ends
     penetration-free and coherent (p95 edge stretch < 10%).

Run:  .venv\\Scripts\\python.exe tests\\wardrobe_smoke.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, refit                             # noqa: E402
from dt_core.collision import signed_offset                    # noqa: E402
from dt_core.surface import build_surface                      # noqa: E402
from dt_core.wardrobe import bind_garment, apply_binding       # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def make_uv_sphere(radius=1.0, S=48, R=28):
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
    v = np.array(verts, float); f = np.array(faces, np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    cent = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3.0
    inward = np.einsum('ij,ij->i', n, cent) < 0
    f[inward] = f[inward][:, ::-1]
    return meshlib.Mesh(vertices=v, faces=f)


def make_band(radius=1.05, th0=80.0, th1=100.0, S=40, R=5):
    """Open tube band around the equator, at `radius` (a garment at standoff)."""
    thetas = np.radians(np.linspace(th0, th1, R))
    verts = [(radius * math.sin(th) * math.cos(2 * math.pi * j / S),
              radius * math.sin(th) * math.sin(2 * math.pi * j / S),
              radius * math.cos(th))
             for th in thetas for j in range(S)]
    vid = lambda i, j: i * S + (j % S)
    faces = []
    for i in range(R - 1):
        for j in range(S):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j), vid(i + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    return meshlib.Mesh(vertices=np.array(verts, float),
                        faces=np.array(faces, np.int64))


def p95_edge_stretch(v0, v1, faces):
    e = np.unique(np.sort(np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    l0 = np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(v1[e[:, 0]] - v1[e[:, 1]], axis=1)
    good = l0 > 1e-9
    return float(np.percentile(np.abs(l1[good] - l0[good]) / l0[good], 95))


def main():
    proxy = make_uv_sphere(1.0)
    proxy_surf = build_surface(proxy.vertices, proxy.faces)
    band = make_band(1.05)
    standoff = 0.05

    # ---------------------------------------------------------------
    # 1. Binding round-trip.
    # ---------------------------------------------------------------
    print("[1] binding round-trip on the undeformed proxy")
    binding = bind_garment(band.vertices, proxy_surf)
    back = apply_binding(binding, proxy.vertices, proxy.faces)
    err = float(np.abs(back - band.vertices).max())
    check("garment reproduced to < 1e-9", err < 1e-9, f"max err {err:.2e}")
    check("offsets ~ standoff", abs(float(binding.offset.mean()) - standoff) < 0.01,
          f"mean offset {binding.offset.mean():.4f}")

    # ---------------------------------------------------------------
    # 2. Transport onto a deformed proxy (ellipsoid).
    # ---------------------------------------------------------------
    print("[2] transport onto an ellipsoid (proxy scaled 1.4, 1.0, 0.8)")
    scale = np.array([1.4, 1.0, 0.8])
    ellip_v = proxy.vertices * scale
    ellip = meshlib.Mesh(vertices=ellip_v, faces=proxy.faces)
    ellip_surf = build_surface(ellip.vertices, ellip.faces)
    transported = apply_binding(binding, ellip_v, proxy.faces)

    _, _, s = signed_offset(transported, ellip_surf)
    rel = np.abs(s.mean() - standoff) / standoff
    check("standoff preserved within 30%", rel < 0.30,
          f"mean standoff {s.mean():.4f} vs {standoff} ({100 * rel:.0f}% off)")
    # Same latitude: the equator band stays near z=0 on the ellipsoid.
    check("stays at the same latitude (band near equator)",
          np.abs(transported[:, 2]).max() < 0.25,
          f"max |z| {np.abs(transported[:, 2]).max():.3f}")

    # ---------------------------------------------------------------
    # 3. End-to-end: transport + short cleanup refit.
    # ---------------------------------------------------------------
    print("[3] transported band + short cleanup refit against the ellipsoid")
    res = refit.run_refit(band, ellip, "skintight", iterations=6,
                          placed_verts=transported)
    _, _, sr = signed_offset(res.verts, ellip_surf)
    pen = int((sr < -0.005).sum())
    st = p95_edge_stretch(transported, res.verts, band.faces)
    check("finite", bool(np.isfinite(res.verts).all()))
    check("penetration-free after cleanup", pen == 0, f"{pen} verts")
    check("band stays coherent (p95 edge stretch < 10%)", st < 0.10,
          f"{100 * st:.1f}%")

    print()
    if failures:
        print(f"FAILURES: {failures}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
