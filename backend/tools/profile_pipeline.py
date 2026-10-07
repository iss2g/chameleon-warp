"""
Pipeline profiler — find where the time goes before optimizing (docs/ROADMAP.md).

Times each stage of the deformation-transfer pipeline on synthetic meshes so
optimization work (Tier 1+) targets the real hotspots instead of guesses. Runs
entirely in-process against dt_core; no server needed.

Usage (from backend/, with the venv python):
  .venv\\Scripts\\python.exe tools\\profile_pipeline.py --src 60 40 --tgt 70 46 --poses 8

--src / --tgt are (segments rings) of a UV sphere. Bigger = denser. --poses is
how many source poses to transfer in the /run phase.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from tools/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dt_core import meshlib  # noqa: E402
from dt_core.correspondence import (  # noqa: E402
    precompute_source, precompute_target, compute_correspondence_with_setup,
)
from dt_core.transformation import Transformation  # noqa: E402


def uv_sphere(segments: int, rings: int, radii=(1.0, 1.0, 1.0)) -> meshlib.Mesh:
    rx, ry, rz = radii
    V = [(0.0, 0.0, rz)]
    for i in range(1, rings):
        th = math.pi * i / rings
        z, r = math.cos(th), math.sin(th)
        for j in range(segments):
            p = 2 * math.pi * j / segments
            V.append((rx * r * math.cos(p), ry * r * math.sin(p), rz * z))
    V.append((0.0, 0.0, -rz))
    top, bot = 0, len(V) - 1

    def rv(i, j):
        return 1 + (i - 1) * segments + (j % segments)

    F = []
    for j in range(segments):
        F.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            F.append((rv(i, j), rv(i + 1, j), rv(i + 1, j + 1)))
            F.append((rv(i, j), rv(i + 1, j + 1), rv(i, j + 1)))
    for j in range(segments):
        F.append((bot, rv(rings - 1, j + 1), rv(rings - 1, j)))
    return meshlib.Mesh(vertices=np.array(V, dtype=float), faces=np.array(F, dtype=np.int64))


class Timer:
    def __init__(self):
        self.rows = []

    def run(self, label, fn):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        self.rows.append((label, dt))
        print(f"  {label:<34} {dt:8.3f}s")
        return out

    def total(self):
        return sum(dt for _, dt in self.rows)


def nearest(mesh: meshlib.Mesh, point) -> int:
    v = mesh.to_third_dimension(copy=False).vertices
    return int(np.argmin(((v - np.array(point)) ** 2).sum(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, nargs=2, default=[60, 40], metavar=("SEG", "RINGS"))
    ap.add_argument("--tgt", type=int, nargs=2, default=[70, 46], metavar=("SEG", "RINGS"))
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--iters", type=int, default=8)
    args = ap.parse_args()

    source = uv_sphere(*args.src)
    target = uv_sphere(*args.tgt, radii=(1.2, 0.9, 1.05))
    print(f"source: {len(source.vertices)} verts / {len(source.faces)} tris   "
          f"target: {len(target.vertices)} verts / {len(target.faces)} tris   "
          f"poses={args.poses}  iters={args.iters}\n")

    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    markers = np.array(
        [[nearest(source, d), nearest(target, (d[0] * 1.2, d[1] * 0.9, d[2] * 1.05))] for d in dirs],
        dtype=np.int64,
    )

    t = Timer()
    print("[setup]")
    src_setup = t.run("precompute_source", lambda: precompute_source(source))
    tgt_setup = t.run("precompute_target", lambda: precompute_target(target))

    print("[wrap / correspondence]")
    _, mapping = t.run(
        "compute_correspondence (+mapping)",
        lambda: compute_correspondence_with_setup(
            src_setup, tgt_setup, markers, iterations=args.iters),
    )

    print("[run / transfer]")
    transf = t.run("Transformation(setup+factor)",
                   lambda: Transformation(source, target, mapping, smoothness=1.0))
    poses = [uv_sphere(*args.src, radii=(1.0, 1.0, 1.0 - 0.03 * k)) for k in range(args.poses)]

    def transfer_all():
        for p in poses:
            transf(p)
    t.run(f"transfer x{args.poses} poses", transfer_all)

    print(f"\n  {'TOTAL':<34} {t.total():8.3f}s")
    print("\nHotspots (desc):")
    for label, dt in sorted(t.rows, key=lambda r: -r[1]):
        print(f"  {dt / t.total() * 100:5.1f}%  {label}")


if __name__ == "__main__":
    main()
