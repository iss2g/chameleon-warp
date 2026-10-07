"""
Headless test for soft markers (relax when the shrink-wrap weight schedule
reaches its strong phase) in compute_correspondence_with_setup.

The scenario soft markers exist for: a marker lands on the WRONG SIDE of a
thin feature (the "ear" case). Source and target are thin flat ellipsoids;
the bad marker maps a +y-face source vertex to a -y-face target vertex,
so a hard pin drags that vertex straight through the plate. Runs:

  A. accurate markers, hard pins   -> ground truth p_good
  B. accurate markers, soft ON     -> must match A (soft is harmless)
  C. bad marker,       hard pins   -> vertex stuck on the wrong side
  D. bad marker,       soft ON     -> surface term must pull it back out

Run:  .venv\\Scripts\\python.exe tests\\soft_markers_smoke.py   (TQDM_DISABLE=1)
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dt_core import meshlib                                            # noqa: E402
from dt_core.correspondence import (                                   # noqa: E402
    compute_correspondence_with_setup, precompute_source, precompute_target,
)
from dt_core.surface import build_surface, closest_points_on_surface   # noqa: E402


def uv_sphere(segments, rings, radii=(1.0, 1.0, 1.0)):
    rx, ry, rz = radii
    verts = [(0.0, 0.0, rz)]
    for i in range(1, rings):
        th = math.pi * i / rings
        z, r = math.cos(th), math.sin(th)
        for j in range(segments):
            ph = 2.0 * math.pi * j / segments
            verts.append((rx * r * math.cos(ph), ry * r * math.sin(ph), rz * z))
    verts.append((0.0, 0.0, -rz))
    top, bottom = 0, len(verts) - 1

    def rv(i, j):
        return 1 + (i - 1) * segments + (j % segments)

    faces = []
    for j in range(segments):
        faces.append((top, rv(1, j), rv(1, j + 1)))
    for i in range(1, rings - 1):
        for j in range(segments):
            a, b, c, d = rv(i, j), rv(i + 1, j), rv(i + 1, j + 1), rv(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))
    for j in range(segments):
        faces.append((bottom, rv(rings - 1, j + 1), rv(rings - 1, j)))
    return meshlib.Mesh(vertices=np.array(verts, dtype=float),
                        faces=np.array(faces, dtype=int))


def nearest(mesh, point):
    return int(np.argmin(np.linalg.norm(mesh.vertices - np.asarray(point), axis=1)))


failures = []


def check(name, cond, detail):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


SRC_R = np.array([1.2, 0.15, 1.0])
TGT_R = np.array([1.25, 0.12, 1.05])
source = uv_sphere(32, 24, tuple(SRC_R))
target = uv_sphere(28, 20, tuple(TGT_R))

DIRS = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
        (0.7, 0, 0.7), (-0.7, 0, 0.7), (0.7, 0, -0.7), (-0.7, 0, -0.7),
        (0, 1, 0), (0, -1, 0)]
good_markers = []
for d in DIRS:
    d = np.array(d, dtype=float)
    d /= np.linalg.norm(d)
    good_markers.append((nearest(source, d * SRC_R), nearest(target, d * TGT_R)))

# The extra marker: a +y-face source vertex. Good target = same spot on the
# +y face; bad target = the mirrored vertex on the -y face (wrong side).
plus_pt = np.array([0.5, SRC_R[1] * 0.9, 0.3])
good_markers.append((nearest(source, plus_pt),
                     nearest(target, plus_pt * TGT_R / SRC_R)))
good_markers = np.array(good_markers)
bad_markers = good_markers.copy()
bad_markers[-1, 1] = nearest(target, plus_pt * TGT_R / SRC_R * np.array([1, -1, 1]))
v_m = int(good_markers[-1, 0])
check("bad marker is on the other face",
      bad_markers[-1, 1] != good_markers[-1, 1],
      f"target vertex {good_markers[-1, 1]} -> {bad_markers[-1, 1]}")

src_setup = precompute_source(source)
tgt_setup = precompute_target(target)
surf = build_surface(target.vertices, target.faces)


def run(markers, soft_weight):
    mesh, _ = compute_correspondence_with_setup(
        src_setup, tgt_setup, markers,
        iterations=8, smoothness=1.0, identity_weight=0.001,
        compute_mapping=False,
        soft_marker_weight=soft_weight,
    )
    return mesh


mesh_good = run(good_markers, 0.0)       # A
mesh_good_soft = run(good_markers, 10.0)  # B
mesh_hard = run(bad_markers, 0.0)         # C
mesh_soft = run(bad_markers, 10.0)        # D

p_good = mesh_good.vertices[v_m]
err_hard = float(np.linalg.norm(mesh_hard.vertices[v_m] - p_good))
err_soft = float(np.linalg.norm(mesh_soft.vertices[v_m] - p_good))
err_good_soft = float(np.linalg.norm(mesh_good_soft.vertices[v_m] - p_good))

check("all finite",
      all(np.isfinite(m.vertices).all()
          for m in (mesh_good, mesh_good_soft, mesh_hard, mesh_soft)),
      "all four results finite")
check("hard pin drags through the plate", err_hard > 0.15,
      f"hard-pin error {err_hard:.3f} (thin plate is ~0.24 thick)")
check("soft markers pull the vertex back out", err_soft < 0.2 * err_hard,
      f"soft error {err_soft:.4f} vs hard {err_hard:.3f} "
      f"({100 * err_soft / err_hard:.0f}% of hard)")
check("soft is harmless for accurate markers", err_good_soft < 0.02,
      f"accurate+soft deviates {err_good_soft:.4f} from accurate+hard")

# The dragged neighborhood must recover too, not just the vertex itself.
ring = np.unique(source.faces[np.any(source.faces == v_m, axis=1)])
nb_hard = float(np.linalg.norm(mesh_hard.vertices[ring] - mesh_good.vertices[ring], axis=1).max())
nb_soft = float(np.linalg.norm(mesh_soft.vertices[ring] - mesh_good.vertices[ring], axis=1).max())
check("neighborhood recovers", nb_soft < 0.5 * nb_hard,
      f"1-ring max deviation soft {nb_soft:.4f} vs hard {nb_hard:.3f}")

# Global fit must not get worse.
_, _, d_hard, _ = closest_points_on_surface(mesh_hard.vertices, surf)
_, _, d_soft, _ = closest_points_on_surface(mesh_soft.vertices, surf)
check("global fit intact", float(d_soft.mean()) < float(d_hard.mean()) * 1.2 + 1e-4,
      f"mean dist soft {d_soft.mean():.5f} vs hard {d_hard.mean():.5f}")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
