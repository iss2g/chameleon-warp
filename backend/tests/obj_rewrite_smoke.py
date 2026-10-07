"""
Headless test for obj_io.rewrite_obj_vertices — polygon-preserving export.

Builds a small OBJ exercising the real-world zoo: quads, a triangle, an
n-gon, UVs, normals, every `f` token form (v, v/vt, v/vt/vn, v//vn),
usemtl/o/s lines, negative-ish ordering. Checks that after rewriting:

  * quads / n-gons survive verbatim (no fan-triangulation)
  * `vt` lines and the vt part of `f` tokens survive (UVs intact)
  * `vn` lines and normal refs are dropped (stale after deformation)
  * new vertex positions land in the `v` lines, order preserved
  * meshlib can still load the rewritten file (viewport keeps working)
  * a vertex-count mismatch raises ObjRewriteError (fallback path)

Run:  .venv\\Scripts\\python.exe tests\\obj_rewrite_smoke.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dt_core import meshlib                                    # noqa: E402
from obj_io import ObjRewriteError, rewrite_obj_vertices       # noqa: E402

ORIGINAL = """\
# authored test cube-ish patch
mtllib mats.mtl
o patch
usemtl skin
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
v 2 1 0
v 3 0.5 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
s 1
f 1/1/1 2/2/1 3/3/1 4/4/1
f 2/2 5/1 6/3 3/4
f 5//1 7 6//1
f 1/1 2/2 6/3 5/4 4/1
"""
# faces: quad, quad, triangle, pentagon (n-gon) -> meshlib fan = 2+2+1+3 = 8 tris

failures = []


def check(name, cond, detail):
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


tmp = Path(tempfile.mkdtemp(prefix="objrw_"))
src_path = tmp / "original.obj"
out_path = tmp / "rewritten.obj"
src_path.write_text(ORIGINAL, encoding="utf-8")

mesh = meshlib.Mesh.load(str(src_path))
check("meshlib triangulates original", len(mesh.faces) == 8,
      f"{len(mesh.faces)} tris from 4 polygons (expected 8)")

new_verts = mesh.vertices * 1.5 + np.array([0.0, 0.0, 2.0])
stats = rewrite_obj_vertices(src_path, new_verts, out_path)

check("stats", stats == {"vertices": 7, "faces": 4, "quads": 2,
                         "ngons": 1, "has_uvs": True}, f"{stats}")

lines = out_path.read_text(encoding="utf-8").splitlines()
f_lines = [ln.split() for ln in lines if ln.startswith("f ")]
check("polygon sizes preserved",
      [len(t) - 1 for t in f_lines] == [4, 4, 3, 5],
      f"face sizes {[len(t) - 1 for t in f_lines]} (expected [4, 4, 3, 5])")

check("vt refs kept, vn refs dropped",
      f_lines[0][1:] == ["1/1", "2/2", "3/3", "4/4"] and
      f_lines[2][1:] == ["5", "7", "6"],
      f"f1={' '.join(f_lines[0][1:])}  f3={' '.join(f_lines[2][1:])}")

check("vt lines kept / vn lines gone",
      sum(1 for ln in lines if ln.startswith("vt ")) == 4 and
      not any(ln.startswith("vn ") for ln in lines),
      "4 vt, 0 vn")

check("metadata lines kept",
      any(ln.startswith("usemtl skin") for ln in lines) and
      any(ln.startswith("o patch") for ln in lines) and
      any(ln.strip() == "s 1" for ln in lines),
      "usemtl / o / s survived")

re_mesh = meshlib.Mesh.load(str(out_path))
check("rewritten still loads in meshlib",
      len(re_mesh.faces) == 8 and len(re_mesh.vertices) == 7,
      f"{len(re_mesh.vertices)} verts / {len(re_mesh.faces)} tris")

check("positions replaced in order",
      bool(np.allclose(re_mesh.vertices, new_verts, atol=1e-5)),
      f"max diff {np.abs(re_mesh.vertices - new_verts).max():.2e}")

try:
    rewrite_obj_vertices(src_path, new_verts[:-1], tmp / "bad.obj")
    check("count mismatch raises", False, "no exception raised")
except ObjRewriteError as e:
    check("count mismatch raises", True, str(e))

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
print("ALL CHECKS PASSED")
