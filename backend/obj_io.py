"""
Polygon-preserving OBJ export for wrap results.

The pipeline computes on triangles (Sumner-Popovic needs per-triangle
affine transforms), so meshlib fan-triangulates on load and save_obj writes
triangles only. But wrap NEVER changes connectivity — it only moves
vertices, in the original order. So the author's real topology (quads,
n-gons, UV seams) can be restored exactly: take the ORIGINAL uploaded .obj
and rewrite just the `v` lines with the wrapped positions.

What survives verbatim: `f` lines (quads / n-gons / their vt references),
`vt` lines (UVs stay valid — connectivity didn't change), `o/g/s/usemtl/
mtllib`, comments. What is dropped: `vn` lines and the normal part of `f`
tokens — normals are stale after deformation and any DCC recomputes them.
Vertex colors / w-components on `v` lines are dropped too (positions only).
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


class ObjRewriteError(ValueError):
    """Original OBJ and new vertex array don't agree (count mismatch etc.)."""


def _strip_normal_ref(token: str) -> str:
    """'7/3/9' -> '7/3', '7//9' -> '7', '7/3' -> '7/3', '7' -> '7'."""
    parts = token.split("/")
    if len(parts) >= 2 and parts[1] != "":
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def rewrite_obj_vertices(original_obj: Union[str, Path],
                         new_vertices: np.ndarray,
                         out_path: Union[str, Path],
                         encoding: str = "utf-8",
                         header: str = None) -> dict:
    """Write `out_path` = `original_obj` with vertex positions replaced by
    `new_vertices` (in file order). Raises ObjRewriteError if the vertex
    counts don't match — callers should fall back to the triangulated export.

    `header` overrides the leading comment line (defaults to the wrap wording);
    /run passes a transfer-specific note.

    Returns stats: {"vertices": n, "faces": n, "quads": n, "ngons": n,
    "has_uvs": bool}.
    """
    new_vertices = np.asarray(new_vertices, dtype=float)
    if new_vertices.ndim != 2 or new_vertices.shape[1] != 3:
        raise ObjRewriteError(f"new_vertices must be (N, 3), got {new_vertices.shape}")

    if header is None:
        header = ("# Rewritten by Chameleon Warp (wrap): original "
                  "topology + UVs, deformed vertex positions")
    out_lines = [header.rstrip("\n") + "\n"]
    vi = 0
    n_faces = n_quads = n_ngons = 0
    has_uvs = False

    with open(original_obj, "rt", encoding=encoding, errors="replace") as fp:
        for line in fp:
            if line.startswith("v "):
                if vi >= len(new_vertices):
                    raise ObjRewriteError(
                        f"original OBJ has more `v` lines than wrapped vertices ({len(new_vertices)})")
                x, y, z = new_vertices[vi]
                out_lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                vi += 1
            elif line.startswith("vn ") or line.startswith("vn\t"):
                continue                      # stale after deformation
            elif line.startswith("vt ") or line.startswith("vt\t"):
                has_uvs = True
                out_lines.append(line)
            elif line.startswith("f ") or line.startswith("f\t"):
                tokens = line.split()
                rebuilt = [tokens[0]] + [_strip_normal_ref(t) for t in tokens[1:]]
                n_verts_in_face = len(rebuilt) - 1
                n_faces += 1
                if n_verts_in_face == 4:
                    n_quads += 1
                elif n_verts_in_face > 4:
                    n_ngons += 1
                out_lines.append(" ".join(rebuilt) + "\n")
            else:
                out_lines.append(line)

    if vi != len(new_vertices):
        raise ObjRewriteError(
            f"vertex count mismatch: original OBJ has {vi} `v` lines, "
            f"wrap result has {len(new_vertices)} vertices")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wt", encoding="utf-8") as fp:
        fp.writelines(out_lines)

    return {
        "vertices": vi,
        "faces": n_faces,
        "quads": n_quads,
        "ngons": n_ngons,
        "has_uvs": has_uvs,
    }
