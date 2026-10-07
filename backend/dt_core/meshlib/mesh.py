"""
Classes to load and handle mesh data.
"""

import os
import re
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from .vector import Vec3f, Vector3D

# Payloads of 'v ' / 'f ' lines. (?m) anchors per line; on CRLF files the
# trailing \r lands in the group and is eaten later by str.split().
_RE_V_LINE = re.compile(r"(?m)^v (.*)$")
_RE_F_LINE = re.compile(r"(?m)^f (.*)$")


def _parse_obj_sequential(text: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reference line-by-line parser. Only used for the rare pathological
    case (negative face indices in a file that interleaves `v` and `f`
    blocks, where an index is relative to the vertices defined SO FAR)."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for line in text.split("\n"):
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            parts = line.split()
            idx = []
            for tok in parts[1:]:
                head = tok.split("/", 1)[0]
                if not head:
                    continue
                i = int(head)
                idx.append(len(verts) + i if i < 0 else i - 1)
            if len(idx) < 3:
                continue
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not verts:
        raise ValueError("No vertices found in .obj file")
    if not faces:
        raise ValueError("No triangle faces found in .obj file")
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _parse_obj(file: str, encoding: str = "utf-8") -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Wavefront .obj parser. Reads only `v` (vertices) and `f` (faces);
    silently ignores everything else (mtllib, usemtl, vt, vn, comments, groups).
    Triangulates non-triangle faces using a fan from the first vertex.
    Returns (vertices: Nx3 float64, faces: Mx3 int64).

    Common layouts ('v x y z' + pure-triangle 'f i j k') parse with one
    C-level split + np.array over the whole block instead of per-line Python
    — ~10x faster on dense meshes, which matters because every preview
    cache miss and every pose in a /run re-parses a file.
    """
    with open(file, "rt", encoding=encoding, errors="replace") as fp:
        text = fp.read()

    # Line classification in C: regex per prefix instead of a Python loop.
    v_payload = _RE_V_LINE.findall(text)
    f_payload = _RE_F_LINE.findall(text)
    first_f = 0 if text.startswith("f ") else text.find("\nf ")
    last_v = 0 if text.startswith("v ") else -1
    last_v = max(last_v, text.rfind("\nv "))
    interleaved = first_f != -1 and last_v > first_f

    if not v_payload:
        raise ValueError("No vertices found in .obj file")
    if not f_payload:
        raise ValueError("No triangle faces found in .obj file")

    # ---- vertices ----
    vtok = " ".join(v_payload).split()
    if len(vtok) == 3 * len(v_payload):
        verts = np.array(vtok, dtype=np.float64).reshape(-1, 3)
    else:
        # some lines carry extras ('v x y z w', vertex colors) — per line
        verts = np.array([p.split()[:3] for p in v_payload], dtype=np.float64)

    # ---- faces ----
    fblob = " ".join(f_payload)
    if interleaved and "-" in fblob:
        # negative index relative to a moving vertex count — bail out to the
        # sequential parser that reproduces that semantics exactly.
        return _parse_obj_sequential(text)

    rows = [p.split() for p in f_payload]
    flat: List[str] = []
    tris_fast = all(len(r) == 3 for r in rows)
    if tris_fast:
        # pure triangles: strip any '/t/n' refs in one pass, convert in C
        flat = [t.partition("/")[0] for r in rows for t in r]
        tris_fast = "" not in flat   # a bare '//n' token needs the slow path
    if tris_fast:
        idx = np.array(flat, dtype=np.int64).reshape(-1, 3)
    else:
        # quads / n-gons (fan-triangulated), possibly with '/t/n' tokens
        tris: List[Tuple[int, int, int]] = []
        for r in rows:
            heads = [h for h in (tok.partition("/")[0] for tok in r) if h]
            if len(heads) < 3:
                continue
            first = int(heads[0])
            prev = int(heads[1])
            for h in heads[2:]:
                cur = int(h)
                tris.append((first, prev, cur))
                prev = cur
        if not tris:
            raise ValueError("No triangle faces found in .obj file")
        idx = np.array(tris, dtype=np.int64)

    faces = np.where(idx < 0, idx + len(verts), idx - 1)
    return verts, faces


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    @classmethod
    def load_obj(cls, file: str, **kwargs) -> "Mesh":
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        encoding = kwargs.get("encoding", "utf-8")
        verts, faces = _parse_obj(file, encoding=encoding)
        return cls(vertices=verts, faces=faces)

    @classmethod
    def load_npz(cls, file: str, **kwargs) -> "Mesh":
        assert os.path.isfile(file), f"Mesh file is missing: {file}"
        data = np.load(file)
        return cls(data["vertices"], data["faces"])

    @classmethod
    def load(cls, file: str, **kwargs) -> "Mesh":
        if file.endswith(".obj") or file.endswith(".pose"):
            return cls.load_obj(file, **kwargs)
        elif file.endswith(".npz"):
            return cls.load_npz(file, **kwargs)
        raise ValueError("Invalid file format")

    def save_obj(self, file: str) -> None:
        """Write the mesh as a Wavefront .obj file (vertices + triangle faces,
        1-indexed). One giant %-format per block runs the formatting loop in
        C — ~10x faster than per-line f-strings on dense meshes."""
        m = self.to_third_dimension(copy=False)
        v = np.ascontiguousarray(m.vertices, dtype=np.float64)
        f = np.ascontiguousarray(m.faces[:, :3], dtype=np.int64) + 1
        with open(file, "wt", encoding="utf-8") as fp:
            fp.write("# Generated by Chameleon Warp\n")
            fp.write(("v %.6f %.6f %.6f\n" * len(v)) % tuple(v.ravel()))
            fp.write(("f %d %d %d\n" * len(f)) % tuple(f.ravel()))

    def get_centroids(self) -> np.ndarray:
        return self.vertices[self.faces[:, :3]].mean(axis=1)

    def scale(self, factor: float):
        self.vertices *= factor
        return self

    def box(self) -> Tuple[Vec3f, Vec3f]:
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def size(self) -> Vec3f:
        a, b = self.box()
        return b - a

    def move(self, offset: Vec3f):
        self.vertices += offset

    def span_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        c = (tmp.T / np.sqrt(np.linalg.norm(tmp, axis=1))).T
        return a, b, c

    @property
    def span(self) -> np.ndarray:
        a, b, c = self.span_components()
        return np.transpose((a, b, c), (1, 2, 0))

    @property
    def v1(self):
        return self.vertices[self.faces[:, 0]]

    def get_dimension(self) -> int:
        return self.faces.shape[1]

    def is_fourth_dimension(self) -> bool:
        return self.get_dimension() == 4

    def to_fourth_dimension(self, copy=True) -> "Mesh":
        if self.is_fourth_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        a, b, c = self.span_components()
        v4 = self.v1 + c
        new_vertices = np.concatenate((self.vertices, v4), axis=0)
        v4_indices = np.arange(len(self.vertices), len(self.vertices) + len(c))
        new_faces = np.concatenate((self.faces, v4_indices.reshape((-1, 1))), axis=1)
        return Mesh(new_vertices, new_faces)

    def is_third_dimension(self) -> bool:
        return self.faces.shape[1] == 3

    def to_third_dimension(self, copy=True) -> "Mesh":
        if self.is_third_dimension():
            if copy:
                return Mesh(np.copy(self.vertices), np.copy(self.faces))
            else:
                return self

        assert self.vertices.shape[1] == 3, f"Some strange error occurred! vertices.shape = {self.vertices.shape}"
        new_faces = self.faces[:, :3]
        new_vertices = self.vertices[:np.max(new_faces) + 1]
        return Mesh(new_vertices, new_faces)

    def transpose(self, shape=(0, 1, 2)):
        shape = np.asarray(shape)
        assert shape.shape == (3,)
        return Mesh(
            vertices=self.vertices[:, shape],
            faces=self.faces
        )

    def normals(self) -> np.ndarray:
        v1, v2, v3 = self.vertices[self.faces.T][:3]
        vns = np.cross(v2 - v1, v3 - v1)
        return (vns.T / np.linalg.norm(vns, axis=1)).T
