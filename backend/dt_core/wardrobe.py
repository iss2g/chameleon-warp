"""
Proxy wardrobe: fit a garment to a basemesh (proxy) ONCE, then wear it on any
body by transporting it through the basemesh->body deformation.

The key reuse: a Wrap run IS the body->proxy registration — the basemesh P
deformed into the target body's shape, same topology as P (call it P'). So a
garment G fitted to P can be transported to the body B by:

  1. bind_garment(G, surface(P))  — record, per garment vertex, WHICH proxy
     triangle it sits over, WHERE on that triangle (barycentric u,v), and how
     far along the triangle normal (signed offset);
  2. apply_binding(binding, P', P.faces) — rebuild those positions on the
     DEFORMED proxy P'. Pure arithmetic, no queries: the garment rides the
     proxy surface wherever the wrap sent it.

A short cleanup refit against B then re-establishes cloth thickness and
resolves any residual collision. `offset` is transported verbatim (not scaled
by local stretch) — a good enough initial placement; the cleanup fixes it.

Pure numpy over (vertices, faces) + surface.SurfaceData — no solver, no
FastAPI, unit-testable headlessly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .surface import SurfaceData, closest_points_on_surface


@dataclass
class GarmentBinding:
    tri: np.ndarray        # (N,) basemesh triangle index per garment vertex
    bary: np.ndarray       # (N,2) barycentric (u,v) of the closest point
    offset: np.ndarray     # (N,) signed distance along the triangle normal


def bind_garment(garment_verts: np.ndarray, proxy_surf: SurfaceData) -> GarmentBinding:
    """Record each garment vertex relative to the proxy surface: closest
    triangle, barycentric coordinate of the closest point, and signed normal
    offset. `proxy_surf` must be built from the UNDEFORMED proxy (basemesh P)."""
    gv = np.asarray(garment_verts, dtype=float)
    _cp, tri, _dist, _reg = closest_points_on_surface(gv, proxy_surf)

    a = proxy_surf.tri_a[tri]
    ab = proxy_surf.tri_ab[tri]
    ac = proxy_surf.tri_ac[tri]
    n = proxy_surf.tri_normals[tri]
    # Decompose the garment vertex against the chosen triangle's PLANE, not its
    # clamped closest point: d = (u*ab + v*ac) + offset*n. This is exact even
    # when the closest point lands on an edge/vertex (u,v may fall just outside
    # the triangle — that only extrapolates the same affine frame), so the
    # round-trip a + u*ab + v*ac + offset*n reproduces the vertex bit-for-bit.
    d = gv - a
    offset = np.einsum('ij,ij->i', d, n)
    planar = d - offset[:, None] * n
    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    p0 = np.einsum('ij,ij->i', planar, ab)
    p1 = np.einsum('ij,ij->i', planar, ac)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1.0, den)
    u = (d11 * p0 - d01 * p1) / den
    v = (d00 * p1 - d01 * p0) / den
    return GarmentBinding(tri=tri.astype(np.int64),
                          bary=np.stack([u, v], axis=1), offset=offset)


def apply_binding(b: GarmentBinding, proxy_verts: np.ndarray,
                  proxy_faces: np.ndarray) -> np.ndarray:
    """Rebuild the garment vertices on a DEFORMED proxy (P'): for each garment
    vertex, a + u*ab + v*ac + offset*n on its bound triangle. `proxy_faces` /
    `proxy_verts` must be the SAME topology the binding was built against."""
    f = np.ascontiguousarray(proxy_faces[:, :3], dtype=np.int64)[b.tri]
    a = proxy_verts[f[:, 0]]
    ab = proxy_verts[f[:, 1]] - a
    ac = proxy_verts[f[:, 2]] - a
    n = np.cross(ab, ac)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln == 0, 1.0, ln)[:, None]
    u = b.bary[:, 0:1]
    v = b.bary[:, 1:2]
    return a + u * ab + v * ac + b.offset[:, None] * n


# ---------------------------------------------------------------------------
# Binding cache (deterministic per garment/proxy vertex buffers)
# ---------------------------------------------------------------------------

def binding_hash(garment_verts: np.ndarray, proxy_verts: np.ndarray) -> str:
    """sha256 over both vertex buffers — a binding is a pure function of the
    garment rest pose and the proxy rest surface, so re-wearing is a cache hit."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(garment_verts, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(proxy_verts, dtype=np.float64).tobytes())
    return h.hexdigest()


def save_binding(path, b: GarmentBinding) -> None:
    np.savez(path, tri=b.tri, bary=b.bary, offset=b.offset)


def load_binding(path) -> GarmentBinding:
    d = np.load(path)
    return GarmentBinding(tri=d["tri"], bary=d["bary"], offset=d["offset"])
