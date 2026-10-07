"""
Collision resolution for refit — hiding the body poking through a garment.
Two strategies, matching how games actually do it:

  push_out(garment, body)  -> deform the GARMENT: any garment vertex inside the
      body (or closer than `offset`) is pushed out onto body_surface + offset.
      Use when the garment should visibly cover the bulge (soft cloth).

  body_hide_mask(body, garment) -> a boolean MASK over BODY vertices that lie
      under the garment. The body mesh is NOT modified; the caller emits a
      per-outfit hide list (the standard game approach). Use for rigid armor —
      no need to inflate the armor, just cull the skin beneath it.

Both reuse surface.py: the closest point on a triangle mesh plus that triangle's
outward normal give a signed offset (positive = outside, negative = inside),
assuming the reference mesh has consistent outward-facing winding.

Caveat: an OPEN shell (a shirt is not closed) has no well-defined interior, so
near the shell's open edges the sign is ambiguous; hide_body therefore also caps
by distance (`max_depth`) so only vertices genuinely tucked under the surface,
not everything on the mesh's negative side out to infinity, get hidden.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .surface import SurfaceData, _on_open_boundary, closest_points_on_surface


def signed_offset(points: np.ndarray, surf: SurfaceData
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point: closest surface point `cp`, that triangle's outward
    normal `n`, and the signed distance `s = dot(point - cp, n)` (>0 outside,
    <0 inside). Magnitude equals the true distance up to the sign."""
    cp, tri, _dist, _reg = closest_points_on_surface(points, surf)
    n = surf.tri_normals[tri]
    s = np.einsum('ij,ij->i', points - cp, n)
    return cp, n, s


def resolve_push_out(garment_verts: np.ndarray, body_surf: SurfaceData,
                     offset: float, rounds: int = 6,
                     violation: float | None = None) -> Tuple[np.ndarray, int]:
    """Push garment vertices that violate the body out onto
    body_surface + offset*normal.

    `violation` is the signed-offset threshold that trips a push (default:
    `offset`, the old behavior). Refit passes violation=0: only vertices
    actually INSIDE the body are lifted, out to the full `offset`. Pushing
    everything below `offset` normalizes double walls whose rest gap is
    smaller than the offset onto one shell — the walls merge and z-fight,
    which is exactly the collar crumble this used to cause.

    Repeated a few rounds because moving one vertex can expose a neighbour.
    Returns (new verts, count still violating on the last check)."""
    vio = offset if violation is None else float(violation)
    v = garment_verts.copy()
    for _ in range(rounds):
        cp, n, s = signed_offset(v, body_surf)
        inside = s < vio
        if not inside.any():
            return v, 0
        v[inside] = cp[inside] + offset * n[inside]
    # Re-measure AFTER the final move so the count reflects the returned mesh
    # (moving a vertex changes its nearest triangle, so convergence is only
    # known after the last push, not from the pre-move count).
    _, _, s = signed_offset(v, body_surf)
    return v, int((s < vio - 1e-6).sum())


def body_hide_mask(body_verts: np.ndarray, garment_surf: SurfaceData,
                   offset: float = 0.0, max_depth: float | None = None,
                   reject_boundary: bool = True) -> np.ndarray:
    """Boolean mask over BODY vertices that lie under the garment surface —
    on its inner (negative-normal) side, within `max_depth` of it. The body is
    never modified; the caller emits this as a per-outfit hide list.

    `offset` (>=0) shrinks the covered band slightly so grazing vertices right
    on the seam aren't culled. `max_depth` (default: unbounded) caps how deep a
    vertex may be and still count as "under" — guards the open-shell case.
    `reject_boundary` drops vertices whose closest garment point lies ON an
    open border (the rim): those sit BESIDE the shell, where the normal's sign
    says nothing about being covered — without this a chest plate "hides" half
    the torso around it."""
    cp, tri, _dist, reg = closest_points_on_surface(body_verts, garment_surf)
    n = garment_surf.tri_normals[tri]
    s = np.einsum('ij,ij->i', body_verts - cp, n)
    under = s < -offset
    if max_depth is not None:
        under &= s > -(offset + max_depth)
    if reject_boundary:
        under &= ~_on_open_boundary(garment_surf, tri, reg)
    return under


def hidden_faces(vertex_mask: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces to drop = those whose every vertex is hidden. (A face with a
    visible corner still needs to render, so only fully-covered faces go.)"""
    f = faces[:, :3]
    return vertex_mask[f].all(axis=1)
