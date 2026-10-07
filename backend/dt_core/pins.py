"""
Pins ("needles") for Refit.

A pin is a needle the user pokes through the placed garment. It pierces one or
more source layers; the layer nearest the body is the INNER wall (grip it onto
the body), the others are OUTER walls (never grip — they must ride the inner
wall's deformation and keep their standoff). The frontend does the ray-pick
against the placed garment + body ghost and sends the pierced source-vertex
indices ALREADY classified (inner / outer); here we turn them into conform
patches that compose onto the base seam field.

Patches are GEODESIC (edge-graph Dijkstra), not euclidean, so an inner-wall
patch never bleeds across the thin gap onto the outer wall of a double shell:
the two walls are euclidean-close but geodesically far (you must travel around
the fold). This is the whole point — it's what makes a collar behave.

Pure geometry over (vertices, faces) + vertex indices — no solver — so it
unit-tests headlessly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from .regions import weighted_adjacency


def geodesic_patch(verts: np.ndarray, faces: np.ndarray, centers: np.ndarray,
                   radius: float, exponent: float = 2.0) -> np.ndarray:
    """Union patch weight in [0, 1]: 1 at any center vertex, decaying to 0 at
    geodesic distance `radius` (and 0 beyond / on unreachable components).
    `centers` empty or radius <= 0 -> all zeros."""
    n = len(verts)
    centers = np.asarray(centers, dtype=np.int64).ravel()
    if len(centers) == 0 or radius <= 0:
        return np.zeros(n, dtype=float)
    g = weighted_adjacency(verts, faces)
    # Multi-source Dijkstra: distance to the NEAREST center (min_only).
    dist = dijkstra(g, indices=centers, min_only=True)
    t = np.clip(dist / radius, 0.0, 1.0)      # inf (unreachable) -> 1 -> weight 0
    return np.power(1.0 - t, max(1.0, float(exponent)))


def apply_pins(field: np.ndarray, verts: np.ndarray, faces: np.ndarray,
               inner: np.ndarray, outer: np.ndarray, radius: float,
               exponent: float = 2.0) -> np.ndarray:
    """Compose pin patches onto a base conform field (returns a new array):

      * outer centers -> RELEASE grip: field *= (1 - patch)  (applied first)
      * inner centers -> GRIP:         field  = max(field, patch)  (wins at seam)

    So a poke's outer layers are protected from any grip (auto seam or another
    pin), while its inner layer is pulled onto the body — regardless of order,
    the inner grip dominates where the two would overlap.
    """
    out = field.astype(float, copy=True)
    inner = np.asarray(inner, dtype=np.int64).ravel()
    outer = np.asarray(outer, dtype=np.int64).ravel()
    if len(outer):
        out = out * (1.0 - geodesic_patch(verts, faces, outer, radius, exponent))
    if len(inner):
        out = np.maximum(out, geodesic_patch(verts, faces, inner, radius, exponent))
    return np.clip(out, 0.0, 1.0)
