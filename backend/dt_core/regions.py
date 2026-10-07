"""
Region selection on a mesh for *regional wrap*: the user draws a boundary
loop around the patch they want to deform (e.g. the nose). Everything
inside the loop is editable; everything outside stays frozen; a feather
band just inside the loop blends the two so the seam is smooth.

This module is pure graph/geometry over (vertices, faces) — no solver, no
FastAPI — so it's easy to unit-test headlessly.

Pipeline:
    adj          = vertex_adjacency(faces, n)          # weighted edge graph
    loop         = loop_from_waypoints(verts, adj, wp) # dense vertex loop
    interior     = flood_fill_interior(adj, loop, seed)# editable set
    feather_w    = feather_weights(verts, adj, interior, loop, width)

`loop` (the boundary ring) is treated as frozen (part of the wall), so the
result is C0-continuous: boundary verts sit exactly on the source, and the
feather pulls interior verts near the boundary toward the source too,
decaying to full freedom deeper inside.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra, connected_components


def vertex_adjacency(faces: np.ndarray, n_verts: int) -> sparse.csr_matrix:
    """Undirected weighted vertex graph; edge weight = Euclidean length is
    filled later by callers that have positions. Here weights are 1 (used
    for connectivity / hop distance). Use `weighted_adjacency` for metric.
    """
    f = faces[:, :3]
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    data = np.ones(len(rows))
    a = sparse.coo_matrix((data, (rows, cols)), shape=(n_verts, n_verts))
    a = a + a.T  # symmetric
    a.data[:] = 1.0
    return a.tocsr()


def weighted_adjacency(verts: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    """Undirected vertex graph with edge weights = Euclidean edge length.
    Used for shortest-path waypoint stitching and geodesic-ish distances.
    """
    n = len(verts)
    f = faces[:, :3]
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0]])
    lengths = np.linalg.norm(verts[rows] - verts[cols], axis=1)
    a = sparse.coo_matrix((lengths, (rows, cols)), shape=(n, n)).tocsr()
    # Symmetrize taking the (equal) edge length; max handles any asymmetry.
    a = a.maximum(a.T)
    return a


def dilate_vertex_mask(
    faces: np.ndarray, n_verts: int, mask: np.ndarray, rings: int,
) -> np.ndarray:
    """Grow or shrink a boolean vertex mask by |rings| steps of vertex adjacency.

    rings > 0 (dilate): each step adds every vertex that has a masked neighbor —
      the standard "make the mask generous so pose changes don't poke through".
    rings < 0 (erode): each step removes any masked vertex that has an UNmasked
      neighbor, i.e. keeps only verts whose full 1-ring is inside the mask.
    rings == 0: unchanged.

    Pure connectivity (edge weights ignored) via boolean adjacency matmul.
    """
    m = np.asarray(mask, dtype=bool).copy()
    steps = abs(int(rings))
    if steps == 0 or m.size == 0:
        return m
    adj = vertex_adjacency(faces, n_verts)          # symmetric, weights 1
    deg = np.asarray(adj.sum(axis=1)).ravel()       # vertex degree
    for _ in range(steps):
        masked_nbrs = adj.dot(m.astype(np.float64))  # # masked neighbors per vert
        if rings > 0:
            m = m | (masked_nbrs > 0.5)
        else:
            # Keep a masked vertex only if every neighbor is also masked.
            full_ring = masked_nbrs >= deg - 1e-9
            m = m & full_ring
    return m


def _neighbors(adj: sparse.csr_matrix) -> List[np.ndarray]:
    adj = adj.tocsr()
    return [adj.indices[adj.indptr[i]:adj.indptr[i + 1]] for i in range(adj.shape[0])]


def loop_from_waypoints(
    verts: np.ndarray, faces: np.ndarray, waypoints: Sequence[int]
) -> np.ndarray:
    """Stitch a closed boundary loop from sparse clicked waypoints by
    connecting consecutive waypoints (and last->first) with shortest paths
    on the edge graph. Returns the unique set of vertex indices on the loop.

    The user only has to click a handful of points around the patch; this
    fills in every vertex along the ring so the flood fill has a solid wall.
    """
    if len(waypoints) < 3:
        raise ValueError("Need at least 3 waypoints to form a closed loop")
    g = weighted_adjacency(verts, faces)
    n = len(verts)
    loop: List[int] = []
    wp = list(waypoints)
    for a, b in zip(wp, wp[1:] + wp[:1]):
        # Dijkstra from a, reconstruct path to b via predecessors.
        dist, pred = dijkstra(g, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            raise ValueError(f"Waypoints {a} and {b} are not connected")
        path = []
        cur = b
        while cur != a and cur != -9999:
            path.append(cur)
            cur = pred[cur]
            if len(path) > n:
                raise ValueError("Path reconstruction failed (cycle?)")
        path.append(a)
        path.reverse()
        # Drop the last vertex of each segment to avoid duplicating the
        # shared waypoint with the next segment's first vertex.
        loop.extend(path[:-1])
    return np.unique(np.array(loop, dtype=np.int64))


def flood_fill_interior(
    adj: sparse.csr_matrix, boundary: np.ndarray, seed: int
) -> np.ndarray:
    """BFS from `seed` over the vertex graph, treating `boundary` vertices as
    impassable walls. Returns the set of reached vertices (the editable
    interior), EXCLUDING the boundary itself.

    `seed` must be a vertex strictly inside the loop. If the loop doesn't
    actually separate the mesh (seed reaches "outside"), the caller will see
    a suspiciously large interior — validated upstream.
    """
    neigh = _neighbors(adj)
    blocked = np.zeros(adj.shape[0], dtype=bool)
    blocked[boundary] = True
    if blocked[seed]:
        raise ValueError("Seed vertex lies on the boundary loop")
    visited = np.zeros(adj.shape[0], dtype=bool)
    stack = [int(seed)]
    visited[seed] = True
    while stack:
        v = stack.pop()
        for w in neigh[v]:
            if not visited[w] and not blocked[w]:
                visited[w] = True
                stack.append(int(w))
    return np.nonzero(visited)[0].astype(np.int64)


def label_sides(adj: sparse.csr_matrix, boundary: np.ndarray, n: int) -> np.ndarray:
    """Split the mesh into the connected regions the boundary loop carves out.

    Returns a per-vertex label array (length n): boundary vertices are -1,
    every other vertex gets the id of its connected component in the graph
    with the boundary removed. A loop that truly separates the mesh yields
    >= 2 components; one that doesn't (open / leaky) yields a single one.

    This lets the caller pick the editable side by seed OR by "smallest
    component" (the patch you drew around) and offer an invert toggle —
    much more forgiving than requiring the seed to land on the right side.
    """
    keep = np.ones(n, dtype=bool)
    keep[boundary] = False
    idx = np.where(keep)[0]
    sub = adj.tocsr()[idx][:, idx]
    _, lab = connected_components(sub, directed=False)
    labels = np.full(n, -1, dtype=np.int64)
    labels[idx] = lab
    return labels


def feather_weights(
    verts: np.ndarray,
    faces: np.ndarray,
    interior: np.ndarray,
    boundary: np.ndarray,
    width: float,
    exponent: float = 2.0,
) -> np.ndarray:
    """Per-interior-vertex soft-anchor weight for the feather band.

    Weight is 1 at the boundary and decays to 0 at geodesic distance `width`
    inside:
        d = geodesic distance from the boundary ring
        t = clamp(d / width, 0, 1)
        w = (1 - t) ** exponent       (1 near seam, 0 deep inside)

    `exponent` controls how sharply the feather concentrates at the seam:
      * 1.0 -> linear falloff (gentle, spread across the whole band);
      * >1  -> steeper gradient right at the boundary, so the seam grips
               harder and the interior gains full wrap freedom faster.
    (The old symmetric smoothstep was a soft, spread-out feather; the higher
    default exponent is the "sharper seam" the region UI now exposes.)

    Returned array is aligned with `interior` (same order, same length).
    Deep-interior verts get 0 -> full wrap freedom; near-seam verts get ~1
    -> pulled toward their own source position for a continuous seam.
    """
    if width <= 0:
        return np.zeros(len(interior), dtype=float)
    g = weighted_adjacency(verts, faces)
    # Multi-source Dijkstra from every boundary vertex at once.
    dist = dijkstra(g, indices=boundary, min_only=True)
    d_int = dist[interior]
    d_int = np.where(np.isfinite(d_int), d_int, width)  # unreachable -> deep
    t = np.clip(d_int / width, 0.0, 1.0)
    return np.power(1.0 - t, max(1.0, float(exponent)))
