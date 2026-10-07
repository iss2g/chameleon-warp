"""
Closest-point-on-SURFACE queries against a triangle mesh, plus mesh-integrity
helpers (fold-over damping).

Why this exists: the correspondence step used to pull every source vertex
toward the nearest target VERTEX (cKDTree over vertex positions). That caps
fit quality at the target's vertex density, clumps several source vertices
onto one target vertex (the high-frequency "wobble" we then smoothed away),
and happily matches through gaps and onto open borders (edge suction at eye
sockets / mouth bags / mesh cuts). Everything here works with the continuous
triangle surface instead:

  build_surface(verts, faces)     -> SurfaceData: KD-tree over triangle
                                     centroids + per-triangle data + boundary
                                     flags, built once per target mesh
  closest_points_on_surface(...)  -> exact nearest point on the mesh surface
                                     (vectorized Ericson point-triangle test
                                     over KD candidates)
  match_points_to_surface(...)    -> the above + robust pruning: normal
                                     compatibility, adaptive distance cut
                                     (3x median, floored), open-boundary
                                     rejection. Pruned vertices are simply
                                     not constrained — smoothness/identity
                                     carry them, which is exactly what should
                                     happen where the target has no
                                     corresponding geometry.
  project_onto_surface(...)       -> weighted snap of vertices onto the
                                     surface (final shrink-wrap pass)
  damp_folds(...)                 -> pull vertices of inverted triangles back
                                     toward their previous position

No new dependencies: numpy + scipy.spatial.cKDTree only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# Broad-phase KD worker count for the crossing-candidate query (query_ball_point).
# -1 = all cores (the default, used on the single-process serial path). When the
# layer signal-A ray cast is parallelized ACROSS processes (dt_core.layers), each
# worker sets this to 1 so N processes don't each spawn N KD threads (N*N thread
# oversubscription). Module-global so it needn't thread through every call site.
_KD_WORKERS = -1

# Region codes of the closest point within a triangle (Ericson 5.1.5).
REG_INSIDE = 0
REG_AB = 1
REG_BC = 2
REG_CA = 3
REG_VA = 4
REG_VB = 5
REG_VC = 6


@dataclass
class SurfaceData:
    """Per-target-mesh acceleration structure for surface queries."""
    faces: np.ndarray          # (T, 3) vertex indices
    tri_a: np.ndarray          # (T, 3) corner A position
    tri_ab: np.ndarray         # (T, 3) B - A
    tri_ac: np.ndarray         # (T, 3) C - A
    tri_normals: np.ndarray    # (T, 3) unit normals (zero rows for degenerate)
    centroid_tree: cKDTree     # KD-tree over triangle centroids
    edge_boundary: np.ndarray  # (T, 3) bool per edge [AB, BC, CA]
    vert_boundary: np.ndarray  # (V,) bool — vertex lies on an open border
    diag: float                # bbox diagonal of the mesh
    # Winding-INDEPENDENT triangle parameterization (corners taken in ascending
    # vertex-index order). Ray/segment crossing tests use these so their result
    # is bitwise-identical under face inversion — the layer signals must never
    # depend on garment orientation. Distinct from tri_a/ab/ac, which keep the
    # authored winding (tri_normals' sign, and closest-point queries, need it).
    tri_ca: np.ndarray         # (T, 3) canonical corner A position
    tri_cab: np.ndarray        # (T, 3) canonical B - A
    tri_cac: np.ndarray        # (T, 3) canonical C - A
    # Crossing-test broad phase, split by triangle size. A single centroid ball
    # sized by the GLOBAL max triangle radius explodes when a handful of huge
    # triangles (cap fans, coarse game-asset faces) inflate it — every query
    # then gathers hundreds of small triangles it can't hit. So "normal" and
    # "big" triangles get their own centroid trees: the small-radius query scans
    # only normal centroids; the large-radius query scans only the few big ones.
    norm_tree: Optional[cKDTree]   # centroids of triangles with corner_r <= cut
    norm_idx: np.ndarray           # their global triangle indices
    norm_r: float                  # the cut radius (query pad for the normal set)
    big_tree: Optional[cKDTree]    # centroids of oversized triangles (may be None)
    big_idx: np.ndarray            # their global triangle indices
    big_r: float                   # max corner radius among big tris (query pad)


def build_surface(verts: np.ndarray, faces: np.ndarray) -> SurfaceData:
    f = np.ascontiguousarray(faces[:, :3], dtype=np.int64)
    a = verts[f[:, 0]]
    ab = verts[f[:, 1]] - a
    ac = verts[f[:, 2]] - a

    n = np.cross(ab, ac)
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln == 0, 1.0, ln)[:, None]

    centroids = a + (ab + ac) / 3.0

    # An undirected edge shared by exactly one face is an open border.
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    edge_is_boundary = counts[inv] == 1            # (3T,) in [AB..., BC..., CA...] order
    edge_boundary = edge_is_boundary.reshape(3, len(f)).T

    vert_boundary = np.zeros(len(verts), dtype=bool)
    vert_boundary[edges[edge_is_boundary].ravel()] = True

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))

    # Canonical (winding-independent) parameterization: order each face's three
    # vertices by ascending index. A face and its inversion share the same index
    # SET, so both yield identical (cA, cB, cC) — hence identical crossing math.
    fc = np.sort(f, axis=1)
    ca = verts[fc[:, 0]]
    cab = verts[fc[:, 1]] - ca
    cac = verts[fc[:, 2]] - ca

    # Split triangles by corner radius for the crossing-test broad phase.
    cen_rel = (ab + ac) / 3.0
    corner_r = np.maximum(np.linalg.norm(ab - cen_rel, axis=1),
                          np.maximum(np.linalg.norm(ac - cen_rel, axis=1),
                                     np.linalg.norm(cen_rel, axis=1)))
    # "Big" = more than 4x the median corner radius: rare on uniform meshes
    # (empty split, zero overhead), captures cap fans / coarse faces otherwise.
    cut = float(np.median(corner_r) * 4.0) if len(corner_r) else 0.0
    big = corner_r > cut
    norm_idx = np.nonzero(~big)[0]
    big_idx = np.nonzero(big)[0]
    norm_tree = cKDTree(centroids[norm_idx]) if len(norm_idx) else None
    big_tree = cKDTree(centroids[big_idx]) if len(big_idx) else None
    norm_r = float(corner_r[norm_idx].max()) if len(norm_idx) else 0.0
    big_r = float(corner_r[big_idx].max()) if len(big_idx) else 0.0

    return SurfaceData(
        faces=f, tri_a=a, tri_ab=ab, tri_ac=ac, tri_normals=n,
        centroid_tree=cKDTree(centroids),
        edge_boundary=edge_boundary, vert_boundary=vert_boundary,
        diag=diag,
        tri_ca=ca, tri_cab=cab, tri_cac=cac,
        norm_tree=norm_tree, norm_idx=norm_idx, norm_r=norm_r,
        big_tree=big_tree, big_idx=big_idx, big_r=big_r,
    )


def _closest_on_triangles(P: np.ndarray, A: np.ndarray, AB: np.ndarray, AC: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point on triangle (A, A+AB, A+AC) for each row. All inputs (M, 3).

    Vectorized version of Ericson, "Real-Time Collision Detection", 5.1.5.
    Returns (closest points (M, 3), region codes (M,) int8).
    """
    B = A + AB
    C = A + AC

    ap = P - A
    d1 = np.einsum('ij,ij->i', AB, ap)
    d2 = np.einsum('ij,ij->i', AC, ap)
    bp = P - B
    d3 = np.einsum('ij,ij->i', AB, bp)
    d4 = np.einsum('ij,ij->i', AC, bp)
    cp = P - C
    d5 = np.einsum('ij,ij->i', AB, cp)
    d6 = np.einsum('ij,ij->i', AC, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    out = np.empty_like(P)
    region = np.empty(len(P), dtype=np.int8)
    todo = np.ones(len(P), dtype=bool)

    m = (d1 <= 0) & (d2 <= 0) & todo               # vertex A
    out[m] = A[m]; region[m] = REG_VA; todo &= ~m

    m = (d3 >= 0) & (d4 <= d3) & todo              # vertex B
    out[m] = B[m]; region[m] = REG_VB; todo &= ~m

    m = (d6 >= 0) & (d5 <= d6) & todo              # vertex C
    out[m] = C[m]; region[m] = REG_VC; todo &= ~m

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & todo   # edge AB
    if m.any():
        den = d1[m] - d3[m]
        v = d1[m] / np.where(den == 0, 1.0, den)
        out[m] = A[m] + v[:, None] * AB[m]
        region[m] = REG_AB
        todo &= ~m

    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & todo   # edge CA
    if m.any():
        den = d2[m] - d6[m]
        w = d2[m] / np.where(den == 0, 1.0, den)
        out[m] = A[m] + w[:, None] * AC[m]
        region[m] = REG_CA
        todo &= ~m

    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & todo   # edge BC
    if m.any():
        den = (d4[m] - d3[m]) + (d5[m] - d6[m])
        w = (d4[m] - d3[m]) / np.where(den == 0, 1.0, den)
        out[m] = B[m] + w[:, None] * (C[m] - B[m])
        region[m] = REG_BC
        todo &= ~m

    m = todo                                        # interior
    if m.any():
        den = va[m] + vb[m] + vc[m]
        den = np.where(den == 0, 1.0, den)
        v = vb[m] / den
        w = vc[m] / den
        out[m] = A[m] + v[:, None] * AB[m] + w[:, None] * AC[m]
        region[m] = REG_INSIDE

    return out, region


def closest_points_on_surface(points: np.ndarray, surf: SurfaceData,
                              k: int = 16, chunk: int = 16384
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest point ON THE SURFACE of the mesh for each query point.

    Candidates come from the k nearest triangle centroids; the exact
    point-triangle test picks the winner. k=16 is plenty unless triangle
    sizes vary wildly. Chunked to bound peak memory on dense meshes.

    Returns (closest_points (N,3), tri_index (N,), distance (N,), region (N,)).
    """
    n = len(points)
    k_eff = max(1, min(k, len(surf.tri_a)))
    cp_out = np.empty((n, 3), dtype=float)
    tri_out = np.empty(n, dtype=np.int64)
    dist_out = np.empty(n, dtype=float)
    reg_out = np.empty(n, dtype=np.int8)

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        pts = points[lo:hi]
        _, tid = surf.centroid_tree.query(pts, k_eff, workers=-1)
        if k_eff == 1:
            tid = tid[:, None]
        flat_t = tid.ravel()
        Pk = np.repeat(pts, k_eff, axis=0)
        cp, reg = _closest_on_triangles(
            Pk, surf.tri_a[flat_t], surf.tri_ab[flat_t], surf.tri_ac[flat_t])
        d2 = np.einsum('ij,ij->i', cp - Pk, cp - Pk).reshape(-1, k_eff)
        pick = d2.argmin(axis=1)
        rows = np.arange(len(pts))
        sel = rows * k_eff + pick
        cp_out[lo:hi] = cp[sel]
        tri_out[lo:hi] = flat_t[sel]
        reg_out[lo:hi] = reg[sel]
        dist_out[lo:hi] = np.sqrt(d2[rows, pick])

    return cp_out, tri_out, dist_out, reg_out


def _on_open_boundary(surf: SurfaceData, tri: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """True where the closest point lies on an open border of the mesh —
    the classic ICP edge-suction case that must not attract anything."""
    f = surf.faces[tri]
    eb = surf.edge_boundary[tri]
    vb = surf.vert_boundary
    return (((reg == REG_AB) & eb[:, 0]) |
            ((reg == REG_BC) & eb[:, 1]) |
            ((reg == REG_CA) & eb[:, 2]) |
            ((reg == REG_VA) & vb[f[:, 0]]) |
            ((reg == REG_VB) & vb[f[:, 1]]) |
            ((reg == REG_VC) & vb[f[:, 2]]))


def match_points_to_surface(points: np.ndarray, point_normals: np.ndarray,
                            surf: SurfaceData,
                            max_angle_deg: float = 60.0,
                            dist_floor: float = 0.0,
                            adaptive_factor: float = 3.0,
                            reject_boundary: bool = True,
                            k: int = 16,
                            offset: float = 0.0,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Robust surface correspondences for the ICP-style closest-point term.

    Pruning, in order:
      * normal compatibility — the matched triangle must face the same way
        as the query vertex (within max_angle_deg), so we never match through
        the mesh onto the far side / inner shell;
      * open-boundary rejection — see _on_open_boundary;
      * adaptive distance cut — drop matches farther than
        max(dist_floor, adaptive_factor * median(distance)). Early iterations
        (everything far) keep their pull; as the fit converges the threshold
        tightens automatically and true outliers (source areas with no target
        coverage) fall out instead of stretching toward a wrong patch.

    `offset` displaces the returned targets along the matched triangle's
    outward normal: cp + offset*n. Refit uses this as cloth thickness — the
    garment rests a standoff ABOVE the skin instead of being painted onto it.
    Pruning always runs on the true (un-offset) closest points.

    Returns (point_indices, closest_points, distances, stats).
    """
    cp, tri, dist, reg = closest_points_on_surface(points, surf, k=k)

    cos_lim = float(np.cos(np.radians(max_angle_deg)))
    ok = np.einsum('ij,ij->i', surf.tri_normals[tri], point_normals) >= cos_lim
    n_angle = int((~ok).sum())

    n_boundary = 0
    if reject_boundary:
        onb = _on_open_boundary(surf, tri, reg)
        n_boundary = int((ok & onb).sum())
        ok &= ~onb

    n_dist = 0
    threshold = float('inf')
    if ok.any():
        threshold = max(float(dist_floor), adaptive_factor * float(np.median(dist[ok])))
        n_dist = int((ok & (dist > threshold)).sum())
        ok &= dist <= threshold

    idx = np.nonzero(ok)[0]
    out_cp = cp[idx]
    if offset != 0.0 and len(idx):
        out_cp = out_cp + offset * surf.tri_normals[tri[idx]]
    stats = {
        "matched": int(len(idx)),
        "rejected_angle": n_angle,
        "rejected_boundary": n_boundary,
        "rejected_dist": n_dist,
        "threshold": threshold,
    }
    return idx, out_cp, dist[idx], stats


def precompute_rings(adj: "sparse.spmatrix", rings: Tuple[int, ...] = (1, 2)
                     ) -> list:
    """Per-ring vertex-index edge lists (row != col) for filter_matches_rigidity.

    Depends only on the mesh adjacency, so it is built ONCE per solve and reused
    across every matching iteration — the boolean ring matmuls dominate the
    filter otherwise (re-running them each iteration blew the perf budget on
    real assets). Returns [(rows_k, cols_k), ...], one entry per requested ring.
    """
    adj_bool = (adj != 0).tocsr()
    reach = adj_bool.copy()
    cur = 1
    out = []
    for K in rings:
        while cur < K:                        # reach = within-K-hops connectivity
            reach = (reach @ adj_bool) + reach
            reach = (reach != 0).tocsr()
            cur += 1
        coo = reach.tocoo()
        m = coo.row != coo.col                # strip the diagonal matmul adds
        out.append((coo.row[m].astype(np.int64), coo.col[m].astype(np.int64)))
    return out


def filter_matches_rigidity(
    verts: np.ndarray,            # (N,3) current source-org vertices
    ring_edges,                   # precompute_rings(...) list, OR a raw adjacency
    m_idx: np.ndarray,            # (M,) matched vertex indices
    m_cp: np.ndarray,             # (M,3) match targets (already offset)
    sigma_k: float = 0.5,
    min_neighbors: int = 3,
    max_neighbors: int = 24,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, list]]:
    """Rigidity-consistency filter (LUIVITON's noise-filtering module): drop
    matches that disagree with their neighbourhood's local rigid motion — a
    sleeve vertex captured by the wrong limb, one collar side gripping while
    the other doesn't. Each such match passes the per-vertex normal / distance
    tests; only the *coherence of the displacement field* exposes it.

    For each matched vertex i with source position v_i and target c_i, over its
    matched neighbours j: p_j = v_j - v_i (source edge), q_j = c_j - c_i
    (target edge). The optimal rotation R_i (Kabsch, det-corrected) gives the
    scale-invariant residual

        E_i = sum_j ||q_j - R_i p_j||^2 / sum_j ||p_j||^2

    Normalising by sum||p_j||^2 makes E independent of mesh units (LUIVITON's
    raw E works for them only because SMPL is metric-normalised; we are not).
    Matches with E_i > mu_E + sigma_k*sigma_E (statistics over currently-kept,
    judgeable matches) are dropped. Passes grow the neighbourhood ring and
    recompute stats on survivors; early-stop when a pass drops < 0.5% of M.

    `ring_edges` is the output of precompute_rings (cached per solve); a raw
    adjacency matrix is also accepted (rings built inline) for standalone use.

    Returns (keep mask over the M matches, {"pass_drops": [...]}).
    """
    if not isinstance(ring_edges, (list, tuple)):   # a raw adjacency was passed
        ring_edges = precompute_rings(ring_edges)

    m_idx = np.asarray(m_idx, dtype=np.int64).ravel()
    M = len(m_idx)
    keep = np.ones(M, dtype=bool)
    info: Dict[str, list] = {"pass_drops": []}
    if M < min_neighbors + 1:
        return keep, info

    N = len(verts)
    src_v = verts[m_idx]                      # (M,3) source positions of matches
    tgt_c = np.asarray(m_cp, dtype=float)     # (M,3) match targets
    row_of = np.full(N, -1, dtype=np.int64)   # vertex index -> match row
    row_of[m_idx] = np.arange(M)
    rng = np.random.default_rng(rng_seed)

    for rows, cols in ring_edges:
        # Restrict edges to (kept match) -> (kept match); map to match rows.
        # rows != cols is guaranteed by precompute_rings and row_of is injective
        # on matched verts, so ci != cj holds wherever both map >= 0.
        ci, cj = row_of[rows], row_of[cols]
        e_ok = (ci >= 0) & (cj >= 0) & keep[np.maximum(ci, 0)] \
            & keep[np.maximum(cj, 0)]
        ci, cj = ci[e_ok], cj[e_ok]
        if len(ci) == 0:
            info["pass_drops"].append(0)
            continue

        # Cap per-centre neighbours at max_neighbors by a seeded random subset
        # (a shuffle then keep the first N per centre-group = uniform sample).
        perm = rng.permutation(len(ci))
        cs = ci[perm]
        order = np.argsort(cs, kind="stable")
        cs_sorted = cs[order]
        _u, first, counts = np.unique(cs_sorted, return_index=True, return_counts=True)
        rank = np.arange(len(cs_sorted)) - np.repeat(first, counts)
        sel = perm[order[rank < max_neighbors]]
        ci, cj = ci[sel], cj[sel]

        # Per-edge source / target edge vectors and Kabsch covariance per centre.
        p = src_v[cj] - src_v[ci]             # (E,3)
        q = tgt_c[cj] - tgt_c[ci]
        contrib = p[:, :, None] * q[:, None, :]   # outer(p, q); R maps p -> q
        S = np.zeros((M, 3, 3))
        np.add.at(S, ci, contrib)
        n_nbr = np.zeros(M, dtype=np.int64)
        np.add.at(n_nbr, ci, 1)
        sum_pp = np.zeros(M)
        np.add.at(sum_pp, ci, np.einsum('ij,ij->i', p, p))

        U, _sig, Vt = np.linalg.svd(S)
        R = np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
        neg = np.linalg.det(R) < 0
        if neg.any():
            Vt_fix = Vt[neg].copy()
            Vt_fix[:, 2, :] *= -1.0
            R[neg] = np.matmul(Vt_fix.transpose(0, 2, 1), U[neg].transpose(0, 2, 1))

        resid = q - np.einsum('eab,eb->ea', R[ci], p)
        sum_resid = np.zeros(M)
        np.add.at(sum_resid, ci, np.einsum('ij,ij->i', resid, resid))
        E = sum_resid / np.where(sum_pp > 1e-18, sum_pp, 1.0)

        # Judgeable = kept, enough neighbours, non-degenerate source ring.
        judge = keep & (n_nbr >= min_neighbors) & (sum_pp > 1e-18)
        if judge.sum() < 2:
            info["pass_drops"].append(0)
            continue
        mu, sigma = float(E[judge].mean()), float(E[judge].std())
        # Near-uniform E (a globally rigid or globally scaled displacement field
        # has the SAME residual everywhere) — spread is pure float noise, so
        # drop nothing rather than culling a random tail at mu + k*sigma.
        if sigma <= 1e-9 * abs(mu) + 1e-12:
            info["pass_drops"].append(0)
            continue
        drop = judge & (E > mu + sigma_k * sigma)
        keep[drop] = False
        n_drop = int(drop.sum())
        info["pass_drops"].append(n_drop)
        if n_drop < 0.005 * M:                # diminishing returns; stop growing
            break

    return keep, info


def project_onto_surface(verts: np.ndarray, vert_normals: np.ndarray,
                         surf: SurfaceData,
                         max_dist: float,
                         max_angle_deg: float = 60.0,
                         weight: Optional[np.ndarray] = None,
                         reject_boundary: bool = True,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Final shrink-wrap: snap vertices onto the target surface.

    Uses the same safety filters as matching (normal compatibility, boundary
    rejection, distance cap) so vertices with no trustworthy surface under
    them are left where the solve put them. `weight` (per-vertex, 0..1)
    scales the snap — used for feather bands in regional wraps; None = full.

    Returns (new vertices, snapped mask).
    """
    cp, tri, dist, reg = closest_points_on_surface(verts, surf)

    cos_lim = float(np.cos(np.radians(max_angle_deg)))
    ok = np.einsum('ij,ij->i', surf.tri_normals[tri], vert_normals) >= cos_lim
    if reject_boundary:
        ok &= ~_on_open_boundary(surf, tri, reg)
    ok &= dist <= max_dist
    if weight is not None:
        ok &= weight > 0.0

    out = verts.copy()
    if ok.any():
        if weight is None:
            out[ok] = cp[ok]
        else:
            out[ok] = verts[ok] + weight[ok, None] * (cp[ok] - verts[ok])
    return out, ok


def _gather_crossing_candidates(surf: SurfaceData, mid: np.ndarray, half: np.ndarray
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """(query, triangle) candidate pairs for a batch of segments/rays whose
    midpoints are `mid` (Q,3) and half-extents `half` (Q,). Queries the normal
    and big centroid trees separately (see SurfaceData) so a few oversized
    triangles don't inflate the radius for the whole normal set.

    Returns (q_idx, tri_idx) flat global-index arrays (empty if no candidates).
    """
    q_parts, t_parts = [], []
    for tree, gidx, pad in ((surf.norm_tree, surf.norm_idx, surf.norm_r),
                            (surf.big_tree, surf.big_idx, surf.big_r)):
        if tree is None:
            continue
        cand = tree.query_ball_point(mid, half + pad, workers=_KD_WORKERS)
        counts = [len(c) for c in cand]
        if not any(counts):
            continue
        q = np.repeat(np.arange(len(mid), dtype=np.int64), counts)
        loc = np.concatenate([np.asarray(c, dtype=np.int64) for c in cand if len(c)])
        q_parts.append(q)
        t_parts.append(gidx[loc])
    if not q_parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(q_parts), np.concatenate(t_parts)


def ray_hits_ordered(
    origins: np.ndarray,          # (R,3)
    dirs: np.ndarray,             # (R,3) unit directions
    t_max: np.ndarray,            # (R,) per-ray cap (world units)
    surf: SurfaceData,            # mesh to intersect (the garment)
    exclude_verts: Optional[np.ndarray] = None,   # per-ray vertex to skip fans of
    t_eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ORDERED ray-mesh crossings — the body-side layer signal's primitive.

    Unlike segments_hit_mesh ("crossed: yes/no"), this returns EVERY crossing of
    each ray with `surf`, so a caller can read the 1st hit (driven vote) apart
    from the 2nd+ (follower votes), and later order stacked layers / buttons.

    Rays are `origins + t*dirs` (dirs unit, so t is a world distance) with hits
    kept for t in (t_eps, t_max[r]); t_max is per-ray because body erosion makes
    the useful cap ray-dependent. An edge crossing reports both incident
    triangles at (nearly) the same t — that is ONE cloth crossing, so after
    sorting we drop any hit whose t is within t_eps of the previous hit of the
    same ray.

    `exclude_verts` (aligned with rays) gives a vertex index per ray whose
    incident triangles are skipped — a ray leaving a vertex always "hits" its
    own fan.

    Returns (ray_idx, t, tri): flat arrays lexsorted by (ray_idx, t). Callers get
    first hits via np.unique(ray_idx, return_index=True).
    """
    R = len(origins)
    empty = (np.zeros(0, np.int64), np.zeros(0, float), np.zeros(0, np.int64))
    if R == 0 or len(surf.tri_a) == 0:
        return empty
    origins = np.asarray(origins, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    t_max = np.asarray(t_max, dtype=float).ravel()

    half = t_max * 0.5
    mid = origins + dirs * half[:, None]
    ray_idx, tri_idx = _gather_crossing_candidates(surf, mid, half)
    if len(ray_idx) == 0:
        return empty
    if exclude_verts is not None:
        exclude_verts = np.asarray(exclude_verts, dtype=np.int64)
        f = surf.faces[tri_idx]
        own = (f == exclude_verts[ray_idx, None]).any(axis=1)
        ray_idx, tri_idx = ray_idx[~own], tri_idx[~own]
        if len(ray_idx) == 0:
            return empty

    # Möller–Trumbore over (ray, candidate) pairs; d is unit so t is world dist.
    # Canonical (winding-independent) corners: identical under face inversion.
    o = origins[ray_idx]
    d = dirs[ray_idx]
    e1 = surf.tri_cab[tri_idx]
    e2 = surf.tri_cac[tri_idx]
    a = surf.tri_ca[tri_idx]
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, det, 1.0)
    tvec = o - a
    u = np.einsum('ij,ij->i', tvec, p) / inv
    q = np.cross(tvec, e1)
    v = np.einsum('ij,ij->i', d, q) / inv
    t = np.einsum('ij,ij->i', e2, q) / inv
    ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    ok &= (t > t_eps) & (t < t_max[ray_idx])
    ray_idx, tri_idx, t = ray_idx[ok], tri_idx[ok], t[ok]
    if len(ray_idx) == 0:
        return empty

    order = np.lexsort((t, ray_idx))
    ray_idx, t, tri_idx = ray_idx[order], t[order], tri_idx[order]
    # Dedup shared-edge double reports: same ray AND t within t_eps of the prev.
    dup = np.zeros(len(t), dtype=bool)
    dup[1:] = (ray_idx[1:] == ray_idx[:-1]) & ((t[1:] - t[:-1]) < t_eps)
    keep = ~dup
    return ray_idx[keep], t[keep], tri_idx[keep]


def segments_hit_mesh(starts: np.ndarray, ends: np.ndarray, surf: SurfaceData,
                      exclude_verts: Optional[np.ndarray] = None,
                      t_eps: float = 1e-4) -> np.ndarray:
    """For each segment start->end: does it cross a triangle of `surf`?

    Used by refit's layer classification: a garment vertex whose segment to
    its closest BODY point crosses the garment itself is an OUTER layer (there
    is cloth between it and the skin), so it must not grip. Segments there are
    short (contact-band only), so the centroid-KD candidate gathering stays
    cheap.

    `exclude_verts` (aligned with segments) gives a vertex index per segment;
    candidate triangles containing that vertex are skipped — a vertex's own
    fan always "intersects" a segment leaving from it. Möller–Trumbore over
    (segment, candidate-triangle) pairs; a hit counts only for t in
    (t_eps, 1 - t_eps) so grazing the endpoints doesn't flag.
    """
    n = len(starts)
    hit = np.zeros(n, dtype=bool)
    if n == 0 or len(surf.tri_a) == 0:
        return hit
    seg = ends - starts
    seg_len = np.linalg.norm(seg, axis=1)
    mid = 0.5 * (starts + ends)
    # Candidate gathering: half the segment plus a per-set triangle radius, split
    # normal/big so oversized triangles don't inflate the query (see
    # _gather_crossing_candidates), missing no potentially-crossed triangle.
    seg_idx, tri_idx = _gather_crossing_candidates(surf, mid, seg_len * 0.5)
    if len(seg_idx) == 0:
        return hit
    if exclude_verts is not None:
        f = surf.faces[tri_idx]
        own = (f == exclude_verts[seg_idx, None]).any(axis=1)
        seg_idx, tri_idx = seg_idx[~own], tri_idx[~own]
        if len(seg_idx) == 0:
            return hit

    # Möller–Trumbore, vectorized over pairs. d = full segment vector, so the
    # ray parameter t is already normalized to [0, 1] along the segment.
    # Canonical (winding-independent) corners: identical under face inversion.
    o = starts[seg_idx]
    d = seg[seg_idx]
    e1 = surf.tri_cab[tri_idx]
    e2 = surf.tri_cac[tri_idx]
    a = surf.tri_ca[tri_idx]
    p = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, det, 1.0)
    tvec = o - a
    u = np.einsum('ij,ij->i', tvec, p) / inv
    q = np.cross(tvec, e1)
    v = np.einsum('ij,ij->i', d, q) / inv
    t = np.einsum('ij,ij->i', e2, q) / inv
    ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    ok &= (t > t_eps) & (t < 1.0 - t_eps)
    hit[seg_idx[ok]] = True
    return hit


def damp_folds(prev_verts: np.ndarray, new_verts: np.ndarray, faces: np.ndarray,
               rounds: int = 3) -> Tuple[np.ndarray, int]:
    """Anti-fold safeguard: any triangle whose normal flipped relative to
    `prev_verts` gets its vertices pulled halfway back toward their previous
    (unflipped) position, repeatedly. Pinned vertices are naturally immune —
    their prev == new, so the blend is a no-op for them.

    Returns (damped vertices copy, number of flipped triangles detected).
    """
    f = faces[:, :3]

    def raw_normals(v: np.ndarray) -> np.ndarray:
        return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])

    ref = raw_normals(prev_verts)
    out = new_verts.copy()
    first = 0
    for round_no in range(rounds):
        flipped = np.einsum('ij,ij->i', raw_normals(out), ref) < 0
        count = int(flipped.sum())
        if round_no == 0:
            first = count
        if count == 0:
            break
        vidx = np.unique(f[flipped])
        out[vidx] = 0.5 * (out[vidx] + prev_verts[vidx])
    return out, first
