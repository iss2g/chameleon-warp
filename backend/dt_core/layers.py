"""
Layer auto-segmentation: score every garment vertex on how DRIVEN it is (grips
and inflates via correspondence) vs. how much it merely FOLLOWS an inner wall.

The governing principle (see docs/design/layer-autoseg.md):

    The driven (inner) wall is the garment surface the BODY sees first.

Rays start on the body and travel along the body's own outward normals; the
first cloth crossing is driven, later crossings are followers. Garment normals
are never read — game assets flip them; body normals are ours and trusted.

Two orientation-independent signals, both continuous, both computed against a
penetration-corrected ("eroded") reference body:

  A  body-side first-hit votes   — primary; body rays vote driven for the FIRST
                                    garment crossing, follower for later ones.
  B  fan-out occlusion fraction  — the legacy single-segment occlusion test, but
                                    averaged over k body directions per vertex.

`layer_scores` combines them into a per-vertex driven-ness in [0, 1];
`released_mask_auto` (L3) turns that into a released mask + per-vertex layer id
via smoothing, hysteresis and a connected-component policy.

Pure numpy/scipy over (vertices, faces) — no solver, no FastAPI, headless.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Combined-score weights (module constants, NOT user knobs — see plan L2).
_W_VOTES = 0.5          # weight of signal A (first-hit votes) where covered
_W_OCCL = 0.5           # weight of signal B (occlusion fraction)


def erosion_depth(body_signed: np.ndarray, thickness: float, body_diag: float,
                  penetration_gate: float = 0.02) -> float:
    """Depth to deflate the reference body along its normals before classifying
    (and, in run_refit, before growing).

    If fewer than `penetration_gate` of the garment verts penetrate the body,
    the reference is the true body (returns 0.0). Otherwise the reference is
    deflated by `min(0.15*body_diag, quantile_0.95(penetration) + thickness)`.
    Inflating/deflating the body a few cm along its normals never reorders cloth
    layers along the skin-outward direction, so a classification (or growth)
    computed against the eroded body is valid for the true one.

    ONE definition, called from both layer_scores and run_refit's growth block.
    """
    s = np.asarray(body_signed, dtype=float)
    n = len(s)
    pen = -s[s < 0.0]
    if n == 0 or len(pen) < penetration_gate * n:
        return 0.0
    return float(min(0.15 * body_diag,
                     float(np.quantile(pen, 0.95)) + float(thickness)))


def _tangent_frame(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal tangent/bitangent per unit normal (R,3):
    Gram–Schmidt against global up (0,0,1), falling back to X where n ∥ up."""
    up = np.array([0.0, 0.0, 1.0])
    t = up[None, :] - n * np.einsum('ij,j->i', n, up)[:, None]
    tl = np.linalg.norm(t, axis=1)
    parallel = tl < 1e-6
    if parallel.any():
        xf = np.array([1.0, 0.0, 0.0])
        t[parallel] = xf[None, :] - n[parallel] * np.einsum('ij,j->i', n[parallel], xf)[:, None]
        tl = np.linalg.norm(t, axis=1)
    t = t / np.where(tl == 0, 1.0, tl)[:, None]
    b = np.cross(n, t)
    return t, b


def _cone_dirs(n: np.ndarray, cone_deg: float, n_cone: int) -> Tuple[np.ndarray, np.ndarray]:
    """For each unit normal, the normal itself plus `n_cone` directions tilted
    `cone_deg` around it (evenly spaced in azimuth). Returns
    (dirs (R*(1+n_cone), 3), src_ray (R*(1+n_cone),) index back to the origin).
    """
    R = len(n)
    dirs = [n]
    src = [np.arange(R)]
    if n_cone > 0 and cone_deg > 0:
        t, b = _tangent_frame(n)
        ca, sa = np.cos(np.radians(cone_deg)), np.sin(np.radians(cone_deg))
        for c in range(n_cone):
            phi = 2 * np.pi * c / n_cone
            tang = np.cos(phi) * t + np.sin(phi) * b
            dirs.append(ca * n + sa * tang)
            src.append(np.arange(R))
    D = np.concatenate(dirs, axis=0)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return D, np.concatenate(src)


def body_first_hit_votes(
    ref_verts: np.ndarray,        # (B,3) reference body verts (eroded or true)
    body_vn: np.ndarray,          # (B,3) TRUE body vertex normals
    gsurf,                        # build_surface of the PLACED garment
    body_surf_ref,                # build_surface of the reference body
    cap: float,                   # scalar ray length (world units)
    n_verts_g: int,
    keep_origin: Optional[np.ndarray] = None,   # bool (B,) prefilter
    cone_deg: float = 15.0, n_cone: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Signal A. Fire rays from the reference body along its OWN outward normals
    (plus a small cone). For each ray the FIRST garment crossing votes driven on
    that triangle's 3 verts; every later crossing votes follower. Body self-
    occlusion is respected: garment hits beyond the first body self-hit are
    discarded (armpit skin facing skin does not see through flesh).

    Returns (driven_votes, follower_votes) per garment vertex (len n_verts_g).
    No garment normals are read anywhere.
    """
    from .surface import ray_hits_ordered

    driven = np.zeros(n_verts_g, dtype=float)
    follower = np.zeros(n_verts_g, dtype=float)
    idx = (np.nonzero(keep_origin)[0] if keep_origin is not None
           else np.arange(len(ref_verts)))
    if len(idx) == 0 or len(gsurf.tri_a) == 0:
        return driven, follower

    origins0 = ref_verts[idx]
    n0 = body_vn[idx]
    n0 = n0 / np.where(np.linalg.norm(n0, axis=1) == 0, 1.0,
                       np.linalg.norm(n0, axis=1))[:, None]
    dirs, src = _cone_dirs(n0, cone_deg, n_cone)
    origins = origins0[src]
    origin_vert = idx[src]                     # body vertex each ray leaves from
    n_ray = len(dirs)
    t_max = np.full(n_ray, float(cap))

    # Garment crossings first (ordered). If a ray never reaches the garment it
    # contributes nothing, so we cast the (much more expensive) body self-
    # occlusion ONLY up to each ray's farthest garment hit — a short distance
    # (~the cloth standoff) instead of the full cap. Grazing rays along the dense
    # body would otherwise gather an enormous local patch for the full length.
    g_ray, g_t, g_tri = ray_hits_ordered(origins, dirs, t_max, gsurf)
    if len(g_ray) == 0:
        return driven, follower
    gmax = np.zeros(n_ray, dtype=float)
    np.maximum.at(gmax, g_ray, g_t)            # per-ray farthest garment hit
    # Body self-occlusion capped per-ray at gmax (0 => no garment hit => no cast,
    # the t>t_eps & t<t_max=0 filter yields nothing). exclude the origin's fan.
    b_ray, b_t, _b_tri = ray_hits_ordered(origins, dirs, gmax, body_surf_ref,
                                          exclude_verts=origin_vert)
    # Nearest body self-hit per ray (inf where none) — the occlusion horizon.
    horizon = np.full(n_ray, np.inf)
    if len(b_ray):
        u, first = np.unique(b_ray, return_index=True)
        horizon[u] = b_t[first]

    # Keep garment hits in front of the body horizon; re-rank per ray.
    vis = g_t < horizon[g_ray]
    g_ray, g_t, g_tri = g_ray[vis], g_t[vis], g_tri[vis]
    if len(g_ray) == 0:
        return driven, follower
    order = np.lexsort((g_t, g_ray))
    g_ray, g_t, g_tri = g_ray[order], g_t[order], g_tri[order]
    is_first = np.zeros(len(g_ray), dtype=bool)
    is_first[0] = True
    is_first[1:] = g_ray[1:] != g_ray[:-1]

    gf = gsurf.faces                            # (T,3)
    first_tris = g_tri[is_first]
    later_tris = g_tri[~is_first]
    if len(first_tris):
        np.add.at(driven, gf[first_tris].ravel(), 1.0)
    if len(later_tris):
        np.add.at(follower, gf[later_tris].ravel(), 1.0)
    return driven, follower


# -----------------------------------------------------------------------------
# Signal A, parallelized. The ray cast in body_first_hit_votes is per-ray
# independent and its votes accumulate additively (+1.0 on garment-triangle
# verts), so splitting the body origins across processes and summing the partial
# vote arrays is BITWISE-identical to the single-process result (a float sum of
# 1.0s is exact below 2^53, order-independent). This just spreads the CURRENT
# work over cores — the deep fix is a tighter candidate broad-phase in
# surface.ray_hits_ordered (see docs/ROADMAP.md, "Performance").
# -----------------------------------------------------------------------------

# Minimum kept-origin count to bother with process spawn: spawning + pickling
# the body/garment surfaces (~10s of MB each) costs ~1-2s, only worth amortizing
# above a few thousand origins.
_MP_MIN_ORIGINS = 4000

_VOTE_CTX = None   # per-worker read-only inputs, set by _vote_init under spawn


def _vote_init(ref_verts, body_vn, gsurf, body_surf_ref, cap, n_g, cone_deg, n_cone):
    """Pool initializer: stash the (large, read-only) inputs in a process global
    so pool.map only ships the small per-chunk origin indices. Also pin this
    worker's KD broad-phase to a single thread — N processes each defaulting to
    all-cores would oversubscribe N*N threads."""
    global _VOTE_CTX
    from . import surface as _surface
    _surface._KD_WORKERS = 1
    _VOTE_CTX = (ref_verts, body_vn, gsurf, body_surf_ref, cap, n_g, cone_deg, n_cone)


def _vote_chunk(origin_idx):
    """Worker: vote using only the body origins in this chunk."""
    (ref_verts, body_vn, gsurf, body_surf_ref, cap, n_g, cone_deg, n_cone) = _VOTE_CTX
    keep = np.zeros(len(ref_verts), dtype=bool)
    keep[origin_idx] = True
    return body_first_hit_votes(ref_verts, body_vn, gsurf, body_surf_ref, cap, n_g,
                                keep_origin=keep, cone_deg=cone_deg, n_cone=n_cone)


def _votes_parallel(ref_verts, body_vn, gsurf, body_surf_ref, cap, n_g,
                    keep_origin, cone_deg, n_cone, n_workers):
    """body_first_hit_votes across processes, split by body origin. Falls back to
    the serial primitive when it isn't worth it or spawn fails — result identical."""
    idx = (np.nonzero(keep_origin)[0] if keep_origin is not None
           else np.arange(len(ref_verts)))

    def _serial():
        return body_first_hit_votes(ref_verts, body_vn, gsurf, body_surf_ref, cap,
                                    n_g, keep_origin=keep_origin,
                                    cone_deg=cone_deg, n_cone=n_cone)

    if n_workers <= 1 or len(idx) < _MP_MIN_ORIGINS:
        return _serial()
    n_workers = min(int(n_workers), len(idx))
    # Per-origin ray cost is very uneven (grazing rays over dense body regions
    # gather huge candidate sets), and origins are in spatially-contiguous vertex
    # order, so contiguous chunks straggle badly. Shuffle deterministically to
    # spread hot origins, and cut MORE chunks than workers so the pool balances
    # stragglers dynamically. Shuffle/rechunk changes nothing numerically —
    # votes are an order-independent additive sum.
    idx = np.random.default_rng(0).permutation(idx)
    n_chunks = min(len(idx), n_workers * 4)
    chunks = [c for c in np.array_split(idx, n_chunks) if len(c)]
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers, initializer=_vote_init,
                      initargs=(ref_verts, body_vn, gsurf, body_surf_ref, cap,
                                n_g, cone_deg, n_cone)) as pool:
            parts = pool.map(_vote_chunk, chunks)
    except Exception:
        # Frozen exe, restricted sandbox, pickling surprise -> exact serial path.
        return _serial()
    driven = np.zeros(n_g, dtype=float)
    follower = np.zeros(n_g, dtype=float)
    for d, f in parts:
        driven += d
        follower += f
    return driven, follower


def occluded_fraction(
    g_verts: np.ndarray,          # (V,3) placed garment verts
    active_idx: np.ndarray,       # (A,) verts with field > 0
    ref_verts: np.ndarray,        # (B,3) reference body verts
    gsurf,                        # build_surface of the placed garment
    k: int = 8,
) -> np.ndarray:
    """Signal B. For each active garment vertex, the fraction of its k nearest
    reference-body directions whose segment to the body crosses the garment
    itself (i.e. there is cloth between it and the skin). Fan-out upgrade of the
    legacy single-segment occlusion test — one gap to the shoulder no longer
    whitewashes an outer wall.

    Returns a (len(active_idx),) fraction in [0, 1].
    """
    from scipy.spatial import cKDTree
    from .surface import segments_hit_mesh

    A = len(active_idx)
    if A == 0:
        return np.zeros(0, dtype=float)
    tree = cKDTree(ref_verts)
    k_eff = max(1, min(k, len(ref_verts)))
    starts = g_verts[active_idx]
    _d, nbr = tree.query(starts, k_eff, workers=-1)
    if k_eff == 1:
        nbr = nbr[:, None]

    # Batch all A*k segments into ONE crossing test: candidates are gathered
    # once and Möller runs in a single vectorized pass, instead of k separate
    # calls each re-gathering (the Python per-call overhead dominated otherwise).
    starts_b = np.repeat(starts, k_eff, axis=0)
    ends_b = ref_verts[nbr.ravel()]
    excl_b = np.repeat(active_idx, k_eff)
    hit = segments_hit_mesh(starts_b, ends_b, gsurf, exclude_verts=excl_b)
    return hit.reshape(A, k_eff).mean(axis=1)


@dataclass
class LayerScores:
    score: np.ndarray        # per-vertex driven-ness in [0,1] (0 where inactive)
    a_driven: np.ndarray     # raw first-hit votes (stats/preview/debug)
    a_follower: np.ndarray
    occl_frac: np.ndarray    # over all verts, -1 where not computed
    coverage: np.ndarray     # bool: vertex received any A votes
    e_cls: float             # erosion depth used for the reference body


def layer_scores(
    placed_verts: np.ndarray,
    faces: np.ndarray,
    target_org,                   # meshlib.Mesh — the body
    body_signed: np.ndarray,      # signed distance of placed verts to the body
    body_vn: np.ndarray,          # TRUE body vertex normals (len == body verts)
    active: np.ndarray,           # bool over garment verts: field > 0
    contact_free: float,          # world units (RefitField.free_d)
    thickness: float,
    body_diag: float,
    cone_deg: float = 15.0,
    n_cone: int = 4,
    occl_k: int = 6,
    max_origins: Optional[int] = 12000,
    rng_seed: int = 0,
    n_workers: int = 1,
) -> LayerScores:
    """Combine signals A and B into a per-vertex driven-ness score in [0, 1].

    Reference body: deflated by erosion_depth so deep penetration at placement
    doesn't invert the layer ordering (submerged inner wall still classifies as
    driven). Score = 0.5*a + 0.5*b where covered (a = driven/(driven+follower)),
    b = 1 - occl_frac; where A saw nothing, b alone.
    """
    from . import meshlib
    from .surface import build_surface
    from .correspondence import get_vertex_normals

    n_g = len(placed_verts)
    active = np.asarray(active, dtype=bool)
    e_cls = erosion_depth(body_signed, thickness, body_diag)

    # Reference body verts (eroded) + build its surface for self-occlusion.
    body_v = np.asarray(target_org.vertices, dtype=float)
    bvn = body_vn / np.where(np.linalg.norm(body_vn, axis=1) == 0, 1.0,
                             np.linalg.norm(body_vn, axis=1))[:, None]
    ref_verts = body_v - e_cls * bvn
    body_surf_ref = build_surface(ref_verts, target_org.faces)
    gsurf = build_surface(placed_verts, faces)

    # Ray reach for signal A. The votes only need to reach STACKED walls (a
    # folded collar's inner+outer gap is small); the full loose-cloth free band
    # (2*contact_free) would make every ray's candidate ball huge on a dense
    # body and blow the perf budget for no classification gain — genuinely
    # occluded loose cloth is caught by signal B (whose segments to k-nearest
    # body verts are short by construction). So cap the rays at the erosion depth
    # plus a layer-stacking distance.
    cap = float(e_cls + max(0.03 * body_diag, 8.0 * thickness))
    from scipy.spatial import cKDTree
    active_idx = np.nonzero(active)[0]
    keep_origin = None
    if len(active_idx):
        gtree = cKDTree(placed_verts[active_idx])
        d_ref, _ = gtree.query(ref_verts, 1, workers=-1)
        keep_origin = d_ref <= cap
        # Decimate origins on a dense body: votes accumulate on garment triangles
        # and are then smoothed over the garment graph, so a seeded random subset
        # of body origins still covers every garment triangle several times over
        # (the body is denser than the garment). Bounds the per-solve ray budget.
        n_keep = int(keep_origin.sum())
        if max_origins and n_keep > max_origins:
            sel = np.nonzero(keep_origin)[0]
            drop = np.random.default_rng(rng_seed).choice(
                sel, n_keep - max_origins, replace=False)
            keep_origin[drop] = False

    a_driven, a_follower = _votes_parallel(
        ref_verts, bvn, gsurf, body_surf_ref, cap, n_g,
        keep_origin=keep_origin, cone_deg=cone_deg, n_cone=n_cone,
        n_workers=n_workers)
    coverage = (a_driven + a_follower) > 0

    # Signal B over active verts (against the eroded reference).
    occl_frac = np.full(n_g, -1.0, dtype=float)
    if len(active_idx):
        occl_frac[active_idx] = occluded_fraction(
            placed_verts, active_idx, ref_verts, gsurf, k=occl_k)

    # Combine. a in [0,1] where covered (neutral 0.5 else); b = 1 - occl_frac.
    a = np.full(n_g, 0.5, dtype=float)
    tot = a_driven + a_follower
    a[coverage] = a_driven[coverage] / tot[coverage]
    b = np.where(occl_frac >= 0.0, 1.0 - occl_frac, 0.5)

    score = np.zeros(n_g, dtype=float)
    both = active & coverage
    score[both] = _W_VOTES * a[both] + _W_OCCL * b[both]
    only_b = active & ~coverage
    score[only_b] = b[only_b]                  # A saw nothing -> occlusion alone
    return LayerScores(score=score, a_driven=a_driven, a_follower=a_follower,
                       occl_frac=occl_frac, coverage=coverage, e_cls=e_cls)


def released_mask_auto(
    scores: LayerScores,
    verts: np.ndarray,
    faces: np.ndarray,
    active: np.ndarray,
    hi: float = 0.6, lo: float = 0.4,
    comp_driven_min: float = 0.15,
    comp_area_min: float = 0.02,
    frozen_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Turn raw per-vertex scores into a released mask + per-vertex layer id.

    layer: 0 = driven, 1 = follower (wall), 2 = follower (whole component).
    Steps: (1) smooth the score over the vertex graph; (2) hysteresis on
    (lo, hi) with graph-neighbor majority in the band; (3) component policy —
    a connected component with too little driven area (or too small overall) is
    demoted whole to layer 2 (pouches, buckles, buttons, straps).

    Returns (released bool over all verts, layer int8 over all verts, stats).
    Release happens only on evidence: undecided verts stay driven (legacy).
    """
    from . import regions
    from scipy.sparse.csgraph import connected_components

    n = len(verts)
    active = np.asarray(active, dtype=bool)
    layer = np.zeros(n, dtype=np.int8)
    released = np.zeros(n, dtype=bool)
    active_idx = np.nonzero(active)[0]
    if len(active_idx) == 0:
        return released, layer, {"n_active": 0}

    # ---- (1) smoothing: 3 rounds of row-normalized neighbor averaging ----
    adj = regions.weighted_adjacency(verts, faces)
    a_bool = (adj != 0)
    deg = np.asarray(a_bool.sum(axis=1)).ravel()
    inv_deg = 1.0 / np.where(deg == 0, 1.0, deg)
    s = scores.score.copy()
    amask = active.astype(float)
    for _ in range(3):
        nb_sum = a_bool.dot(s * amask)
        nb_cnt = a_bool.dot(amask)
        avg = nb_sum / np.where(nb_cnt == 0, 1.0, nb_cnt)
        s = np.where(active & (nb_cnt > 0), avg, s)

    # ---- (2) hysteresis with graph-neighbor majority in the band ----
    # label: -1 undecided, 0 driven, 1 follower. Only active verts get labeled.
    label = np.full(n, -1, dtype=np.int8)
    label[active & (s >= hi)] = 0
    label[active & (s <= lo)] = 1
    band = active & (s > lo) & (s < hi)
    neigh_idx = a_bool.tocsr()
    for _ in range(10):
        undecided = np.nonzero(band & (label < 0))[0]
        if len(undecided) == 0:
            break
        changed = False
        # majority vote of already-decided neighbors
        driven_vote = a_bool.dot((label == 0).astype(float))
        follow_vote = a_bool.dot((label == 1).astype(float))
        for v in undecided:
            dv, fv = driven_vote[v], follow_vote[v]
            if dv == 0 and fv == 0:
                continue
            label[v] = 0 if dv >= fv else 1
            changed = True
        if not changed:
            break
    # leftovers in the band with no decided neighbor -> driven (legacy behavior)
    label[band & (label < 0)] = 0
    # any active vert still unlabeled (shouldn't happen) -> driven
    label[active & (label < 0)] = 0

    # ---- (3) component policy ----
    _nc, comp = connected_components(adj, directed=False)
    # per-vertex area = 1/3 sum of incident triangle areas
    f = faces[:, :3]
    tri_area = 0.5 * np.linalg.norm(
        np.cross(verts[f[:, 1]] - verts[f[:, 0]],
                 verts[f[:, 2]] - verts[f[:, 0]]), axis=1)
    varea = np.zeros(n, dtype=float)
    np.add.at(varea, f.ravel(), np.repeat(tri_area, 3) / 3.0)
    total_area = float(varea[active].sum()) if active.any() else 0.0

    layer[active_idx] = np.where(label[active_idx] == 0, 0, 1).astype(np.int8)
    comp_stats = {"components": int(_nc), "demoted_components": 0}
    for c in range(_nc):
        cverts = np.nonzero(comp == c)[0]
        ca = cverts[active[cverts]]
        if len(ca) == 0:
            continue
        comp_area = float(varea[ca].sum())
        driven_area = float(varea[ca[label[ca] == 0]].sum())
        frac = driven_area / comp_area if comp_area > 0 else 0.0
        too_small = total_area > 0 and comp_area < comp_area_min * total_area
        if frac < comp_driven_min or too_small:
            layer[ca] = 2
            comp_stats["demoted_components"] += 1

    released = layer >= 1
    # Frozen paint is the absolute override elsewhere; keep it consistent here by
    # not counting frozen verts as driven handles (they are released downstream).
    if frozen_idx is not None and len(frozen_idx):
        released[np.asarray(frozen_idx, np.int64)] = True

    stats = {
        "n_active": int(len(active_idx)),
        "driven": int((layer == 0).sum()),
        "follower_wall": int((layer == 1).sum()),
        "follower_component": int((layer == 2).sum()),
        "coverage_frac": float(scores.coverage[active].mean()) if active.any() else 0.0,
        "e_cls": float(scores.e_cls),
        **comp_stats,
    }
    return released, layer, stats
