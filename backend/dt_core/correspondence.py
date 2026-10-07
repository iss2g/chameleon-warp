"""
Computes the correspondence between vertices of two models.
Adapted from the original project for API use:
  - removed imports of `config` and `render.plot`
  - removed plot=True hooks (the API never plots)
  - compute_correspondence now returns BOTH the inflated source mesh and the triangle mapping
  - split into precompute_source / precompute_target / compute_correspondence_with_setup
    so the session can cache expensive per-mesh setup across runs
  - vectorized get_vertex_normals (was a Python loop over ~50k verts)
  - cKDTree.query now uses all cores (workers=-1)
  - sparse LU solve via dt_core.solver (pypardiso if available, else SuperLU)
"""
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Dict, Set, List, Optional

import numpy as np
import tqdm
import scipy.sparse as sparse
from scipy.spatial import cKDTree

from . import meshlib
from . import regions
from .meshlib.cache import SparseMatrixCache
from .solver import spsolve, backend_name
from .surface import (SurfaceData, build_surface, match_points_to_surface,
                      damp_folds, filter_matches_rigidity, precompute_rings)


# -----------------------------------------------------------------------------
# Mesh adjacency
# -----------------------------------------------------------------------------

def compute_adjacent_by_edges(mesh: meshlib.Mesh):
    """Computes the adjacent triangles by using the edges"""
    candidates = defaultdict(set)
    for n, f in enumerate(mesh.faces):
        f0, f1, f2 = sorted(f)
        candidates[(f0, f1)].add(n)
        candidates[(f0, f2)].add(n)
        candidates[(f1, f2)].add(n)

    faces_adjacent: Dict[int, Set[int]] = defaultdict(set)
    for faces in candidates.values():
        for f in faces:
            faces_adjacent[f].update(faces)

    faces_sorted = sorted([(f, [a for a in adj if a != f]) for f, adj in faces_adjacent.items()], key=lambda e: e[0])
    return [adj for f, adj in faces_sorted]


# -----------------------------------------------------------------------------
# Closest point search
# -----------------------------------------------------------------------------

def get_aec(columns, rows):
    return sparse.identity(columns, dtype=float, format="csc")[:rows]


def get_bec(closest_points: np.array, verts: np.array):
    return verts[closest_points]


def get_closest_points(kd_tree: cKDTree, verts: np.array, vert_normals: np.array, target_normals: np.array,
                       max_angle: float = np.radians(90), ks: int = 200) -> np.ndarray:
    assert len(verts) == len(vert_normals)
    closest_points: List[Tuple[int, int]] = []
    # `workers=-1` -> all cores; this scales well on a multi-core box.
    dists, indicies = kd_tree.query(verts, min(len(target_normals), ks), workers=-1)
    for v, (dist, ind) in enumerate(zip(dists, indicies)):
        # clip: dots of near-parallel unit normals can land at 1+eps -> arccos NaN,
        # which silently discarded the BEST candidate and fell back to a worse one.
        angles = np.arccos(np.clip(np.dot(target_normals[ind], vert_normals[v]), -1.0, 1.0))
        angles_cond = np.abs(angles) < max_angle
        if angles_cond.any():
            cind = ind[angles_cond][0]
            closest_points.append((v, cind))
    return np.array(closest_points)


# -----------------------------------------------------------------------------
# Geometry helpers (vectorized)
# -----------------------------------------------------------------------------

def get_triangle_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vns = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    return (vns.T / np.linalg.norm(vns, axis=1)).T


def get_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Mean of incident triangle normals per vertex, normalized.

    Vectorized: instead of a Python loop building sets per vertex (slow on
    ~50k verts) we accumulate triangle normals into per-vertex buckets via
    np.add.at, then divide by the per-vertex degree.
    """
    max_index = int(np.max(faces[:, :3])) + 1
    tri_normals = get_triangle_normals(verts, faces)  # (T, 3)

    vertex_sum = np.zeros((max_index, 3), dtype=float)
    vertex_count = np.zeros(max_index, dtype=np.int64)

    # Three indices per face contribute the same triangle normal.
    for col in range(3):
        np.add.at(vertex_sum, faces[:, col], tri_normals)
        np.add.at(vertex_count, faces[:, col], 1)

    # Avoid divide-by-zero for orphan vertices
    safe_count = np.where(vertex_count == 0, 1, vertex_count)[:, None]
    mean = vertex_sum / safe_count

    norms = np.linalg.norm(mean, axis=1)
    safe_norms = np.where(norms == 0, 1, norms)[:, None]
    return mean / safe_norms


def max_triangle_length(mesh: meshlib.Mesh):
    a, b, c = mesh.span_components()
    return max(np.max(np.linalg.norm(a, axis=1)), np.max(np.linalg.norm(b, axis=1)))


def match_triangles(source: meshlib.Mesh, target: meshlib.Mesh, factor=2) -> List[Tuple[int, int]]:
    source_centroids = source.get_centroids()
    target_centroids = target.get_centroids()
    source_normals = source.normals()
    target_normals = target.normals()
    radius = max(max_triangle_length(source), max_triangle_length(target)) * factor
    triangles = get_closest_triangles(source_normals, target_normals, source_centroids, target_centroids, radius)
    tmp_triangles = get_closest_triangles(target_normals, source_normals, target_centroids, source_centroids, radius)
    triangles.update((t[1], t[0]) for t in tmp_triangles)
    return list(triangles)


def get_closest_triangles(
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        source_centroids: np.ndarray,
        target_centroids: np.ndarray,
        max_angle: float = np.radians(90),
        k: int = 500,
        radius: float = np.inf
) -> Set[Tuple[int, int]]:
    assert len(source_normals) == len(source_centroids)
    assert len(target_normals) == len(target_centroids)
    triangles = set()
    kd_tree = cKDTree(target_centroids)
    dists, indicies = kd_tree.query(source_centroids, min(len(target_centroids), k),
                                    distance_upper_bound=radius, workers=-1)
    for index_source, (dist, ind) in enumerate(zip(dists, indicies)):
        angles = np.arccos(np.clip(np.dot(target_normals[ind], source_normals[index_source]), -1.0, 1.0))
        angles_cond = angles < max_angle
        if angles_cond.any():
            index_target = ind[angles_cond][0]
            triangles.add((index_source, index_target))
    return triangles


# -----------------------------------------------------------------------------
# Sparse matrix construction for triangle transformations
# -----------------------------------------------------------------------------

class TransformMatrix:
    __row_partial_baked = np.array([0, 1, 2] * 4)

    @classmethod
    def expand(cls, f: np.ndarray, inv: np.ndarray, size: int):
        i0, i1, i2, i3 = f
        col = np.array([i0, i0, i0, i1, i1, i1, i2, i2, i2, i3, i3, i3])
        data = np.concatenate([-inv.sum(axis=0), *inv])
        return sparse.coo_matrix((data, (cls.__row_partial_baked, col)), shape=(3, size), dtype=float)

    @classmethod
    def construct(cls, faces: np.ndarray, invVs: np.ndarray, size: int, desc="Building Transformation Matrix"):
        assert len(faces) == len(invVs)
        return sparse.vstack([
            cls.expand(f, inv, size) for f, inv in tqdm.tqdm(zip(faces, invVs), total=len(faces), desc=desc)
        ], dtype=float)


def apply_pins(A: sparse.spmatrix, b: np.ndarray, pin_idx: np.ndarray, pin_pos: np.ndarray):
    """Move hard-constrained ("pinned") columns to the right-hand side.

    Generalizes apply_markers: instead of always pinning to a target
    vertex position, the caller supplies explicit pin positions. This lets
    regional wrap pin some vertices to the *target* (user markers) and
    others to their own *source* position (the frozen region) in one pass.
    """
    assert pin_idx.ndim == 1 and pin_pos.shape == (len(pin_idx), 3)
    invpin = np.setdiff1d(np.arange(A.shape[1]), pin_idx)
    zb = b - A[:, pin_idx] * pin_pos
    return A[:, invpin].tocsc(), zb


def revert_pins(A: sparse.spmatrix, x: np.ndarray, pin_idx: np.ndarray, pin_pos: np.ndarray,
                *, out=None):
    if out is None:
        out = np.zeros((A.shape[1] + len(pin_idx), 3))
    else:
        assert out.shape == (A.shape[1] + len(pin_idx), 3)
    invpin = np.setdiff1d(np.arange(len(out)), pin_idx)
    out[invpin] = x
    out[pin_idx] = pin_pos
    return out


def apply_markers(A: sparse.spmatrix, b: np.ndarray, target: meshlib.Mesh, markers: np.ndarray):
    assert markers.ndim == 2 and markers.shape[1] == 2
    return apply_pins(A, b, markers[:, 0], target.vertices[markers[:, 1]])


def revert_markers(A: sparse.spmatrix, x: np.ndarray, target: meshlib.Mesh, markers: np.ndarray,
                   *, out=None):
    return revert_pins(A, x, markers[:, 0], target.vertices[markers[:, 1]], out=out)


def construct_identity_cost(subject, invVs) -> Tuple[sparse.spmatrix, np.ndarray]:
    shape = (len(subject.faces) * 3, len(subject.vertices))
    hashid = hashlib.sha256()
    hashid.update(b"identity")
    hashid.update(np.array(shape).data)
    hashid.update(subject.vertices.data)
    hashid = hashid.hexdigest()

    cache = SparseMatrixCache(suffix="_aei").entry(hashid=hashid, shape=shape)
    AEi = cache.get()
    if AEi is None:
        AEi = TransformMatrix.construct(
            subject.faces, invVs, len(subject.vertices),
            desc="Building Identity Cost"
        ).tocsr()
        AEi.eliminate_zeros()
        cache.store(AEi)
    Bi = np.tile(np.identity(3, dtype=float), (len(subject.faces), 1))
    assert AEi.shape[0] == Bi.shape[0]
    return AEi.tocsr(), Bi


def construct_smoothness_cost(subject, invVs, adjacent) -> Tuple[sparse.spmatrix, np.ndarray]:
    count_adjacent = sum(len(a) for a in adjacent)
    shape = (count_adjacent * 3, len(subject.vertices))
    hashid = hashlib.sha256()
    hashid.update(b"smoothness")
    hashid.update(np.array(shape).data)
    hashid.update(subject.vertices.data)
    hashid = hashid.hexdigest()

    cache = SparseMatrixCache(suffix="_aes").entry(hashid=hashid, shape=shape)
    AEs = cache.get()
    if AEs is None:
        size = len(subject.vertices)
        transforms = [
            TransformMatrix.expand(f, inv, size).tocsr() for (f, inv) in
            tqdm.tqdm(zip(subject.faces, invVs), total=len(subject.faces), desc="Building TransformMatrices")
        ]

        def construct(index):
            a = transforms[index]
            for adj in adjacent[index]:
                yield a, transforms[adj]

        lhs, rhs = zip(*(adjacents for index in
                         tqdm.trange(len(subject.faces), desc="Building Smoothness Cost")
                         for adjacents in construct(index)))
        AEs = (sparse.vstack(lhs).tocsr() - sparse.vstack(rhs).tocsr()).tocsc()
        AEs.eliminate_zeros()
        cache.store(AEs)
    Bs = np.zeros((count_adjacent * 3, 3))
    assert AEs.shape[0] == Bs.shape[0]
    return AEs, Bs


# -----------------------------------------------------------------------------
# Precomputed setup (cacheable per mesh)
# -----------------------------------------------------------------------------

@dataclass
class SourceSetup:
    """Per-source-mesh precomputed data, reusable across markers / iterations."""
    source: meshlib.Mesh           # 4D-extended source
    source_org: meshlib.Mesh       # original 3D source
    adjacent: List[List[int]]
    invVs: np.ndarray
    AEi_pre: sparse.spmatrix       # before apply_markers
    Bi_pre: np.ndarray
    AEs_pre: sparse.spmatrix
    Bs_pre: np.ndarray


@dataclass
class TargetSetup:
    """Per-target-mesh precomputed data, reusable across markers / iterations."""
    target: meshlib.Mesh           # 4D-extended target
    target_org: meshlib.Mesh
    kd_tree: cKDTree
    target_normals: np.ndarray
    surface: SurfaceData           # triangle-surface queries (matching / projection)


@dataclass
class RegionSpec:
    """Partition of the SOURCE vertices for a regional (masked) wrap.

    Built upstream from a boundary loop (see dt_core.regions):
      frozen_idx    — hard-pinned to their own source position (outside the
                      loop + the loop ring itself). The result is byte-
                      identical to the source here.
      editable_mask — bool over source-org verts; only True (interior)
                      vertices receive the closest-point / shrink-wrap pull.
      soft_idx      — interior vertices in the feather band that also get a
                      soft anchor to their source position, for a smooth seam.
      soft_w        — per-vertex feather weight (aligned with soft_idx), 1
                      near the boundary decaying to 0 deeper inside.
    """
    frozen_idx: np.ndarray
    editable_mask: np.ndarray
    soft_idx: np.ndarray
    soft_w: np.ndarray


def precompute_source(source_org: meshlib.Mesh) -> SourceSetup:
    """Build everything that depends only on the source mesh (no markers / no target)."""
    t0 = time.perf_counter()
    source = source_org.to_fourth_dimension()
    adjacent = compute_adjacent_by_edges(source_org)
    invVs = np.linalg.inv(source.span)
    AEi_pre, Bi_pre = construct_identity_cost(source, invVs)
    AEs_pre, Bs_pre = construct_smoothness_cost(source, invVs, adjacent)
    dt = time.perf_counter() - t0
    print(f"[precompute_source] done in {dt:.2f}s  (source faces={len(source_org.faces)})")
    return SourceSetup(
        source=source, source_org=source_org,
        adjacent=adjacent, invVs=invVs,
        AEi_pre=AEi_pre, Bi_pre=Bi_pre,
        AEs_pre=AEs_pre, Bs_pre=Bs_pre,
    )


def precompute_target(target_org: meshlib.Mesh) -> TargetSetup:
    """Build everything that depends only on the target mesh."""
    t0 = time.perf_counter()
    target = target_org.to_fourth_dimension()
    kd_tree = cKDTree(target_org.vertices)
    target_normals = get_vertex_normals(target_org.vertices, target_org.faces)
    surface = build_surface(target_org.vertices, target_org.faces)
    dt = time.perf_counter() - t0
    print(f"[precompute_target] done in {dt:.2f}s  (target verts={len(target_org.vertices)})")
    return TargetSetup(
        target=target, target_org=target_org,
        kd_tree=kd_tree, target_normals=target_normals,
        surface=surface,
    )


# -----------------------------------------------------------------------------
# Correspondence iteration (using pre-built source/target setups)
# -----------------------------------------------------------------------------

def compute_correspondence_with_setup(
        src: SourceSetup,
        tgt: TargetSetup,
        markers: np.ndarray,
        iterations: int = 8,
        smoothness: float = 1.0,
        identity_weight: float = 0.001,
        use_closest_point: bool = True,
        region: Optional[RegionSpec] = None,
        anchor_strength: float = 8000.0,
        match_max_angle_deg: float = 60.0,
        match_dist_floor: Optional[float] = None,
        anti_fold: bool = True,
        compute_mapping: bool = True,
        soft_marker_weight: float = 10.0,
        soft_marker_min_wc: float = 1000.0,
        conform_weight: Optional[np.ndarray] = None,
        wc_schedule: Optional[List[float]] = None,
        match_offset: float = 0.0,
        weak_anchor_weight: float = 0.0,
        rigidity_filter: bool = False,
        init_vertices: Optional[np.ndarray] = None,
        solve_info: Optional[dict] = None,
        progress_callback=None,
) -> Tuple[meshlib.Mesh, Optional[np.ndarray]]:
    """
    Run the iterative inflation using a precomputed SourceSetup and TargetSetup.
    Caller is responsible for caching `src` / `tgt`.
    Returns (inflated source mesh, triangle mapping Nx2).

    compute_mapping=False skips the final match_triangles pass and returns
    (mesh, None) — for callers that only need the inflated mesh (/wrap). The
    mapping is a legacy per-triangle Python loop and costs seconds on dense
    meshes.

    Soft markers: hard marker pins guarantee global registration while the
    closest-point weight Wc is still small, but once Wc dominates, an
    imprecisely placed marker only damages the surface around itself — the
    classic case being a marker on the WRONG SIDE of a thin feature (ear,
    fold), which hard pins drag straight through the mesh. From the first
    iteration where Wc >= `soft_marker_min_wc`, markers therefore turn into
    weak penalty rows (`soft_marker_weight`), and the surface term pulls
    such vertices back out; accurate markers are unaffected (the surface
    holds them where they already are). The weight must be SMALL (~10):
    with a strong weight the wrong pin keeps winning against smoothness.
    Frozen-region pins always stay hard. soft_marker_weight=0 (or Fit
    mode, or a Wc schedule that never reaches min_wc) = classic hard pins
    throughout.

    The closest-point term matches against the target SURFACE (nearest point
    on a triangle, not nearest vertex) with robust pruning — see
    dt_core.surface.match_points_to_surface. `match_max_angle_deg` is the
    normal-compatibility limit; `match_dist_floor` the floor of the adaptive
    distance cut (None = 2% of the target's bbox diagonal). `anti_fold`
    enables per-iteration damping of triangles whose orientation flipped.

    use_closest_point=False drops the shrink-wrap term entirely: the solve
    becomes "pin marker vertices to their target positions, minimize
    distortion everywhere else". That is Fit mode — attaching an accessory
    (beard -> head) where the bulk of the source must NOT snap onto the
    target surface. Without that term the system is identical on every
    iteration, so callers should pass iterations=1.

    region (RegionSpec) enables *regional wrap*: vertices outside the user's
    boundary loop are hard-pinned to their own source position (frozen), the
    closest-point pull is restricted to the editable interior, and a feather
    band of soft source-anchors near the seam keeps the transition smooth.
    `anchor_strength` scales those feather anchors (per-vertex × soft_w).

    conform_weight (per source-org vertex, 0..1) scales the closest-point pull
    per vertex for *refit* (garment / accessory fitting): 1 = stick to the
    target surface, 0 = no pull (the vertex is carried by stiffness =
    identity_weight and smoothness only). A beard graft uses a feather field
    (1 at the attachment seam decaying to 0 into the bulk) with high stiffness;
    tight cloth uses ~1 everywhere with low stiffness. None = uniform pull
    (classic wrap). Matching is restricted to conform_weight > 0.

    wc_schedule overrides the default closest-point weight ramp. Refit passes
    a CAPPED ramp: the classic one ends at 5000, which out-shouts any nonzero
    conform weight by ~4 orders of magnitude over the stiffness term — the
    mathematical root of "the collar shrink-wraps flat". A cap keeps the
    trade-off between grip and shape a real trade-off.

    match_offset displaces every closest-point target along the matched
    triangle's outward normal (cloth thickness — see
    surface.match_points_to_surface).

    solve_info (a caller-supplied dict) is filled with facts about the solve
    the result mesh alone can't tell — currently 'matched_idx': the source-org
    vertices that received a surface constraint on the LAST matching
    iteration. Refit needs it to pick ARAP bind handles: a vertex with a high
    conform weight whose matches were always PRUNED (normal-opposed outer
    wall) was never actually placed by the fit and must not be treated as one.

    weak_anchor_weight > 0 adds one tiny penalty row per vertex anchoring it
    to its INITIAL (placed) position. The gradient-based identity/smoothness
    terms are translation-invariant, so a marker-less solve (refit with no
    pins) is singular on iteration 1 (Wc=0): the solver returns the placed
    mesh plus an arbitrary translation. Every other term has exactly zero
    energy along that null space, so any small weight picks the intended
    solution without measurably biasing the fit.

    rigidity_filter (refit) drops surface matches that disagree with their
    neighbourhood's local rigid motion (see surface.filter_matches_rigidity)
    before they become closest-point rows — a sleeve caught by the wrong limb,
    a one-sided collar grip. It only ever REMOVES constraints (worst case =
    more freedom), so refit ships it ON; wrap keeps it OFF. The last
    iteration's dropped count lands in solve_info["rigidity_dropped"].

    init_vertices (refit growth continuation) warm-starts the iterate from a
    previous solve (org-vertex array, len = source_org) instead of the placed
    source. The rest shape (identity/smoothness/weak-anchor targets) is
    UNCHANGED — it always encodes the placed garment — so only the starting
    positions and hence the first closest-point match move. Growth stages must
    pass a schedule whose first weight is > 0, so matching fires on iteration 0
    from the warm shape (otherwise iteration 0, with Wc=0, resolves the pure
    stiffness system straight back to the placed pose and discards the warm
    start).
    """
    Ws = float(smoothness)
    Wi = float(identity_weight)
    default_wc = ([float(w) for w in wc_schedule] if wc_schedule is not None
                  else [0, 10, 50, 250, 1000, 2000, 3000, 5000])
    if iterations <= len(default_wc):
        Wc = default_wc[:iterations]
    else:
        Wc = default_wc + [default_wc[-1]] * (iterations - len(default_wc))

    dist_floor = (match_dist_floor if match_dist_floor is not None
                  else 0.02 * tgt.surface.diag)

    print(f"[correspondence] solver={backend_name()}  iters={iterations}  "
          f"smoothness={smoothness}  identity_weight={identity_weight}  "
          f"markers={len(markers)}  closest_point={use_closest_point}  "
          f"region={'yes' if region is not None else 'no'}  "
          f"match_angle={match_max_angle_deg}deg  dist_floor={dist_floor:.4g}")

    # ---- hard pins: user markers (-> target) + frozen region (-> source) ----
    # In regional mode, only markers INSIDE the editable interior apply: a
    # marker on a frozen vertex would both pin-to-target (marker) and
    # pin-to-source (freeze), i.e. the same column twice with conflicting
    # positions — which breaks the pin bookkeeping. Frozen verts are already
    # held at the source, so dropping those markers is also the right
    # semantics ("outside the loop stays put").
    eff_markers = markers
    if region is not None and len(markers) > 0:
        keep_m = region.editable_mask[markers[:, 0]]
        eff_markers = markers[keep_m]
        dropped = len(markers) - len(eff_markers)
        if dropped:
            print(f"[correspondence] region: dropped {dropped} marker(s) outside the editable area")

    marker_idx = eff_markers[:, 0]
    marker_pos = tgt.target.vertices[eff_markers[:, 1]]
    if region is not None and len(region.frozen_idx) > 0:
        frozen_pos = src.source_org.vertices[region.frozen_idx]
        pin_idx = np.concatenate([marker_idx, region.frozen_idx])
        pin_pos = np.concatenate([marker_pos, frozen_pos])
    else:
        pin_idx = np.asarray(marker_idx)
        pin_pos = np.asarray(marker_pos)

    # Dedupe pinned columns (keep first occurrence). A source vertex can repeat
    # across pins — two markers sharing a source, a symmetry-plane vertex that
    # mirrors onto itself, or a marker that also lands in the frozen set. The
    # apply/revert_pins bookkeeping derives sizes from len(pin_idx) while the
    # column selection dedupes via setdiff1d, so duplicates desync the shapes
    # (AssertionError in revert_pins) and double-count the pin in the RHS.
    if len(pin_idx):
        uniq = np.sort(np.unique(pin_idx, return_index=True)[1])
        pin_idx = pin_idx[uniq]
        pin_pos = pin_pos[uniq]

    t0 = time.perf_counter()
    AEi, Bi = apply_pins(src.AEi_pre, src.Bi_pre, pin_idx, pin_pos)
    AEs, Bs = apply_pins(src.AEs_pre, src.Bs_pre, pin_idx, pin_pos)
    print(f"[correspondence] apply_pins: {time.perf_counter() - t0:.2f}s  "
          f"(pins={len(pin_idx)})")

    # ---- feather: soft anchors to source position, built once (constant) ----
    AEa = Ba = None
    sel = anchor_pos = None
    if region is not None and len(region.soft_idx) > 0:
        sel = get_aec(len(src.source.vertices), len(src.source_org.vertices))[region.soft_idx]
        w = region.soft_w.astype(float)
        sel = sparse.diags(w) @ sel
        anchor_pos = src.source_org.vertices[region.soft_idx] * w[:, None]
        AEa, Ba = apply_pins(sel, anchor_pos, pin_idx, pin_pos)

    # ---- weak positional anchor (kills the translation null space) ----
    AEw = Bw = None
    if weak_anchor_weight > 0:
        n_org = len(src.source_org.vertices)
        sel_w = get_aec(len(src.source.vertices), n_org)
        AEw, Bw = apply_pins(sel_w, src.source_org.vertices.astype(float),
                             pin_idx, pin_pos)

    # ---- soft-marker configuration for the final iterations ----
    # A second set of pinned matrices where only the frozen region stays
    # hard; the markers become weighted penalty rows instead. See docstring.
    soft_from = None
    if use_closest_point and len(marker_idx) > 0 and soft_marker_weight > 0:
        past = np.nonzero(np.asarray(Wc, dtype=float) >= soft_marker_min_wc)[0]
        if len(past) > 0:
            soft_from = max(1, int(past[0]))
        if soft_from is not None and soft_from >= iterations:
            soft_from = None
    if soft_from is not None:
        if region is not None and len(region.frozen_idx) > 0:
            spin_idx = np.asarray(region.frozen_idx)
            spin_pos = src.source_org.vertices[spin_idx]
        else:
            spin_idx = np.array([], dtype=np.int64)
            spin_pos = np.zeros((0, 3))
        AEi_soft, Bi_soft = apply_pins(src.AEi_pre, src.Bi_pre, spin_idx, spin_pos)
        AEs_soft, Bs_soft = apply_pins(src.AEs_pre, src.Bs_pre, spin_idx, spin_pos)
        sel_m = get_aec(len(src.source.vertices), len(src.source_org.vertices))[marker_idx]
        AEm_soft, Bm_soft = apply_pins(sel_m, np.asarray(marker_pos, dtype=float),
                                       spin_idx, spin_pos)
        AEa_soft = Ba_soft = None
        if sel is not None:
            AEa_soft, Ba_soft = apply_pins(sel, anchor_pos, spin_idx, spin_pos)
        AEw_soft = Bw_soft = None
        if weak_anchor_weight > 0:
            sel_w = get_aec(len(src.source.vertices), len(src.source_org.vertices))
            AEw_soft, Bw_soft = apply_pins(sel_w, src.source_org.vertices.astype(float),
                                           spin_idx, spin_pos)
        print(f"[correspondence] soft markers: hard pins it1..{soft_from}, "
              f"soft w={soft_marker_weight:g} it{soft_from + 1}..{iterations}")

    if init_vertices is not None:
        vertices = np.copy(meshlib.Mesh(
            np.asarray(init_vertices, dtype=float), src.source_org.faces
        ).to_fourth_dimension().vertices)
    else:
        vertices = np.copy(src.source.vertices)
    last_matched: np.ndarray = np.zeros(0, dtype=np.int64)
    rigidity_dropped = 0
    src_rings = None  # per-ring edge lists, built lazily on first match

    iter_t = np.zeros(4)  # [closest, combine, solve, total]
    result = meshlib.Mesh(
        vertices=vertices[:len(src.source_org.vertices)],
        faces=src.source_org.faces,
    )
    pBar = tqdm.tqdm(total=iterations * 3)

    for iteration in range(iterations):
        def pbar_next(msg: str):
            pBar.set_description(f"[{iteration + 1}/{iterations}] {msg}")
            pBar.update()

        soft_iter = soft_from is not None and iteration >= soft_from
        if soft_iter:
            it_pin_idx, it_pin_pos = spin_idx, spin_pos
            Astack = [AEi_soft * Wi, AEs_soft * Ws, AEm_soft * soft_marker_weight]
            Bstack = [Bi_soft * Wi, Bs_soft * Ws, Bm_soft * soft_marker_weight]
            if AEa_soft is not None:
                Astack.append(AEa_soft * anchor_strength)
                Bstack.append(Ba_soft * anchor_strength)
            if AEw_soft is not None:
                Astack.append(AEw_soft * weak_anchor_weight)
                Bstack.append(Bw_soft * weak_anchor_weight)
        else:
            it_pin_idx, it_pin_pos = pin_idx, pin_pos
            Astack = [AEi * Wi, AEs * Ws]
            Bstack = [Bi * Wi, Bs * Ws]
            if AEa is not None:
                Astack.append(AEa * anchor_strength)
                Bstack.append(Ba * anchor_strength)
            if AEw is not None:
                Astack.append(AEw * weak_anchor_weight)
                Bstack.append(Bw * weak_anchor_weight)

        # ---- closest point cost (surface matching + robust pruning) ----
        t_cp = time.perf_counter()
        pbar_next("Closest Point Costs")
        # Normally iteration 0 (Wc=0) skips matching. A warm start (growth
        # continuation) with Wc[0] > 0 matches immediately from the warm shape.
        if (use_closest_point and Wc[iteration] != 0
                and (iteration > 0 or init_vertices is not None)):
            vertices_clipped = vertices[:len(src.source_org.vertices)]
            vnormals = get_vertex_normals(vertices_clipped, src.source_org.faces)
            # Restrict matching to the vertices that actually receive a surface
            # pull, then map indices back. Two drivers:
            #   * region mode    -> the editable interior (frozen excluded);
            #   * conform_weight -> vertices with weight > 0 (refit): a beard's
            #     attachment band pulls, its rigid bulk (weight 0) does not, so
            #     it never gets fed to the matcher.
            # Subsetting first keeps the adaptive distance cut honest (its 3x
            # median is computed over the kept population, not one we discard)
            # and skips wasted point-triangle work.
            if region is not None:
                match_sub = np.nonzero(region.editable_mask)[0]
            elif conform_weight is not None:
                match_sub = np.nonzero(conform_weight > 0.0)[0]
            else:
                match_sub = None
            if match_sub is not None:
                match_pts, match_norms = vertices_clipped[match_sub], vnormals[match_sub]
            else:
                match_pts, match_norms = vertices_clipped, vnormals
            m_idx, m_cp, _m_dist, m_stats = match_points_to_surface(
                match_pts, match_norms, tgt.surface,
                max_angle_deg=match_max_angle_deg,
                dist_floor=dist_floor,
                offset=match_offset,
            )
            if match_sub is not None and len(m_idx) > 0:
                m_idx = match_sub[m_idx]
            # Rigidity-consistency filter (refit): remove matches that fight
            # their neighbourhood's local rigid motion before they constrain
            # the solve. Only removes rows, so it is safe to run unconditionally
            # for refit; needs a source-vertex adjacency (built once, lazily).
            if rigidity_filter and len(m_idx) >= 50:
                if src_rings is None:
                    src_rings = precompute_rings(regions.weighted_adjacency(
                        src.source_org.vertices, src.source_org.faces))
                keep, rinfo = filter_matches_rigidity(
                    vertices_clipped, src_rings, m_idx, m_cp)
                rigidity_dropped = int((~keep).sum())
                if rigidity_dropped:
                    m_idx, m_cp = m_idx[keep], m_cp[keep]
                print(f"[correspondence]   it{iteration + 1}: rigidity dropped "
                      f"{rigidity_dropped}/{len(keep)} (passes {rinfo['pass_drops']})")
            last_matched = np.asarray(m_idx, dtype=np.int64)
            if len(m_idx) > 0:
                AEc = get_aec(len(src.source.vertices), len(src.source_org.vertices))[m_idx]
                Bc = m_cp
                # Per-vertex conform weight scales each surface constraint (a
                # soft, spatially-varying pull): 1 = stick to the target, 0 =
                # ignored (carried by stiffness/smoothness only). This is what
                # turns a beard into a graft — the seam is pulled onto the head
                # while the bulk free-rides and keeps its shape.
                if conform_weight is not None:
                    cw = conform_weight[m_idx].astype(float)
                    AEc = sparse.diags(cw) @ AEc
                    Bc = cw[:, None] * Bc
                mAEc, mBc = apply_pins(AEc, Bc, it_pin_idx, it_pin_pos)
                Astack.append(mAEc * Wc[iteration])
                Bstack.append(mBc * Wc[iteration])
            print(f"[correspondence]   it{iteration + 1}: matched {len(m_idx)}"
                  f"/{len(match_pts)}  pruned: angle={m_stats['rejected_angle']} "
                  f"boundary={m_stats['rejected_boundary']} dist={m_stats['rejected_dist']} "
                  f"(thr={m_stats['threshold']:.4g})")
        iter_t[0] += time.perf_counter() - t_cp

        # ---- combine ----
        t_cb = time.perf_counter()
        pbar_next("Combining Costs")
        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        b = np.concatenate(Bstack)
        iter_t[1] += time.perf_counter() - t_cb

        # ---- solve ----
        t_sv = time.perf_counter()
        pbar_next("Solving")
        # Snapshot BEFORE revert_pins overwrites `vertices` in place — the
        # anti-fold damper needs the previous iterate as the safe state.
        prev_org = vertices[:len(src.source_org.vertices)].copy()
        A = A.tocsc()
        x = spsolve((A.T @ A).tocsc(), A.T @ b)
        revert_pins(A, x, it_pin_idx, it_pin_pos, out=vertices)
        new_org = vertices[:len(src.source_org.vertices)]
        if anti_fold:
            new_org, n_flip = damp_folds(prev_org, new_org, src.source_org.faces)
            if n_flip:
                print(f"[correspondence]   it{iteration + 1}: damped {n_flip} flipped triangle(s)")
        result = meshlib.Mesh(
            vertices=new_org,
            faces=src.source_org.faces,
        )
        vertices = result.to_fourth_dimension().vertices
        iter_t[2] += time.perf_counter() - t_sv

        # Publish coarse progress for the async job poller. Best-effort — a
        # callback that raises must not abort the solve.
        if progress_callback is not None:
            try:
                progress_callback(iteration + 1, iterations)
            except Exception:  # noqa: BLE001
                pass

    iter_t[3] = time.perf_counter() - t0 - iter_t[0:3].sum() + iter_t[0:3].sum()
    total = time.perf_counter() - t0
    print(f"[correspondence] iterations total: {total:.2f}s  "
          f"(closest={iter_t[0]:.1f}s  combine={iter_t[1]:.1f}s  solve={iter_t[2]:.1f}s)")

    if solve_info is not None:
        solve_info["matched_idx"] = last_matched
        solve_info["rigidity_dropped"] = rigidity_dropped

    if not compute_mapping:
        return result, None

    t1 = time.perf_counter()
    mapping = np.array(match_triangles(result, tgt.target_org))
    print(f"[correspondence] match_triangles: {time.perf_counter() - t1:.2f}s  -> {len(mapping)} pairs")
    return result, mapping


# -----------------------------------------------------------------------------
# Legacy entry points (kept for API compatibility)
# -----------------------------------------------------------------------------

def compute_correspondence(
        source_org: meshlib.Mesh,
        target_org: meshlib.Mesh,
        markers: np.ndarray,
        iterations: int = 8,
        smoothness: float = 1.0,
        identity_weight: float = 0.001,
        use_closest_point: bool = True,
):
    """Convenience wrapper that builds source/target setups inline (no cache)."""
    src = precompute_source(source_org)
    tgt = precompute_target(target_org)
    return compute_correspondence_with_setup(
        src, tgt, markers,
        iterations=iterations,
        smoothness=smoothness,
        identity_weight=identity_weight,
        use_closest_point=use_closest_point,
    )


def get_correspondence(source_org: meshlib.Mesh, target_org: meshlib.Mesh, markers: np.ndarray,
                       **kwargs) -> np.ndarray:
    _, mapping = compute_correspondence(source_org, target_org, markers, **kwargs)
    return mapping
