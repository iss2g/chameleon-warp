"""
Refit: fitting a garment / accessory onto a body by REUSING the wrap solver
with a per-vertex conform field (see correspondence.compute_correspondence_with
_setup's `conform_weight`).

Unlike wrap (which shrink-wraps the whole source ONTO the target), refit keeps
the garment's own shape and only grips it to the body at its INTERFACE. The key
observation: a garment's interface is exactly the OPEN BOUNDARIES of its mesh —
a shirt's neck / hem / cuffs, a beard's face seam are all mesh holes (edges used
by a single face). So the conform field is built automatically:

  * seam mode (accessory / loose cloth / armor): conform = 1 on the open-
    boundary seam, feathering to 0 into the bulk over `grip_width`. The seam is
    pulled onto the body; the bulk free-rides and keeps its relief — baggy, not
    clingy. Paired with a high stiffness (identity_weight) the bulk holds shape.
  * full mode (skintight): conform = 1 everywhere -> classic shrink-wrap, low
    stiffness so it hugs detail.

On top of the seam field, the fit is CONTACT-DRIVEN (see run_refit):

  1. contact field — the placed garment's signed distance to the body decides
     grip strength (pressed = tight, hanging = free), frozen at placement so
     the user's manual place/scale is the specification;
  2. layer occlusion — a vertex with garment cloth between itself and the body
     (outer wall of a folded collar) is auto-released, the needle/pin idea run
     densely for every vertex;
  3. capped Wc ramp + thickness-offset targets in the solver;
  4. ARAP bind — released vertices are re-posed as-rigidly-as-possible from
     the placed shape with the gripped verts as handles (surface-deform
     semantics: folds keep their angles, drape keeps its authored shape).

Collision handling (push the garment out, or hide the body underneath) is a
separate stage and lives elsewhere; the preset only records the intended mode.

Pure geometry over (vertices, faces) — no solver, no FastAPI — so it unit-tests
headlessly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import regions


def open_boundary_vertices(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Vertices lying on an open border of the mesh (an undirected edge used by
    exactly one face). These are the garment's interfaces — neck/hem/cuffs of a
    shirt, the face seam of a beard. Same rule as surface.build_surface, but
    standalone (no KD-tree) so it's cheap to call for the conform field."""
    f = np.ascontiguousarray(faces[:, :3], dtype=np.int64)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    on_boundary = counts[inv] == 1
    vb = np.zeros(n_verts, dtype=bool)
    vb[edges[on_boundary].ravel()] = True
    return np.nonzero(vb)[0].astype(np.int64)


@dataclass
class RefitPreset:
    """A named starting point that maps a use-case to solver parameters.

    conform_mode  — "seam" (grip the open-boundary interface, keep the bulk) or
                    "full" (shrink-wrap everywhere).
    grip_width_frac — seam feather width as a fraction of the mesh bbox diagonal.
    stiffness     — identity_weight: how hard the bulk keeps its rest shape.
    smoothness    — surface-continuity weight.
    collision     — "none" | "push_out" | "hide_body" (resolved in a later stage).
    """
    conform_mode: str
    grip_width_frac: float
    stiffness: float
    smoothness: float
    collision: str
    # Proximity gate for the seam: an open boundary only counts as a real
    # interface (and grips the body) if it sits within this band of the body,
    # expressed as a fraction of the body's bbox diagonal. Far-standing open
    # rims — e.g. the OUTER edge of a double-wall collar — fall outside the band
    # and are left free, so they ride the identity/smoothness terms and keep
    # their rest offset from the inner wall instead of collapsing onto the body.
    # 0 disables the gate (every open boundary grips, the old behavior).
    seam_depth_frac: float = 0.0
    # ---- contact field (fractions of the BODY bbox diagonal) ----
    # The user's manual place/scale is a statement of intent: what they pushed
    # against the body should sit tight, what they left hanging should hang.
    # The conform field therefore comes from the PLACED garment's signed
    # distance s to the body: s <= tight -> full grip (the body pokes out /
    # presses the cloth), decaying to 0 by `free` (hanging cloth, no pull).
    # The field is frozen at placement — recomputing it per iteration would
    # grow the contact zone as the fit tightens, i.e. progressive shrink-wrap.
    contact_tight_frac: float = 0.01
    contact_free_frac: float = 0.05
    # Cloth thickness: closest-point targets sit at cp + thickness*n instead of
    # exactly on the skin (no more painted-on look).
    thickness_frac: float = 0.003
    # Cap of the closest-point weight ramp. The classic wrap ramp ends at 5000,
    # which out-shouts stiffness by ~4 orders of magnitude for ANY nonzero
    # conform weight — the mathematical root of "the collar shrink-wraps flat".
    wc_max: float = 500.0
    # Run the ARAP bind pass: vertices below the grip threshold are
    # re-posed as-rigidly-as-possible from the PLACED shape, with the gripped
    # vertices as handles (surface-deform semantics: the outer wall follows
    # the inner wall, folds keep their angles).
    bind: bool = True
    # Growth continuation: solve against a SEQUENCE of eroded bodies growing to
    # the true shape, so a deeply-penetrating placement (shirt on a fat body,
    # most verts starting INSIDE) tracks the body outward with local, side-
    # correct closest points instead of tearing. 1 = today's single-body solve.
    # Auto-collapses to 1 when < 2% of the placed garment penetrates. Presets
    # stay at 1 until real-asset validation; opt in via the payload.
    growth_steps: int = 1


# Starting values — tuned empirically later against real assets.
PRESETS: dict[str, RefitPreset] = {
    # loose accessory (beard, fur trim): seam grips the face, bulk keeps shape.
    "accessory": RefitPreset("seam", 0.15, stiffness=0.05, smoothness=1.0,
                             collision="none", seam_depth_frac=0.04,
                             contact_tight_frac=0.008, contact_free_frac=0.05,
                             thickness_frac=0.002),
    # loose garment (shirt): interface grips, drapes baggy; hide/keep body optional.
    "cloth":     RefitPreset("seam", 0.12, stiffness=0.03, smoothness=1.0,
                             collision="push_out", seam_depth_frac=0.04,
                             contact_tight_frac=0.012, contact_free_frac=0.06,
                             thickness_frac=0.004),
    # rigid armor: interface grips, bulk very stiff, body hidden underneath.
    "armor":     RefitPreset("seam", 0.10, stiffness=0.20, smoothness=1.0,
                             collision="hide_body", seam_depth_frac=0.03,
                             contact_tight_frac=0.01, contact_free_frac=0.04,
                             thickness_frac=0.006),
    # second skin (tight suit): conform everywhere, low stiffness. Occlusion
    # still releases hidden layers, and bind re-poses them rigidly, so even
    # here a folded collar keeps its fold. With no occluded layers the grip
    # covers every vertex and the bind pass is skipped automatically.
    # push_out (true penetrations only) lifts the half of the noise band that
    # lands under the skin back to the thickness standoff — no z-fighting.
    "skintight": RefitPreset("full", 0.00, stiffness=0.001, smoothness=0.5,
                             collision="push_out", seam_depth_frac=0.0,
                             thickness_frac=0.001, wc_max=1000.0),
}


def contact_weights(signed_dist: np.ndarray, tight: float, free: float
                    ) -> np.ndarray:
    """Per-vertex grip weight from the PLACED garment's signed distance to the
    body: 1 where the cloth presses the body or the body pokes through it
    (s <= tight, including s < 0), a smooth (1-t)^2 falloff across the band,
    0 where the cloth hangs free (s >= free)."""
    s = np.asarray(signed_dist, dtype=float)
    free = max(free, tight + 1e-9)
    cw = np.zeros(len(s), dtype=float)
    cw[s <= tight] = 1.0
    band = (s > tight) & (s < free)
    t = (s[band] - tight) / (free - tight)
    cw[band] = (1.0 - t) ** 2
    return cw


def build_conform_field(
    verts: np.ndarray,
    faces: np.ndarray,
    preset: RefitPreset,
    grip_width: float | None = None,
    seam: np.ndarray | None = None,
    body_dist: np.ndarray | None = None,
    seam_depth: float | None = None,
    interior_gate: np.ndarray | None = None,
) -> np.ndarray:
    """Per-vertex conform weight in [0, 1] for compute_correspondence_with_setup.

    full  -> ones (shrink-wrap everywhere).
    seam  -> 1 on the seam (open boundary by default, or an explicit `seam`
             vertex set), feathering to 0 into the bulk over `grip_width`
             (default: preset.grip_width_frac * bbox diagonal). A mesh with no
             open boundary and no explicit seam yields all-zeros (nothing grips;
             the caller falls back to pins).

    Proximity gate (auto open-boundary seam only): if `body_dist` (per-vertex
    distance from the PLACED garment to the body) and `seam_depth` are given,
    only open-boundary vertices within `seam_depth` of the body seed the grip.
    A far-standing open rim (the outer edge of a double-wall collar) is thus
    left free — it keeps its rest offset via the elastic terms instead of being
    pulled onto the body. If the gate would drop everything (garment placed far
    from the body), it's ignored so the fit still has something to grip.

    `interior_gate` (per-vertex, 0..1) multiplies the FEATHER (never the seam
    verts themselves): the feather's job is a smooth grip transition NEAR THE
    BODY — a collar standing off the neck is inside the feather's geodesic
    band but must not inherit its pull. Refit passes the contact gate here.
    """
    n = len(verts)
    if preset.conform_mode == "full":
        return np.ones(n, dtype=float)

    boundary = seam if seam is not None else open_boundary_vertices(faces, n)
    field = np.zeros(n, dtype=float)
    if len(boundary) == 0:
        return field
    # Gate the AUTO seam by proximity to the body (never an explicit user seam).
    if (seam is None and body_dist is not None
            and seam_depth is not None and seam_depth > 0):
        near = body_dist[boundary] <= seam_depth
        if near.any():                # else: placed far — keep the full boundary
            boundary = boundary[near]
    field[boundary] = 1.0

    interior = np.setdiff1d(np.arange(n), boundary)
    if len(interior) == 0:
        return field
    if grip_width is None:
        diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        grip_width = max(preset.grip_width_frac, 1e-6) * diag
    field[interior] = regions.feather_weights(
        verts, faces, interior, boundary, grip_width, exponent=2.0)
    if interior_gate is not None:
        field[interior] *= np.clip(interior_gate[interior], 0.0, 1.0)
    return field


def occluded_layer_mask(verts: np.ndarray, faces: np.ndarray,
                        body_cp: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Automatic layer classification — the needle idea, run densely for every
    active vertex instead of a hand-poked ray. A vertex whose segment to its
    closest BODY point crosses the garment itself has cloth between it and the
    skin: it is an OUTER layer (the visible wall of a folded collar) and must
    not grip — it rides the inner wall via stiffness / the ARAP bind instead.

    `active` limits the (ray-cast) test to vertices that could grip at all;
    returns a bool mask over ALL verts (True = outer layer, release)."""
    from .surface import build_surface, segments_hit_mesh

    out = np.zeros(len(verts), dtype=bool)
    idx = np.nonzero(np.asarray(active))[0]
    if len(idx) == 0:
        return out
    gsurf = build_surface(verts, faces)
    out[idx] = segments_hit_mesh(verts[idx], body_cp[idx], gsurf,
                                 exclude_verts=idx)
    return out


def apply_transform(verts: np.ndarray, matrix: Optional[np.ndarray]) -> np.ndarray:
    """Apply a 4x4 (or 3x4) affine to Nx3 verts. None = identity. This is the
    manual place / non-uniform resize the user dials in with the gizmo before
    the solve grips the garment."""
    if matrix is None:
        return verts.astype(float, copy=True)
    m = np.asarray(matrix, dtype=float)
    h = np.c_[verts, np.ones(len(verts))]
    return (h @ m[:3].T)[:, :3]


def _rot_from_axisangle(r: np.ndarray) -> np.ndarray:
    """3x3 rotation from an axis-angle vector (Rodrigues). |r| is the angle."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _delta_matrix(x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """9-DOF placement delta as a row-major 4x4, applied around `centroid`:
    v' = R @ (s ⊙ (v - c)) + c + t, with x = (t[3], r[3] axis-angle, logs[3])."""
    t = x[0:3]
    R = _rot_from_axisangle(x[3:6])
    s = np.exp(x[6:9])
    L = R @ np.diag(s)                       # linear part (rotate then scale)
    M = np.eye(4)
    M[:3, :3] = L
    M[:3, 3] = centroid - L @ centroid + t
    return M


def polish_placement(
    placed_verts: np.ndarray,
    faces: np.ndarray,
    body_surf,                               # surface.SurfaceData of the body
    thickness: float,
    contact_free: float,
    rng_seed: int = 0,
    n_sample: int = 4000,
    w_pen: float = 4.0,
    w_fit: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Low-DOF pre-alignment: nudge the user's placement by a small, trust-
    regioned 9-DOF transform (translation, rotation, per-axis scale) so the
    contact region sits at the cloth standoff instead of penetrating or floating.

    The user's placement is a PRIOR, not a suggestion — hard bounds keep the
    delta tiny (|t| <= 0.03·body_diag/axis, |r| <= 6°, scale in [0.92, 1.08]).
    Only verts already near the body at placement (|s_placed| < contact_free)
    vote, so hanging cloth cannot suck the whole garment inward. Powell
    (derivative-free; the closest-point energy is piecewise-smooth).

    Returns (delta 4x4 row-major around the garment centroid, stats). Identity
    delta when the polish would not help (too few contact verts, or Powell
    fails to beat the starting energy).
    """
    from scipy.optimize import minimize
    from .collision import signed_offset

    N = len(placed_verts)
    centroid = placed_verts.mean(axis=0)
    body_diag = float(body_surf.diag)

    rng = np.random.default_rng(rng_seed)
    sample = (placed_verts if N <= n_sample
              else placed_verts[rng.choice(N, n_sample, replace=False)])

    # Contact gate frozen at placement: only near-body verts participate in rho.
    _, _, s_placed = signed_offset(sample, body_surf)
    participate = np.abs(s_placed) < contact_free
    stats = {"evals": 0, "e_before": 0.0, "e_after": 0.0,
             "n_contact": int(participate.sum())}
    if participate.sum() < 200:
        stats["note"] = "skipped: < 200 contact verts"
        return np.eye(4), stats
    sample = sample[participate]

    # Work in units of the cloth thickness so the energy is O(1): distances are
    # ~thickness, whose square (~1e-3 at delta=0.03) sits below Powell's default
    # tolerances and would make the optimiser treat noise as signal.
    delta_th = max(thickness, 1e-9)
    t_b = 0.03 * body_diag
    r_b = np.radians(6.0)
    s_b = float(np.log(1.08))
    bounds = [(-t_b, t_b)] * 3 + [(-r_b, r_b)] * 3 + [(-s_b, s_b)] * 3
    scale_x = np.array([t_b, t_b, t_b, r_b, r_b, r_b, s_b, s_b, s_b])

    def energy(x: np.ndarray) -> float:
        M = _delta_matrix(np.asarray(x, dtype=float), centroid)
        tv = (sample @ M[:3, :3].T) + M[:3, 3]
        _, _, s = signed_offset(tv, body_surf)
        sn = s / delta_th                    # signed offset in thickness units
        pen = w_pen * np.square(np.maximum(0.0, 1.0 - sn))   # <delta = too tight
        a = sn - 1.0
        absa = np.abs(a)
        huber = np.where(absa <= 1.0, 0.5 * a * a, absa - 0.5)   # huber, delta=1
        data = float(np.mean(pen + w_fit * huber))
        # Gentle tie-breaker toward identity, each DOF normalised to ~1 at its
        # bound so no single component dominates the regulariser.
        xn = np.asarray(x, dtype=float) / scale_x
        return data + 0.05 * float(np.mean(xn * xn))

    x0 = np.zeros(9)
    e_before = energy(x0)
    res = minimize(energy, x0, method="Powell", bounds=bounds,
                   options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-4})
    e_after = float(res.fun)
    stats.update(evals=int(res.nfev), e_before=float(e_before), e_after=e_after)
    if not res.success or e_after >= e_before:
        stats["note"] = "no improvement; identity"
        return np.eye(4), stats
    return _delta_matrix(np.asarray(res.x, dtype=float), centroid), stats


@dataclass
class RefitResult:
    verts: np.ndarray                    # fitted garment vertices
    faces: np.ndarray                    # garment faces (topology preserved)
    conform_field: np.ndarray            # the per-vertex grip field used
    body_hide_mask: Optional[np.ndarray]  # bool over BODY verts: covered by the fitted garment (every preset)
    stats: dict


@dataclass
class RefitField:
    """The placed garment + the per-vertex grip field, plus the intermediate
    geometry the solve and the dry-run preview both key off. Produced by
    build_refit_field; consumed by run_refit (which then solves) and by the
    /refit_field preview endpoint (which just returns the field + stats)."""
    field: np.ndarray                    # conform/grip field 0..1 (post release/pins/frozen)
    garment: object                      # meshlib.Mesh — placed (+ polished) garment
    tgt: object                          # precompute_target result (has .surface)
    body_signed: np.ndarray              # signed distance of placed garment verts to body
    body_cp: np.ndarray                  # closest body points
    body_n: np.ndarray                   # closest body normals
    body_dist: np.ndarray                # |signed|
    body_diag: float
    contact: np.ndarray                  # contact-band weights
    gate: np.ndarray                     # wider feather gate
    tight: float
    free_d: float
    eff_thickness: float
    eff_seam_depth: float
    eff_stiffness: float
    eff_smoothness: float
    eff_collision: str
    preset: object                       # the resolved RefitPreset
    frozen_idx: np.ndarray
    opposed: np.ndarray                  # per-vertex: normal points into the body
    occluded: np.ndarray                 # per-vertex: garment cloth between it and the body
    polish_stats: Optional[dict]
    # Layer auto-classification (layer_mode="auto"): per-vertex layer id
    # (0 driven / 1 follower-wall / 2 follower-component), None in legacy mode.
    layer: Optional[np.ndarray] = None
    layer_stats: Optional[dict] = None


def build_refit_field(
    source_org,                          # meshlib.Mesh — the garment / accessory
    target_org,                          # meshlib.Mesh — the body
    preset_name: str = "accessory",
    *,
    pre_transform: Optional[np.ndarray] = None,
    placed_verts: Optional[np.ndarray] = None,
    grip_width: Optional[float] = None,
    seam_depth: Optional[float] = None,
    seam: Optional[np.ndarray] = None,
    preserve: Optional[np.ndarray] = None,
    frozen: Optional[np.ndarray] = None,
    pin_radius: Optional[float] = None,
    stiffness: Optional[float] = None,
    smoothness: Optional[float] = None,
    collision: Optional[str] = None,
    contact_tight: Optional[float] = None,
    contact_free: Optional[float] = None,
    thickness: Optional[float] = None,
    auto_polish: bool = False,
    layer_mode: str = "legacy",
    layer_dry_run: bool = False,
    layer_workers: Optional[int] = None,
) -> RefitField:
    """Place/resize the garment and build its per-vertex grip field: contact
    band + seam, minus the released layers (occlusion / opposed normals), pins
    and frozen paint. This is everything before the solver — factored out so the
    /refit_field dry-run can show the grip field without running a solve. No
    behavior change to run_refit, which calls this and then solves.

    `layer_mode` selects layer classification: "legacy" = the per-vertex
    occlusion | opposed signals (default, unchanged); "auto" = the score-based
    classifier (dt_core.layers) — body-side first-hit votes + fan-out occlusion
    on an eroded reference, smoothed + hysteresis + component policy. Auto mode
    does NOT compute the legacy signals (they cost raycasts). `layer_dry_run`
    decimates the vote cone (n_cone=2) for the synchronous /refit_field preview.
    """
    from . import meshlib
    from . import collision as _collision   # aliased: `collision` is a param name
    from .correspondence import precompute_target, get_vertex_normals

    if preset_name not in PRESETS:
        raise ValueError(f"unknown refit preset {preset_name!r}; "
                         f"expected one of {sorted(PRESETS)}")
    if layer_mode not in ("legacy", "auto"):
        raise ValueError(f"unknown layer_mode {layer_mode!r}; "
                         f"expected 'legacy' or 'auto'")
    base = PRESETS[preset_name]
    # Effective params: preset defaults, optionally overridden by the caller
    # (UI advanced controls). conform_mode/grip_width_frac come from the preset.
    eff_stiffness = base.stiffness if stiffness is None else float(stiffness)
    eff_smoothness = base.smoothness if smoothness is None else float(smoothness)
    eff_collision = base.collision if collision is None else str(collision)
    preset = base

    # Placement: an explicit placed-vertex array (proxy-wardrobe transport) wins
    # over the gizmo transform; otherwise apply the manual place/resize matrix.
    if placed_verts is not None:
        placed_arr = np.asarray(placed_verts, dtype=float)
        if placed_arr.shape != source_org.vertices.shape:
            raise ValueError(
                f"placed_verts shape {placed_arr.shape} != source "
                f"{source_org.vertices.shape}")
        garment = meshlib.Mesh(vertices=placed_arr.copy(), faces=source_org.faces)
    else:
        garment = meshlib.Mesh(
            vertices=apply_transform(source_org.vertices, pre_transform),
            faces=source_org.faces)

    # Target first: everything below keys off the PLACED garment's signed
    # distance to the body — the seam gate, the contact field, the layer test.
    tgt = precompute_target(target_org)
    body_diag = float(np.linalg.norm(target_org.vertices.max(axis=0)
                                     - target_org.vertices.min(axis=0)))
    body_cp, body_n, body_signed = _collision.signed_offset(garment.vertices,
                                                            tgt.surface)
    body_dist = np.abs(body_signed)
    eff_seam_depth = (preset.seam_depth_frac * body_diag
                      if seam_depth is None else float(seam_depth))

    # Contact field: the user's rough place/scale IS the specification — cloth
    # pressed against the body grips tight, hanging cloth stays free. Frozen
    # at placement (per-iteration recompute would creep into shrink-wrap).
    tight = (preset.contact_tight_frac * body_diag
             if contact_tight is None else float(contact_tight))
    free_d = (preset.contact_free_frac * body_diag
              if contact_free is None else float(contact_free))
    eff_thickness = (preset.thickness_frac * body_diag
                     if thickness is None else float(thickness))

    # ---- placement auto-polish (feature C) ----
    # A small, trust-regioned pre-alignment of the user's placement so the
    # contact region sits at the cloth standoff. Runs BEFORE the conform field /
    # contact / seam gate so all of them see the polished pose. Off by default.
    polish_stats = None
    if auto_polish:
        delta, polish_stats = polish_placement(
            garment.vertices, garment.faces, tgt.surface,
            eff_thickness, free_d)
        if not np.allclose(delta, np.eye(4)):
            garment = meshlib.Mesh(
                vertices=apply_transform(garment.vertices, delta),
                faces=garment.faces)
            body_cp, body_n, body_signed = _collision.signed_offset(
                garment.vertices, tgt.surface)
            body_dist = np.abs(body_signed)
        # Full effective placement (pre_transform ∘ delta), row-major, so the
        # frontend can adopt what was actually fitted into the gizmo.
        pre = (np.asarray(pre_transform, dtype=float)
               if pre_transform is not None else np.eye(4))
        polish_stats["matrix"] = (delta @ pre).reshape(-1).tolist()

    contact = contact_weights(body_signed, tight, free_d)
    gate = contact_weights(body_signed, tight, 2.0 * free_d)  # wider, for feather

    field = build_conform_field(garment.vertices, garment.faces, preset,
                                grip_width=grip_width, seam=seam,
                                body_dist=body_dist, seam_depth=eff_seam_depth,
                                interior_gate=gate)
    if preset.conform_mode != "full":
        field = np.maximum(field, contact)

    # Layer release. Two modes:
    #  legacy — two independent per-vertex signals, either one frees the vertex:
    #    * occlusion — garment cloth between the vertex and the body (parallel
    #      double wall): the needle's job done densely, without hand pokes;
    #    * normal opposition — the vertex's PLACED normal points INTO the body
    #      (dot < 0 against the closest body triangle). A folded collar's outer
    #      wall faces the body, so it can never legitimately grip; without this
    #      a FLARED outer wall (line of sight to the shoulders past the inner
    #      wall, so no occlusion) kept a high field, its matches were pruned by
    #      the solver's angle filter anyway, smoothness quietly unfolded it —
    #      and the bind pass then pinned it in that flattened state.
    #  auto   — score-based classifier (dt_core.layers): body-side first-hit
    #    votes + fan-out occlusion on an eroded reference, then smoothing +
    #    hysteresis + component policy. Never reads garment normals; the legacy
    #    signals are not computed (they cost raycasts).
    layer = None
    layer_stats: dict = {"mode": layer_mode}
    if layer_mode == "auto":
        from . import layers as _layers
        body_vn = get_vertex_normals(target_org.vertices, target_org.faces)
        # Signal A's ray cast dominates the cost and is embarrassingly parallel;
        # split it over processes (result is identical — see layers._votes_parallel).
        # It's memory-bandwidth-bound (the Möller pass streams ~10^8 candidate
        # pairs through RAM), so throughput plateaus around 6-8 processes and MORE
        # then regress (contention + spawn cost) — cap the default there. The real
        # lever is a tighter broad-phase (see docs/ROADMAP.md, "Performance").
        # None -> env DT_LAYER_WORKERS, else the capped core count.
        if layer_workers is None:
            layer_workers = (int(os.environ.get("DT_LAYER_WORKERS", "0"))
                             or min(os.cpu_count() or 1, 8))
        scores = _layers.layer_scores(
            garment.vertices, garment.faces, target_org,
            body_signed, body_vn, active=field > 0.0,
            contact_free=free_d, thickness=eff_thickness, body_diag=body_diag,
            # Dry-run (synchronous /refit_field preview) trades a little
            # precision for latency: fewer cone rays, fewer occlusion segments,
            # a tighter origin budget. The solve path uses the full settings.
            n_cone=2 if layer_dry_run else 4,
            occl_k=4 if layer_dry_run else 6,
            max_origins=6000 if layer_dry_run else 12000,
            n_workers=int(layer_workers))
        released, layer, mask_stats = _layers.released_mask_auto(
            scores, garment.vertices, garment.faces, active=field > 0.0)
        layer_stats.update(mask_stats)
        # Keep the RefitField shape stable; legacy signals unused in auto mode.
        opposed = np.zeros(len(garment.vertices), dtype=bool)
        occluded = released.copy()
        field[released] = 0.0
    else:
        vnorm = get_vertex_normals(garment.vertices, garment.faces)
        opposed = np.einsum('ij,ij->i', vnorm, body_n) < 0.0
        occluded = occluded_layer_mask(garment.vertices, garment.faces,
                                       body_cp, field > 0.0)
        released = occluded | opposed
        field[released] = 0.0

    # Pin "preserve" regions: the OUTER wall(s) a needle pierced. They must not
    # grip the body (the inner wall does that, via a hard marker) — so we RELEASE
    # a geodesic patch there, protecting the double-wall standoff from the auto
    # seam. Geodesic so it can't bleed across a thin gap. Applied over the seam.
    if preserve is not None and len(preserve):
        from . import pins as _pins
        gdiag = float(np.linalg.norm(garment.vertices.max(axis=0)
                                     - garment.vertices.min(axis=0)))
        prad = pin_radius if pin_radius is not None else 0.06 * gdiag
        field = _pins.apply_pins(
            field, garment.vertices, garment.faces,
            np.zeros(0, np.int64), np.asarray(preserve, np.int64), prad)

    # Frozen paint: the user brushed "do not touch this at all". Absolute —
    # wins over the seam, the contact field and any pin grip. The verts get no
    # surface pull, can't become bind handles (never matched), and the ARAP
    # pass treats the patch as near-rigid, so it rides the garment with its
    # authored shape intact (a collar the automation keeps getting wrong).
    frozen_idx = (np.asarray(frozen, dtype=np.int64).ravel()
                  if frozen is not None and len(frozen) else np.zeros(0, np.int64))
    if len(frozen_idx):
        field[frozen_idx] = 0.0

    return RefitField(
        field=field, garment=garment, tgt=tgt,
        body_signed=body_signed, body_cp=body_cp, body_n=body_n,
        body_dist=body_dist, body_diag=body_diag,
        contact=contact, gate=gate, tight=tight, free_d=free_d,
        eff_thickness=eff_thickness, eff_seam_depth=eff_seam_depth,
        eff_stiffness=eff_stiffness, eff_smoothness=eff_smoothness,
        eff_collision=eff_collision, preset=preset,
        frozen_idx=frozen_idx, opposed=opposed, occluded=occluded,
        polish_stats=polish_stats, layer=layer, layer_stats=layer_stats)


def run_refit(
    source_org,                          # meshlib.Mesh — the garment / accessory
    target_org,                          # meshlib.Mesh — the body
    preset_name: str = "accessory",
    *,
    pre_transform: Optional[np.ndarray] = None,
    placed_verts: Optional[np.ndarray] = None,
    grip_width: Optional[float] = None,
    seam_depth: Optional[float] = None,
    seam: Optional[np.ndarray] = None,
    preserve: Optional[np.ndarray] = None,
    frozen: Optional[np.ndarray] = None,
    pin_radius: Optional[float] = None,
    markers: Optional[np.ndarray] = None,
    iterations: int = 10,
    offset: Optional[float] = None,
    stiffness: Optional[float] = None,
    smoothness: Optional[float] = None,
    collision: Optional[str] = None,
    contact_tight: Optional[float] = None,
    contact_free: Optional[float] = None,
    thickness: Optional[float] = None,
    auto_polish: bool = False,
    growth_steps: Optional[int] = None,
    layer_mode: str = "legacy",
    layer_workers: Optional[int] = None,
    progress_callback=None,
) -> RefitResult:
    """End-to-end refit: place/resize the garment, grip its interface to the
    body while the bulk keeps its shape, then resolve collision per the preset.

    Reuses the wrap solver (compute_correspondence_with_setup) with a per-vertex
    conform field, so it inherits the robust surface matching / anti-fold work.
    Imports are local to avoid importing the heavy solver stack at module load.
    """
    # Local imports: keep dt_core.refit importable (for field/geometry helpers)
    # without pulling tqdm/scipy-heavy correspondence unless a solve is run.
    from . import meshlib
    from . import collision as _collision   # aliased: `collision` is a param name
    from .arap import arap_bind
    from .correspondence import (
        precompute_source, precompute_target, compute_correspondence_with_setup,
        get_vertex_normals,
    )
    from .surface import build_surface

    # Place the garment and build the grip field (contact + seam − released
    # layers − pins − frozen). Factored into build_refit_field so the
    # /refit_field dry-run can preview the same field without solving.
    rf = build_refit_field(
        source_org, target_org, preset_name,
        pre_transform=pre_transform, placed_verts=placed_verts,
        grip_width=grip_width, seam_depth=seam_depth, seam=seam,
        preserve=preserve, frozen=frozen, pin_radius=pin_radius,
        stiffness=stiffness, smoothness=smoothness, collision=collision,
        contact_tight=contact_tight, contact_free=contact_free,
        thickness=thickness, auto_polish=auto_polish, layer_mode=layer_mode,
        layer_workers=layer_workers)
    field = rf.field
    garment = rf.garment
    tgt = rf.tgt
    body_signed = rf.body_signed
    body_diag = rf.body_diag
    contact = rf.contact
    eff_thickness = rf.eff_thickness
    eff_seam_depth = rf.eff_seam_depth
    eff_stiffness = rf.eff_stiffness
    eff_smoothness = rf.eff_smoothness
    eff_collision = rf.eff_collision
    preset = rf.preset
    frozen_idx = rf.frozen_idx
    opposed = rf.opposed
    occluded = rf.occluded
    polish_stats = rf.polish_stats

    src = precompute_source(garment)
    mk = markers if markers is not None else np.zeros((0, 2), dtype=np.int64)
    if len(frozen_idx) and len(mk):
        # A hard marker inside a frozen patch would drag a single vertex out of
        # the untouchable zone — the paint wins, the marker is dropped.
        fz = np.zeros(len(garment.vertices), dtype=bool)
        fz[frozen_idx] = True
        keep = ~fz[mk[:, 0]]
        if (~keep).any():
            print(f"[refit] frozen paint dropped {int((~keep).sum())} marker(s)")
        mk = mk[keep]
    # Capped closest-point ramp (see RefitPreset.wc_max) + a tiny positional
    # anchor: without markers the first iteration (Wc=0) has no positional
    # constraint at all — gradient terms are translation-invariant, so the
    # solve is singular and the placed pose can drift before the fit begins.
    wc_ramp = [min(w, preset.wc_max)
               for w in (0, 10, 50, 100, 250, 500, 750, 1000)]
    solve_info: dict = {}

    # ---- growth continuation (feature B) ----
    # Decide the number of erosion stages. Opt-in (payload > preset default 1);
    # auto-collapse to 1 when the placement barely penetrates (nothing to grow
    # through), so a well-placed garment runs exactly today's single solve.
    eff_growth = preset.growth_steps if growth_steps is None else int(growth_steps)
    eff_growth = max(1, min(6, eff_growth))
    n_pen = int((body_signed < 0.0).sum())
    if eff_growth > 1 and n_pen < 0.02 * len(garment.vertices):
        eff_growth = 1
    growth_stats = {"steps": eff_growth, "e0": 0.0, "penetrating_at_start": n_pen}

    if eff_growth == 1:
        result, _ = compute_correspondence_with_setup(
            src, tgt, mk,
            iterations=iterations, smoothness=eff_smoothness,
            identity_weight=eff_stiffness, use_closest_point=True,
            conform_weight=field, compute_mapping=False,
            wc_schedule=wc_ramp, match_offset=eff_thickness,
            weak_anchor_weight=0.01, rigidity_filter=True, solve_info=solve_info,
            progress_callback=progress_callback,
        )
        verts = result.vertices
    else:
        # Solve against eroded bodies growing to the true shape. The garment
        # tracks each small step so closest points stay local and side-correct.
        # The conform field / occlusion / frozen paint above were built ONCE
        # against the TRUE body — they encode user intent and do NOT change.
        K = eff_growth
        body_vn = get_vertex_normals(target_org.vertices, target_org.faces)
        # Same erosion-depth formula the layer classifier uses — ONE definition.
        # Past the >=2% penetration guard above, so this returns the nonzero e0.
        from . import layers as _layers
        e0 = _layers.erosion_depth(body_signed, eff_thickness, body_diag)
        growth_stats["e0"] = float(e0)

        # Warm-continuation schedules have NO leading zero (that iteration would
        # resolve the stiffness system back to the placed pose and drop the warm
        # start). Intermediate stages ride the low half; the final stage runs
        # the full capped ramp against the true body.
        inter_ramp = [min(w, preset.wc_max) for w in (10, 50, 100)]
        final_ramp = [min(w, preset.wc_max)
                      for w in (10, 50, 100, 250, 500, 750, 1000)]
        inter_iters = max(3, iterations // K)
        final_iters = max(1, iterations - inter_iters * (K - 1))
        total_iters = inter_iters * (K - 1) + final_iters

        cur = garment.vertices.copy()
        offset_i = 0
        for k in range(K):
            t_k = (k + 1) / K
            e_k = e0 * (1.0 - t_k)          # k=0 -> e0(1-1/K), last -> 0
            is_final = (k == K - 1)
            if is_final:
                tgt_k = tgt                 # true body, already precomputed
            else:
                body_k = meshlib.Mesh(
                    vertices=target_org.vertices - e_k * body_vn,
                    faces=target_org.faces)
                tgt_k = precompute_target(body_k)
            # Start the stage penetration-free against its (eroded) body.
            cur, _ = _collision.resolve_push_out(
                cur, tgt_k.surface, offset=eff_thickness, violation=0.0)
            stage_iters = final_iters if is_final else inter_iters
            sched = final_ramp if is_final else inter_ramp
            dist_floor_k = max(0.02 * body_diag, e_k + 2.0 * eff_thickness)
            off = offset_i

            def stage_cb(i, _n, _off=off):
                if progress_callback is not None:
                    progress_callback(_off + i, total_iters)

            result, _ = compute_correspondence_with_setup(
                src, tgt_k, mk,
                iterations=stage_iters, smoothness=eff_smoothness,
                identity_weight=eff_stiffness, use_closest_point=True,
                conform_weight=field, compute_mapping=False,
                wc_schedule=sched, match_offset=eff_thickness,
                weak_anchor_weight=0.01, rigidity_filter=True,
                match_dist_floor=dist_floor_k, init_vertices=cur,
                solve_info=solve_info, progress_callback=stage_cb)
            cur = result.vertices
            offset_i += stage_iters
            print(f"[refit] growth stage {k + 1}/{K}: e={e_k:.4g} "
                  f"iters={stage_iters} matched={len(solve_info.get('matched_idx', []))}")
        verts = cur

    # ---- bind pass (surface-deform semantics) ----
    # The solver placed the GRIPPED vertices; everything else (outer walls,
    # hanging cloth) is now re-posed as-rigidly-as-possible from the PLACED
    # shape with the gripped verts as handles. This is what actually preserves
    # a collar's fold: ARAP penalizes non-rigid one-ring changes at the
    # crease, which the solver's gradient terms cannot see.
    bind_stats: dict = {"free": 0, "handles": 0, "iterations": 0}
    if preset.bind:
        # A handle must have been PLACED by the fit: a decent conform weight
        # AND an actual surface match on the final iteration. A high-field
        # vertex whose matches were pruned throughout ended up wherever
        # smoothness dragged it — pinning it would bake that error in.
        grip = np.zeros(len(garment.vertices), dtype=bool)
        matched = solve_info.get("matched_idx", np.zeros(0, np.int64))
        grip[matched[field[matched] >= 0.25]] = True
        if mk is not None and len(mk):
            grip[mk[:, 0]] = True           # needle/marker verts are handles too
        n_grip = int(grip.sum())
        if 3 <= n_grip < len(garment.vertices):
            handle_idx = np.nonzero(grip)[0]
            # A connected component with no handle can't be bound — keep the
            # solver's output there by promoting all its verts to handles.
            from scipy.sparse.csgraph import connected_components
            _nc, labels = connected_components(
                regions.weighted_adjacency(garment.vertices, garment.faces),
                directed=False)
            handled = np.zeros(labels.max() + 1, dtype=bool)
            handled[labels[handle_idx]] = True
            orphan = ~handled[labels]
            if orphan.any():
                handle_idx = np.union1d(handle_idx, np.nonzero(orphan)[0])
            verts, bind_stats = arap_bind(
                garment.vertices, garment.faces,
                handle_idx, verts[handle_idx],
                init_verts=verts, iterations=6,
                rigid_verts=frozen_idx if len(frozen_idx) else None)

    # push_out is a safety net for TRUE penetrations only (violation=0): the
    # solver already placed contact verts at the thickness standoff, and
    # pushing everything below `off` would merge double walls whose rest gap
    # is smaller than the offset into one crumpled shell.
    off = offset if offset is not None else eff_thickness
    if eff_collision == "push_out":
        pre_push = verts[frozen_idx].copy() if len(frozen_idx) else None
        verts, _rem = _collision.resolve_push_out(verts, tgt.surface, offset=off,
                                                  violation=0.0)
        if pre_push is not None:
            # Frozen paint outranks collision: the painted patch is returned
            # exactly as the bind pass placed it, even if it grazes the body.
            verts[frozen_idx] = pre_push

    # Body-hide mask is now an INDEPENDENT output, computed for every preset
    # (not just hide_body). It's cheap (one signed-offset raycast batch) and
    # the preview/editor needs it regardless of `eff_collision`: which body
    # verts sit under the fitted garment, so the covered skin can be hidden.
    # Computed on the FINAL garment verts (after any push_out).
    gsurf = build_surface(verts, source_org.faces)
    body_mask = _collision.body_hide_mask(
        target_org.vertices, gsurf, offset=0.0, max_depth=0.1 * body_diag)

    return RefitResult(
        verts=verts, faces=source_org.faces, conform_field=field,
        body_hide_mask=body_mask,
        stats={"preset": preset_name, "collision": eff_collision,
               "stiffness": eff_stiffness, "smoothness": eff_smoothness,
               "seam_depth": eff_seam_depth,
               "seam_grip_verts": int((field >= 1.0 - 1e-9).sum()),
               "contact_verts": int((contact > 0).sum()),
               "occluded_verts": int(occluded.sum()),
               "opposed_verts": int(opposed.sum()),
               "matched_verts": int(len(solve_info.get("matched_idx", []))),
               "rigidity_dropped": int(solve_info.get("rigidity_dropped", 0)),
               "polish": polish_stats, "growth": growth_stats,
               "thickness": eff_thickness, "wc_max": preset.wc_max,
               "bind": bind_stats,
               "preserve_verts": int(len(preserve)) if preserve is not None else 0,
               "frozen_verts": int(len(frozen_idx)),
               "hidden_body_verts": int(body_mask.sum()) if body_mask is not None else 0,
               "layers": rf.layer_stats},
    )
