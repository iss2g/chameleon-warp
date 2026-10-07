# Refit upgrade plan: A / B / C / D

Four transplants into the geometric refit pipeline, inspired by LUIVITON
(arXiv 2509.05030) but implemented **without ML** (A–C) or on top of existing
machinery (D). Written to be self-contained: everything an implementer needs
is either in this file or in the referenced source files.

Implementation order: **A → C → B → D** (independent; each ships separately).

---

## Context: the pipeline as it exists today

Refit fits a garment (`source_ref`) onto a body (`target_ref`):

1. `backend/dt_core/refit.py :: run_refit()` — orchestrator.
   - applies the user's manual placement (`pre_transform`, row-major 4x4);
   - builds a per-vertex **conform field** in [0,1]: seam grip (open mesh
     boundaries gated by body proximity) ∨ **contact field**
     (`contact_weights()`: signed distance of the *placed* garment to the
     body → 1 pressed / 0 hanging, frozen at placement);
   - releases layered walls: `occluded_layer_mask()` (segment to the closest
     body point crosses the garment itself) and normal-opposition
     (placed vertex normal · body normal < 0);
   - zeroes the field on `frozen` (user paint) verts, drops markers there;
   - runs the wrap solver with the field, a **capped Wc ramp** and
     thickness-offset targets;
   - **ARAP bind pass** (`backend/dt_core/arap.py :: arap_bind()`): vertices
     that did NOT receive a real surface match (`solve_info["matched_idx"]`
     ∩ field ≥ 0.25 are the handles) are re-posed as-rigidly-as-possible from
     the placed shape; `frozen` verts form a near-rigid patch
     (`rigid_verts`, edge weights ×20);
   - collision: `resolve_push_out(violation=0)` (true penetrations only,
     frozen verts restored afterwards) or `body_hide_mask`.
2. `backend/dt_core/correspondence.py :: compute_correspondence_with_setup()`
   — the iterative solver. Energy = identity (triangle gradients → I) +
   smoothness (neighbor gradients equal) + closest-point term scaled by the
   conform field, with hard pins for markers. Refit-relevant params already
   present: `wc_schedule`, `match_offset`, `weak_anchor_weight`,
   `conform_weight`, `solve_info` (returns `matched_idx` of the last
   iteration).
3. `backend/dt_core/surface.py :: match_points_to_surface()` — robust
   matching: normal compatibility (60°), open-boundary rejection, adaptive
   distance cut `max(dist_floor, 3·median)`; optional `offset` displaces
   targets along the triangle normal (cloth thickness).
4. `backend/main.py :: RefitPayload / _refit_compute()` — API. Result is
   written into the session's wrap slot; `residual.json` carries
   `distances_frac` + `grip_frac` for the viewport heatmaps.
5. Frontend: `frontend/src/App.tsx` (refit panel: gizmo placement →
   `refitMatrix`, needle pins, frozen-paint brush), `api.ts :: RefitOpts`,
   `components/MeshViewport.tsx` (paint brush, fly camera),
   `components/PreviewPanel.tsx` (error/grip heatmap).

### Repo conventions (follow them)

- Backend is numpy/scipy only in `dt_core` (no torch). Heavy deps import
  lazily inside functions.
- Tests are standalone headless scripts in `backend/tests/*_smoke.py`
  printing `PASS`/`FAIL` lines and `ALL PASS`, exiting 1 on failure. Run from
  `backend/`: `.venv\Scripts\python.exe tests\refit_fold_smoke.py`.
- Regression suite to keep green: `refit_fold_smoke, refit_e2e_smoke,
  refit_conform_smoke, refit_field_smoke, pins_smoke, collision_smoke,
  wrap_quality_smoke, soft_markers_smoke, region_smoke` (HTTP tests need a
  running server; skip them).
- Frontend: `npm run build` must pass (tsc strict). The backend serves the
  built `frontend/dist`; there is no dev server in production use, and the
  backend runs WITHOUT `--reload` — remind the user to restart it.
- Comments explain constraints, not narration. Every new behavior ships
  behind a parameter that defaults to the old behavior (except A, see below).
- Real-asset sanity data lives in `testfit/` (tg.obj = garment 10.6k verts,
  body_clean.obj = body 28.8k verts). Use it for perf/behavior checks, don't
  commit new large files.

---

## A. Rigidity-consistency match filter

**Problem it solves:** matches that pass the normal and distance filters but
disagree with their neighborhood — a sleeve vertex captured by the wrong limb,
one collar side gripping while the other doesn't. These wrong matches are
individually plausible; only the *local coherence of the displacement field*
exposes them.

**Mechanic (LUIVITON's noise-filtering module, Eq. numbers theirs):** for each
matched vertex i with source position v_i and match target c_i, consider its
neighbors j (matched only). Let p_j = v_j − v_i (source edge) and
q_j = c_j − c_i (target-of-match edge). Find the optimal rotation R_i
(Kabsch: SVD of Σ p_j q_jᵀ, det-corrected) and the residual energy

    E_i = Σ_j ‖q_j − R_i p_j‖² / Σ_j ‖p_j‖²          (normalize! see below)

Drop matches with E_i > μ_E + 0.5·σ_E (statistics over currently-kept
matches). Iterate: each pass recomputes stats on survivors and *grows* the
neighborhood ring (LUIVITON uses K = iter², 4 iterations; K-rings of 9/16 are
too heavy for our meshes — use rings [1, 2, 3] and early-stop when a pass
drops < 0.5% of matches).

Normalization by Σ‖p_j‖² makes E scale-invariant (LUIVITON's raw E is fine
for them because SMPL is metric-normalized; we are not).

### Implementation

New function in `backend/dt_core/surface.py`:

```python
def filter_matches_rigidity(
    verts: np.ndarray,            # (N,3) current source-org vertices
    adj: "sparse.csr_matrix",     # vertex adjacency (regions.weighted_adjacency)
    m_idx: np.ndarray,            # (M,) matched vertex indices
    m_cp: np.ndarray,             # (M,3) match targets (already offset)
    rings: tuple[int, ...] = (1, 2, 3),
    sigma_k: float = 0.5,
    min_neighbors: int = 3,
) -> np.ndarray:                  # boolean keep-mask over the M matches
```

- Build a dense lookup `target_of = np.full(N, -1)` → per-vertex match row.
- Ring growth: boolean sparse matmul of the adjacency (`adj_bool @ mask`),
  restricted to matched verts. For each pass, per-vertex neighbor lists via
  the CSR structure of `adj_bool**k` is too dense at k=3 on 100k meshes — cap
  per-vertex neighbors at 24 by random subsample (rng seeded, deterministic).
- Vectorized covariance exactly like `arap.py :: arap_bind` local step
  (`np.add.at` on (E,3,3) contributions, batched `np.linalg.svd`,
  det sign fix on the last row of Vt).
- Vertices with fewer than `min_neighbors` matched neighbors in the current
  ring: **keep** (cannot judge), do not include in μ/σ.
- Return the final keep mask; also return (second value or via a stats dict)
  counts per pass for logging.

Call site — `correspondence.py :: compute_correspondence_with_setup()`, right
after `match_points_to_surface` and after `m_idx` is mapped back through
`match_sub`:

```python
if rigidity_filter and len(m_idx) >= 50:
    keep = filter_matches_rigidity(vertices_clipped, src_adjacency, m_idx, m_cp)
    m_idx, m_cp = m_idx[keep], m_cp[keep]
```

- New param `rigidity_filter: bool = False` on
  `compute_correspondence_with_setup`. `run_refit` passes `True`
  unconditionally (this is the one feature that ships ON for refit — it only
  *removes* constraints, worst case equals more freedom; wrap keeps False).
- `src_adjacency`: build once per solve from
  `regions.weighted_adjacency(src.source_org.vertices, src.source_org.faces)`
  — cache it on the SourceSetup? No: SourceSetup is a shared cache across
  tools; compute lazily inside the solve on first use (cheap, sparse COO).
- Log line per iteration: `rigidity: dropped X/M (pass sizes ...)`.
- Add dropped count to `solve_info["rigidity_dropped"]` (last iteration) and
  surface it in refit stats (`main.py` already spreads `res.stats`).

**Perf budget:** ≤ 20% of the matching step on testfit (10.6k garment). If the
ring-3 pass busts the budget, drop to rings (1, 2).

### Tests — new `backend/tests/rigidity_filter_smoke.py`

1. Synthetic wrong-limb: flat grid source; matches = identity displacement
   for 95% of verts, and for a contiguous 5% patch matches displaced by a
   large constant offset sideways (simulates capture by the wrong limb).
   Assert ≥ 90% of the contaminated patch is dropped, ≤ 2% of clean matches
   dropped.
2. Rotation invariance: apply a global rigid rotation to ALL targets — assert
   nothing is dropped (the field is globally rigid → E ≈ 0 everywhere).
3. Scale invariance: same with a global 10× scale on all coordinates.
4. Integration: `refit_fold_smoke.py` must stay green with the filter ON
   (it is on via run_refit); add an assertion in the flared-collar case that
   `stats["rigidity_dropped"]` is present.

---

## C. Placement auto-polish (low-DOF pre-alignment)

**Problem it solves:** the fit quality depends on the manual placement being
decent; users place sloppily. LUIVITON solves global per-axis scale jointly
with registration. We polish the user's transform by a small deterministic
optimization *before* the dense solve. The user's placement is a **prior**,
not a suggestion — the polish is trust-regioned.

### Mechanic

Optimize 9 DOF `x = (t ∈ ℝ³, r ∈ ℝ³ axis-angle, s ∈ ℝ³ log-scale)` applied
around the placed garment's centroid (so rotation/scale don't translate).

Energy over a subsample of placed garment verts (uniform random, seeded,
`min(4000, N)` points), s_i = signed offset (collision.signed_offset) of the
transformed sample against the body surface, δ = effective cloth thickness:

    E(x) = mean_i ρ(s_i) + λ_reg · ‖x‖²_W

    ρ(s) = w_pen · max(0, δ − s)²        (penetration / too-tight: strong)
         + w_fit · huber(s − δ, δ)       (near-contact pull: gentle)
    only verts with placed |s_placed| < contact_free participate in ρ
    (hanging cloth must NOT vote, else the whole garment gets sucked in)

Defaults: w_pen = 4.0, w_fit = 1.0, λ_reg with per-component weights W such
that the optimum stays inside the trust region: |t| ≤ 0.03·body_diag per
axis, |r| ≤ 6°, s ∈ [0.92, 1.08] (implement as hard bounds, not just λ_reg —
`scipy.optimize.minimize(method="Powell", bounds=...)`; Powell is
derivative-free, the energy is piecewise-smooth due to closest-point
switches). ~100–200 evals × 4k closest-point queries ≈ well under a second.

### Implementation

- New module function `backend/dt_core/refit.py :: polish_placement(
  placed_verts, faces, body_surf, thickness, contact_free, rng_seed=0)
  -> (matrix4x4 row-major, stats dict)`. Returns the DELTA transform (around
  centroid) to compose onto the placement, plus
  `{"evals": int, "e_before": float, "e_after": float}`.
- `run_refit`: new kwarg `auto_polish: bool = False`. When True, run polish
  right after the first `signed_offset` call, apply the delta to
  `garment.vertices`, recompute `signed_offset` once (field building uses the
  polished placement), and record `stats["polish"] = {...,
  "matrix": <16 floats>}` — the FULL effective placement
  (pre_transform ∘ delta), row-major, so the frontend can adopt it.
- `main.py :: RefitPayload`: `auto_polish: bool = False`, pass through.
  Response: expose `res.stats` as today (matrix rides inside `refit.polish`).
- Frontend (small): checkbox "Snap placement" in the refit panel
  (`App.tsx`, near the Place/Resize block) → `auto_polish` in `RefitOpts`
  (`api.ts`). On job completion, if the response carries
  `refit.polish.matrix`, call `setRefitMatrix(matrix)` so the gizmo state
  matches what was actually fitted. Keep the checkbox OFF by default.

**Guard:** if the polish worsens E (Powell can stall on plateaus), return
identity delta. If fewer than 200 sample verts are inside `contact_free`,
skip polish entirely (nothing to align against) with a stats note.

### Tests — new `backend/tests/polish_smoke.py`

1. Sphere body (r=1), garment = spherical shell patch at standoff δ.
   Perturb placement by t=(0.05, −0.03, 0.02)·diag? No — keep inside trust
   region: t = 0.02·diag, uniform scale 1.06, rotation 3°. Assert polish
   recovers ≥ 70% of each error component and `e_after < e_before`.
2. Hanging-skirt immunity: garment = the fold-test skirt (hem far from the
   body). Assert the polish delta is near-identity (‖t‖ < 0.005·diag,
   |s−1| < 0.01).
3. Degenerate: garment fully outside `contact_free` → polish skipped,
   identity, stats note present.

---

## B. Growth continuation (solve against an inflating body)

**Problem it solves:** the deep-penetration case — a shirt placed on a fat
body starts with most verts INSIDE the target; closest-point directions are
side-ambiguous, matching/occlusion/opposed tests degrade, the solve tears the
garment. Measured on the user's real session: 82% of placed verts inside.
LUIVITON's equivalent: they interpolate the body from the garment's canonical
fit to the target and let the simulator track the sequence.

### Mechanic

Solve refit against a SEQUENCE of eroded bodies growing to the true shape.
The garment tracks each small step; closest points stay local and
side-correct throughout.

- Erosion: `body_t = body_verts − e(t)·n_v` with per-vertex normals
  (`correspondence.get_vertex_normals`). e(t) = e0·(1 − t), stages
  t = 1/K, 2/K, …, 1 for K = `growth_steps` (K=1 ≡ today's behavior, one
  stage at e=0).
- e0 (max erosion) from the PLACED garment's penetration profile:
  `e0 = min(0.15·body_diag, quantile(−s[s<0], 0.95) + thickness)`; if fewer
  than 2% of garment verts penetrate, force K=1 (nothing to grow through).
- Erosion self-intersects on thin features (ears, fingers) — acceptable: the
  eroded mesh is only a closest-point attractor, never rendered. Do NOT try
  to make it clean.
- Conform field, occlusion, opposed-release, frozen paint: computed ONCE
  against the TRUE body at placement (they encode user intent; unchanged).
- Per stage k:
  1. `tgt_k = precompute_target(Mesh(body_t_verts, body_faces))` (~0.5 s on
     28k — acceptable ×3; do not cache across stages);
  2. feasibility push: `resolve_push_out(cur_verts, tgt_k.surface,
     offset=thickness, violation=0)` so the stage starts penetration-free;
  3. run the solver for `iters_k` iterations warm-started from `cur_verts`
     (see solver change below) with a per-stage wc schedule: intermediate
     stages use the LOW half of the refit ramp (`[0, 10, 50, 100]`-ish),
     only the final stage runs the full capped ramp. Intermediate stages:
     `iterations=max(3, iterations//K)`; final stage: the remainder.
  4. `cur_verts = result`.
- After the final stage: ARAP bind, collision, exactly as today (unchanged
  code path — the growth loop replaces only the single solver call).

### Solver change (correspondence.py)

`compute_correspondence_with_setup` needs a warm start:
`init_vertices: Optional[np.ndarray] = None` (org-vertex array, len =
source_org). Implementation: where `vertices` is initialized from
`src.source.vertices`, if `init_vertices is not None`, build
`meshlib.Mesh(init_vertices, src.source_org.faces).to_fourth_dimension().vertices`
instead. Everything else (identity/smoothness rest = the PLACED garment via
`src`) stays untouched — the rest shape must NOT change across stages.

Also per-stage `dist_floor`: pass `match_dist_floor = max(0.02·diag_true,
e_k + 2·thickness)` so early stages don't drop everything by the adaptive cut
(e_k = current erosion depth).

### API

- `RefitPreset`: `growth_steps: int = 1`. Set `cloth` preset to 3 ONLY after
  the real-asset check below passes; otherwise keep all presets at 1 and let
  the payload opt in.
- `RefitPayload`: `growth_steps: Optional[int]` (1..6), overrides preset.
- Progress: `job.set_progress("iterations", stage_offset + i, total_iters)` —
  keep the total = sum over stages so the UI bar is monotone.
- Stats: `stats["growth"] = {"steps": K, "e0": e0, "penetrating_at_start": n}`.

### Tests — new `backend/tests/refit_growth_smoke.py`

1. Fat-body synthetic: body = sphere r=1.3; garment = open cylinder shirt
   (radius 1.0, so ~everything starts inside). Run K=1 and K=3 with the same
   iteration budget.
   Assert for K=3: final penetration count (s < −0.005) == 0; garment stays
   coherent — p95 edge stretch vs placed < 60% (K=1 on this setup will tear:
   document its numbers in the test output as the baseline, assert K=3
   strictly better on both metrics).
2. No-penetration shortcut: the fold-test collar (placed outside) with
   growth_steps=4 must auto-collapse to K=1 (`stats["growth"]["steps"] == 1`)
   and produce byte-identical-ish results to today (same PASS set).
3. Real asset: script may load `../testfit/tg.obj` + body_clean with a
   pre_transform that scales the garment down until ≥60% penetrates, then
   assert K=3 ends with < 0.5% penetrating and finite verts. (Keep runtime
   < 3 min; 8 iterations total.)

---

## D. Proxy wardrobe (fit once to the basemesh, wear on every body)

**Problem it solves:** every garment is currently fitted per body from
scratch. LUIVITON's core idea — garment→proxy and body→proxy decouple — maps
onto machinery we already have: a Wrap run IS the body→proxy registration
(basemesh deformed into the target body's shape, same topology as basemesh).

**Total mechanic:** bind the fitted garment to the basemesh surface once;
transport it through any basemesh→body deformation; clean up with a short
refit.

### Scope for v1 (no "character project" model yet)

Works inside one session: the user (a) runs Wrap with source_ref = basemesh
P, target_ref = body B → the wrap slot holds P′ (P's topology in B's shape);
(b) uploads/keeps a garment G that was fitted to P (its rest pose already
sits on P). Then "wear via proxy" = transport G by the P→P′ deformation and
run a short refit against B.

### Implementation

New module `backend/dt_core/wardrobe.py`:

```python
@dataclass
class GarmentBinding:
    tri: np.ndarray        # (N,) basemesh triangle index per garment vertex
    bary: np.ndarray       # (N,2) barycentric (u,v) of the closest point
    offset: np.ndarray     # (N,) signed distance along the triangle normal

def bind_garment(garment_verts, proxy_surf: SurfaceData) -> GarmentBinding:
    # closest_points_on_surface gives cp + tri; recover (u,v) by solving the
    # 2x2 system against tri_ab/tri_ac (regions already computed the closest
    # point, so the system is consistent); offset = dot(v - cp, tri_normal).

def apply_binding(b: GarmentBinding, proxy_verts, proxy_faces) -> np.ndarray:
    # rebuild per-triangle a/ab/ac + unit normals from the DEFORMED proxy,
    # return a + u*ab + v*ac + offset*n. Pure numpy, no queries.
```

Notes:
- `offset` transported verbatim (not scaled by local stretch) — good enough
  as an initial placement; the cleanup refit re-establishes thickness.
- Binding is deterministic per (garment_hash, proxy_hash): cache the arrays
  under the session workspace (`wardrobe/bind_<hash>.npz`) so re-wearing is
  instant. Hash = sha256 of the vertex buffer bytes (see
  `meshlib.cache.SparseMatrixCache` for the hashing pattern used elsewhere).

`run_refit` change: accept `placed_verts: Optional[np.ndarray]` as an
alternative to `pre_transform` (mutually exclusive; placed_verts wins). It
replaces the `apply_transform` line — everything downstream already works
from `garment.vertices`.

Endpoint `POST /api/sessions/{sid}/refit_via_proxy` (`main.py`):
- Preconditions: session has a wrap result (`s.wrap_path`,
  `s.wrap_mode in ("wrap","region")` — refit results in the slot don't
  qualify; keep the original wrap source available as `s.source_ref_path`)
  and both refs uploaded. source_ref at the time of the wrap = the proxy
  rest P; the CURRENT source_ref = the garment G (the user re-uploads source
  after wrapping). That's fragile — instead require an explicit upload:
  payload carries nothing; the endpoint loads P from a NEW session artifact.
  **Decision for the implementer:** v1 keeps it simple and explicit — add an
  upload role `proxy_ref` (`/upload/proxy_ref`, same plumbing as
  source/target) holding the basemesh rest P. The endpoint then:
  1. loads G = source_ref, B = target_ref, P = proxy_ref,
     P′ = wrap result verts (must have len(P.verts) — else 400);
  2. binding = cached `bind_garment(G.vertices, build_surface(P))`;
  3. `placed = apply_binding(binding, P′_verts, P.faces)`;
  4. `run_refit(G, B, preset, placed_verts=placed,
     iterations=min(payload.iterations, 6), ...)` — short cleanup;
  5. persists exactly like `_refit_compute` (reuse it: factor the
     save/residual/hide-mask tail of `_refit_compute` into a helper and call
     it from both endpoints).
- Payload: same RefitPayload minus `pre_transform` (ignored if sent).

Frontend v1 (keep minimal): in the refit panel, a collapsible "Wear via
proxy" block: upload button for the proxy basemesh + a Run button that is
enabled when a wrap result exists; calls the new endpoint. New `api.ts`
function `refitViaProxy(opts)`. No gizmo/pins in this path (placement comes
from the binding).

### Tests — new `backend/tests/wardrobe_smoke.py`

1. Binding round-trip: proxy = sphere; garment = band at standoff 0.05.
   `apply_binding(bind_garment(...), undeformed proxy)` must reproduce the
   garment verts to < 1e-9.
2. Transport: deform proxy sphere → ellipsoid (scale 1.4, 1.0, 0.8).
   Transported band must sit on the ellipsoid at the same parametric
   latitude, standoff preserved within 30% (offset is not stretch-aware).
3. End-to-end: transported band + short refit against the ellipsoid →
   penetration 0, band coherent (edge-stretch p95 < 10%).

---

## Cross-cutting

- **Flags & defaults:** A ON for refit only (OFF for wrap), C/B/D opt-in via
  payload; presets untouched until real-asset validation.
- **Definition of done per item:** new smoke test green; full regression list
  (top of this file) green; `npm run build` clean when the item touches the
  frontend; a manual run on `testfit/` assets with stats printed (before vs
  after) pasted into the PR/commit message.
- **Perf guardrails:** refit on testfit (10.6k garment / 28.8k body,
  8 iterations) currently runs ~35–40 s. A ≤ +10%, C ≤ +2 s, B scales
  roughly linearly with growth_steps (document the measured factor), D's
  transport+bind ≤ 2 s on cached binding.
- **Commit style:** one commit per item, message summarizing behavior change +
  test evidence; do not amend the checkpoint history.
- The backend runs without --reload; after landing each item, restart uvicorn
  and rebuild the frontend if touched (`npm run build`; the server serves
  `frontend/dist`).
