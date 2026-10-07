# Layer auto-segmentation plan: L1–L6

Goal: make the driven/follower wall separation in refit (inner wall grips and
inflates via correspondence; outer walls, folds and attachments only FOLLOW it)
reliable enough that manual frozen paint becomes a correction tool, not the
workflow. Today the automatic release (`occluded | opposed` in
`build_refit_field`) is per-vertex, binary, single-ray and trusts garment
normals — real assets defeat it, so users hand-paint. This plan replaces it
with a score-based classification built on one principle:

> **The driven (inner) wall is the garment surface the BODY sees first.**
> Rays start on the body and travel along the body's own outward normals; the
> first cloth crossing is driven, later crossings are followers. Garment
> normals are never read (game assets flip them); body normals are ours and
> trustworthy.

The follower mechanics need no invention — the ARAP bind pass already gives
surface-deform semantics. Only the CLASSIFICATION is upgraded, plus one bind
fix for disconnected attachments (L5).

Implementation order: **L1 → L2 → L3** (ship together as "auto layer mode"),
then **L5**, **L4**, **L6** (independent, each optional).

---

## Context: what exists today and where it breaks

`backend/dt_core/refit.py :: build_refit_field()` releases layered walls with
two signals, either one frees the vertex:

1. `occluded_layer_mask()` — ONE segment per vertex to its closest body point,
   crossing test via `surface.segments_hit_mesh` (Möller–Trumbore, centroid-KD
   candidates). Breaks when: the outer wall has line of sight past the inner
   wall (flared collar — documented in the code comment), the segment slips
   through a hole/gap in the inner wall, or the closest-point direction grazes
   a fold.
2. normal opposition — placed vertex normal · closest body normal < 0. A
   patch for the flared collar that fully trusts the garment's normals;
   flipped normals (ubiquitous in game meshes, incl. single-wall shirts with
   only rim/collar doubled) break it in both directions, and concave body
   areas (armpit, sleeve wall facing the torso) false-positive.

Both are per-vertex booleans with no spatial coherence (speckled masks) and no
policy for disconnected components (pouches, buttons, buckles). Deep
penetration at placement additionally inverts the occlusion semantics — the
growth-continuation machinery already solves this for the SOLVER via body
erosion; classification must use the same trick (L2).

Downstream, released verts get `field = 0`, ride identity/smoothness, and the
bind pass (`arap.py :: arap_bind`) re-poses them from the placed shape with
gripped verts as handles. Connected followers are therefore fine. Orphan
components (no handle) are currently promoted whole to handles — i.e. they
stay at the solver output ≈ placed pose and do NOT follow the body (L5).

Prior art this design leans on (references, not dependencies):
* Robust Skin Weights Transfer via Weight Inpainting (Abdrashitov et al.,
  SIGGRAPH Asia 2023) — confident sparse correspondence + interpolation for
  the rest, instead of forcing a per-vertex decision.
* Generalized winding numbers (Jacobson et al. 2013; Barill et al. 2018) —
  hole/orientation-robust "is there cloth in between" (L6 upgrade).

### Repo conventions (follow them)

- Backend numpy/scipy only in `dt_core`; heavy deps import lazily inside
  functions. No torch, no libigl.
- Every new behavior ships behind a parameter defaulting to the old behavior.
- Tests: standalone headless `backend/tests/*_smoke.py`, print `PASS`/`FAIL` +
  `ALL PASS`, exit 1 on failure. Run from `backend/`:
  `.venv\Scripts\python.exe tests\layers_smoke.py`.
- Regression suite to keep green: `refit_fold_smoke, refit_e2e_smoke,
  refit_conform_smoke, refit_field_smoke, pins_smoke, collision_smoke,
  wrap_quality_smoke, soft_markers_smoke, region_smoke` (+ the new ones below).
- Frontend `npm run build` (tsc strict) when touched; backend runs without
  `--reload` — restart after landing.
- Real-asset checks on `testfit/` (tg.obj 10.6k verts garment, body_clean.obj
  28.8k body); don't commit new large assets.

---

## L1. Ordered ray-hits primitive

**Problem:** `segments_hit_mesh` answers only "crossed: yes/no". The body-side
signal needs ORDERED crossings per ray (1st hit = driven vote, 2nd+ =
follower vote), and the same primitive later serves button/stacked-layer
ordering.

### Implementation

New function in `backend/dt_core/surface.py`, next to `segments_hit_mesh` and
sharing its candidate-gathering pattern:

```python
def ray_hits_ordered(
    origins: np.ndarray,          # (R,3)
    dirs: np.ndarray,             # (R,3) unit directions
    t_max: np.ndarray,            # (R,) per-ray cap (world units)
    surf: SurfaceData,            # mesh to intersect (the garment)
    exclude_verts: Optional[np.ndarray] = None,   # per-ray vertex to skip fans of
    t_eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns (ray_idx, t, tri) flat arrays lexsorted by (ray_idx, t):
    # all crossings with t in (t_eps, t_max). Callers get first hits via
    # np.unique(ray_idx, return_index=True).
```

- Candidates: centroid-KD ball query at `origins + dirs * t_max/2` with radius
  `t_max/2 + r_tri` — same recipe as `segments_hit_mesh` (reuse its `corner_r`
  computation; factor it into a tiny helper rather than copy).
- Möller–Trumbore over (ray, candidate) pairs, batched; keep ALL hits, not
  `any()`. Dedup near-identical t on shared edges: after lexsort, drop hits
  whose t differs from the previous hit of the same ray by < `t_eps` (an edge
  crossing reports both incident triangles — it is ONE cloth crossing).
- Per-ray `t_max` as an array (erosion makes caps ray-dependent).

### Tests — new `backend/tests/ray_hits_smoke.py`

1. Two concentric spheres (r=1, r=2), rays from center outward: every ray
   reports exactly 2 hits, ordered t≈1 then t≈2, correct tri ownership.
2. Edge dedup: ray aimed exactly at a shared edge → 1 crossing, not 2.
3. Cap: t_max=1.5 → only the inner sphere reported.
4. exclude_verts: ray leaving a vertex does not hit its own fan.
5. Perf print (no assert): 100k rays vs a 20k-tri mesh, seconds logged.

---

## L2. Layer scores: body-side voting + fan-out occlusion, on an eroded reference

**Problem it solves:** the two brittle legacy signals. Two new signals, both
orientation-independent w.r.t. the garment, both continuous, both computed
against a penetration-corrected body.

### Mechanic

New module `backend/dt_core/layers.py` (pure geometry, headless-testable).

**Reference body (erosion).** If < 2% of placed garment verts penetrate
(same gate as growth), the reference is the true body, `e_cls = 0`. Otherwise
`e_cls = min(0.15·body_diag, quantile(−s[s<0], 0.95) + thickness)` and the
reference verts are `body_verts − e_cls·body_vn`. Factor this formula OUT of
`run_refit`'s growth block into `layers.erosion_depth(body_signed, thickness,
body_diag)` and call it from both places — one definition. Rationale:
inflating/deflating the body a few cm along its normals never reorders cloth
layers along the skin-outward direction, so a classification computed against
the eroded body is valid for the true one; the eroded mesh is a computational
reference only (self-intersections on ears/fingers are fine — same argument
as growth).

**Signal A — body-side first-hit votes** (primary):

```python
def body_first_hit_votes(
    ref_verts, body_vn,            # reference body verts + TRUE body vertex normals
    gsurf,                         # build_surface of the PLACED garment
    body_surf_ref,                 # build_surface of the reference body (self-occlusion)
    cap,                           # scalar: e_cls + 2*contact_free
    n_verts_g: int,
    cone_deg: float = 15.0, n_cone: int = 4,
) -> tuple[np.ndarray, np.ndarray]:   # (driven_votes, follower_votes) per garment vertex
```

- Ray origins: reference body verts pre-filtered by a cKDTree query — keep
  only verts within `cap` of any placed garment vert (skips feet when fitting
  a collar). Directions: the vertex normal plus `n_cone` directions tilted
  `cone_deg` around it (deterministic tangent frame: Gram–Schmidt against
  global up, fallback X when parallel).
- For each ray: garment crossings via `ray_hits_ordered`; body self-occlusion
  via the same primitive against `body_surf_ref` (exclude the origin vertex
  fan) — garment hits farther than the first body self-hit are discarded
  (armpit: skin facing skin, no vote through flesh).
- First surviving garment hit → `driven_votes += 1` on the hit triangle's 3
  verts (`np.add.at`); every later hit → `follower_votes += 1`. No garment
  normals anywhere.
- Body normals are trusted; optional one-time sanity (behind a flag, default
  off): flip body normals whose `p + ε·n` lands inside per a coarse
  winding-number check. Not needed for our own basemeshes.

**Signal B — fan-out occlusion fraction** (upgrade of the legacy single
segment; runs on the garment side):

```python
def occluded_fraction(
    g_verts, active_idx,           # placed garment verts; verts with field > 0
    ref_verts_tree,                # cKDTree over reference body verts
    gsurf, k: int = 8,
) -> np.ndarray:                   # (len(active_idx),) fraction in [0,1]
```

- k nearest reference-body verts per active garment vert, k segments, existing
  `segments_hit_mesh` verbatim, occluded fraction = mean over k. The single
  "gap to the shoulder" no longer whitewashes an outer wall; against the
  eroded reference a submerged vertex's segments go INWARD to the surface
  below it, so the semantics never invert.

**Combined score:**

```python
@dataclass
class LayerScores:
    score: np.ndarray        # per-vertex driven-ness in [0,1] (0 where inactive)
    a_driven: np.ndarray     # raw votes, kept for stats/preview/debug
    a_follower: np.ndarray
    occl_frac: np.ndarray    # over all verts, -1 where not computed
    coverage: np.ndarray     # bool: vertex received any A votes
    e_cls: float

def layer_scores(placed_verts, faces, target_org, body_signed, body_vn,
                 active, contact_free, thickness, body_diag, ...) -> LayerScores
```

- `a = a_driven / (a_driven + a_follower)` where coverage, else neutral 0.5;
  `b = 1 − occl_frac`; `score = 0.5·a + 0.5·b` (coverage-weighted: `b` alone
  where A saw nothing). Weights are module constants, not user knobs.
- Legacy `opposed` (garment-normal test) is NOT part of the score — retiring
  it is the point. Signal A already covers the flared collar it was patched
  in for (a flared outer wall gets zero first-hit votes).

### Tests — new `backend/tests/layers_smoke.py` (cases 1–4; L3 adds more)

Synthetic rigs, built in-test like `refit_fold_smoke`:

1. **Double-wall collar, closed shell**: body = cylinder (neck); garment =
   cylindrical band with inner wall at r+δ and outer wall at r+3δ joined by a
   fold — a watertight two-layer tube (the user's "fully double-walled" case).
   Assert inner-wall verts score > 0.7, outer-wall < 0.3.
2. **Flipped normals**: same rig with every garment face inverted — scores
   must be IDENTICAL (bitwise, same rays): the signals never read garment
   orientation. (The legacy `opposed` test provably fails here; print its
   error rate as documentation.)
3. **Flared collar**: outer wall tilted 40° outward (line of sight to the
   body past the inner rim) — outer wall still < 0.3 (zero first-hit votes).
4. **Submerged placement**: shrink the body… rather, scale the garment down
   so the inner wall sits INSIDE the cylinder (≥ 50% penetration). Assert
   `e_cls > 0` and the same classification as case 1 (erosion restores
   ordering).

Perf note printed (no assert): full L2 on testfit-scale (10.6k garment,
28.8k body, ~5 rays/body vert) — budget ≤ 4 s.

---

## L3. Released-mask v2: smoothing + hysteresis + component policy, wired into refit and the UI

**Problem it solves:** raw per-vertex scores are still noisy at folds and
undecided in low-coverage areas; disconnected attachments need a policy; and
the result must land where users already work — the conform field and the
frozen-paint brush.

### Mechanic

```python
def released_mask_auto(
    scores: LayerScores, verts, faces, active,
    hi: float = 0.6, lo: float = 0.4,
    comp_driven_min: float = 0.15,     # min driven AREA fraction per component
    comp_area_min: float = 0.02,       # components smaller than this fraction of
                                       # total garment area → follower outright
) -> tuple[np.ndarray, np.ndarray, dict]:
    # (released bool over all verts, layer int8 over all verts, stats)
    # layer: 0 = driven, 1 = follower (wall), 2 = follower (whole component)
```

1. **Smoothing**: 3 rounds of neighbor averaging of `score` over the vertex
   graph (`regions.weighted_adjacency`, row-normalized), active verts only.
2. **Hysteresis**: score ≥ hi → driven; ≤ lo → follower; the (lo, hi) band
   joins whichever label the majority of its already-decided graph neighbors
   holds (iterate to fixation, ≤ 10 passes; leftovers → driven, i.e. legacy
   behavior — release only on evidence).
3. **Component policy** (`scipy.sparse.csgraph.connected_components`, already
   used in the bind pass): per component compute driven area fraction over its
   active verts (vertex area = 1/3 incident triangle areas). Fraction <
   `comp_driven_min` OR component area < `comp_area_min`·total → the WHOLE
   component is layer 2 (pouches, buckles, buttons, straps). Note buttons on
   a shirt are already layer-1 by ordering (the body ray hits the shirt wall
   first, the button second) — the area rule is a second net, and layer 2 is
   what L5's rigid attach keys off.

### Integration

- `build_refit_field(..., layer_mode: str = "legacy")` — `"auto"` computes
  `layer_scores` + `released_mask_auto` and uses that as `released` instead
  of `occluded | opposed` (which are then not computed — they cost raycasts).
  Everything downstream (field zeroing, preserve pins, frozen paint, bind
  handle selection) is untouched; frozen paint remains the absolute override.
- `RefitField` gains `layer: Optional[np.ndarray]` and `layer_stats: dict`
  (vote coverage, e_cls, per-layer counts); `run_refit` forwards `layer_mode`
  and spreads `layer_stats` into `stats["layers"]`.
- `main.py :: RefitPayload.layer_mode: str = "legacy"` (validated against
  {"legacy", "auto"}), passed through by `_refit_compute`, the proxy endpoint
  and `refit_field_endpoint`.
- `/refit_field` response adds `"layer": [...]` (int8 list) and the stats —
  the dry-run is synchronous, so keep its budget: if L2 exceeds ~3 s on the
  dry-run path, decimate ray origins ×2 there (`n_cone=2`) — dry-run
  precision can be slightly looser than the solve's.
- **Frontend (the reliability payoff)**: in the refit panel a "Layers: auto"
  toggle (`api.ts :: RefitOpts.layerMode`); on dry-run response
  `MeshViewport` colors verts by `layer` (reuse the grip-heatmap vertex-color
  path in `PreviewPanel`/`MeshViewport`: driven = base, follower = blue,
  component = purple); a button **"→ frozen paint"** merges layer ∈ {1, 2}
  verts into the existing frozen set so the user EDITS a prefilled mask with
  the brush they already know instead of painting from scratch. Painted state
  still persists via `refit_state` (size bound already exists).

### Tests

Extend `backend/tests/layers_smoke.py`:

5. **Sailor collar**: flat collar lying on the shoulders over the shirt back —
   collar-under-side verts above shirt cloth → follower; collar patch
   touching bare skin (no shirt beneath) → driven. Assert both.
6. **Pouch component**: body cylinder + gripping band + disconnected box next
   to the band, near the body. Assert the whole box is layer 2 even though
   its body-facing wall gets first-hit votes.
7. **Speckle robustness**: case-1 collar with 5% of scores randomly flipped
   pre-smoothing (rng seeded) — final mask identical to case 1.
8. **No-double-wall regression**: single-wall open cylinder shirt, auto mode
   → zero released verts (score high everywhere; nothing to release).

New `backend/tests/layer_groundtruth_check.py` (manual harness, NOT in the
regression suite — real assets aren't committed): loads a garment + body +
a session's `refit_state` frozen set (paths as argv), runs auto
classification, prints IoU of (layer ≥ 1) vs the hand-painted frozen set +
per-signal stats. Run it on the user's real painted sessions; target IoU
≥ 0.8 before flipping any preset default.

HTTP: `refit_field_smoke` extended with `layer_mode="auto"` round-trip
(response carries `layer`, len == n_verts). `npm run build` green.

**Perf budget:** solve path ≤ +5 s on testfit (baseline 35–40 s); dry-run
≤ 3 s.

---

## L4. Graph-cut boundary refinement (optional quality upgrade)

**Problem it solves:** smoothing + hysteresis put the driven/follower boundary
NEAR the fold; a min-cut puts it ON the fold — the same line a careful human
paints, and the visible difference between "ok" and "clean" on collars/cuffs.

### Mechanic

Binary min-cut labeling on the active-vertex graph:

- unary: terminal capacities `c_src = w_u · score`, `c_sink = w_u · (1 −
  score)` (w_u ≈ 4; frozen-painted verts get an infinite sink edge — paint
  stays absolute).
- pairwise: capacity per mesh edge ∝ edge length, ×0.2 when the dihedral
  angle across the edge exceeds 60° (fold crease = cheap place to cut), ×0.2
  when the two verts' body distances differ by > thickness·4 (standoff jump).
- solver: `scipy.sparse.csgraph.maximum_flow` (int32 capacities — quantize
  ×1024). Graph = active verts + 2 terminals; testfit-scale is trivial for it.

```python
def graphcut_labels(score, verts, faces, active, frozen_idx,
                    w_unary=4.0, crease_deg=60.0, dist_jump=4.0) -> np.ndarray
```

Slots into `released_mask_auto` between smoothing and the component policy,
behind `layer_graphcut: bool = False` (payload knob rides `RefitPayload`).

### Tests — new `backend/tests/layer_graphcut_smoke.py`

1. Fold accuracy: case-1 collar — every label-change edge lies within 1 ring
   of the true fold loop (constructed, so known).
2. All-driven no-op: single-wall shirt — cut releases nothing.
3. Determinism: two runs, identical labels.
4. Budget: ≤ +2 s on testfit-scale (print, assert < 5 s).

---

## L5. Rigid attachment for released components in the bind pass

**Problem it solves:** orphan components (no bind handle) are currently
promoted whole to handles → they keep the solver output ≈ placed pose and do
not follow the body at all. A pouch must ride its belt; a buckle must ride its
strap — child-of semantics, shape preserved exactly.

### Mechanic

In `run_refit`'s bind section, replace the orphan-promotion branch behind
`orphan_bind: str = "handles"` (default) | `"rigid"`:

```python
def rigid_attach_components(placed, fitted, labels, orphan_comp_ids,
                            handle_idx, k: int = 32,
                            max_reach: float) -> np.ndarray
```

- Per orphan component: anchor = its placed vert closest to any handle vert;
  patch = k nearest handle verts to the anchor (cKDTree over placed handles);
  Kabsch best-fit rigid transform of that patch's placed→fitted motion (same
  SVD recipe as `arap_bind`'s local step / the rigidity filter); apply to the
  whole component's PLACED verts.
- `max_reach` (default `0.1·body_diag`): anchor farther than this from every
  handle → leave the component at its placed pose (nothing to ride).
- Components classified layer 2 by L3 are exactly the intended customers, but
  the function keys only off "component with no handles" — it also fixes the
  legacy path when enabled.

### Tests — new `backend/tests/bind_rigid_smoke.py`

1. Pouch-on-belt: cylinder body, gripping belt band, disconnected pouch box
   0.02·diag from the belt. Deform the body (bend/scale), run refit with
   `orphan_bind="rigid"`. Assert pouch edge lengths preserved to 1e-6 (rigid),
   and pouch centroid displacement ≈ its belt patch displacement (within 20%).
2. Out-of-reach component stays at placed pose.
3. Legacy default: `orphan_bind="handles"` → byte-identical to today on the
   fold-test rigs (regression).

---

## L6. Winding-number occlusion backend (only if real assets demand it)

**Problem it solves:** ray-based crossing tests leak through holes/gaps in
the inner wall (torn game meshes). The generalized winding number jump along
a segment counts cloth crossings smoothly and is robust to holes,
self-intersections and orientation — the exact failure modes of rays.

### Mechanic

`layers.py`, behind `occlusion_backend: str = "rays"` | `"winding"`, consumed
by `occluded_fraction`:

- exact GWN in chunked numpy (van Oosterom–Strackee signed solid angle of
  each garment triangle at each query point; sum / 4π). Queries: 4 sample
  points per segment; measure `max |w(p_i) − w(p_0)|`; crossing ⇔ jump ≥ 0.5.
- cost: queries × all garment tris; chunk to keep peak memory < ~500 MB.
  testfit-scale ≈ (8k active × 8 segs × 4 pts) ≈ 256k queries × 20k tris —
  chunked einsum, seconds-range; if it busts the budget, pre-prune far
  triangles per chunk with the centroid KD tree (they contribute ~0 to the
  JUMP even though they contribute to w).

### Tests — new `backend/tests/layer_winding_smoke.py`

1. Holed inner wall: case-1 collar with 30% of inner-wall faces deleted —
   assert `"winding"` still classifies the outer wall < 0.3 while `"rays"`
   documents its leak rate (print, no assert on rays).
2. Agreement: on the intact collar, both backends produce the same mask.
3. Budget print; assert < 10 s on testfit-scale.

---

## Cross-cutting

- **Flags & defaults:** everything defaults to legacy behavior
  (`layer_mode="legacy"`, `layer_graphcut=False`, `orphan_bind="handles"`,
  `occlusion_backend="rays"`). Flip `layer_mode="auto"` per-preset only after
  the ground-truth harness shows IoU ≥ 0.8 on the real painted sessions AND
  the full regression suite is green in auto mode.
- **Definition of done per item:** its smoke test green; regression list
  green; `npm run build` clean when the frontend is touched; a manual
  `testfit/` run with `stats["layers"]` (and before/after screenshots for L3's
  overlay) pasted into the commit message.
- **Perf guardrails:** refit on testfit currently ~35–40 s. L2+L3 ≤ +5 s
  (solve) / ≤ 3 s (dry-run); L4 ≤ +2 s; L5 negligible; L6 ≤ +10 s and only
  when opted in.
- **Docs:** update docs/ROADMAP.md's refit section (layer classification
  paragraph) and the refit panel help text when L3 lands.
- **Commit style:** one commit per item; do not amend checkpoint history.
- **Explicitly out of scope:** watertight completion / mesh splitting (the
  functional boundary is the fold curve, not an open boundary — and splitting
  breaks the solver's rest state); trusting garment normals anywhere in the
  new path; multi-garment stacked outfits (the ordered-hits primitive L1
  leaves the door open — first/second/third crossings — but no code here).
