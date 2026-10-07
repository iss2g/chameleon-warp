# Roadmap

Ideas and known limitations, roughly grouped. Nothing here is a promise;
implementation notes for larger pieces live in [design/](design/).

## Core tools

- **Reverse attraction (detail capture)** — a second wrap term where target
  vertices pull the nearest point of the source surface, so details between
  source vertices (a bump, a fold) are captured without a marker on each.
  Slower; would be an opt-in for the final iterations.
- **Straight-edge preservation** — keep garment borders that are straight in
  the source (collar, cuffs, hem) straight after correspondence, instead of
  inheriting the target's local curvature.
- **Region wrap:** compute the adaptive match-distance median over editable
  vertices only (today it includes frozen ones).
- **Quad twist metric** in the wrap report, as a diagnostic for quad export.
- **Fit / Refit** are beta: solver stabilization and better defaults.
- Soft cap with a warning (instead of a hard error) when an FBX has more poses
  than the session limit.
- Built-in "try an example" flow using the meshes in `demo_assets/`.

## Garments

- Material/texture pass-through for refit results.
- Layer rules between garments (a shirt under a jacket: mask priorities).
- **Garments in motion:**
  - transfer of skin weights from body to garment, with weight smoothing;
  - "dressed range-of-motion" QA: run a dressed character through a ROM
    animation and report per-pose penetrations;
  - body mask as the union over all ROM poses, not only the bind pose;
  - pose-driven corrective shapes for problem areas (elbows, hips).

## Character pipeline (longer term)

Chameleon Warp covers retopology-by-wrap, blendshape transfer and garment
fitting. Stages around it that are not covered yet:

- scale/axis normalization and pose normalization (A/T-pose) of input meshes;
- mesh repair (holes, non-manifold geometry, duplicate shells);
- UV transfer and texture baking onto the wrapped topology;
- rig and skin-weight transfer from a reference character;
- LODs and export conventions for game engines (FBX/glTF scale, axes, tangents).

## Performance

Profile first (`backend/tools/profile_pipeline.py`): on typical inputs the
correspondence solve dominates.

- **Layer auto broad-phase.** Refit's automatic layer classification gathers
  ray-cast candidates with one sphere around each ray's midpoint, which on a
  dense body tests ~10⁸ ray/triangle pairs (minutes for a 42k-triangle body).
  Gathering candidates in small segments along the ray (or a slab/AABB test)
  should cut that by orders of magnitude with an identical result; the current
  process-parallel split only gives 3–4.5×.
- float32 in the matching phase; early exit from correspondence iterations once
  vertices stop moving.
- Vectorized assembly of the sparse system; reuse of Pardiso's symbolic
  factorization across iterations (the sparsity pattern barely changes).
- Batched right-hand sides for multi-pose transfer.
- A decimated proxy of very dense targets for matching only.
- Cache source/target setups across sessions by mesh hash.
- GPU matching/projection (e.g. NVIDIA Warp mesh queries) with CPU fallback.
