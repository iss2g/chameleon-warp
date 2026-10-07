# Core-tools finish line: refit/wrap/dt — closing feature set

Goal: bring refit/wrap/dt-transfer to "feature complete for v1" so work can
move on to the character-pipeline stages (see docs/ROADMAP.md). Five items,
ordered. Written to be implemented standalone; read refit-abcd.md's
"Context" + "Repo conventions" sections first — they apply verbatim (test
style, regression list, no --reload, frontend served from dist, one commit
per item).

Status of prior plan: A/B/C/D are all landed (commits e593667..cf041b8).

Implementation order: **1 → 2 → 3 → 4 → 5**. Items 1–4 are small/medium;
item 5 is the architecture jump and can be a separate work stream.

---

## 1. Combined result preview: garment ON the body

**Problem:** after a refit the preview shows the fitted garment alone; the
user downloads and assembles in Blender to see how it sits. All the pieces
already exist in the viewer.

**Mechanic:** in `frontend/src/components/PreviewPanel.tsx`, the non-transfer
branch renders the result in a `MeshViewport`. `MeshViewport` already
supports a second mesh via the `ghostMesh` prop (used in the editor for
placement). Add a "Show body" toggle (default ON when the session's last
result came from refit — the store knows `wrapMesh` came with a `refit`
stats block; plumb a boolean through the store when the job result lands):

- body = the session's target mesh (`useStore.targetMesh` — already loaded
  for the editor; if null in preview-only flows, fetch via the existing
  target mesh endpoint).
- Render the body SOLID (not semi-transparent): honest preview — where the
  garment penetrates, the user sees it. Use a distinct neutral color
  (e.g. the 0x8fa6c4 used for "before" meshes).
- `ghostMesh` currently renders semi-transparent and non-pickable; add a
  prop `ghostStyle?: 'ghost' | 'solid'` (default 'ghost' — editor behavior
  unchanged) and use 'solid' here.
- Camera: frame on the UNION bounding sphere of result+body (extend
  `handleFrame`/auto-framing to account for the ghost when present).
- Heatmaps (error/grip) keep working on the garment unchanged.

**Body-mask application in preview (bridge to item 2):** when the session has
a body mask (see item 2's GET endpoint), a second toggle "Apply body mask"
hides masked faces of the body: client-side filter — drop faces whose three
vertices are all in the mask (mirrors `collision.hidden_faces`). Implement by
building a filtered index buffer for the ghost geometry; keep the unfiltered
one to toggle back without re-fetch.

**Tests/DoD:** `npm run build` clean; manual: refit on testfit assets →
preview shows shirt on body, toggles work, mask hides the covered belly.
No backend change except none. Screenshot in the commit message.

---

## 2. Body-mask lifecycle (compute → edit → preview → persist → export)

**Problem:** hide-body masks are computed only by the `armor` preset, saved
as `wrap/body_hide.json`, and there is no way to see or fix them. The user
needs a mask per fitted garment, editable by painting on the TARGET, because
fits are imperfect and garments don't follow the body 100%.

### Backend

- `run_refit`: compute the mask for **every** preset (not just
  `hide_body` collision mode) — it's cheap (`collision.body_hide_mask`).
  Keep `eff_collision` semantics unchanged (push_out still pushes); the mask
  is now an independent output. Params already exist (`offset`,
  `max_depth=0.1*body_diag`).
- New session artifact: keep writing `wrap/body_hide.json`
  (`hidden_vertices`, `body_vertex_count`) for every refit.
- New endpoints in `main.py`:
  - `GET  /api/sessions/{sid}/body_mask` → `{hidden_vertices: [...],
    body_vertex_count: n}` (404 if none).
  - `PUT  /api/sessions/{sid}/body_mask` body `{hidden_vertices: [...]}` —
    replaces the file (validated: ints in range). This is how paint edits
    persist.
  - `POST /api/sessions/{sid}/body_mask/dilate` body `{rings: +N | -N}` —
    grow/shrink by vertex-adjacency rings on the target
    (`regions.weighted_adjacency` boolean matmul; negative rings = erode:
    keep verts whose full ring is inside). Returns the new mask. Dilation is
    the standard "make the mask generous so pose changes don't poke through"
    knob (see docs/ROADMAP.md, "Garments in motion" — pose-union masks come later, dilate is
    the manual stopgap).
- Download: `GET /api/sessions/{sid}/body_mask.json` as a file download, and
  include it in the OBJ download flow unchanged (no OBJ encoding — JSON
  sidecar; GLB embedding arrives with item 5).

### Frontend

- In the refit RESULT view (PreviewPanel, after item 1): "Edit body mask"
  button switches the right viewport into mask-paint mode ON THE BODY mesh:
  reuse the existing generic brush (`paintMode`, `onPaintStroke`,
  `frozenVerts` highlight — they are not frozen-specific in MeshViewport).
  Left-drag adds to mask, right-drag removes; masked verts tinted (reuse the
  ice-blue points or better: vertex-color the masked faces dark). "Save mask"
  → PUT. Buttons: Dilate +1 / Erode −1 / Reset to computed.
- Stats line: `hidden N of M verts`.

**Tests:** extend `backend/tests/collision_smoke.py` or new
`body_mask_smoke.py`: dilate +1 grows the set (superset, bounded), erode −1
shrinks, dilate then erode ⊆ dilate; PUT round-trip via the session state is
covered by an HTTP smoke only if a server harness exists — otherwise unit-test
the dilate helper directly.

---

## 3. UX-debt bundle (small, ships as one commit or three)

### 3a. Persist refit working state per session

`refitMatrix` (gizmo placement), pins, frozen-paint set, chosen preset — all
live in React state and die on F5 / view switches / session restore. Persist
them SERVER-side with the session (they are session data, like markers):

- Backend: `refit_state: Optional[dict]` on the Session (persisted in
  `state.json` like markers are), endpoints
  `PUT /api/sessions/{sid}/refit_state` (whole blob) and include it in the
  session summary GET. Blob: `{pre_transform: [16]|null, pins: [...],
  frozen: [...], preset: str, advanced: {...}}` — opaque to the backend
  (validate size < 2 MB, drop out-of-range indices on use as already done).
- Frontend: on any change to refitMatrix/pins/frozenSet/preset — debounced
  (1 s) PUT; on session load — hydrate the React state from the summary.
  Remove the current "reset on sourceRefName change" only when the incoming
  state matches the current source (store the source name inside the blob;
  if it differs — discard, the mesh changed).

### 3b. Refit stats + warnings in the result panel

The job result already carries `refit: {contact_verts, occluded_verts,
opposed_verts, matched_verts, rigidity_dropped?, frozen_verts, bind, ...}`.
Render a compact block in PreviewPanel (refit case) + WARNINGS derived
client-side:

- `matched_verts == 0` → "Nothing gripped — check placement/scale or lower
  contact thresholds."
- `opposed_verts > 0.5 * vertex_count` → "Most normals face away from the
  body — the garment (or body) may have inverted normals."
- `contact_verts < 0.02 * vertex_count` → "Garment placed far from the body —
  move/scale it closer or raise contact_free."

Also add the three advanced sliders to the refit panel (Advanced collapsible):
`contact_tight`, `contact_free`, `thickness` — world units, default = preset
(send null when untouched). Payload fields already exist.

### 3c. "Preview grip" dry-run

Button in the refit panel: build ONLY the conform field (no solve) and show
it on the source viewport. Backend: `POST /api/sessions/{sid}/refit_field`
with the same RefitPayload → runs the run_refit pipeline up to (and
including) the release/frozen steps, returns `{grip_frac: [...],
stats: {contact_verts, occluded_verts, opposed_verts, seam_grip_verts}}` —
synchronous (it's seconds: signed_offset + occlusion raycasts; no job
queue; add a lock or reuse the job queue with a fast lane if contention
bites). Factor the field-construction part of `run_refit` into a helper
`build_refit_field(...) -> (field, aux)` used by both paths (no behavior
change to run_refit — assert via existing smoke tests).
Frontend: color the source mesh via the existing heat machinery
(`heatValuesFrac` + `heatScaleFrac=1`), legend "free ↔ gripped", auto-clear
on any input change.

---

## 4. Opt-in interaction dataset (training_archive)

Decision recorded in project memory: collect refit/wrap interaction records
for (a) future auto-placement learning, (b) an immediate replay benchmark.
STRICTLY opt-in.

- **Consent:** checkbox in the frontend footer area of the refit/wrap panels
  ("Allow using my uploads & actions to improve the tool"), stored in
  localStorage, sent as `X-Data-Consent: 1` header on job submissions and
  the download endpoints. One sentence added to the Terms page (there is an
  InfoPages.tsx) — placeholder text, the user will lawyer it later.
- **Sink:** `STATE_DIR / "training_archive"`:
  - `blobs/<sha256>.npz` — mesh dedupe store (vertices float32 + faces int32,
    compressed). Written once per unique mesh.
  - `records.jsonl` — append-only: `{ts, tool, session_hash, client_hash,
    source_blob, target_blob, payload: {pre_transform, markers, preserve,
    frozen, preset, overrides}, result_blob, stats, signal: "run"}`.
    `session_hash`/`client_hash` = sha256 truncated — no raw ids, no
    filenames anywhere.
  - Download events append `{..., signal: "download"}` referencing the last
    record's ids (positive label).
- Hook points: end of `_refit_compute` / `_wrap_compute` (only when the
  header was present on submission — carry the flag on the Job), and the
  wrap download endpoint.
- Failure-tolerant: archive errors are logged and swallowed (never fail the
  job). Size guard: skip meshes > 50 MB.
- **Replay harness** (the immediate payoff): `backend/tools/replay.py` — CLI
  that re-runs every archived refit record against the CURRENT code and
  prints per-record deltas (penetration count, mean standoff, edge-stretch
  p95) vs the archived result. Not a smoke test; a manual regression tool.

**Tests:** unit-test the sink (tmp STATE_DIR): record written, blobs deduped,
no consent → nothing written; replay.py runs on a synthetic archive of 2
records.

---

## 5. GLB as the project container (v1)

**Problem:** tools operate on bare OBJ pairs inside a throwaway session. The
target state (user's words): "my GLB *is* the project — I open the full-body
GLB, pick the head part, work on it, and get the same GLB back with the
changes (e.g. new blendshapes) baked in". Also: body masks for multiple
garment layers must ACCUMULATE on the body GLB.

This is the bridge from "session" to "character project" without building a
whole project system: the GLB file carries the state.

### 5a. GLB read + part selection + geometry write-back

- Dependency: `pygltflib` (add to backend requirements). Rationale: lossless
  pass-through — we only append/patch accessors we touch; skins, animations,
  materials, textures we don't understand stay byte-identical. Do NOT use
  trimesh for GLB (lossy for skins/extras).
- New module `backend/dt_core/glbio.py`:
  - `list_parts(path) -> [{index, node_name, mesh_name, primitive, n_verts,
    n_tris, has_skin, has_targets, material}]` — one entry per (mesh,
    primitive) with a triangle POSITION accessor.
  - `extract_part(path, part_id) -> meshlib.Mesh` (+ return an opaque
    `PartRef` with accessor bookkeeping). Apply node world transform? NO —
    v1 requirement: operate in the node's LOCAL space and document it; the
    editor shows the part where its local coords put it. (Applying and
    un-applying world transforms is where round-trips rot; defer.)
  - `replace_part_positions(path_in, part_ref, new_verts, path_out)` —
    topology must be identical (assert vertex count); writes a new POSITION
    accessor (+ recompute accessor min/max), appends to the binary buffer
    with 4-byte alignment, leaves normals: recompute flat from faces into
    the existing NORMAL accessor if present (or drop NORMAL and let the
    engine compute — decide: recompute, engines dislike missing normals).
    Everything else untouched.
  - `add_morph_target(path_in, part_ref, name, delta_verts, path_out)` —
    appends a sparse-free POSITION-delta accessor to `primitive.targets`,
    appends the name to `mesh.extras.targetNames` (create if absent), sets
    `mesh.weights` entry 0.0.
  - `set_extras_mask(path_in, name, indices, path_out)` — writes/updates
    `gltf.extras["chameleonwarp"]["bodyMasks"][name] = [...]` (top-level
    extras; JSON, no buffer changes).
- Upload plumbing: accept `.glb` for source_ref/target_ref/proxy_ref. On
  upload, if GLB: store the original file as the session artifact
  (`source_ref.glb`), run `list_parts`, return the part list in the upload
  response, and DEFER mesh extraction until the client picks a part
  (`POST /api/sessions/{sid}/select_part {role, part_id}` → extracts to the
  internal OBJ path the tools already use, remembers `(role → part_ref)` in
  session state). Single-part GLBs auto-select part 0.
- Frontend: after a GLB upload, a part dropdown appears under the viewport
  (names + vert counts); selection triggers select_part and loads the part
  mesh into the viewport. Everything downstream (markers, refit, wrap) is
  unchanged — it sees a normal mesh.

### 5b. Result write-back modes

For a session whose source (or target) came from a GLB, the result panel
gains download options (backend endpoints; the OBJ download stays):

1. **"Download GLB (geometry replaced)"** — `replace_part_positions` with
   the result verts into the ORIGINAL uploaded GLB → same file, part
   deformed. Valid for refit/wrap/fit results (source topology preserved).
   Skin weights/UVs/materials of the part survive by construction (same
   vertex order).
2. **"Download GLB (+ blendshape)"** — `add_morph_target(name)` with
   `delta = result_verts − part_rest_verts`; name from a text input
   (default: `wrap_<target_name>`). This is the user's head-morphs workflow:
   wrap the head part to N reference heads → N morph targets accumulated in
   the same GLB by repeating the flow.
3. **"Save body mask into GLB"** — for the TARGET GLB: `set_extras_mask`
   with a garment name (text input, default from source filename). Multiple
   refits against the same body GLB accumulate named masks. The mask editor
   (item 2) edits are what get saved.

Accumulation rule: each write-back produces a NEW session artifact
(`project_rev_<n>.glb`) AND the download; the next write-back in the same
session chains from the latest revision, so masks + morphs accumulate.
Show the revision chain in the panel ("project: rev 3 — 2 masks, 1 morph").

### 5c. Out of scope for v1 (state explicitly in code comments)

Skeleton-aware editing (weights transfer, bone edits), texture BAKE, world
transforms, multi-primitive parts spanning materials, Draco-compressed GLBs
(reject with a clear 400: "re-export without compression").

**Tests — new `backend/tests/glbio_smoke.py`:** build a tiny GLB
programmatically with pygltflib (two parts, one skinned with dummy weights,
one morph target, one material reference): extract→replace round-trip keeps
untouched bytes-equal parts (compare accessors count, skin JSON, material
JSON), vertex count mismatch → error; add_morph_target → loadable back,
targetNames grows; set_extras_mask twice with different names → both
present; alignment: buffer length % 4 == 0 after every write.

---

## Explicitly deferred (tracked in docs/ROADMAP.md, do not start here)

- Texture/material carry-through and texture BAKE (needs 5a as base).
- Skeleton preservation through tools; garment weight transfer + weight
  brush + LBS preview (needs rigs in GLB).
- Dressed-ROM QA and pose-union body masks (needs rig + animation).
- XPBD/ContourCraft drape relax.
