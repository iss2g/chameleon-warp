# Tools guide

Every tool follows the same idea: a **source** mesh in the left viewport, a
**target** mesh in the right one, and **marker pairs** that tell the solver
which part of the source corresponds to which part of the target. The app
walks you through each tool as a wizard; the stepper at the top is clickable,
so you can jump back to any step without losing work.

- [The basics](#the-basics) — viewports, markers, sessions
- [Wrap](#wrap)
- [Wrap Region](#wrap-region)
- [DT Transfer](#dt-transfer)
- [Refit](#refit-beta) *(beta)*
- [Fit](#fit-beta) *(beta)*
- [Which tool do I need?](#which-tool-do-i-need)

---

## The basics

### Loading meshes

Drag an `.obj` (or `.fbx`, if [Blender is available](INSTALL.md#fbx-support-blender))
onto a viewport, or use the upload buttons.

- Meshes are triangulated internally, but **quads, n-gons and UVs of the
  original file are preserved** in the results wherever the output has the
  same topology as an input.
- Uploading an **FBX as the source** extracts its shape keys: the basis
  becomes the source reference and every shape key becomes a pose (used by
  DT Transfer).
- Default caps: source ≤ 100k triangles, target ≤ 1.5M (≤ 100k for DT
  Transfer). See [CONFIGURATION.md](CONFIGURATION.md) to change them.

### Navigating

Orbit with the left mouse button, pan with the right, zoom with the wheel.
**⊡ Frame** re-centers the view.

### Markers

1. Click a vertex on the **left** mesh — it lights up as a pending pick.
2. Click the matching spot on the **right** mesh — the two become one
   colored pair.

- **Right-click** a marker in the viewport to delete its pair; **Ctrl+Z /
  Ctrl+Y** undo and redo.
- **Symmetric markers** (with a mirror axis X/Y/Z) places every pair a second
  time on the mirrored side — half the clicking for symmetric characters.
- **Export / Import** saves the marker list as JSON
  (`{"version": 1, "markers": [{"source": 12, "target": 345}, …], …}`) so you
  can reuse a set on the next mesh with the same topology.

Good markers sit on features you can find unambiguously on **both** meshes
(eye corners, nose tip, mouth corners, ear roots, chin, fingertips) and are
**spread over the whole mesh**. A cluster in one area leaves the rest
unguided.

If source and target have **identical topology**, Wrap, Wrap Region and DT
Transfer need no markers at all — vertex *i* already corresponds to vertex *i*.

### Sessions

Your work (uploads, markers, settings, results) lives in a session on the
backend and survives page reloads and backend restarts. Idle sessions are
deleted after 7 days (the header shows the countdown). **New session** clears
everything and returns to the tool picker.

Heavy jobs run in the background with a progress bar; you can close the tab
and come back later. One heavy job runs at a time; others wait in a queue.

### Results

**Preview results** shows the output next to the input. After a wrap, the
**heatmap** colors each vertex by its distance to the target surface (blue →
red, scaled to 2 % of the target's bounding-box diagonal), and the panel
reports mean / p95 / max distance.

---

## Wrap

**Conforms the whole source onto the target's shape while keeping the
source's vertex count, vertex order, topology and UVs.** The output drops
straight into a rig as a shape key.

Use it to wrap a clean basemesh onto a scan (retopology), or to make a
character variant from your base mesh.

**Markers:** 10–40 well-spread pairs on stable features. Fewer are fine when
the meshes are already similar.

**Settings**

| Setting | Effect |
|---|---|
| Iterations | Number of solver rounds (default 8). More = closer fit, slower. |
| Smoothness | Resistance to local distortion. Raise it if the result has spikes. |
| Align target to source | Brings the target into the source's scale/rotation/position using the markers (similarity transform) before wrapping. Leave on unless both are already aligned. |
| Smooth result | Taubin smoothing passes after the solve, to remove fine wobble; with projection on it alternates smooth → re-project so the result stays on the surface. |

**Advanced** (presets *Loose / Normal / Tight*):

| Setting | Effect |
|---|---|
| Match angle | Max angle between source and target normals for a closest-point match (default 60°). Prevents snapping through the mesh onto the far side or an inner shell. |
| Match distance | Floor of the adaptive distance cut (fraction of the target's size). Matches farther than `max(floor, 3 × median)` are dropped, so areas the target doesn't cover ride along on smoothness instead of stretching toward a wrong patch. |
| Final projection / distance | Snaps the result onto the target surface at the end, up to this distance. |
| Soft markers | In the late, strong phase of the solve, marker pins relax into weak constraints, so a slightly misplaced marker (e.g. wrong side of an ear fold) gets pulled onto the right surface instead of denting it. |

Robustness built in: matches landing on open borders (eye sockets, mouth
bags, mesh cuts) are rejected, and triangles that flip during an iteration
are damped back.

**Output:** one `.obj`, `<source>_wrapped_to_<target>.obj`.

---

## Wrap Region

**Like Wrap, but only inside a loop you draw on the source.** Everything
outside stays exactly where it was; a feathered seam blends the two.

Use it to borrow a detailed ear onto a plainer head, or reshape a nose or
cheek without disturbing the rest.

**Outline the patch**

1. Press **✎ Outline** and click points around the area on the left mesh.
   The line follows the surface between clicks; 6–12 points are usually
   enough.
2. Close the ring by clicking the first (magenta) point again, or press
   **Close loop**.
3. The smaller enclosed side becomes editable — **⇄ Invert** if the wrong
   side lit up.
4. **Preview region** colors it: green = edited, grey = frozen, in between =
   the seam.

**Markers** are optional: a few pairs inside the patch to steer where it
lands (none if the topology is identical).

| Setting | Effect |
|---|---|
| Feather | Width of the blend band, as a share of the mesh size. Wider = softer transition, less freedom near the edge. |
| Seam sharpness | 1 = gentle linear ramp; up to 5–6 concentrates the blend at the boundary and frees the interior sooner. |

**Output:** the full source mesh with the patch replaced, as `.obj`.

---

## DT Transfer

**Classic deformation transfer:** reproduces the deformations of the source's
poses (expressions, blendshapes) on the target. The results have the
**target's** topology.

Use it when you have blendshapes for one character and want the same set on
another.

1. **Meshes** — source and target, both in their **neutral** pose.
2. **Poses** — one or many `.obj` files of the source in different poses
   (same vertex and face count as the source reference). If you uploaded the
   source as FBX, its shape keys are already here. File names carry through
   to the results.
3. **Markers** — at least 3; use 15–40 when proportions differ. Put them
   where deformation matters (mouth corners, eyelids, brows, jaw, nostrils)
   plus a few on stable parts to anchor the rest. Not needed for identical
   topology.
4. **Run** — every pose is transferred in one job.

| Setting | Effect |
|---|---|
| Iterations | Rounds of the correspondence solve. 8 is a good default. |
| Smoothness | Resists sharp local distortion in the correspondence. |

**Output:** each pose as `<pose>_deformed.obj`, all of them as a `.zip`, or —
with Blender — one `.fbx` whose basis is the target and which carries one
shape key per pose.

Repeated runs are fast: the correspondence is cached per session, so adding
poses or re-running with the same markers only transfers what is new.

---

## Refit *(beta)*

**Body-aware garment fitting.** The source is a garment, armor piece or
accessory; the target is the **body** it goes on. Your markers anchor the
interface, and around them the solver builds a contact field, decides which
garment walls are driven by the body and which only follow, binds the
garment so it keeps its own shape and relief (ARAP), and resolves collisions.

1. **Place** — the body is ghosted so you can see through it. Use **⤧ Move /
   ⟳ Rotate / ⤢ Scale** and drag the gizmo. How close you leave it matters:
   pressed against the body reads as tight, standing off reads as loose.
   **Snap placement to body** lets the solver nudge the placement onto the
   contact standoff.
2. **Refine** — tie the garment to the body:
   - **Marker pairs** along the interface (collar, hem, cuffs, the seam an
     accessory attaches at) — 6–20 well-spread pairs.
   - **📌 Poke** — when the garment already sits close to the body, a needle
     through it derives the correspondence: it grips where the inner layer
     meets the body and marks any outer wall it pierces as *preserve*, so a
     double-wall collar keeps its standoff.
   - **◉ Preview grip** — a dry run that colors the garment without solving:
     red = gripped onto the body, blue = left free.
   - **❄ Freeze paint** — brush over parts that must not deform at all (a
     buckle, a rigid collar). Left-drag paints, right-drag erases.
   - **Layers: auto** — a score-based classifier that decides which walls are
     driven (the inner wall the body sees first) and which follow. It doesn't
     trust garment normals, so it copes with flipped game-asset normals.
     **→ frozen paint** pre-fills the brush with the detected followers.
3. **Refit** — pick a preset:

| Preset | Behavior |
|---|---|
| Accessory | Grips at its seam, the bulk keeps its shape (beard, fur trim, pouch). |
| Cloth | Drapes loosely; the garment is pushed out where the body pokes through. |
| Armor | Very stiff, keeps its form; the body underneath is hidden instead of inflating the shell. |
| Skin-tight | Conforms everywhere, like a second skin. |

**Advanced** overrides the preset: contact distances (closer than *contact
tight* grips fully, the grip fades out by *contact free*), cloth thickness,
stiffness, smoothness and the collision mode (*none*, *push out* the garment,
or *hide body* under it).

**Body mask.** Presets that hide the body produce a mask of covered body
vertices. In the result preview you can show it on the body, **✎ Edit** it
with a brush, **Dilate** it by N rings, and download it as a JSON sidecar to
hide those vertices in your engine.

**Wear via proxy.** Fit a garment once to a base character, then wear it on
any body: Wrap the proxy basemesh onto the new body first, upload the same
basemesh as the *proxy*, and run — the garment rides the wrap, followed by a
short cleanup fit.

**Output:** the fitted garment as `.obj`; the preview can also show it on the
body.

---

## Fit *(beta)*

**Purely geometric point attachment.** Pins the source's marker vertices
exactly onto the target's marker points and deforms the rest minimally. No
surface snapping, contact, collision or layers — the markers are the whole
mechanism.

Use it for a rigid-ish piece that attaches at a few known points: glasses on
a face, a badge, horns, a prop.

**Markers:** at least 3, and at least one (ideally 3+) on **every**
disconnected piece — a part without a marker does not move. Place them around
the ring where the accessory touches the target.

| Setting | Effect |
|---|---|
| Rigidity | High (toward 10⁰) = keeps its shape, only the area near the anchors bends. Low (toward 10⁻⁴) = stretches more freely to reach the anchors. Start high. |

**Output:** the attached accessory as `.obj`.

---

## Which tool do I need?

| I want to… | Tool |
|---|---|
| Give my basemesh the shape of a scan or another head, keep my topology | Wrap |
| Change only one part (ear, nose) and freeze the rest | Wrap Region |
| Copy expressions/blendshapes from one character to another | DT Transfer |
| Retarget blendshapes to a wrapped variant of the same base | Wrap, then DT Transfer (no markers needed) |
| Dress a character in clothing or armor | Refit |
| Re-use one outfit across many bodies | Refit → *Wear via proxy* |
| Stick an accessory onto exact points | Fit |
