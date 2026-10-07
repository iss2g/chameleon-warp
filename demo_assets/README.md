# Demo assets

Redistributable sample meshes for trying the tools — see **Licensing** below.
Load them into the app by dragging the `.obj` files onto the viewports.

## What's here

| File | Verts | For |
|---|---|---|
| `ict-facekit/skin/generic_neutral_mesh.obj` | 14 062 | DT Transfer **source** (neutral head) |
| `ict-facekit/skin/{jawOpen,mouthSmile_L,mouthSmile_R,eyeBlink_L,eyeBlink_R}.obj` | 14 062 | DT Transfer **poses** (the blendshapes to transfer) |
| `makehuman/head.obj` | 4 244 | DT Transfer / Wrap **target** — a different face on different topology |
| `makehuman/body.obj` | 13 380 | Refit **target** (neutral body) |
| `makehuman/male_casualsuit01.obj` | 8 426 | Refit **source** (garment authored for a normal body) |

All six ICT files share one topology, which is what the pose upload requires.

Try it:

- **DT Transfer** — source `generic_neutral_mesh.obj`, target `makehuman/head.obj`,
  poses = the five ICT expressions. Place ~20 marker pairs on the face.
- **Wrap** — the same two heads: the ICT head takes the MakeHuman head's shape
  while keeping its own topology.
- **Refit** — source `male_casualsuit01.obj`, target `makehuman/body.obj`.

## Regenerating

The upstream originals are not committed (they are large and re-downloadable);
only the trimmed files we ship are. To rebuild everything:

```bash
python demo_assets/prepare.py all      # fetch + trim, ~19 MB of downloads
```

What the trimming does, and why:

- `ict` — keeps only the `M_Face` / `M_BackHead` material runs. The full ICT
  model also carries teeth, tongue, gums, eyeballs, lacrimal fluid and lashes,
  which float around unshaded in the viewport and double the file size. The
  same filter runs over every expression so they stay vertex-compatible.
- `body` — keeps only the `body` group of the MakeHuman base mesh, dropping the
  46 helper groups (hair / tights / teeth / eyelash proxies) that ship inside
  the same OBJ.
- `head` — cuts `body.obj` at y ≥ 6.0, which lands in the neck.

## Licensing

**ICT-FaceKit** (`ict-facekit/`) — © USC Institute for Creative Technologies,
**MIT License**, full text in `ict-facekit/LICENSE-ICT-FaceKit.txt`, which must
travel with any copy or substantial portion. Commercial use and redistribution
are permitted. Source: <https://github.com/USC-ICT/ICT-FaceKit>.

> Only the *light* model in that repository is MIT-licensed. The full ICT face
> model is under a separate USC license — do not pull files from it.

**MakeHuman** (`makehuman/`) — **CC0 1.0** (public domain), text in
`makehuman/LICENSE-MakeHuman-assets.txt`. No attribution required. The
MakeHuman project licenses its code under AGPL but all bundled *assets* — base
mesh, targets, proxies, clothes — under CC0, and models exported from the app
inherit CC0. Sources:
<https://github.com/makehumancommunity/makehuman> (base mesh),
<https://github.com/makehumancommunity/makehuman-assets> (clothes).

Both files here are modified: see "Regenerating" for exactly how.

If you add an asset under an attribution license (e.g. CC-BY), credit it here,
in `NOTICE` and on the in-app **Credits** page (`frontend/src/InfoPages.tsx`).
