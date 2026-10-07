# Design notes

Implementation plans written before (and kept after) the larger features
landed. They explain *why* the code is shaped the way it is — useful when
changing the refit pipeline. They describe the plan at the time; the code is
the source of truth where they differ.

| Plan | Covers | Code |
|---|---|---|
| [refit-abcd.md](refit-abcd.md) | Refit upgrades: rigidity-consistency match filter (A), growth continuation (B), placement auto-polish (C), proxy wardrobe (D) | `dt_core/refit.py`, `dt_core/correspondence.py`, `dt_core/wardrobe.py` |
| [core-finish.md](core-finish.md) | Combined preview, body-mask lifecycle, refit stats/warnings, grip preview, persisted refit state | `frontend/src/components/PreviewPanel.tsx`, `backend/main.py` |
| [layer-autoseg.md](layer-autoseg.md) | Automatic driven/follower layer classification (L1–L6) | `dt_core/layers.py`, `dt_core/surface.py` |

The plans mention a local `testfit/` folder: a real garment/body pair used for
performance and behavior checks. It is not redistributable and therefore not
in the repository; any garment (~10k vertices) and body (~30k vertices) pair
works the same way.
