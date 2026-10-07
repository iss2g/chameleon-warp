"""
Manual ground-truth harness for auto layer classification — NOT in the
regression suite (real assets aren't committed).

Loads a garment + body + a session's hand-painted frozen set, runs the auto
classifier, and prints the IoU of (auto layer >= 1) against the painted frozen
set plus per-signal stats. Run it on the user's real painted sessions; target
IoU >= 0.8 before flipping any preset default to layer_mode="auto".

Usage:
  .venv\\Scripts\\python.exe tests\\layer_groundtruth_check.py \\
      GARMENT.obj BODY.obj FROZEN.json [--preset cloth] [--placed PLACED.npy]

FROZEN.json: either a JSON list of vertex indices, or a refit_state dict with a
"frozen" key (that's what put_refit_state persists). PLACED.npy (optional): an
(N,3) array of placed garment vertices (proxy-wardrobe transport / a saved
gizmo pose); without it the garment is used as-authored.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt_core import meshlib, refit                             # noqa: E402


def load_frozen(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("frozen", [])
    return np.asarray(data, dtype=np.int64).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("garment")
    ap.add_argument("body")
    ap.add_argument("frozen")
    ap.add_argument("--preset", default="cloth")
    ap.add_argument("--placed", default=None)
    args = ap.parse_args()

    garment = meshlib.Mesh.load(args.garment)
    body = meshlib.Mesh.load(args.body)
    nv = len(garment.vertices)
    frozen = load_frozen(args.frozen)
    frozen = frozen[(frozen >= 0) & (frozen < nv)]
    placed = np.load(args.placed) if args.placed else None

    rf = refit.build_refit_field(garment, body, args.preset,
                                 placed_verts=placed, layer_mode="auto")
    layer = rf.layer
    if layer is None:
        print("FAIL: auto classification produced no layer array")
        return 1

    auto_follower = layer >= 1
    gt = np.zeros(nv, dtype=bool)
    gt[frozen] = True

    inter = int((auto_follower & gt).sum())
    union = int((auto_follower | gt).sum())
    iou = inter / union if union else 1.0
    # Precision/recall of auto followers against the hand-painted set.
    prec = inter / auto_follower.sum() if auto_follower.any() else 0.0
    rec = inter / gt.sum() if gt.any() else 0.0

    print(f"garment verts     : {nv}")
    print(f"painted frozen    : {int(gt.sum())}")
    print(f"auto followers    : {int(auto_follower.sum())} "
          f"(wall {int((layer == 1).sum())}, component {int((layer == 2).sum())})")
    print(f"IoU(auto>=1, paint): {iou:.3f}   precision {prec:.3f}  recall {rec:.3f}")
    print(f"layer_stats       : {rf.layer_stats}")
    print(f"TARGET IoU >= 0.8  -> {'OK' if iou >= 0.8 else 'below target'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
