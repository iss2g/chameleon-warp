"""
Marker-based rigid+scale alignment (Umeyama similarity).

Two meshes uploaded from different sources rarely share a coordinate frame:
one head might span ~20 units around z=6, another ~0.4 units around the
origin. Deformation transfer's closest-point step assumes the meshes overlap
in space, so on mismatched inputs it collapses the result toward the tiny /
offset target.

The user's marker pairs are exact point correspondences, which is exactly
what we need to recover the similarity transform (uniform scale, rotation,
translation) that best maps one mesh's marker points onto the other's. We
use it to bring the TARGET into the SOURCE's frame before wrapping, so the
output stays in source space and closest-point behaves.

Reference: Umeyama, "Least-squares estimation of transformation parameters
between two point patterns", IEEE PAMI 1991.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True
            ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Best similarity transform mapping src -> dst in the least-squares sense.

    Returns (scale, R, t) such that  dst ≈ scale * (src @ R.T) + t.

    src, dst: (N, 3) corresponding point sets (N >= 3, not all collinear).
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n                      # (3,3)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0                       # reflection fix

    R = U @ S @ Vt
    if with_scale and var_s > 1e-12:
        scale = float((D * np.diag(S)).sum() / var_s)
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def apply_similarity(verts: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray
                     ) -> np.ndarray:
    """Apply  scale * (verts @ R.T) + t  to an (N,3) array."""
    return scale * (verts @ R.T) + t
