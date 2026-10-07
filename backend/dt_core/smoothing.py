"""
Taubin (λ/μ) mesh smoothing — used to denoise wrap results.

The closest-point step snaps every editable vertex independently onto the
nearest target point, which injects high-frequency wobble (visible as ragged
creases across the patch and at the seam). A few Taubin passes remove that
wobble while preserving the overall shape and, unlike plain Laplacian
smoothing, WITHOUT shrinking the volume (the μ step inflates back what the
λ step pulled in).

Only the `movable` vertices are updated; everything else (the frozen region
and the boundary ring in a regional wrap) stays exactly put and acts as a
Dirichlet anchor, so the smoothing eases the patch into the fixed surround
and the seam comes out smooth too.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from . import meshlib


def _averaging_matrix(faces: np.ndarray, n: int) -> sparse.csr_matrix:
    """Row-normalized neighbor-averaging operator W (umbrella weights)."""
    f = faces[:, :3]
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                       # collapse duplicate edges to 1
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sparse.diags(1.0 / deg)
    return (Dinv @ A).tocsr()


def taubin_smooth(
    mesh: meshlib.Mesh,
    movable: np.ndarray,
    iterations: int,
    lam: float = 0.5,
    mu: float = -0.53,
) -> meshlib.Mesh:
    """Return a copy of `mesh` with `movable` vertices Taubin-smoothed.

    `movable` is a bool mask over vertices (True = may move). `iterations`
    is the number of λ/μ passes (each pass = one smoothing + one inflating
    step). 5–10 is usually plenty.
    """
    if iterations <= 0 or not movable.any():
        return mesh
    V = mesh.vertices.astype(float).copy()
    W = _averaging_matrix(mesh.faces, len(V))
    mov = movable.astype(bool)
    for _ in range(iterations):
        for step in (lam, mu):
            lap = W @ V - V               # umbrella Laplacian
            V[mov] += step * lap[mov]
    return meshlib.Mesh(vertices=V, faces=mesh.faces)
