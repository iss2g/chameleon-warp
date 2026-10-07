"""
As-rigid-as-possible (ARAP) surface binding for Refit — the "grip then bind"
second pass.

The wrap solver's identity/smoothness terms are built on triangle deformation
GRADIENTS: a smoothly varying rotation field (unfolding a collar flat onto the
skin) costs almost nothing there, which is why a double-wall fold collapses.
ARAP is the missing bending resistance: it penalizes any non-rigid change of
each vertex's one-ring,

    E(v) = sum_i sum_{j in N(i)} w_ij || (v_i - v_j) - R_i (v^_i - v^_j) ||^2

with the per-vertex rotations R_i free. Flattening a fold necessarily deforms
the one-rings AT THE CREASE non-rigidly, so the fold pushes back — while a
rigid ride-along (the outer wall following the inner wall) is free. That is
exactly Blender's Surface Deform behaviour the outer wall needs.

Usage in refit: HANDLES = the gripped vertices (high conform weight + marker
verts), fixed at their solver-fitted positions; FREE = everything else (outer
walls, hanging cloth), reconstructed from the PLACED rest shape v^.

Sorkine & Alexa, "As-Rigid-As-Possible Surface Modeling" (SGP 2007), classic
local/global alternation:
  local  — best rotation per vertex via SVD of the one-ring covariance;
  global — one sparse Laplacian solve per coordinate (factorized once).

Pure numpy/scipy over (vertices, faces) — no solver stack, unit-tests
headlessly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _edge_weights(verts: np.ndarray, faces: np.ndarray,
                  min_w: float = 1e-3) -> sparse.csr_matrix:
    """Symmetric cotangent edge weights, clamped to `min_w` so garment meshes
    with sliver triangles (negative / exploding cotangents) stay stable."""
    f = faces[:, :3]
    n = len(verts)
    rows, cols, vals = [], [], []
    for c in range(3):
        i = f[:, (c + 1) % 3]
        j = f[:, (c + 2) % 3]
        k = f[:, c]                       # vertex opposite edge (i, j)
        u = verts[i] - verts[k]
        v = verts[j] - verts[k]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        cot = np.einsum('ij,ij->i', u, v) / np.where(cross < 1e-12, 1.0, cross)
        w = 0.5 * cot
        rows.append(i); cols.append(j); vals.append(w)
        rows.append(j); cols.append(i); vals.append(w)
    W = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n)).tocsr()
    W.data = np.maximum(W.data, min_w)
    return W


def arap_bind(rest_verts: np.ndarray, faces: np.ndarray,
              handle_idx: np.ndarray, handle_pos: np.ndarray,
              init_verts: Optional[np.ndarray] = None,
              iterations: int = 6,
              rigid_verts: Optional[np.ndarray] = None,
              rigid_scale: float = 20.0) -> Tuple[np.ndarray, dict]:
    """Deform `rest_verts` so that `handle_idx` land exactly on `handle_pos`
    and every other vertex moves as-rigidly-as-possible with its one-ring.

    init_verts warm-starts the local step (pass the wrap solver's output);
    None starts from the rest shape. Handles are hard (eliminated from the
    system). Returns (new verts, stats).

    rigid_verts (refit "frozen paint"): edges whose BOTH endpoints are in this
    set get their weight multiplied by `rigid_scale`, so the painted patch
    behaves like one near-rigid piece — its internal shape survives verbatim
    while its border edges (normal weight) let it hinge and ride the handles.
    """
    n = len(rest_verts)
    handle_idx = np.asarray(handle_idx, dtype=np.int64).ravel()
    assert handle_pos.shape == (len(handle_idx), 3)
    free = np.setdiff1d(np.arange(n), handle_idx)
    out = rest_verts.astype(float).copy()
    out[handle_idx] = handle_pos
    if len(free) == 0 or len(handle_idx) == 0:
        return out, {"free": int(len(free)), "handles": int(len(handle_idx)),
                     "iterations": 0}

    W = _edge_weights(rest_verts, faces)
    if rigid_verts is not None and len(rigid_verts) and rigid_scale > 1.0:
        rigid = np.zeros(n, dtype=bool)
        rigid[np.asarray(rigid_verts, dtype=np.int64)] = True
        Wc = W.tocoo()
        both = rigid[Wc.row] & rigid[Wc.col]
        Wc.data[both] *= float(rigid_scale)
        W = Wc.tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sparse.diags(deg) - W                       # weighted graph Laplacian
    L_ff = L[free][:, free].tocsc()
    L_fh = L[free][:, handle_idx].tocsr()
    lu = splu(L_ff)

    v = (init_verts.astype(float).copy() if init_verts is not None
         else rest_verts.astype(float).copy())
    v[handle_idx] = handle_pos

    W_csr = W.tocsr()
    indptr, indices, wdata = W_csr.indptr, W_csr.indices, W_csr.data
    # Flat neighbor arrays for the vectorized covariance accumulation.
    src_of_edge = np.repeat(np.arange(n), np.diff(indptr))
    rest_e = rest_verts[src_of_edge] - rest_verts[indices]   # v^_i - v^_j

    for _ in range(iterations):
        # ---- local: best rotation per vertex (SVD of covariance) ----
        cur_e = v[src_of_edge] - v[indices]
        contrib = wdata[:, None, None] * (rest_e[:, :, None] * cur_e[:, None, :])
        S = np.zeros((n, 3, 3))
        np.add.at(S, src_of_edge, contrib)
        U, _sig, Vt = np.linalg.svd(S)
        R = np.matmul(Vt.transpose(0, 2, 1), U.transpose(0, 2, 1))
        neg = np.linalg.det(R) < 0
        if neg.any():
            Vt_fix = Vt[neg].copy()
            Vt_fix[:, 2, :] *= -1.0
            R[neg] = np.matmul(Vt_fix.transpose(0, 2, 1), U[neg].transpose(0, 2, 1))

        # ---- global: L v = b with b_i = sum_j w_ij/2 (R_i + R_j)(v^_i - v^_j)
        Rsum = R[src_of_edge] + R[indices]
        rhs_e = 0.5 * wdata[:, None] * np.einsum('eab,eb->ea', Rsum, rest_e)
        b = np.zeros((n, 3))
        np.add.at(b, src_of_edge, rhs_e)
        b_f = b[free] - L_fh @ handle_pos
        v[free] = lu.solve(b_f)

    out[free] = v[free]
    return out, {"free": int(len(free)), "handles": int(len(handle_idx)),
                 "iterations": int(iterations)}
