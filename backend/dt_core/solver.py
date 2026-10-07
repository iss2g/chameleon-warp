"""
Sparse linear solver wrapper.

Prefers Intel MKL Pardiso (pypardiso) for ~2-4x speedup on multi-core boxes,
falls back to scipy SuperLU if pypardiso is not installed.

Two interfaces:
  - spsolve(A, b)            : one-shot solve
  - PardisoFactor(A).solve(b): factor once, solve many (useful for per-pose transfer)
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

# --- Backend detection ---------------------------------------------------------

try:
    import pypardiso  # type: ignore
    _HAS_PARDISO = True
except Exception as _err:  # noqa: BLE001
    _HAS_PARDISO = False
    _PARDISO_ERR = repr(_err)


def backend_name() -> str:
    return "pypardiso (MKL)" if _HAS_PARDISO else "scipy.splu (SuperLU)"


# --- One-shot solve ------------------------------------------------------------

def spsolve(A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b for a square, non-singular sparse A. b can be 1-D or 2-D.
    Uses Pardiso if available, otherwise SuperLU via splu.
    """
    A = A.tocsc()
    if _HAS_PARDISO:
        return pypardiso.spsolve(A, b)
    LU = sparse_linalg.splu(A)
    return LU.solve(b)


# --- Factor-once, solve-many ---------------------------------------------------

class PardisoFactor:
    """
    Factorize a square sparse A once, then solve A x = b for many right-hand sides.
    Falls back to scipy.sparse.linalg.splu if pypardiso is unavailable.
    """

    def __init__(self, A: sparse.spmatrix) -> None:
        A = A.tocsc()
        self._A = A
        if _HAS_PARDISO:
            self._pardiso: Optional["pypardiso.PyPardisoSolver"] = pypardiso.PyPardisoSolver()
            # mtype 11 = real, unsymmetric (safe default; A.T @ A could be SPD
            # but we don't enforce that).
            self._pardiso.factorize(A)
            self._splu: Optional["sparse_linalg.SuperLU"] = None
        else:
            self._splu = sparse_linalg.splu(A)
            self._pardiso = None

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self._pardiso is not None:
            # pypardiso's solve() needs both A and b; internal LU cache is keyed
            # on the matrix, so passing the same A back reuses our factorization.
            return self._pardiso.solve(self._A, b)
        assert self._splu is not None
        return self._splu.solve(b)
