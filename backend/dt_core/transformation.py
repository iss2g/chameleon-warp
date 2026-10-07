"""
Deformation transfer step.

Adapted for API use:
  - imports cleaned, no __main__
  - **factor reuse**: the sparse system matrix A doesn't depend on the pose,
    only b does. We factor A.T @ A ONCE in __init__ via dt_core.solver
    (Pardiso if available, else SuperLU) and reuse it for every pose,
    turning per-pose work from "factor + solve" into just "solve".
"""
import time
from typing import Optional

import numpy as np
import tqdm
from scipy import sparse

from . import meshlib
from .correspondence import compute_adjacent_by_edges, TransformMatrix
from .solver import PardisoFactor, backend_name


class Transformation:
    def __init__(
            self,
            source: meshlib.Mesh,
            target: meshlib.Mesh,
            mapping: np.ndarray,
            smoothness: float = 1.0,
    ):
        self.source = source.to_third_dimension(copy=False)
        self.target = target.to_third_dimension(copy=False)
        self.mapping = mapping
        self.Wm = 1.0
        self.Ws = max(1e-8, float(smoothness))

        # Centroid of the target reference. Used in __call__ to snap every
        # solved pose to the same world centroid as the basis — see the long
        # comment there for why this is needed.
        self._target_centroid = self.target.vertices.mean(axis=0)

        t0 = time.perf_counter()
        self._Am = self._compute_mapping_matrix(self.target, mapping)
        self._As, self._Bs = self._compute_missing_smoothness(self.target, mapping)

        # Stack once (Am and As don't depend on the pose), then factor A.T@A once.
        Astack = [self._Am * self.Wm, self._As * self.Ws]
        A: sparse.spmatrix = sparse.vstack(Astack, format="csc")
        A.eliminate_zeros()
        A = A.tocsc()
        self._A_csc = A
        self._At = A.T

        t_f = time.perf_counter()
        self._factor: Optional[PardisoFactor] = PardisoFactor((A.T @ A).tocsc())
        print(f"[transformation] solver={backend_name()}  init total={time.perf_counter() - t0:.2f}s  "
              f"factor={time.perf_counter() - t_f:.2f}s  "
              f"A shape={A.shape}  Am nnz={self._Am.nnz}  As nnz={self._As.nnz}")

    @classmethod
    def _compute_mapping_matrix(cls, target: meshlib.Mesh, mapping: np.ndarray):
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)
        Am = TransformMatrix.construct(
            target.faces[mapping[:, 1]], inv_target_span[mapping[:, 1]],
            len(target.vertices), desc="Building Mapping",
        )
        return Am.tocsc()

    @classmethod
    def _compute_missing_smoothness(cls, target: meshlib.Mesh, mapping: np.ndarray):
        adjacent = compute_adjacent_by_edges(target)
        target = target.to_fourth_dimension(copy=False)
        inv_target_span = np.linalg.inv(target.span)
        missing = np.setdiff1d(np.arange(len(target.faces)), np.unique(mapping[:, 1]))
        count_adjacent = sum(len(adjacent[m]) for m in missing)

        if count_adjacent == 0:
            return sparse.csc_matrix((0, len(target.vertices)), dtype=float), np.zeros((0, 3))

        size = len(target.vertices)

        def construct(f, inv, index):
            a = TransformMatrix.expand(f, inv, size).tocsc()
            for adj in adjacent[index]:
                yield a, TransformMatrix.expand(target.faces[adj], inv_target_span[adj], size).tocsc()

        lhs, rhs = zip(*(adjacents for index, m in
                         enumerate(tqdm.tqdm(missing, total=len(missing),
                                             desc="Fixing Missing Mapping with Smoothness"))
                         for adjacents in construct(target.faces[m], inv_target_span[m], index)))

        As = (sparse.vstack(lhs) - sparse.vstack(rhs)).tocsc()
        Bs = np.zeros((As.shape[0], 3))
        return As, Bs

    def __call__(self, pose: meshlib.Mesh) -> meshlib.Mesh:
        t0 = time.perf_counter()
        s = (pose.span @ np.linalg.inv(self.source.span)).transpose(0, 2, 1)
        Bm = np.concatenate(s[self.mapping[:, 0]])

        Bstack = [Bm * self.Wm, self._Bs * self.Ws]
        b = np.concatenate(Bstack)

        assert self._A_csc.shape[0] == b.shape[0]
        assert b.shape[1] == 3

        assert self._factor is not None
        x = self._factor.solve(self._At @ b)
        verts = x[:len(self.target.vertices)]

        # Pin centroid to the target reference.
        #
        # The system matrix A only encodes per-triangle deformation gradients
        # (mapping rows Am, smoothness rows As). Gradients are invariant under
        # a global translation: A @ (x + t·1) == A @ x for any constant t.
        # So the all-ones vector is in ker(A), making A.T@A singular and the
        # least-squares solution unique only up to an additive constant per
        # column. The solver picks SOME element of that affine subspace —
        # different for every pose — which leaks into Blender shape keys as
        # a per-key world translation:
        #
        #   blender_result = basis + Σ value_i · (key_i − basis)
        #                  = basis + Σ value_i · ((ideal_i + t_i) − basis)
        #                                        ^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                        the t_i drifts the model.
        #
        # We kill that DoF by translating every result so its centroid
        # matches the target reference's centroid (== basis centroid). The
        # gradient field — i.e. the actual deformation — is unchanged.
        shift = self._target_centroid - verts.mean(axis=0)
        verts = verts + shift

        out = meshlib.Mesh(
            vertices=verts,
            faces=self.target.faces,
        )
        print(f"[transformation] one pose in {time.perf_counter() - t0:.2f}s  centroid shift={np.linalg.norm(shift):.4f}")
        return out
