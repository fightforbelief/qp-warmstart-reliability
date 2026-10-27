"""Canonical convex QP instances and validators.

OSQP form:
    minimize  1/2 x' P x + q' x
    subject to l <= A x <= u
with P symmetric positive semidefinite. Infinite bounds are allowed.
A small numeric residual is not a formal optimality certificate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as spa


class QPValidationError(ValueError):
    """Raised when a QP is structurally invalid. No solver call is made."""


def _as_csc64(mat: spa.spmatrix, shape: tuple[int, int], name: str) -> spa.csc_matrix:
    if mat.shape != shape:
        raise QPValidationError(f"{name} shape {mat.shape} != {shape}")
    if not np.issubdtype(mat.dtype, np.floating):
        mat = mat.astype(np.float64)
    csc = mat.tocsc().astype(np.float64)
    if not np.all(np.isfinite(csc.data)):
        raise QPValidationError(f"{name} contains NaN or infinite numeric entries")
    return csc


def symmetrize(P: spa.spmatrix) -> spa.csc_matrix:
    """Return the full symmetric matrix (P + P.T)/2 in CSC."""
    full = ((P + P.T) * 0.5).tocsc().astype(np.float64)
    full.sum_duplicates()
    return full


def upper_triangular(P_full: spa.spmatrix) -> spa.csc_matrix:
    """Upper-triangular CSC view used by OSQP's 1/2 x'Px convention."""
    tri = spa.triu(P_full, format="csc").astype(np.float64)
    tri.sum_duplicates()
    return tri


def max_asymmetry(P: spa.spmatrix) -> float:
    diff = (P - P.T).tocsc()
    if diff.nnz == 0:
        return 0.0
    return float(np.max(np.abs(diff.data))) if diff.data.size else 0.0


def min_eigenvalue(P_full: spa.csc_matrix) -> float:
    """Smallest eigenvalue. Dense for moderate n; sparse estimate otherwise."""
    n = P_full.shape[0]
    if n == 0:
        return 0.0
    if n <= 400:
        dense = P_full.toarray()
        dense = 0.5 * (dense + dense.T)
        return float(np.linalg.eigvalsh(dense)[0])
    # A few Rayleigh quotients. This is a screen for large sparse P, not a certificate.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(12):
        x = rng.normal(size=n)
        x /= np.linalg.norm(x) + 1e-30
        worst = min(worst, float(x @ (P_full @ x)))
    return worst


@dataclass
class QPInstance:
    instance_id: str
    group_id: str
    family: str
    source_repo_sha: str
    source_file: str
    source_seed: int | None
    step: int
    P_full: spa.csc_matrix
    q: np.ndarray
    A: spa.csc_matrix
    l: np.ndarray
    u: np.ndarray
    problem_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q = np.ascontiguousarray(self.q, dtype=np.float64).reshape(-1)
        self.l = np.ascontiguousarray(self.l, dtype=np.float64).reshape(-1)
        self.u = np.ascontiguousarray(self.u, dtype=np.float64).reshape(-1)
        n = self.q.shape[0]
        m = self.l.shape[0]
        if self.u.shape[0] != m:
            raise QPValidationError("l and u lengths differ")
        self.P_full = _as_csc64(self.P_full, (n, n), "P")
        self.A = _as_csc64(self.A, (m, n), "A")
        # Sum duplicate CSC entries. Equivalent A storage then shares qp_content_hash.
        # An already canonical CSC matrix keeps the same index and data bytes.
        self.A.sum_duplicates()
        if not np.all(np.isfinite(self.q)):
            raise QPValidationError("q contains NaN or inf")
        if np.any(np.isnan(self.l)) or np.any(np.isnan(self.u)):
            raise QPValidationError("l or u contains NaN")
        finite = np.isfinite(self.l) & np.isfinite(self.u)
        if np.any(self.l[finite] > self.u[finite] + 1e-12):
            raise QPValidationError("inconsistent bounds l > u on a finite pair")
        # Also reject +inf lower or -inf upper, which cannot be satisfied.
        if np.any(np.isposinf(self.l)) or np.any(np.isneginf(self.u)):
            raise QPValidationError("bound orientation is impossible (+inf lower or -inf upper)")
        asym = max_asymmetry(self.P_full)
        tol = float(self.metadata.get("symmetry_tol", 1e-8))
        if asym > tol * max(1.0, float(np.max(np.abs(self.P_full.data))) if self.P_full.nnz else 1.0):
            raise QPValidationError(f"P is not symmetric within tolerance (max |P-P.T|={asym})")
        self.P_full = symmetrize(self.P_full)
        psd_tol = float(self.metadata.get("psd_tol", 1e-7))
        if self.metadata.get("check_psd", True):
            lam_min = min_eigenvalue(self.P_full)
            if lam_min < -psd_tol * max(1.0, abs(float(np.max(np.abs(self.P_full.data))) if self.P_full.nnz else 1.0)):
                raise QPValidationError(f"P is not PSD within tolerance (lambda_min={lam_min})")
            self.metadata = dict(self.metadata)
            self.metadata["lambda_min_est"] = lam_min
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("n", n)
        self.metadata.setdefault("m", m)
        self.metadata["nnzP"] = int(self.P_triu.nnz)
        self.metadata["nnzA"] = int(self.A.nnz)
        self.metadata["asymmetry"] = asym

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def m(self) -> int:
        return int(self.l.shape[0])

    @property
    def P_triu(self) -> spa.csc_matrix:
        return upper_triangular(self.P_full)

    def sparsity_pattern(self) -> tuple[bytes, bytes]:
        p = self.P_triu
        return (
            np.ascontiguousarray(p.indices).tobytes() + np.ascontiguousarray(p.indptr).tobytes(),
            np.ascontiguousarray(self.A.indices).tobytes() + np.ascontiguousarray(self.A.indptr).tobytes(),
        )


def _update_hash(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())


def qp_content_hash(instance: QPInstance) -> str:
    """Stable hash of the numeric QP. Identical across solver policies."""
    h = hashlib.sha256()
    p = instance.P_triu
    _update_hash(h, p.data)
    _update_hash(h, p.indices)
    _update_hash(h, p.indptr)
    _update_hash(h, np.array(p.shape))
    _update_hash(h, instance.q)
    _update_hash(h, instance.A.data)
    _update_hash(h, instance.A.indices)
    _update_hash(h, instance.A.indptr)
    _update_hash(h, np.array(instance.A.shape))
    _update_hash(h, instance.l)
    _update_hash(h, instance.u)
    h.update(instance.instance_id.encode())
    h.update(instance.group_id.encode())
    return h.hexdigest()


def objective_value(instance: QPInstance, x: np.ndarray) -> float:
    x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != instance.n:
        raise QPValidationError(f"x has length {x.shape[0]} but n={instance.n}")
    quad = float(x @ (instance.P_full @ x))
    return 0.5 * quad + float(instance.q @ x)
