"""Original-coordinate numerical checks.

These are floating-point diagnostics. They are not formal KKT certificates
and they are not a proof of a duality gap when P is singular or bounds are
two-sided boxes.

Dual-sign convention, matching OSQP: the stationarity vector is
``P_sym x + q + A.T @ y``. The objective uses the same symmetrized matrix,
``P_sym = (P + P.T) / 2``, so ``f(x) = 0.5 x' P_sym x + q' x``.
The skew part of P does not change that scalar, but it does change ``P x``.
Stationarity therefore always multiplies by ``P_sym``, not by the raw stored P.

``r_p_norm_candidate`` is an exploratory scale. It is not an approved
scientific threshold and it is not OSQP's primal residual ``||Ax - z||``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import scipy.sparse as spa

from .instance import QPInstance, symmetrize

METRIC_IMPLEMENTATION_VERSION = "qpwr-quality-metrics-1"
DUAL_SIGN_CONVENTION = "OSQP: P_sym x + q + A.T @ y"
OBJECTIVE_CONVENTION = "0.5 * x.T @ P_sym @ x + q.T @ x, P_sym = (P + P.T) / 2"
PRIMAL_NORM_CANDIDATE_NOTE = (
    "r_p / (1 + ||(A x)_F||_inf), F = rows with at least one finite bound. "
    "If F is empty the candidate is 0. Not an approved final criterion."
)


class QualityInputError(ValueError):
    """Shape or non-finite candidate data. A large residual is not this error."""


# Status strings that mean the requested tolerance was met.
SOLVED_EXACT = {"solved"}
# Not a clean success. Retained in every latency table.
NOT_CLEAN = {
    "solved inaccurate",
    "maximum iterations reached",
    "run time limit reached",
    "primal infeasible",
    "dual infeasible",
    "primal infeasible inaccurate",
    "dual infeasible inaccurate",
    "non convex",
    "unsolved",
    "error",
}


def normalize_status(status: str | None) -> str:
    if status is None:
        return "error"
    return " ".join(str(status).strip().lower().split())


def _finite_vector(name: str, value: np.ndarray, length: int | None = None) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).reshape(-1)
    if length is not None and arr.shape[0] != length:
        raise QualityInputError(f"{name} has length {arr.shape[0]} but expected {length}")
    if not np.all(np.isfinite(arr)):
        raise QualityInputError(f"{name} contains NaN or infinity")
    return arr


def _bound_vector(name: str, value: np.ndarray, length: int) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64)).reshape(-1)
    if arr.shape[0] != length:
        raise QualityInputError(f"{name} has length {arr.shape[0]} but expected {length}")
    if np.any(np.isnan(arr)):
        raise QualityInputError(f"{name} contains NaN")
    return arr


def _as_linear(name: str, mat: Any, shape: tuple[int, int]) -> spa.csc_matrix | np.ndarray:
    if spa.issparse(mat):
        if mat.shape != shape:
            raise QualityInputError(f"{name} shape {mat.shape} != {shape}")
        out = mat.tocsc().astype(np.float64)
        if not np.all(np.isfinite(out.data)):
            raise QualityInputError(f"{name} contains NaN or infinity")
        return out
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != shape:
        raise QualityInputError(f"{name} shape {arr.shape} != {shape}")
    if not np.all(np.isfinite(arr)):
        raise QualityInputError(f"{name} contains NaN or infinity")
    return arr


def symmetric_objective_matrix(P: Any) -> spa.csc_matrix:
    """Return ``(P + P.T) / 2`` in CSC float64."""
    if spa.issparse(P):
        if P.shape[0] != P.shape[1]:
            raise QualityInputError(f"P shape {P.shape} is not square")
        if not np.all(np.isfinite(P.tocoo().data)):
            raise QualityInputError("P contains NaN or infinity")
        return symmetrize(P)
    arr = np.asarray(P, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise QualityInputError(f"P shape {arr.shape} is not square")
    if not np.all(np.isfinite(arr)):
        raise QualityInputError("P contains NaN or infinity")
    return symmetrize(spa.csc_matrix(arr))


def _matvec(mat: spa.csc_matrix | np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.asarray(mat @ vec, dtype=np.float64).reshape(-1)


def raw_primal_parts(A: Any, l: np.ndarray, u: np.ndarray, x: np.ndarray) -> dict[str, float | int]:
    """Bound violation in the original coordinates.

    Only finite entries of ``l`` and ``u`` can contribute. A row with both
    bounds infinite is outside ``F`` and cannot create a violation.
    """
    x = _finite_vector("x", x)
    n = x.shape[0]
    l_arr = np.ascontiguousarray(np.asarray(l, dtype=np.float64)).reshape(-1)
    m = l_arr.shape[0]
    u_arr = _bound_vector("u", u, m)
    if np.any(np.isnan(l_arr)):
        raise QualityInputError("l contains NaN")
    A_mat = _as_linear("A", A, (m, n))
    ax = _matvec(A_mat, x)
    finite_l = np.isfinite(l_arr)
    finite_u = np.isfinite(u_arr)
    viol = 0.0
    if np.any(finite_l):
        viol = max(viol, float(np.max(np.maximum(l_arr[finite_l] - ax[finite_l], 0.0))))
    if np.any(finite_u):
        viol = max(viol, float(np.max(np.maximum(ax[finite_u] - u_arr[finite_u], 0.0))))
    active = finite_l | finite_u
    n_finite = int(np.count_nonzero(active))
    if n_finite == 0:
        ax_f = 0.0
        r_p_norm = 0.0
    else:
        ax_f = float(np.linalg.norm(ax[active], ord=np.inf))
        r_p_norm = float(viol / (1.0 + ax_f))
    return {
        "r_p": float(viol),
        "n_finite_bound_rows": n_finite,
        "ax_f_inf": float(ax_f),
        "r_p_norm_candidate": float(r_p_norm),
    }


def raw_stationarity_parts(
    P: Any, q: np.ndarray, A: Any, x: np.ndarray, y: np.ndarray
) -> dict[str, float | np.ndarray]:
    """Stationarity in original coordinates, using ``P_sym`` and OSQP's dual sign."""
    x = _finite_vector("x", x)
    n = x.shape[0]
    q_arr = _finite_vector("q", q, n)
    P_sym = symmetric_objective_matrix(P)
    if P_sym.shape != (n, n):
        raise QualityInputError(f"P shape {P_sym.shape} != {(n, n)}")
    y_arr = _finite_vector("y", y)
    m = y_arr.shape[0]
    A_mat = _as_linear("A", A, (m, n))
    px = _matvec(P_sym, x)
    aty = _matvec(A_mat.T, y_arr)
    residual = px + q_arr + aty
    r_d = float(np.linalg.norm(residual, ord=np.inf)) if residual.size else 0.0
    norm_px = float(np.linalg.norm(px, ord=np.inf)) if px.size else 0.0
    norm_q = float(np.linalg.norm(q_arr, ord=np.inf)) if q_arr.size else 0.0
    norm_aty = float(np.linalg.norm(aty, ord=np.inf)) if aty.size else 0.0
    denom = 1.0 + norm_px + norm_q + norm_aty
    return {
        "residual": np.ascontiguousarray(residual, dtype=np.float64),
        "r_d": r_d,
        "norm_px": norm_px,
        "norm_q": norm_q,
        "norm_aty": norm_aty,
        "r_d_norm": float(r_d / denom),
    }


def objective_from_arrays(P: Any, q: np.ndarray, x: np.ndarray) -> float:
    x = _finite_vector("x", x)
    q_arr = _finite_vector("q", q, x.shape[0])
    P_sym = symmetric_objective_matrix(P)
    if P_sym.shape[0] != x.shape[0]:
        raise QualityInputError(f"P shape {P_sym.shape} is incompatible with x")
    quad = float(x @ _matvec(P_sym, x))
    return 0.5 * quad + float(q_arr @ x)


def relative_objective_gap(objective: float, reference_objective: float) -> float:
    if not np.isfinite(objective):
        raise QualityInputError("objective is not finite")
    if not np.isfinite(reference_objective):
        raise QualityInputError("reference objective is not finite")
    return float(abs(objective - reference_objective) / (1.0 + abs(reference_objective)))


@dataclass
class MetricValue:
    """One metric. ``unavailable`` never uses 0 as a stand-in."""

    state: str
    value: float | None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _available(value: float) -> MetricValue:
    return MetricValue("available", float(value), None)


def _missing(reason: str) -> MetricValue:
    return MetricValue("unavailable", None, reason)


@dataclass
class QualityReport:
    implementation_version: str
    input_status: str
    dual_sign_convention: str
    objective_convention: str
    r_p: MetricValue
    r_d: MetricValue
    r_d_norm: MetricValue
    norm_px: MetricValue
    norm_q: MetricValue
    norm_aty: MetricValue
    r_p_norm_candidate: MetricValue
    ax_f_inf: MetricValue
    n_finite_bound_rows: MetricValue
    objective: MetricValue
    reference_objective: MetricValue
    relative_gap: MetricValue
    failed_criteria: tuple[str, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_criteria"] = list(self.failed_criteria)
        payload["notes"] = list(self.notes)
        return payload


def evaluate_arrays(
    *,
    P: Any,
    q: np.ndarray,
    A: Any,
    l: np.ndarray,
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray | None = None,
    reference_objective: float | None = None,
    thresholds: dict[str, float] | None = None,
    status: str | None = None,
) -> QualityReport:
    """Compute every documented component from arrays. Does not call a solver."""
    primal = raw_primal_parts(A, l, u, x)
    objective = objective_from_arrays(P, q, x)
    if y is None:
        stationarity_reason = "dual vector y was not supplied"
        r_d = _missing(stationarity_reason)
        r_d_norm = _missing(stationarity_reason)
        norm_px = _missing(stationarity_reason)
        norm_q = _missing(stationarity_reason)
        norm_aty = _missing(stationarity_reason)
    else:
        parts = raw_stationarity_parts(P, q, A, x, y)
        r_d = _available(float(parts["r_d"]))
        r_d_norm = _available(float(parts["r_d_norm"]))
        norm_px = _available(float(parts["norm_px"]))
        norm_q = _available(float(parts["norm_q"]))
        norm_aty = _available(float(parts["norm_aty"]))
    if reference_objective is None:
        ref = _missing("reference objective was not supplied")
        gap = _missing("reference objective was not supplied")
    else:
        ref = _available(float(reference_objective))
        gap = _available(relative_objective_gap(objective, float(reference_objective)))
    report = QualityReport(
        implementation_version=METRIC_IMPLEMENTATION_VERSION,
        input_status="valid",
        dual_sign_convention=DUAL_SIGN_CONVENTION,
        objective_convention=OBJECTIVE_CONVENTION,
        r_p=_available(float(primal["r_p"])),
        r_d=r_d,
        r_d_norm=r_d_norm,
        norm_px=norm_px,
        norm_q=norm_q,
        norm_aty=norm_aty,
        r_p_norm_candidate=_available(float(primal["r_p_norm_candidate"])),
        ax_f_inf=_available(float(primal["ax_f_inf"])),
        n_finite_bound_rows=_available(float(primal["n_finite_bound_rows"])),
        objective=_available(objective),
        reference_objective=ref,
        relative_gap=gap,
        failed_criteria=(),
        notes=(PRIMAL_NORM_CANDIDATE_NOTE,),
    )
    if thresholds:
        report.failed_criteria = tuple(
            criteria_failures(report, status=status, **thresholds)
        )
    return report


def criteria_failures(
    report: QualityReport,
    *,
    status: str | None = None,
    primal_tol: float | None = None,
    stationarity_tol: float | None = None,
    gap_tol: float | None = None,
    normalized_stationarity_tol: float | None = None,
    normalized_primal_tol: float | None = None,
    require_solved: bool = False,
) -> list[str]:
    """Name the checks that fail. Omitted tolerances are not applied.

    This function does not choose a project threshold. An unavailable
    component requested by a tolerance is reported as unavailable, not as zero
    and not as a pass.
    """
    failed: list[str] = []
    if require_solved and normalize_status(status) not in SOLVED_EXACT:
        failed.append("status_not_exactly_solved")
    pairs = (
        ("raw_primal", report.r_p, primal_tol),
        ("raw_stationarity", report.r_d, stationarity_tol),
        ("relative_objective_gap", report.relative_gap, gap_tol),
        ("normalized_stationarity_candidate", report.r_d_norm, normalized_stationarity_tol),
        ("normalized_primal_candidate", report.r_p_norm_candidate, normalized_primal_tol),
    )
    for name, metric, tol in pairs:
        if tol is None:
            continue
        if metric.state != "available" or metric.value is None:
            failed.append(f"unavailable:{name}")
            continue
        if not np.isfinite(metric.value) or metric.value > tol:
            failed.append(name)
    return failed


def historical_report_from_row(row: dict[str, Any]) -> QualityReport:
    """Scalars from a preflight row. Normalized terms stay unavailable.

    The stored ``quality_pass`` flag is copied into ``notes``. It is not
    recomputed and it is not a KKT certificate.
    """
    missing = (
        "historical preflight stored scalar columns only; "
        "P, A, q, x, and y were not saved"
    )

    def stored(key: str) -> MetricValue:
        if key not in row or row[key] is None or row[key] == "":
            return _missing(f"historical column {key} is absent")
        value = float(row[key])
        if not np.isfinite(value):
            return _missing(f"historical column {key} is not finite")
        return _available(value)

    objective = stored("objective")
    reference = stored("reference_objective")
    if objective.state == "available" and reference.state == "available":
        gap = _available(relative_objective_gap(float(objective.value), float(reference.value)))
    else:
        gap = _missing("historical objective or reference objective is absent")
    flag = row.get("quality_pass")
    notes = (
        f"stored quality_pass={flag!r} was not recomputed from matrices",
        "normalized residuals were not reconstructed",
    )
    return QualityReport(
        implementation_version=METRIC_IMPLEMENTATION_VERSION,
        input_status="historical_scalars_only",
        dual_sign_convention=DUAL_SIGN_CONVENTION,
        objective_convention=OBJECTIVE_CONVENTION,
        r_p=stored("primal_residual"),
        r_d=stored("dual_residual"),
        r_d_norm=_missing(missing),
        norm_px=_missing(missing),
        norm_q=_missing(missing),
        norm_aty=_missing(missing),
        r_p_norm_candidate=_missing(missing),
        ax_f_inf=_missing(missing),
        n_finite_bound_rows=_missing(missing),
        objective=objective,
        reference_objective=reference,
        relative_gap=gap,
        failed_criteria=(),
        notes=notes,
    )


def primal_violation(instance: QPInstance, x: np.ndarray) -> float:
    """max(0, l - Ax, Ax - u) over finite bounds. Infinite bounds do not contribute."""
    return float(raw_primal_parts(instance.A, instance.l, instance.u, x)["r_p"])


def stationarity_residual(instance: QPInstance, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """r = P_sym x + q + A'y in the original coordinates. Sign follows OSQP's dual."""
    parts = raw_stationarity_parts(instance.P_full, instance.q, instance.A, x, y)
    return np.ascontiguousarray(parts["residual"], dtype=np.float64)


def stationarity_inf(instance: QPInstance, x: np.ndarray, y: np.ndarray) -> float:
    return float(raw_stationarity_parts(instance.P_full, instance.q, instance.A, x, y)["r_d"])


def relative_stationarity(instance: QPInstance, x: np.ndarray, y: np.ndarray) -> float:
    return float(raw_stationarity_parts(instance.P_full, instance.q, instance.A, x, y)["r_d_norm"])


def quality_pass(
    status: str | None,
    primal_inf: float,
    stationarity: float,
    *,
    primal_tol: float,
    stationarity_tol: float,
) -> bool:
    """Clean pass requires OSQP 'solved' and independent residual checks.

    'solved inaccurate' is intentionally not a pass.
    """
    if normalize_status(status) not in SOLVED_EXACT:
        return False
    if not np.isfinite(primal_inf) or not np.isfinite(stationarity):
        return False
    return primal_inf <= primal_tol and stationarity <= stationarity_tol
