"""Analytic QP checks. These are correctness tests, not external evidence."""

import numpy as np
import pytest
import scipy.sparse as spa

from qpwr.instance import QPInstance, QPValidationError, objective_value
from qpwr.quality import primal_violation, stationarity_inf


def _box(P, q, l, u):
    P = spa.csc_matrix(P, dtype=float)
    A = spa.eye(P.shape[0], format="csc")
    return QPInstance(
        instance_id="toy",
        group_id="toy",
        family="mpc",
        source_repo_sha="none",
        source_file="unit",
        source_seed=0,
        step=0,
        P_full=P,
        q=np.array(q, dtype=float),
        A=A,
        l=np.array(l, dtype=float),
        u=np.array(u, dtype=float),
        metadata={"check_psd": True},
    )


def test_unconstrained_1d_optimum():
    # 1/2 (x-3)^2 = 1/2 x^2 - 3x + const, nonbinding bounds.
    inst = _box([[1.0]], [-3.0], [-100.0], [100.0])
    x = np.array([3.0])
    assert objective_value(inst, x) == pytest.approx(-4.5)
    assert primal_violation(inst, x) == pytest.approx(0.0)
    # No active constraint, dual is zero: Px + q = 0.
    assert stationarity_inf(inst, x, np.array([0.0])) == pytest.approx(0.0, abs=1e-12)


def test_upper_and_lower_dual_signs():
    # Upper active: min 1/2 (x-3)^2 s.t. x <= 1. y >= 0.
    upper = _box([[1.0]], [-3.0], [-np.inf], [1.0])
    x = np.array([1.0])
    y = np.array([2.0])
    assert stationarity_inf(upper, x, y) == pytest.approx(0.0, abs=1e-12)
    assert y[0] > 0
    # Lower active: min 1/2 (x+2)^2 s.t. x >= 0. P=1, q=2, y <= 0.
    lower = _box([[1.0]], [2.0], [0.0], [np.inf])
    x0 = np.array([0.0])
    y0 = np.array([-2.0])
    assert stationarity_inf(lower, x0, y0) == pytest.approx(0.0, abs=1e-12)
    assert y0[0] < 0


def test_equality_and_semidefinite():
    # Equality x = 1. Stationarity Px + q + A'y = 0 gives y = -1. The multiplier may be negative.
    eq = _box([[1.0]], [0.0], [1.0], [1.0])
    assert stationarity_inf(eq, np.array([1.0]), np.array([-1.0])) == pytest.approx(0.0, abs=1e-12)
    # P = 0, q = 0, bounds [0, 1]. Every feasible x is optimal. Do not claim uniqueness.
    psd = _box([[0.0]], [0.0], [0.0], [1.0])
    assert objective_value(psd, np.array([0.2])) == pytest.approx(0.0)
    assert objective_value(psd, np.array([0.8])) == pytest.approx(0.0)


def test_rejects_inconsistent_bounds_and_nan_without_solving():
    with pytest.raises(QPValidationError):
        _box([[1.0]], [0.0], [1.0], [0.0])
    with pytest.raises(QPValidationError):
        _box([[1.0]], [np.nan], [-1.0], [1.0])


def test_triu_nnz_and_objective_uses_full_symmetric_matrix():
    P = np.array([[2.0, 0.5], [0.5, 2.0]])
    inst = _box(P, [0.0, 0.0], [-10, -10], [10, 10])
    assert inst.P_triu.nnz == 3
    x = np.array([1.0, 1.0])
    assert objective_value(inst, x) == pytest.approx(0.5 * float(x @ P @ x))
