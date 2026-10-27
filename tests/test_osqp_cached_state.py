"""Cache identity and warm-start accounting on one frozen two-step sequence."""

import numpy as np
import scipy.sparse as spa

from qpwr.baselines.cached_cold import solve_cached_cold
from qpwr.baselines.previous import solve_auto_previous, solve_explicit_previous
from qpwr.instance import QPInstance, objective_value
from qpwr.solver import DEFAULT_SETTINGS, solve_fresh_cold


def _seq():
    P = spa.csc_matrix([[2.0]])
    A = spa.eye(1, format="csc")
    base = dict(
        group_id="g",
        family="mpc",
        source_repo_sha="none",
        source_file="unit",
        source_seed=0,
        P_full=P,
        A=A,
        l=np.array([-20.0]),
        u=np.array([20.0]),
        metadata={"check_psd": True},
    )
    a = QPInstance(instance_id="g/s0", step=0, q=np.array([-2.0]), **base)  # optimum 1
    b = QPInstance(instance_id="g/s1", step=1, q=np.array([-10.0]), **base)  # optimum 5
    return [a, b]


def test_cached_cold_keeps_one_workspace_and_resets():
    settings = dict(DEFAULT_SETTINGS)
    settings["eps_abs"] = 1e-8
    settings["eps_rel"] = 1e-8
    rows = solve_cached_cold(_seq(), settings, primal_tol=1e-6, stationarity_tol=1e-5)
    assert len({r.solver_id for r in rows}) == 1
    assert rows[-1].setup_count == 1
    assert rows[-1].warm_start_calls == 2
    assert rows[0].objective == pytest_approx_opt(1.0, rows[0])
    assert rows[1].objective == pytest_approx_opt(5.0, rows[1])
    assert all(r.quality_pass for r in rows)


def pytest_approx_opt(xstar, rec):
    import pytest

    inst_obj = objective_value_from_star(xstar)
    return pytest.approx(inst_obj, abs=1e-6)


def objective_value_from_star(xstar: float) -> float:
    # 1/2 * 2 * x^2 + q x with q = -2 xstar, optimum value = -xstar^2
    return -(xstar ** 2)


def test_auto_and_explicit_previous_match_optimum_and_accounting():
    settings = dict(DEFAULT_SETTINGS)
    settings["eps_abs"] = 1e-8
    settings["eps_rel"] = 1e-8
    auto = solve_auto_previous(_seq(), settings, primal_tol=1e-6, stationarity_tol=1e-5)
    explicit = solve_explicit_previous(_seq(), settings, primal_tol=1e-6, stationarity_tol=1e-5)
    assert auto[-1].warm_start_calls == 0
    assert explicit[-1].warm_start_calls == 1
    assert auto[0].solver_id == auto[1].solver_id
    assert explicit[0].solver_id == explicit[1].solver_id
    for rec, star in ((auto[0], 1.0), (auto[1], 5.0), (explicit[0], 1.0), (explicit[1], 5.0)):
        assert rec.objective == __import__("pytest").approx(-(star ** 2), abs=1e-5)
        assert rec.quality_pass


def test_fresh_cold_builds_a_new_solver_each_qp():
    settings = dict(DEFAULT_SETTINGS)
    settings["eps_abs"] = 1e-8
    settings["eps_rel"] = 1e-8
    seq = _seq()
    rows = [solve_fresh_cold(inst, settings, primal_tol=1e-6, stationarity_tol=1e-5) for inst in seq]
    assert rows[0].solver_id != rows[1].solver_id
    assert rows[0].cache_policy == "fresh_setup_each_qp"
    assert rows[0].total_streaming_ns is None


def test_warm_start_shape_mismatch():
    import pytest

    from qpwr.solver import OSQPWorkspace, StageTimer
    from qpwr.states import StateError, WarmStartState

    settings = dict(DEFAULT_SETTINGS)
    ws = OSQPWorkspace(settings)
    timer = StageTimer()
    ws.setup_or_update(_seq()[0], timer)
    with pytest.raises(StateError):
        ws.insert(WarmStartState(np.zeros(3), np.zeros(1), source="bad"), timer)
