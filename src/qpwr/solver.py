"""OSQP wrapper with explicit cache and warm-start policies.

Policies
--------
C0 FRESH_COLD
    New OSQP object and setup on every QP. One-off baseline.
C1 CACHED_COLD
    One workspace and factorization. Explicit primal/dual reset to zero
    before every solve. This is the no-previous-answer cached control.
C2 CACHED_AUTO_PREVIOUS
    One workspace. OSQP reuses the previous solution when warm_starting
    is true. Strong sequential baseline.
C3 CACHED_EXPLICIT_PREVIOUS
    One workspace. warm_start(x, y) with the previous primal and dual.

Adaptive rho is a solver-state confound: the workspace can change rho
across steps. The preflight freezes adaptive_rho and reports it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import osqp

from .instance import QPInstance, objective_value, qp_content_hash
from .quality import normalize_status, primal_violation, quality_pass, stationarity_inf
from .states import StateError, WarmStartState, zeros_state
from .timing import StageTimer, streaming_total_ns


def _seconds_to_ns(value: Any) -> int | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return int(round(v * 1e9))


def settings_sha(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


DEFAULT_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "warm_starting": True,
    "adaptive_rho": False,
    "scaling": 10,
    "eps_abs": 1e-4,
    "eps_rel": 1e-4,
    "eps_prim_inf": 1e-4,
    "eps_dual_inf": 1e-4,
    "max_iter": 4000,
    "rho": 0.1,
    "sigma": 1e-6,
    "alpha": 1.6,
    "check_termination": 25,
    "polishing": False,
}


@dataclass
class SolveRecord:
    method: str
    instance_id: str
    group_id: str
    step: int
    family: str
    qp_hash: str
    status: str
    status_val: int | None
    x: np.ndarray | None
    y: np.ndarray | None
    objective: float | None
    primal_residual: float | None
    dual_residual: float | None
    quality_pass: bool
    iterations: int | None
    stages_ns: dict[str, int]
    osqp_setup_ns: int | None
    osqp_update_ns: int | None
    osqp_solve_ns: int | None
    osqp_run_ns: int | None
    cache_policy: str
    solver_id: int | None
    setup_count: int
    warm_start_calls: int
    n: int
    m: int
    nnzP: int
    nnzA: int
    notes: str = ""
    method_failed: bool = False
    timeout: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_streaming_ns(self) -> int | None:
        if self.cache_policy == "fresh_setup_each_qp":
            return None
        return streaming_total_ns(self.stages_ns)

    @property
    def oneoff_complete_ns(self) -> int:
        return int(
            self.stages_ns.get("setup_ns", 0)
            + self.stages_ns.get("parameter_update_ns", 0)
            + self.stages_ns.get("solve_wall_ns", 0)
            + self.stages_ns.get("verify_ns", 0)
            + self.stages_ns.get("state_insert_ns", 0)
            + self.stages_ns.get("feature_ns", 0)
            + self.stages_ns.get("lookup_ns", 0)
            + self.stages_ns.get("infer_ns", 0)
            + self.stages_ns.get("fallback_ns", 0)
        )


def _call_warm_start(solver: Any, state: WarmStartState, n: int, m: int) -> None:
    state.assert_shapes(n, m)
    try:
        solver.warm_start(x=state.x, y=state.y)
    except TypeError:
        solver.warm_start(state.x, state.y)
    except Exception as exc:
        raise StateError(f"OSQP warm_start rejected the state: {exc}") from exc


def _read_info(info: Any, *names: str) -> Any:
    for name in names:
        if hasattr(info, name):
            return getattr(info, name)
    return None


class OSQPWorkspace:
    """One OSQP object. setup_count stays 1 when only q/l/u are updated."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = dict(settings)
        self.solver: Any | None = None
        self.solver_id: int | None = None
        self.setup_count = 0
        self.pattern: tuple[bytes, bytes] | None = None
        self.warm_start_calls = 0
        self._n = 0
        self._m = 0

    def _new_solver(self) -> Any:
        return osqp.OSQP()

    def setup_or_update(self, instance: QPInstance, timer: StageTimer) -> str:
        pattern = instance.sparsity_pattern()
        P = instance.P_triu
        if self.solver is None or pattern != self.pattern:
            with timer.stage("setup_ns"):
                self.solver = self._new_solver()
                self.solver.setup(
                    P,
                    instance.q,
                    instance.A,
                    instance.l,
                    instance.u,
                    **self.settings,
                )
            self.solver_id = id(self.solver)
            self.setup_count += 1
            self.pattern = pattern
            self._n = instance.n
            self._m = instance.m
            return "setup"
        with timer.stage("parameter_update_ns"):
            self.solver.update(q=instance.q, l=instance.l, u=instance.u)
        return "update"

    def insert(self, state: WarmStartState, timer: StageTimer) -> None:
        assert self.solver is not None
        with timer.stage("state_insert_ns"):
            _call_warm_start(self.solver, state, self._n, self._m)
        self.warm_start_calls += 1

    def solve(self, timer: StageTimer) -> Any:
        assert self.solver is not None
        with timer.stage("solve_wall_ns"):
            return self.solver.solve()


def _record_from_result(
    *,
    method: str,
    instance: QPInstance,
    result: Any,
    timer: StageTimer,
    cache_policy: str,
    workspace: OSQPWorkspace,
    primal_tol: float,
    stationarity_tol: float,
    notes: str,
) -> SolveRecord:
    info = getattr(result, "info", None)
    status = normalize_status(_read_info(info, "status") if info is not None else None)
    status_val = _read_info(info, "status_val") if info is not None else None
    x = None if getattr(result, "x", None) is None else np.array(result.x, dtype=np.float64, copy=True)
    y = None if getattr(result, "y", None) is None else np.array(result.y, dtype=np.float64, copy=True)
    with timer.stage("verify_ns"):
        if x is None or y is None:
            obj = None
            prim = None
            dual = None
            passed = False
        else:
            obj = objective_value(instance, x)
            prim = primal_violation(instance, x)
            dual = stationarity_inf(instance, x, y)
            passed = quality_pass(
                status,
                prim,
                dual,
                primal_tol=primal_tol,
                stationarity_tol=stationarity_tol,
            )
    timeout = "time limit" in status or "maximum iterations" in status
    failed = (not passed) or x is None
    stages = {
        "preparation_ns": timer.get("preparation_ns"),
        "parameter_update_ns": timer.get("parameter_update_ns"),
        "feature_ns": timer.get("feature_ns"),
        "lookup_ns": timer.get("lookup_ns"),
        "infer_ns": timer.get("infer_ns"),
        "state_insert_ns": timer.get("state_insert_ns"),
        "solve_wall_ns": timer.get("solve_wall_ns"),
        "verify_ns": timer.get("verify_ns"),
        "fallback_ns": timer.get("fallback_ns"),
        "setup_ns": timer.get("setup_ns"),
    }
    return SolveRecord(
        method=method,
        instance_id=instance.instance_id,
        group_id=instance.group_id,
        step=instance.step,
        family=instance.family,
        qp_hash=qp_content_hash(instance),
        status=status,
        status_val=None if status_val is None else int(status_val),
        x=x,
        y=y,
        objective=obj,
        primal_residual=prim,
        dual_residual=dual,
        quality_pass=passed,
        iterations=None if info is None else _read_info(info, "iter", "iterations"),
        stages_ns=stages,
        osqp_setup_ns=_seconds_to_ns(_read_info(info, "setup_time")) if info is not None else None,
        osqp_update_ns=_seconds_to_ns(_read_info(info, "update_time")) if info is not None else None,
        osqp_solve_ns=_seconds_to_ns(_read_info(info, "solve_time")) if info is not None else None,
        osqp_run_ns=_seconds_to_ns(_read_info(info, "run_time")) if info is not None else None,
        cache_policy=cache_policy,
        solver_id=workspace.solver_id,
        setup_count=workspace.setup_count,
        warm_start_calls=workspace.warm_start_calls,
        n=instance.n,
        m=instance.m,
        nnzP=int(instance.metadata.get("nnzP", instance.P_triu.nnz)),
        nnzA=int(instance.metadata.get("nnzA", instance.A.nnz)),
        notes=notes,
        method_failed=failed,
        timeout=timeout,
    )


def solve_fresh_cold(
    instance: QPInstance,
    settings: dict[str, Any],
    *,
    primal_tol: float,
    stationarity_tol: float,
) -> SolveRecord:
    ws = OSQPWorkspace(settings)
    timer = StageTimer()
    ws.setup_or_update(instance, timer)
    result = ws.solve(timer)
    return _record_from_result(
        method="C0_FRESH_COLD",
        instance=instance,
        result=result,
        timer=timer,
        cache_policy="fresh_setup_each_qp",
        workspace=ws,
        primal_tol=primal_tol,
        stationarity_tol=stationarity_tol,
        notes="New OSQP setup every instance. Not a streaming warm-start comparison.",
    )


def solve_sequence(
    instances: list[QPInstance],
    settings: dict[str, Any],
    *,
    method: str,
    mode: str,
    primal_tol: float,
    stationarity_tol: float,
) -> list[SolveRecord]:
    """Replay one frozen sequence on one workspace.

    mode:
      cached_cold — explicit zero primal/dual before every solve
      auto_previous — no explicit warm_start; OSQP keeps its solution
      explicit_previous — warm_start with the previous accepted (x, y)
    """
    if mode not in {"cached_cold", "auto_previous", "explicit_previous"}:
        raise ValueError(mode)
    if not instances:
        return []
    ws = OSQPWorkspace(settings)
    prev: WarmStartState | None = None
    records: list[SolveRecord] = []
    for inst in instances:
        timer = StageTimer()
        action = ws.setup_or_update(inst, timer)
        if mode == "cached_cold":
            ws.insert(zeros_state(inst.n, inst.m), timer)
            note = "factorization reused; primal/dual reset to zero"
        elif mode == "explicit_previous":
            if prev is not None:
                ws.insert(prev, timer)
                note = "explicit warm_start of previous primal and dual"
            else:
                note = "first step has no previous solution"
        else:
            note = "OSQP automatic previous solution; no explicit warm_start"
        if action == "setup" and mode != "cached_cold":
            note = "initial setup; " + note
        result = ws.solve(timer)
        rec = _record_from_result(
            method=method,
            instance=inst,
            result=result,
            timer=timer,
            cache_policy=mode,
            workspace=ws,
            primal_tol=primal_tol,
            stationarity_tol=stationarity_tol,
            notes=note,
        )
        records.append(rec)
        if rec.x is not None and rec.y is not None:
            prev = WarmStartState(rec.x, rec.y, source=method, algorithm="osqp_primal_dual")
        else:
            prev = None
    return records
