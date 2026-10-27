"""C2 automatic previous solution and C3 explicit previous primal/dual."""

from __future__ import annotations

from ..instance import QPInstance
from ..solver import SolveRecord, solve_sequence


def solve_auto_previous(
    instances: list[QPInstance],
    settings: dict,
    *,
    primal_tol: float,
    stationarity_tol: float,
) -> list[SolveRecord]:
    return solve_sequence(
        instances,
        settings,
        method="C2_CACHED_AUTO_PREVIOUS",
        mode="auto_previous",
        primal_tol=primal_tol,
        stationarity_tol=stationarity_tol,
    )


def solve_explicit_previous(
    instances: list[QPInstance],
    settings: dict,
    *,
    primal_tol: float,
    stationarity_tol: float,
) -> list[SolveRecord]:
    return solve_sequence(
        instances,
        settings,
        method="C3_CACHED_EXPLICIT_PREVIOUS",
        mode="explicit_previous",
        primal_tol=primal_tol,
        stationarity_tol=stationarity_tol,
    )
