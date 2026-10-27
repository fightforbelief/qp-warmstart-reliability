"""C1: cached factorization, explicit zero primal/dual."""

from __future__ import annotations

from ..instance import QPInstance
from ..solver import SolveRecord, solve_sequence


def solve_cached_cold(
    instances: list[QPInstance],
    settings: dict,
    *,
    primal_tol: float,
    stationarity_tol: float,
) -> list[SolveRecord]:
    return solve_sequence(
        instances,
        settings,
        method="C1_CACHED_COLD",
        mode="cached_cold",
        primal_tol=primal_tol,
        stationarity_tol=stationarity_tol,
    )
