"""Primal/dual warm-start states.

OSQP's public warm_start(x, y) is the user primal/dual pair.
The JMLR 2024 L2WS fixed-point state w is a different object (operator
splitting iterates, scaling, and auxiliary z). Matching the length of x
is not a state mapping. Until a tested mapping exists, external learned
methods must be named L2WS_INSPIRED and must not be reported as the
author algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class StateError(ValueError):
    pass


@dataclass
class WarmStartState:
    x: np.ndarray
    y: np.ndarray
    source: str
    algorithm: str = "osqp_primal_dual"

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float64).reshape(-1).copy()
        self.y = np.ascontiguousarray(self.y, dtype=np.float64).reshape(-1).copy()
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.y)):
            raise StateError("warm-start state contains NaN or inf")

    def assert_shapes(self, n: int, m: int) -> None:
        if self.x.shape[0] != n or self.y.shape[0] != m:
            raise StateError(
                f"warm start shape x:{self.x.shape[0]} y:{self.y.shape[0]} does not match n={n} m={m}"
            )


def zeros_state(n: int, m: int) -> WarmStartState:
    return WarmStartState(np.zeros(n), np.zeros(m), source="explicit_zero", algorithm="osqp_primal_dual")


L2WS_MAPPING_STATUS = "UNMAPPED"
L2WS_MAPPING_NOTE = (
    "Author L2WS predicts an internal fixed-point initialization w0 for a "
    "differentiable operator splitting method. OSQP Python warm_start accepts "
    "only primal x and dual y. No equality between those objects has been "
    "proved in this repository. External use must be labeled L2WS_INSPIRED."
)
