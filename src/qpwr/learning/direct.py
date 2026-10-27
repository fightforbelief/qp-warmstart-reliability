"""Direct regression is implemented for later approved training only.

It predicts the OSQP primal/dual pair from pre-solve features. It is not
the L2WS fixed-point method. Preflight does not call fit().
"""

from __future__ import annotations

from ..manifest import LeakageError


class DirectRegressor:
    name = "C5_DIRECT_REGRESSION"

    def __init__(self) -> None:
        self._fitted = False
        self._partitions: set[str] = set()

    def fit(self, rows: list[dict]) -> None:
        parts = {r.get("partition") for r in rows}
        if parts - {"train"}:
            raise LeakageError(f"direct regression fit saw partitions {parts}")
        if any(r.get("oracle") for r in rows):
            raise LeakageError("direct regression cannot train on oracle routing labels")
        self._partitions = parts
        self._fitted = True

    def predict(self, feature: dict) -> None:
        if not self._fitted:
            raise RuntimeError("direct regression is not fitted; preflight must not train it")
        raise NotImplementedError(
            "Deployment inference is intentionally unavailable until a frozen protocol approves training."
        )
