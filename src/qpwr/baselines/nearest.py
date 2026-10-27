"""Nearest stored train solution. The index is fit on train ids only."""

from __future__ import annotations

import numpy as np

from ..manifest import LeakageError


class NearestNeighborIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.features: np.ndarray | None = None
        self.payloads: list[dict] = []

    def fit(self, rows: list[dict]) -> "NearestNeighborIndex":
        if any(r.get("partition") != "train" for r in rows):
            raise LeakageError("nearest neighbor index received a non-train row")
        self.ids = [r["instance_id"] for r in rows]
        self.features = np.vstack([np.asarray(r["feature"], dtype=np.float64) for r in rows])
        self.payloads = [r["payload"] for r in rows]
        return self

    def query(self, feature: np.ndarray, instance_id: str) -> dict:
        if self.features is None:
            raise LeakageError("nearest neighbor queried before train fit")
        if instance_id in self.ids:
            raise LeakageError("query id is inside the training index; do not look up test labels this way during fit")
        f = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        dist = np.linalg.norm(self.features - f, axis=1)
        return self.payloads[int(np.argmin(dist))]
