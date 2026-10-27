"""Monotonic stage timers. Nanoseconds internally; milliseconds only in reports."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class StageTimer:
    def __init__(self) -> None:
        self.ns: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.ns[name] = self.ns.get(name, 0) + (time.perf_counter_ns() - t0)

    def get(self, name: str, default: int = 0) -> int:
        return int(self.ns.get(name, default))


# Streaming estimand E1. Setup is recorded separately and is not added again.
STREAMING_STAGES = (
    "parameter_update_ns",
    "feature_ns",
    "lookup_ns",
    "infer_ns",
    "state_insert_ns",
    "solve_wall_ns",
    "verify_ns",
    "fallback_ns",
)


def streaming_total_ns(stages: dict[str, int]) -> int:
    return int(sum(int(stages.get(k, 0)) for k in STREAMING_STAGES))


def stages_nonnegative(stages: dict[str, int]) -> bool:
    return all(int(v) >= 0 for v in stages.values())
