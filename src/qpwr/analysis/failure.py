"""Failure rows stay in every table. Nothing is dropped for being slow or infeasible."""

from __future__ import annotations

import pandas as pd


def retain_failures(frame: pd.DataFrame) -> pd.DataFrame:
    if "method_failed" not in frame.columns:
        raise KeyError("method_failed column required")
    out = frame.copy()
    # Explicitly keep failed and timeout rows. Do not dropna on residuals.
    return out.reset_index(drop=True)


def failure_rate(frame: pd.DataFrame) -> float:
    kept = retain_failures(frame)
    if kept.empty:
        return float("nan")
    return float(kept["method_failed"].astype(bool).mean())
