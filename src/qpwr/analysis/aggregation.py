"""Summaries. Oracle-best is labeled and is not a deployable policy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def streaming_ms(frame: pd.DataFrame) -> pd.Series:
    return frame["total_streaming_ns"].astype(float) / 1e6


def summarize_method(frame: pd.DataFrame, method: str) -> dict:
    sub = frame.loc[frame["method"] == method].copy()
    if sub.empty:
        return {"method": method, "n": 0}
    # Streaming summaries exclude the first step of each sequence, which is
    # an initial setup transient, and exclude fresh-cold rows whose streaming
    # time is null.
    seq = sub.loc[sub["step"] > 0]
    vals = seq["total_streaming_ns"].dropna().astype(float) / 1e6
    statuses = sub["status"].value_counts().to_dict()
    out = {
        "method": method,
        "n_rows": int(len(sub)),
        "n_streaming_steps": int(vals.shape[0]),
        "n_groups": int(sub["group_id"].nunique()),
        "failure_rate": float(sub["method_failed"].astype(bool).mean()),
        "quality_pass_rate": float(sub["quality_pass"].astype(bool).mean()),
        "status_counts": {str(k): int(v) for k, v in statuses.items()},
        "median_streaming_ms": float(vals.median()) if len(vals) else None,
        "p95_streaming_ms": float(np.quantile(vals, 0.95)) if len(vals) else None,
        "mean_streaming_ms": float(vals.mean()) if len(vals) else None,
        "median_iters": float(sub["iterations"].median()) if sub["iterations"].notna().any() else None,
        "median_setup_ms": float(sub["setup_ns"].median() / 1e6),
        "median_update_ms": float(sub["parameter_update_ns"].median() / 1e6),
        "median_solve_wall_ms": float(sub["solve_wall_ns"].median() / 1e6),
        "median_osqp_solve_ms": (
            float(sub["osqp_solve_ns"].dropna().median() / 1e6) if sub["osqp_solve_ns"].notna().any() else None
        ),
        "median_osqp_run_ms": (
            float(sub["osqp_run_ns"].dropna().median() / 1e6) if sub["osqp_run_ns"].notna().any() else None
        ),
        "median_oneoff_ms": float(sub["oneoff_complete_ns"].median() / 1e6),
        "median_objective_abs_gap": (
            float((sub["objective"] - sub["reference_objective"]).abs().median())
            if sub["objective"].notna().any() and sub["reference_objective"].notna().any()
            else None
        ),
    }
    return out


def oracle_best_note() -> str:
    return (
        "ORACLE_BEST would pick the fastest policy after seeing each test time. "
        "It is oracle_only and is not an admissible deployed baseline."
    )
