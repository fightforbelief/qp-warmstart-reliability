"""Figures are drawn from result tables, including negative differences."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_stage_bars(summary_rows: list[dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["method"] for r in summary_rows if r.get("n_rows", 0)]
    solve = [r.get("median_solve_wall_ms") or 0 for r in summary_rows if r.get("n_rows", 0)]
    update = [r.get("median_update_ms") or 0 for r in summary_rows if r.get("n_rows", 0)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, update, label="parameter update")
    ax.bar(labels, solve, bottom=update, label="solve wall")
    ax.set_ylabel("median milliseconds")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_paired_difference(frame: pd.DataFrame, path: Path, title: str) -> None:
    """C1 streaming minus C2 streaming on matched steps. Positive means cold is slower."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["group_id", "instance_id", "step", "timing_repeat"]
    cols = keys + ["total_streaming_ns"]
    a = frame.loc[frame["method"] == "C1_CACHED_COLD", cols]
    b = frame.loc[frame["method"] == "C2_CACHED_AUTO_PREVIOUS", cols]
    merged = a.merge(b, on=keys, suffixes=("_c1", "_c2"))
    merged = merged.loc[merged["step"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    if merged.empty:
        ax.text(0.5, 0.5, "no paired rows", ha="center")
    else:
        delta_ms = (merged["total_streaming_ns_c1"] - merged["total_streaming_ns_c2"]) / 1e6
        ax.axhline(0.0, color="black", linewidth=1)
        ax.scatter(merged["step"], delta_ms, s=12, alpha=0.7)
        ax.set_xlabel("step within frozen sequence")
        ax.set_ylabel("C1 minus C2 streaming ms")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
