"""Corrected preflight figures. Drawn only from an existing result table.

Component-wise medians are plotted separately. They are not stacked into a
single bar, because the median of a sum is not the sum of medians.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preflight_audit import (
    C0,
    C1,
    C2,
    C3,
    CLASSICAL,
    OVERHEAD_STAGES,
    paired_c1_minus_c2,
    short_method,
    streaming_rows,
)

_COLORS = {
    C0: "#9c755f",
    C1: "#4c78a8",
    C2: "#f58518",
    C3: "#54a24b",
}


def _ms(series: pd.Series) -> pd.Series:
    return series.astype(float) / 1e6


def plot_streaming_stages(frame: pd.DataFrame, path: Path) -> None:
    """Figure 1. Solver wall and small overheads on separate panels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = streaming_rows(frame)
    families = list(dict.fromkeys(stream["family"].astype(str)))
    methods = CLASSICAL
    labels = [short_method(m) for m in methods]
    fig, axes = plt.subplots(len(families), 2, figsize=(11.2, 3.4 * max(len(families), 1)), squeeze=False)
    for row, family in enumerate(families):
        sub = stream.loc[stream["family"].astype(str) == family]
        wall = [_ms(sub.loc[sub["method"] == m, "solve_wall_ns"]).median() for m in methods]
        axes[row, 0].bar(labels, wall, color=[_COLORS[m] for m in methods])
        axes[row, 0].set_ylabel("median ms")
        axes[row, 0].set_title(f"{family}: solver wall time")
        x = np.arange(len(methods))
        width = 0.25
        for j, stage in enumerate(OVERHEAD_STAGES):
            vals = [_ms(sub.loc[sub["method"] == m, stage]).median() for m in methods]
            axes[row, 1].bar(x + (j - 1) * width, vals, width=width, label=stage.replace("_ns", ""))
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(labels)
        axes[row, 1].set_ylabel("median ms")
        axes[row, 1].set_title(f"{family}: overhead medians, separate from wall time")
        axes[row, 1].legend(fontsize=7)
    fig.suptitle(
        "Streaming stages for classical cached policies (step 0 excluded)\n"
        "Each bar is its own median. Do not add the panels: medians do not sum to the median total.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_oneoff(frame: pd.DataFrame, path: Path) -> None:
    """Figure 2. Setup, solve wall, and complete one-off cost as separate medians."""
    path.parent.mkdir(parents=True, exist_ok=True)
    families = list(dict.fromkeys(frame["family"].astype(str)))
    methods = [C0, C1, C2, C3]
    labels = [short_method(m) for m in methods]
    fig, axes = plt.subplots(1, len(families), figsize=(5.4 * len(families), 4.6), squeeze=False)
    for ax, family in zip(axes[0], families):
        sub = frame.loc[frame["family"].astype(str) == family]
        step0 = sub.loc[sub["step"].astype(int) == 0]
        setup = [_ms(step0.loc[step0["method"] == m, "setup_ns"]).median() for m in methods]
        wall = [_ms(sub.loc[sub["method"] == m, "solve_wall_ns"]).median() for m in methods]
        oneoff = [_ms(sub.loc[sub["method"] == m, "oneoff_complete_ns"]).median() for m in methods]
        x = np.arange(len(methods))
        width = 0.25
        ax.bar(x - width, setup, width, label="step-0 setup")
        ax.bar(x, wall, width, label="solve wall, all steps")
        ax.bar(x + width, oneoff, width, label="complete one-off, all steps")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("median ms")
        ax.set_title(family)
        ax.legend(fontsize=7)
    fig.suptitle(
        "E2 one-off view. Blue is step-0 setup. Orange is solve wall over every step.\n"
        "Green is the one-off median over every step, so cached policies look cheap: most of their rows have no setup. Lasso C0 does not.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_paired_by_group(frame: pd.DataFrame, path: Path) -> None:
    """Figure 3. Classical C1-minus-C2 diagnostic, split by independent group."""
    path.parent.mkdir(parents=True, exist_ok=True)
    paired = paired_c1_minus_c2(frame)
    groups = list(dict.fromkeys(paired["group_id"].astype(str)))
    fig, axes = plt.subplots(1, max(len(groups), 1), figsize=(4.4 * max(len(groups), 1), 4.4), squeeze=False)
    if paired.empty:
        axes[0, 0].text(0.5, 0.5, "no paired rows", ha="center")
    for ax, group_id in zip(axes[0], groups):
        sub = paired.loc[paired["group_id"].astype(str) == group_id]
        for repeat, chunk in sub.groupby(sub["timing_repeat"].astype(int)):
            chunk = chunk.sort_values("step")
            ax.plot(
                chunk["step"].astype(int),
                chunk["delta_ms"],
                marker="o",
                markersize=3.5,
                linewidth=1,
                label=f"timing repeat {int(repeat)}",
            )
        ax.axhline(0.0, color="black", linewidth=1)
        median = float(sub["delta_ms"].median())
        ax.axhline(median, color="gray", linestyle="--", linewidth=1, label=f"group median {median:.2f} ms")
        ax.set_title(str(group_id), fontsize=9)
        ax.set_xlabel("step inside the sequence (step 0 excluded)")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("C1 minus C2 streaming ms")
    fig.suptitle(
        "Classical baseline diagnostic: paired C1 − C2. Not a learned-model result.\n"
        "One panel is one independent source group. Timing repeats are technical replicates.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_latency_by_group(frame: pd.DataFrame, path: Path) -> None:
    """Figure 4. C1/C2/C3 streaming latency inside each independent group."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = streaming_rows(frame)
    groups = list(dict.fromkeys(stream["group_id"].astype(str)))
    methods = CLASSICAL
    fig, axes = plt.subplots(1, max(len(groups), 1), figsize=(4.3 * max(len(groups), 1), 4.6), squeeze=False)
    for ax, group_id in zip(axes[0], groups):
        sub = stream.loc[stream["group_id"].astype(str) == group_id]
        data = [_ms(sub.loc[sub["method"] == m, "total_streaming_ns"]).to_numpy() for m in methods]
        ax.boxplot(data, tick_labels=[short_method(m) for m in methods], showfliers=True)
        for i, values in enumerate(data, start=1):
            if values.size == 0:
                continue
            p95 = float(np.quantile(values, 0.95))
            ax.scatter([i], [p95], marker="D", color="black", s=18, zorder=3)
        ax.set_title(str(group_id), fontsize=9)
        ax.set_ylabel("streaming ms, step > 0")
    fig.suptitle(
        f"Classical streaming latency by independent group (n = {len(groups)}). Diamonds are descriptive p95.\n"
        "Timing repeats stay inside the group. This plot does not pick a per-QP winner.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_quality(frame: pd.DataFrame, path: Path) -> None:
    """Figure 5. Status, residuals, and objective gap. No row is dropped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    status_counts = frame["status"].astype(str).value_counts()
    axes[0, 0].bar(status_counts.index.astype(str), status_counts.to_numpy())
    axes[0, 0].set_title(f"solver status (all {len(frame)} rows)")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set_ylabel("rows")

    methods = [C0, C1, C2, C3]
    labels = [short_method(m) for m in methods]

    def _log_boxes(ax, column: str, title: str) -> None:
        data = []
        for method in methods:
            values = frame.loc[frame["method"] == method, column].astype(float).to_numpy()
            values = values[np.isfinite(values)]
            data.append(np.maximum(values, 1e-16))
        ax.boxplot(data, tick_labels=labels, showfliers=True)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("value (log, floor 1e-16 for display)")

    _log_boxes(axes[0, 1], "primal_residual", "primal bound violation, inf norm")
    _log_boxes(axes[1, 0], "dual_residual", "stationarity inf norm (stored as dual_residual)")
    gap = (frame["objective"].astype(float) - frame["reference_objective"].astype(float)).abs()
    gap = gap / (1.0 + frame["reference_objective"].astype(float).abs())
    data = []
    for method in methods:
        values = gap.loc[frame["method"] == method].to_numpy()
        values = values[np.isfinite(values)]
        data.append(np.maximum(values, 1e-16))
    axes[1, 1].boxplot(data, tick_labels=labels, showfliers=True)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("relative objective gap vs high-accuracy reference")
    axes[1, 1].set_ylabel("|f-fref| / (1+|fref|)")
    failed = int(frame["method_failed"].map(lambda v: str(v).strip().lower() in {"1", "true", "yes"}).sum())
    fig.suptitle(
        f"Numerical diagnostics on every stored row. method_failed rows = {failed}. Slow rows are kept.\n"
        "The log floor is only a display device. Exact zeros remain zeros in the tables.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
