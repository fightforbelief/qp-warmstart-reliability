"""Cluster bootstrap over independent source groups, not over timesteps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_medians(values: np.ndarray, groups: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"v": values, "g": groups})
    return frame.groupby("g", sort=True)["v"].median()


def median_of_group_medians(values: np.ndarray, groups: np.ndarray) -> float:
    med = group_medians(values, groups)
    if med.empty:
        return float("nan")
    return float(np.median(med.to_numpy()))


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> dict:
    """Percentile interval for the median of per-group medians.

    Resamples groups with replacement. Repeats inside a group do not
    become extra independent units. With very few groups the interval is
    descriptive, not a precise confidence statement.
    """
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    point = median_of_group_medians(values, groups)
    uniq = np.array(sorted(set(groups.tolist())))
    if uniq.size == 0:
        return {"estimate": point, "low": point, "high": point, "n_groups": 0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    grouped = {g: values[groups == g] for g in uniq}
    for b in range(n_boot):
        draw = rng.choice(uniq, size=uniq.size, replace=True)
        parts_v = []
        parts_g = []
        for i, g in enumerate(draw):
            parts_v.append(grouped[g])
            parts_g.append(np.full(grouped[g].shape[0], i))
        stats[b] = median_of_group_medians(np.concatenate(parts_v), np.concatenate(parts_g))
    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return {
        "estimate": point,
        "low": low,
        "high": high,
        "n_groups": int(uniq.size),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "note": "cluster bootstrap over groups; few clusters make this exploratory",
    }
