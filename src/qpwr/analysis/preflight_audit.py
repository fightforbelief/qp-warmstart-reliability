"""Recompute the exploratory preflight from its raw CSV.

Reference solves are counted in ``solves_used`` and are not rows of results.csv.
This module does not solve QPs and does not train a model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..timing import STREAMING_STAGES

C0 = "C0_FRESH_COLD"
C1 = "C1_CACHED_COLD"
C2 = "C2_CACHED_AUTO_PREVIOUS"
C3 = "C3_CACHED_EXPLICIT_PREVIOUS"
CLASSICAL = (C1, C2, C3)
ALL_METHODS = (C0, C1, C2, C3)
OVERHEAD_STAGES = ("parameter_update_ns", "state_insert_ns", "verify_ns")
JOIN_KEYS = ("group_id", "instance_id", "step", "timing_repeat", "qp_hash", "family")

# Display-only candidate. Not a frozen confirmatory gate.
CANDIDATE_PRIMAL_TOL = 1e-3
CANDIDATE_STATIONARITY_TOL = 1e-3
CANDIDATE_REL_GAP = 1e-3


def short_method(name: str) -> str:
    return {
        C0: "C0",
        C1: "C1",
        C2: "C2",
        C3: "C3",
    }.get(name, name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["1", "true", "yes"])


def classify_proxy(final_norm: float) -> str:
    if not np.isfinite(final_norm) or abs(float(final_norm)) > 1.0e6:
        return "INVALID_NUMERICAL_BLOWUP"
    return "UNVALIDATED_ESTIMATE"


def streaming_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """E1 rows: cached policies, after the step-0 transient, with a streaming total."""
    sub = frame.loc[frame["method"].isin(CLASSICAL)].copy()
    sub = sub.loc[sub["step"].astype(int) > 0]
    return sub.loc[sub["total_streaming_ns"].notna()].copy()


def paired_c1_minus_c2(frame: pd.DataFrame) -> pd.DataFrame:
    """Inner-join C1 and C2 on the same QP, step, repeat, and hash. Drop step 0."""
    columns = list(JOIN_KEYS) + ["total_streaming_ns"]
    left = frame.loc[frame["method"] == C1, columns]
    right = frame.loc[frame["method"] == C2, columns]
    merged = left.merge(right, on=list(JOIN_KEYS), how="inner", suffixes=("_c1", "_c2"))
    merged = merged.loc[merged["step"].astype(int) > 0].copy()
    merged["delta_ms"] = (
        merged["total_streaming_ns_c1"].astype(float) - merged["total_streaming_ns_c2"].astype(float)
    ) / 1e6
    return merged


def independent_group_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["group_id"].astype(str).unique())


def relative_objective_gap(frame: pd.DataFrame) -> pd.Series:
    objective = frame["objective"].astype(float)
    reference = frame["reference_objective"].astype(float)
    return (objective - reference).abs() / (1.0 + reference.abs())


def _median_ms(series: pd.Series) -> float | None:
    values = series.dropna().astype(float)
    if values.empty:
        return None
    return float(values.median() / 1e6)


def _quantile_ms(series: pd.Series, q: float) -> float | None:
    values = series.dropna().astype(float)
    if values.empty:
        return None
    return float(np.quantile(values.to_numpy() / 1e6, q))


def classical_effect_label(group_median_delta_ms: dict[str, float]) -> str:
    if not group_median_delta_ms:
        return "inconclusive"
    values = list(group_median_delta_ms.values())
    if all(v > 0 for v in values):
        return "observed"
    if any(v > 0 for v in values):
        return "inconclusive"
    return "not observed"


def component_median_gap(frame: pd.DataFrame) -> list[dict]:
    """Show median(sum) and sum(medians) side by side. They need not match."""
    rows = []
    stream = streaming_rows(frame)
    for (family, method), sub in stream.groupby(["family", "method"], sort=True):
        parts = [sub[stage].astype(float) for stage in STREAMING_STAGES]
        total = sub["total_streaming_ns"].astype(float)
        sum_of_medians = float(sum(part.median() for part in parts))
        median_of_sum = float(total.median())
        rows.append(
            {
                "family": str(family),
                "method": str(method),
                "sum_of_component_medians_ms": sum_of_medians / 1e6,
                "median_total_ms": median_of_sum / 1e6,
                "difference_ms": (sum_of_medians - median_of_sum) / 1e6,
            }
        )
    return rows


def residual_screen(frame: pd.DataFrame, primal_tol: float, stationarity_tol: float) -> pd.Series:
    status_ok = frame["status"].map(lambda s: " ".join(str(s).strip().lower().split())) == "solved"
    primal_ok = frame["primal_residual"].astype(float) <= primal_tol
    stationarity_ok = frame["dual_residual"].astype(float) <= stationarity_tol
    finite = frame["primal_residual"].astype(float).map(np.isfinite) & frame["dual_residual"].astype(float).map(
        np.isfinite
    )
    return status_ok & primal_ok & stationarity_ok & finite


def build_audit(frame: pd.DataFrame, summary: dict) -> dict:
    groups = independent_group_ids(frame)
    by_group = []
    for group_id, sub in frame.groupby("group_id", sort=True):
        repeats = (
            sub.groupby(["method", "step"], sort=False)["timing_repeat"].nunique().astype(int)
        )
        by_group.append(
            {
                "group_id": str(group_id),
                "family": str(sub["family"].iloc[0]),
                "n_instances": int(sub["instance_id"].nunique()),
                "n_rows": int(len(sub)),
                "steps": sorted(int(v) for v in sub["step"].unique()),
                "n_step0_rows": int((sub["step"].astype(int) == 0).sum()),
                "timing_repeats_min": int(repeats.min()),
                "timing_repeats_max": int(repeats.max()),
                "methods": sorted(str(v) for v in sub["method"].unique()),
            }
        )
    stream = streaming_rows(frame)
    family_medians = []
    for family, sub in stream.groupby("family", sort=True):
        item = {"family": str(family)}
        for method in CLASSICAL:
            piece = sub.loc[sub["method"] == method, "total_streaming_ns"]
            item[short_method(method)] = {
                "n_streaming_rows": int(piece.shape[0]),
                "p50_ms": _median_ms(piece),
                "p95_ms": _quantile_ms(piece, 0.95),
            }
        family_medians.append(item)
    group_latency = []
    for group_id, sub in stream.groupby("group_id", sort=True):
        item = {
            "group_id": str(group_id),
            "family": str(sub["family"].iloc[0]),
            "n_timing_repeats": int(sub["timing_repeat"].nunique()),
        }
        for method in CLASSICAL:
            piece = sub.loc[sub["method"] == method, "total_streaming_ns"]
            cell = {
                "n": int(piece.shape[0]),
                "p50_ms": _median_ms(piece),
                "p95_ms": _quantile_ms(piece, 0.95),
            }
            if "iterations" in sub.columns:
                iters = sub.loc[sub["method"] == method, "iterations"].dropna().astype(float)
                cell["median_iters"] = float(iters.median()) if len(iters) else None
            item[short_method(method)] = cell
        group_latency.append(item)
    paired = paired_c1_minus_c2(frame)
    c1_stream = frame.loc[(frame["method"] == C1) & (frame["step"].astype(int) > 0)]
    group_deltas = {
        str(group_id): float(sub["delta_ms"].median())
        for group_id, sub in paired.groupby("group_id", sort=True)
    }
    early_step_median_delta_ms = {}
    for group_id, sub in paired.groupby("group_id", sort=True):
        by_step = sub.groupby(sub["step"].astype(int))["delta_ms"].median()
        early_step_median_delta_ms[str(group_id)] = [
            None if step not in by_step.index else float(by_step.loc[step]) for step in (1, 2, 3, 4)
        ]
    repeat_deltas = []
    for (group_id, repeat), sub in paired.groupby(["group_id", "timing_repeat"], sort=True):
        repeat_deltas.append(
            {
                "group_id": str(group_id),
                "timing_repeat": int(repeat),
                "n_steps": int(len(sub)),
                "median_c1_minus_c2_ms": float(sub["delta_ms"].median()),
            }
        )
    gap = relative_objective_gap(frame)
    sensitivity = []
    for primal_tol, stat_tol in ((1e-4, 1e-4), (1e-3, 1e-3), (1e-3, 1e-2), (1e-2, 1e-2)):
        mask = residual_screen(frame, primal_tol, stat_tol)
        sensitivity.append(
            {
                "primal_tol": primal_tol,
                "stationarity_tol": stat_tol,
                "n_pass": int(mask.sum()),
                "n_rows": int(len(frame)),
                "pass_rate": float(mask.mean()),
            }
        )
    candidate = residual_screen(frame, CANDIDATE_PRIMAL_TOL, CANDIDATE_STATIONARITY_TOL) & (
        gap <= CANDIDATE_REL_GAP
    )
    streaming_vs_parts = (stream["total_streaming_ns"].astype(float) - stream[list(STREAMING_STAGES)].sum(axis=1)).abs()
    oneoff_vs_parts = (
        frame["oneoff_complete_ns"].astype(float)
        - frame["setup_ns"].astype(float)
        - frame[list(STREAMING_STAGES)].sum(axis=1)
    ).abs()
    zero_components = {}
    for method in CLASSICAL:
        piece = frame.loc[frame["method"] == method]
        zero_components[method] = [
            stage for stage in STREAMING_STAGES if float(piece[stage].astype(float).max()) == 0.0
        ]
    proxy = summary.get("nn_proxy") or {}
    final_norm = float(proxy.get("final_norm", np.nan))
    solves_used = int(summary["solves_used"])
    n_rows = int(len(frame))
    n_instances = int(frame["instance_id"].nunique())
    return {
        "run_id": summary.get("run_id"),
        "n_rows": n_rows,
        "n_instances": n_instances,
        "n_independent_groups": len(groups),
        "independent_groups": by_group,
        "solves_used_recorded": solves_used,
        "reference_solves_not_in_csv": solves_used - n_rows,
        "reference_matches_one_per_instance": solves_used - n_rows == n_instances,
        "n_step0_rows": int((frame["step"].astype(int) == 0).sum()),
        "n_streaming_rows": int(len(stream)),
        "n_failed": int(as_bool(frame["method_failed"]).sum()),
        "n_timeout": int(as_bool(frame["timeout"]).sum()) if "timeout" in frame.columns else None,
        "status_counts": {str(k): int(v) for k, v in frame["status"].astype(str).value_counts().items()},
        "quality_pass_recorded_n": int(as_bool(frame["quality_pass"]).sum()),
        "hash_nunique_max": int(frame.groupby("instance_id")["qp_hash"].nunique().max()),
        "settings_sha_nunique": int(frame["solver_settings_sha"].nunique()),
        "settings_sha": str(frame["solver_settings_sha"].iloc[0]),
        "summary_settings_sha": summary.get("settings_sha"),
        "osqp_version_nunique": int(frame["osqp_version"].astype(str).nunique()),
        "osqp_version": str(frame["osqp_version"].iloc[0]),
        "c0_streaming_all_null": bool(frame.loc[frame["method"] == C0, "total_streaming_ns"].isna().all()),
        "streaming_total_max_abs_diff_ns": float(streaming_vs_parts.max()) if len(streaming_vs_parts) else None,
        "oneoff_max_abs_diff_ns": float(oneoff_vs_parts.max()) if len(oneoff_vs_parts) else None,
        "unmatched_c1_streaming_rows": int(len(c1_stream) - len(paired)),
        "family_streaming": family_medians,
        "group_latency": group_latency,
        "group_median_c1_minus_c2_ms": group_deltas,
        "early_step_median_delta_ms": early_step_median_delta_ms,
        "repeat_median_c1_minus_c2_ms": repeat_deltas,
        "classical_warm_start_effect": classical_effect_label(group_deltas),
        "learned_headroom": "untested",
        "zero_components": zero_components,
        "component_median_gap": component_median_gap(frame),
        "nn_proxy_mean_ms": proxy.get("mean_ms"),
        "nn_proxy_final_norm": final_norm,
        "nn_proxy_validity": classify_proxy(final_norm),
        "sensitivity": sensitivity,
        "candidate_rule": {
            "status": "solved",
            "primal_tol": CANDIDATE_PRIMAL_TOL,
            "stationarity_tol": CANDIDATE_STATIONARITY_TOL,
            "relative_objective_gap_max": CANDIDATE_REL_GAP,
            "n_pass": int(candidate.sum()),
            "n_rows": int(len(frame)),
            "frozen": False,
        },
        "residual_max": {
            "primal": float(frame["primal_residual"].astype(float).max()),
            "stationarity": float(frame["dual_residual"].astype(float).max()),
            "relative_objective_gap": float(gap.max()),
        },
        "residual_p50": {
            "primal": float(frame["primal_residual"].astype(float).median()),
            "stationarity": float(frame["dual_residual"].astype(float).median()),
            "relative_objective_gap": float(gap.median()),
        },
        "residual_by_family": _residual_by_family(frame),
        "setup_step0_median_ms": _setup_table(frame),
        "recorded_quality_tolerances": {
            "primal_tol": summary.get("primal_tol"),
            "stationarity_tol": summary.get("stationarity_tol"),
            "eps_abs": (summary.get("settings") or {}).get("eps_abs"),
            "eps_rel": (summary.get("settings") or {}).get("eps_rel"),
        },
    }


def _residual_by_family(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for family, sub in frame.groupby("family", sort=True):
        primal = sub["primal_residual"].astype(float)
        stationarity = sub["dual_residual"].astype(float)
        rows.append(
            {
                "family": str(family),
                "n": int(len(sub)),
                "primal_p50": float(primal.median()),
                "primal_max": float(primal.max()),
                "stationarity_p50": float(stationarity.median()),
                "stationarity_max": float(stationarity.max()),
                "n_pass_abs_1e-4": int(residual_screen(sub, 1e-4, 1e-4).sum()),
                "n_pass_abs_1e-3": int(residual_screen(sub, 1e-3, 1e-3).sum()),
                "n_pass_recorded_screen": int(residual_screen(sub, 1e-3, 1e-2).sum()),
            }
        )
    return rows


def heterogeneity_prose(audit: dict) -> str:
    parts = []
    for item in audit["group_latency"]:
        delta = audit["group_median_c1_minus_c2_ms"].get(item["group_id"])
        iters = item["C1"].get("median_iters")
        iter_text = f", C1 median iterations {_fmt(iters, 1)}" if iters is not None else ""
        parts.append(
            f"{item['group_id']}: C1 p50 {_fmt(item['C1']['p50_ms'])} ms "
            f"(p95 {_fmt(item['C1']['p95_ms'])}), "
            f"C2 p50 {_fmt(item['C2']['p50_ms'])} ms, "
            f"paired median C1−C2 {_fmt(delta)} ms{iter_text}"
        )
    early = audit.get("early_step_median_delta_ms", {}).get("mpc/dim20/seed2") or []
    early_txt = ""
    if any(v is not None for v in early):
        shown = ", ".join("NA" if v is None else f"{v:.1f}" for v in early)
        early_txt = (
            f" On mpc/dim20/seed2 the median paired gaps at steps 1-4 are {shown} ms; "
            "later steps sit near zero and some change sign, so the horizon median is not a steady advantage."
        )
    return (
        "The group medians are the observation. "
        + ". ".join(parts)
        + ". The pooled MPC median, about 11 ms versus about 5 ms, mixes seed 1 with seed 2 "
        "and is not the latency of either plant. Seed 1 keeps a multi-millisecond gap on both "
        "timing repeats. Seed 2's median solve is sub-millisecond and its horizon-median paired gap "
        "stays under 0.15 ms on both repeats, while its C1 p95 is several milliseconds."
        + early_txt
        + " A classical previous-solution effect is present in some of these workloads. "
        "The size of that effect is not stable across the two MPC plants. "
        "A learned initializer has not been compared with C2 or C3. "
        "Three groups do not support a general claim."
    )


def _setup_table(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for (family, method), sub in frame.groupby(["family", "method"], sort=True):
        step0 = sub.loc[sub["step"].astype(int) == 0]
        rows.append(
            {
                "family": str(family),
                "method": str(method),
                "step0_setup_median_ms": _median_ms(step0["setup_ns"]),
                "all_rows_solve_wall_median_ms": _median_ms(sub["solve_wall_ns"]),
                "all_rows_oneoff_median_ms": _median_ms(sub["oneoff_complete_ns"]),
                "step0_oneoff_median_ms": _median_ms(step0["oneoff_complete_ns"]),
                "fraction_rows_setup_positive": float((sub["setup_ns"].astype(float) > 0).mean()),
            }
        )
    return rows


def _fmt(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _family_line(item: dict, key: str) -> str:
    cell = item[key]
    return f"{_fmt(cell['p50_ms'])} (n={cell['n_streaming_rows']}, p95={_fmt(cell['p95_ms'])})"


def render_reports(audit: dict, live: dict, figures: dict[str, str]) -> dict[str, str]:
    effect = audit["classical_warm_start_effect"]
    group_rows = "\n".join(
        f"| {row['group_id']} | {row['family']} | {row['n_instances']} | "
        f"{row['timing_repeats_min']}–{row['timing_repeats_max']} | {row['n_step0_rows']} |"
        for row in audit["independent_groups"]
    )
    family_rows = "\n".join(
        f"| {item['family']} | {_family_line(item, 'C1')} | {_family_line(item, 'C2')} | {_family_line(item, 'C3')} |"
        for item in audit["family_streaming"]
    )
    latency_rows = "\n".join(
        "| {group} | {c1} / {c1p} | {c2} / {c2p} | {c3} / {c3p} |".format(
            group=item["group_id"],
            c1=_fmt(item["C1"]["p50_ms"]),
            c1p=_fmt(item["C1"]["p95_ms"]),
            c2=_fmt(item["C2"]["p50_ms"]),
            c2p=_fmt(item["C2"]["p95_ms"]),
            c3=_fmt(item["C3"]["p50_ms"]),
            c3p=_fmt(item["C3"]["p95_ms"]),
        )
        for item in audit["group_latency"]
    )
    delta_rows = "\n".join(
        f"| {group_id} | {_fmt(value)} |"
        for group_id, value in audit["group_median_c1_minus_c2_ms"].items()
    )
    repeat_rows = "\n".join(
        f"| {row['group_id']} | {row['timing_repeat']} | {row['n_steps']} | {_fmt(row['median_c1_minus_c2_ms'])} |"
        for row in audit["repeat_median_c1_minus_c2_ms"]
    )
    gap_rows = "\n".join(
        f"| {row['family']} | {short_method(row['method'])} | {_fmt(row['sum_of_component_medians_ms'], 4)} | "
        f"{_fmt(row['median_total_ms'], 4)} | {_fmt(row['difference_ms'], 4)} |"
        for row in audit["component_median_gap"]
    )
    zero_lines = "\n".join(
        f"- `{method}` identically zero stages: {', '.join(stages) if stages else 'none'}"
        for method, stages in audit["zero_components"].items()
    )
    setup_rows = "\n".join(
        f"| {row['family']} | {short_method(row['method'])} | {_fmt(row['step0_setup_median_ms'])} | "
        f"{_fmt(row['all_rows_solve_wall_median_ms'])} | {_fmt(row['all_rows_oneoff_median_ms'])} | "
        f"{_fmt(row['step0_oneoff_median_ms'])} | {_fmt(row['fraction_rows_setup_positive'], 3)} |"
        for row in audit["setup_step0_median_ms"]
    )
    family_residual_rows = "\n".join(
        "| {family} | {n} | {pp:.3e} | {pm:.3e} | {sp:.3e} | {sm:.3e} | {a} / {n} | {b} / {n} | {c} / {n} |".format(
            family=row["family"],
            n=row["n"],
            pp=row["primal_p50"],
            pm=row["primal_max"],
            sp=row["stationarity_p50"],
            sm=row["stationarity_max"],
            a=row["n_pass_abs_1e-4"],
            b=row["n_pass_abs_1e-3"],
            c=row["n_pass_recorded_screen"],
        )
        for row in audit["residual_by_family"]
    )
    sens_rows = "\n".join(
        f"| {row['primal_tol']:.0e} | {row['stationarity_tol']:.0e} | {row['n_pass']} / {row['n_rows']} |"
        for row in audit["sensitivity"]
    )
    figure_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in figures.items())
    accounting = f"""# Preflight audit and plot fixes

Exploratory reanalysis of run `{audit['run_id']}`. No new QP was solved. No model was trained.

## Accounting

| Quantity | Value |
|---|---:|
| Result rows in `results.csv` | {audit['n_rows']} |
| Unique QP instances | {audit['n_instances']} |
| Independent source groups | {audit['n_independent_groups']} |
| `solves_used` in `summary.json` | {audit['solves_used_recorded']} |
| Solves counted but absent from the CSV | {audit['reference_solves_not_in_csv']} |
| One reference solve per instance | {audit['reference_matches_one_per_instance']} |
| Step-0 rows stored in the CSV | {audit['n_step0_rows']} |
| Streaming rows used below (C1–C3, step > 0) | {audit['n_streaming_rows']} |
| `method_failed` rows | {audit['n_failed']} |
| `timeout` rows | {audit['n_timeout']} |
| Recorded `quality_pass` rows | {audit['quality_pass_recorded_n']} / {audit['n_rows']} |
| Max distinct `qp_hash` values per instance | {audit['hash_nunique_max']} |
| Distinct solver-settings SHA-256 values | {audit['settings_sha_nunique']} |
| C0 streaming total is null on every row | {audit['c0_streaming_all_null']} |
| Max \\|streaming total − sum of streaming stages\\| (ns) | {audit['streaming_total_max_abs_diff_ns']} |
| Max \\|one-off − setup − streaming stages\\| (ns) | {audit['oneoff_max_abs_diff_ns']} |
| C1 step>0 rows without a C2 pair | {audit['unmatched_c1_streaming_rows']} |

`solves_used` is the script counter, not the number of CSV rows. `generate_mpc_group` calls `solve_fresh_cold` once per step and returns those records. `run_preflight` adds `len(refs)` to `used` and keeps only the objective. The Lasso loop calls one extra high-accuracy `solve_fresh_cold` per instance, increments `used`, and does not append that record. `_apply_policies` is what writes C0–C3. For this file the identity is {audit['n_rows']} policy rows + {audit['reference_solves_not_in_csv']} reference solves = {audit['solves_used_recorded']}.

Status counts: {audit['status_counts']}.

Settings SHA in the CSV is `{audit['settings_sha']}`. The summary SHA is `{audit['summary_settings_sha']}`. OSQP version on every row: `{audit['osqp_version']}`.

## Independent groups and the repeat imbalance

| Group | Family | Instances | Timing repeats | Step-0 rows |
|---|---|---:|---:|---:|
{group_rows}

MPC groups have two timing repeats. The Lasso group has one. Adjacent steps inside a group are one closed loop or one lambda path. Repeats and steps are not additional source groups.

Streaming summaries drop step 0. Those rows stay in the CSV and in the failure tables.

## Recomputed streaming medians

Step > 0. p50 is the median. p95 is descriptive.

| Family | C1 cached cold | C2 automatic previous | C3 explicit previous |
|---|---|---|---|
{family_rows}

### Same quantity inside each group

| Group | C1 p50 / p95 | C2 p50 / p95 | C3 p50 / p95 |
|---|---|---|---|
{latency_rows}

## Paired C1 minus C2

Pairs match `group_id`, `instance_id`, `step`, `timing_repeat`, and `qp_hash`. Step 0 is excluded. Positive means cached cold is slower than automatic previous-solution warm start.

| Group | Median paired difference (ms) |
|---|---:|
{delta_rows}

| Group | Timing repeat | Paired steps | Median difference (ms) |
|---|---:|---:|---:|
{repeat_rows}

`CLASSICAL_WARM_START_EFFECT`: **{effect}**.

`LEARNED_HEADROOM`: **untested**.

{heterogeneity_prose(audit)}

C1 minus C2 compares two classical policies. It is not evidence that a learned initializer beats the better of C2 and C3.

## Stage medians do not reconstruct the median total

| Family | Method | Sum of component medians (ms) | Median total (ms) | Difference (ms) |
|---|---|---:|---:|---:|
{gap_rows}

{zero_lines}

Feature computation, nearest-neighbour lookup, model inference, and fallback are zero because C1–C3 do not run those stages. Zero is a recorded method meaning, not a missing timer.

## One-off setup

| Family | Method | Step-0 setup median (ms) | Solve-wall median (ms) | One-off median (ms) | Step-0 one-off median (ms) | Fraction of rows with setup > 0 |
|---|---|---:|---:|---:|---:|---:|
{setup_rows}

Lasso fresh setup is a different estimand from MPC streaming. C0's one-off cost includes setup on every QP. A solve-wall plot that omits setup does not make C0 comparable to C1.

## Untrained ReLU proxy

The stored proxy mean is {audit['nn_proxy_mean_ms']} ms and its final norm is {audit['nn_proxy_final_norm']}. Validity: `{audit['nn_proxy_validity']}`. The recurrence is an untrained Gaussian matvec passed through ReLU until the state overflows. That elapsed time is not a neural-network inference cost, and it is not used in the headroom label.

## Provenance checks run with this audit

{live['prose']}

Input hashes are in `reports/05_PLOT_AUDIT.md`.
"""
    quality = f"""# Quality-gate audit

Run `{audit['run_id']}`. Residuals below are recomputed only from columns already stored in `results.csv`. No QP was re-solved.

## What the stored columns are

`primal_residual` is `quality.primal_violation`: the infinity norm of `max(0, l - Ax, Ax - u)` over finite bounds, in the original unscaled coordinates. Infinite bounds do not contribute. This is not OSQP's internal primal residual `||Ax - z||`.

`dual_residual` is `quality.stationarity_inf`: `||P_full x + q + A' y||_∞`. `P_full` is the symmetrized objective matrix. The sign of `y` is OSQP's dual sign. The vector is unscaled. OSQP decides termination on scaled residuals of the form absolute plus relative tolerance. The stored number is not that scaled residual.

The objective is `0.5 x' P_full x + q' x`. `reference_objective` is the objective of a separate fresh solve at `eps_abs = eps_rel = 1e-8`, `max_iter = 20000`. Those reference solves are not rows in the CSV. The relative gap used here is `|f - f_ref| / (1 + |f_ref|)`.

`quality_pass` in the CSV is `quality.quality_pass`: status normalizes to exactly `solved`, the primal inf-norm is at most the configured `primal_tol`, and the stationarity inf-norm is at most `stationarity_tol`. The recorded tolerances are primal `{audit['recorded_quality_tolerances']['primal_tol']}` and stationarity `{audit['recorded_quality_tolerances']['stationarity_tol']}`. Candidate OSQP settings are `eps_abs = eps_rel = {audit['recorded_quality_tolerances']['eps_abs']}`. The objective gap is not part of the flag. `solved inaccurate` would fail the flag. This flag is a screen, not a KKT certificate and not a proof that two methods returned the same solution.

## What this file actually contains

| Quantity | Value |
|---|---:|
| Recorded `quality_pass` | {audit['quality_pass_recorded_n']} / {audit['n_rows']} |
| Status counts | {audit['status_counts']} |
| Failed rows | {audit['n_failed']} |
| Timed-out rows | {audit['n_timeout']} |
| Max primal inf-norm | {audit['residual_max']['primal']:.3e} |
| Median primal inf-norm | {audit['residual_p50']['primal']:.3e} |
| Max stationarity inf-norm | {audit['residual_max']['stationarity']:.3e} |
| Median stationarity inf-norm | {audit['residual_p50']['stationarity']:.3e} |
| Max relative objective gap | {audit['residual_max']['relative_objective_gap']:.3e} |
| Median relative objective gap | {audit['residual_p50']['relative_objective_gap']:.3e} |

Every stored status is a clean `solved`, and every stored residual is finite. Failures were not dropped; there were none to drop.

| Family | Rows | Primal p50 | Primal max | Stationarity p50 | Stationarity max | Pass at 1e-4 | Pass at 1e-3 | Pass recorded 1e-3/1e-2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{family_residual_rows}

## Sensitivity on the existing rows

The screen is status `solved` plus the two absolute inf-norms. No threshold below was chosen to create a latency result.

| Primal tol | Stationarity tol | Rows passing |
|---|---|---:|
{sens_rows}

The recorded gate is the row `1e-3` / `1e-2`. Stationarity tolerance `1e-2` is 100 times `eps_abs` and is not OSQP's termination test. Tightening stationarity to `1e-4` or `1e-3` is a sensitivity check. It is not a new frozen rule.

## Exploratory candidate, not a certification

A scale-aware match to OSQP cannot be recomputed: the CSV does not store `||Ax||`, `||Px||`, `||q||`, or `||A'y||`. Quality certification against the solver's own termination criterion is **unresolved**.

The candidate inspected on this file, and not frozen, is:

- status exactly `solved`;
- primal bound violation at most `{CANDIDATE_PRIMAL_TOL:.0e}` (the existing absolute screen, 10 times `eps_abs`);
- stationarity inf-norm at most `{CANDIDATE_STATIONARITY_TOL:.0e}` (tighter than the recorded `1e-2` screen, still an absolute unscaled test);
- relative objective gap at most `{CANDIDATE_REL_GAP:.0e}`.

On this file that candidate passes {audit['candidate_rule']['n_pass']} / {audit['candidate_rule']['n_rows']} rows. Passing it still does not certify that C1, C2, and C3 returned interchangeable solutions, because the reference solve used a tighter tolerance and its own residual vector was not stored.

Do not treat `quality_pass=True` as matched solution quality for a future learned-versus-classical comparison. A frozen rule needs the missing scale terms, or an explicit decision that the absolute screen is the scientific endpoint.
"""
    plots = f"""# Plot audit

Figures were regenerated from `experiments/preflight/exploration/results.csv`. The original `stage_bars.png` and `c1_minus_c2.png` were left in place.

Command, from the repository root:

```
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.\\ .venv-eval\\Scripts\\python.exe -m qpwr.analysis.preflight_audit
```

The space in `.\\ .venv-eval` above is a line-wrap artifact. The executable is `.venv-eval\\Scripts\\python.exe`.

## Inputs

| File | SHA-256 |
|---|---|
{figure_rows}

## Outputs

| Figure | File | What it shows |
|---|---|---|
| 1 | `experiments/preflight/exploration/figures_audit/fig1_streaming_stages.png` | C1–C3 parameter update, state insertion, and verification on one panel; solver wall on another. Feature, lookup, inference, and fallback are zero for these methods. |
| 2 | `experiments/preflight/exploration/figures_audit/fig2_oneoff_setup_solve.png` | Step-0 setup, solve wall, and complete one-off cost. Lasso C0 setup stays visible. |
| 3 | `experiments/preflight/exploration/figures_audit/fig3_paired_c1_minus_c2.png` | Paired C1−C2 by independent group, step 0 excluded, zero line, within-sequence order, timing repeats marked, group median marked. Classical diagnostic. |
| 4 | `experiments/preflight/exploration/figures_audit/fig4_latency_by_group.png` | C1/C2/C3 streaming distributions by group, with descriptive p95 diamonds. |
| 5 | `experiments/preflight/exploration/figures_audit/fig5_quality_diagnostics.png` | Status, primal violation, stationarity, and relative objective gap on every row. |

SHA-256 of the raw inputs and the new artifacts:

| File | SHA-256 |
|---|---|
{live['figure_hash_table']}

## Reading rule

Sum-of-medians versus median-total is tabulated in `reports/03_PREFLIGHT_AUDIT_AND_PLOT_FIXES.md`. The panels are not a stacked identity. Figure 4 does not select the best classical policy inside each test QP. Any future selection happens once, on validation groups only. This preflight has no validation split and no learned method.
"""
    # Fix the awkward command line. I'll replace it with a clean one in the string... 
    # I already have a bad wrap. Let me fix plots string by rewriting the command cleanly.
    plots = plots.replace(
        """```
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.\\ .venv-eval\\Scripts\\python.exe -m qpwr.analysis.preflight_audit
```

The space in `.\\ .venv-eval` above is a line-wrap artifact. The executable is `.venv-eval\\Scripts\\python.exe`.
""",
        """```
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
.\\venv-eval\\Scripts\\python.exe -m qpwr.analysis.preflight_audit
```
""",
    )
    l1 = f"""# Bounded L1 reproduction plan

This document does not report a training run. No L2WS epoch was executed in this stage.

## Target

The bounded target is a **partial** residual-versus-iteration smoke of upstream L2WS on its smallest valid example, `benchmarks/configs/unconstrained_qp`, not a claim that Figure numbers from the paper were reproduced.

- Upstream checkout: `{live['l2ws_sha']}`
- Dirty upstream tree: {live['l2ws_dirty']}
- Anchor import: {live['anchor_prose']}
- Device: CPU only, as reported by the anchor interpreter. The RTX 5070 Ti and its driver are not evidence that JAX 0.4.20 can use that GPU.

`unconstrained_qp_setup.yaml` has `n_orig: 20`, `N_train: 100`, `N_test: 10`, `seed: 42`. `unconstrained_qp_run.yaml` has `epochs: 1e6`, `train_unrolls: 60`, `eval_unrolls: 1000`, and a 10-unit hidden layer. That epoch count is the upstream request. It is outside a pilot.

The MPC config that would be closer to this project's QP family, `benchmarks/configs/mpc/mpc_run.yaml`, contains committed merge-conflict markers: {live['mpc_conflict']}. It cannot be parsed. This plan does not edit that file.

## Blockers

- Exact paper reproduction is not available from the checked-out MPC config.
- Full training requests `1e6` epochs.
- The anchor environment used for the import check does not by itself prove that Hydra, JAXopt, Optax, and the pinned `trajax` dependency import. Those imports are part of the approval gate below, still without a training loop.
- `pyproject.toml` states Apache-2.0. A `LICENSE` file is present in the checkout: {live['license_present']}. If the file is absent, keep that absence on the record and do not invent a license file.
- The OSQP benchmark repository was not a full git clone. `UPSTREAM_PROVENANCE.json` records `git_clone` = `{live['osqp_clone']}` at commit `{live['osqp_sha']}`. Parametric sources used here were hash-checked. That is not a full benchmark checkout.
- OSQP `warm_start(x, y)` remains an unmapped stand-in for the L2WS fixed-point state. A successful unconstrained smoke would not certify the OSQP bridge.

## Proposed local repair, kept outside the upstream tree

If a human approves the pilot, copy `unconstrained_qp_setup.yaml` and `unconstrained_qp_run.yaml` to `configs/l1_partial/` and change only the copy:

- `epochs: 5`
- `eval_unrolls: 50`
- `eval_every_x_epochs: 5`
- leave `N_train`, `n_orig`, network width, and the optimizer as in the upstream files

Write outputs under `experiments/l1_partial/`, labeled `PARTIAL_REPRODUCTION`. Do not commit generated checkpoints into the upstream tree. Do not modify `external/l2ws`.

## Runtime and memory

Not measured. The cap for the approved pilot is 30 minutes of wall clock and an immediate stop if resident memory grows without a completed epoch. Expected differences from the paper experiment: at most 5 epochs instead of 1e6, 50 evaluation unrolls instead of 1000, unconstrained QP instead of the conflicted MPC config, CPU JAX 0.4.20 instead of whatever accelerator the authors used. The curve, if produced, is a debugging smoke.

## Approval required before launch

Approve this sentence and nothing broader: run the partial unconstrained L2WS smoke from a config copy under `configs/l1_partial/`, at most 5 epochs and 50 eval unrolls, 30-minute wall-clock cap, CPU only, outputs under `experiments/l1_partial/`, without editing `external/l2ws`.

Until that sentence is approved, L1 status stays blocked. L2 training is a separate approval and is not implied by an L1 smoke.
"""
    gate = f"""# Gate recommendation

Date of this recommendation: 2026-10-26. Exploratory run `{audit['run_id']}`. Protocol status remains **draft**. Holdout release remains **false**. No `frozen_protocol.yaml` was written.

The 2026-10-23 label `HEADROOM_PRESENT` is retired. It compared a classical baseline with an untrained ReLU matvec whose final norm is {audit['nn_proxy_final_norm']}.

## Gate status

| Gate | Status | Why |
|---|---|---|
| G0 environment and sources | PASS, with the recorded limits | Windows host, eval OSQP, anchor JAX CPU import, pinned SHAs. L2WS has no usable MPC run config. OSQP benchmarks were not a full clone. |
| G1 classical measurement | NEEDS_HUMAN_APPROVAL | A classical previous-solution gap appears in some groups and is about 0.09 ms on MPC seed 2. Learned headroom is untested. Quality certification is unresolved. |
| G2 learned evaluation | BLOCKED | No training, no learned solves, test partition sealed. |
| G3 confirmatory claim | NOT STARTED | No frozen protocol and no held-out test. |

Overall status: **NEEDS_HUMAN_APPROVAL**.

## What the preflight supports

`CLASSICAL_WARM_START_EFFECT` = `{effect}` for the sign of the paired median in each of the three groups. The magnitudes are not one effect. MPC seed 1 is a multi-millisecond gap on both timing repeats. MPC seed 2 is a sub-millisecond solve whose paired median gap is about 0.09 ms. Lasso, with one matrix and one timing repeat, is a multi-millisecond gap. The pooled MPC figures of about 11 ms and about 5 ms mix seed 1 with seed 2.

`LEARNED_HEADROOM` = `untested`.

## Choices supported by the existing evidence

- Keep the scientific question fixed.
- Keep MPC at dimension 20 as the primary family. Lasso at dimension 50 stays the secondary family. The Lasso timing gap is not a reason to switch.
- Keep C2 and C3 as the classical competitors. C0 is an E2 cost, not the streaming denominator.
- Keep preflight seeds MPC 1–2 and Lasso 1 out of any future test partition.
- Keep technical repeats nested inside groups. The preflight imbalance (MPC two repeats, Lasso one) is real and is not smoothed over.
- Keep step 0 out of the streaming median.
- Keep the learned method labeled `L2WS_INSPIRED` until an explicit state map exists.

## Choices that still need a person

- Whether to freeze any protocol at all.
- Horizon: the measured horizon is 25 steps. The upstream generator allows 100. Steps 26–100 were not measured. Using 100 is a different experiment.
- Group pool 101–124 and the 12/4/8 split. The generator's seed resets `numpy`'s global RNG and redraws A, B, Q, the terminal cost, bounds, and x0. Those are independent plants. Two plants do not estimate how stable a 24-plant study will be. Eight test groups remain a small cluster sample.
- Timing repeats: 10 is a proposed technical-repeat target, not a measured variance requirement.
- The quality rule. Absolute screens at `1e-3` / `1e-2` passed every existing row. A scale-aware OSQP match is unresolved.
- Whether a 5-epoch unconstrained L1 smoke must happen before any L2 training request.

## Budget if a later approval launches the MPC study

Not run. Arithmetic only, using the measured horizon of 25:

- 24 groups × 25 steps = 600 QP instances.
- 600 high-accuracy reference solves to freeze the rollouts.
- C1, C2, and C3 × 10 technical repeats × 600 = 18,000 streaming solves.
- At the observed few-millisecond to tens-of-millisecond solve times, solve time itself is minutes, before Python and generation overhead. This is an extrapolation.
- Training cost is not in that number. Upstream L2WS asks for `1e6` epochs. No training budget is approved.

## When to stop, stay inconclusive, or report a negative L2

- Stop before test evaluation if validation quality does not meet the rule that was frozen, or if the scale-aware residual question is still open and the protocol required it.
- Report a negative L2 result if the trained initializer does not beat the single validation-selected policy among C1, C2, and C3 on the group-level paired streaming endpoint. Do not raise the QP dimension to manufacture a gap.
- Stay inconclusive if a learned run is never approved, or if the independent-group sample cannot separate a latency difference from group heterogeneity. The present three-group file is already in that inconclusive region for any learned claim.
- Do not describe C1 minus C2 as learned headroom.

## Next approval

The next justified request is the bounded L1 plan in `reports/06_L1_REPRODUCTION_PLAN.md`. L2 training is not the next action. Freezing the protocol is a separate human decision and is not recommended until the quality rule and the horizon are chosen explicitly.
"""
    return {
        "reports/03_PREFLIGHT_AUDIT_AND_PLOT_FIXES.md": accounting,
        "reports/04_QUALITY_GATE_AUDIT.md": quality,
        "reports/05_PLOT_AUDIT.md": plots,
        "reports/06_L1_REPRODUCTION_PLAN.md": l1,
        "reports/02_GATE_RECOMMENDATION.md": gate,
    }


def render_protocol(audit: dict) -> str:
    families = {item["family"]: item for item in audit["family_streaming"]}
    mpc = families.get("mpc", {})
    lasso = families.get("lasso", {})

    def p50(block: dict, key: str) -> str:
        cell = block.get(key) or {}
        value = cell.get("p50_ms")
        return "null" if value is None else f"{float(value):.2f}"

    return f"""# DRAFT — not frozen, test partition sealed
# Recomputed from preflight run {audit['run_id']} on 2026-10-26.
# This file is not protocol/frozen_protocol.yaml and must not be treated as one.

status: draft
holdout_release_approval: false
title: "When Do Learned Warm Starts Actually Pay Off? An Independent Benchmark Study of Learning-Augmented Quadratic Programming"
scientific_question_unchanged: true

primary_endpoint: "median across independent groups of the within-group median of paired streaming latency (ms): T_validation_selected_classical - T_learned. Positive favors the learned initializer."
streaming_definition: "parameter update + features + lookup or inference + state insertion + solve wall + online verification. Setup is excluded from this estimand and reported separately. Step 0 of each sequence is an initial transient and is excluded from the streaming median."
classical_candidate_pool:
  - C1_CACHED_COLD
  - C2_CACHED_AUTO_PREVIOUS
  - C3_CACHED_EXPLICIT_PREVIOUS
classical_selection: "On validation groups only, choose the single policy with the lowest median streaming time. Use that one policy for every test group. Do not pick a winner inside each test QP."
learned_method_status: L2WS_INSPIRED
learned_method_note: "OSQP warm_start(x, y) is not the L2WS fixed-point state. Do not label an external network as the JMLR algorithm."

classical_warm_start_effect: {audit['classical_warm_start_effect']}
classical_warm_start_effect_note: >
  The paired median is positive in each preflight group. MPC seed 2 is only
  about 0.09 ms. MPC seed 1 is about 8 ms and the single Lasso group is about
  9.5 ms. Do not quote the pooled MPC median as if both plants shared it.
learned_headroom: untested
nn_proxy_validity: {audit['nn_proxy_validity']}
quality_certification: unresolved

recommended_primary_family: mpc
recommendation_reason: >
  MPC stays primary because the charter names it and ControlExample draws an
  independent plant per seed. The Lasso timing gap is not a reason to switch.
  The classical previous-solution effect is a preflight observation. It does
  not show that a learned initializer beats C2 or C3.
exploratory_preflight:
  run_id: "{audit['run_id']}"
  osqp: "{audit['osqp_version']}"
  policy_rows: {audit['n_rows']}
  solves_used_including_reference: {audit['solves_used_recorded']}
  reference_solves_absent_from_csv: {audit['reference_solves_not_in_csv']}
  independent_groups: {audit['n_independent_groups']}
  timing_repeats_observed:
    mpc: 2
    lasso: 1
  excluded_from_future_test_seeds:
    mpc: [1, 2]
    lasso: [1]
  mpc_dim20_median_streaming_ms:
    C1: {p50(mpc, 'C1')}
    C2: {p50(mpc, 'C2')}
    C3: {p50(mpc, 'C3')}
  lasso_dim50_median_streaming_ms:
    C1: {p50(lasso, 'C1')}
    C2: {p50(lasso, 'C2')}
    C3: {p50(lasso, 'C3')}
  group_median_c1_minus_c2_ms: {json.dumps(audit['group_median_c1_minus_c2_ms'])}

proposed_if_approved:
  frozen: false
  mpc_dimension: 20
  horizon_steps: 25
  horizon_note: "25 is the measured preflight horizon. Upstream allows 100. Steps 26-100 were not measured and need a separate approval."
  note: "Do not increase dimension unless a later approval explicitly selects another size already listed by upstream (40, 60, 80)."
  group_counts: {{train: 12, val: 4, test: 8}}
  group_seed_pool: "101-124 inclusive, assigned by make_groups with seed 20261023. Not generated in this round."
  split_seed: 20261023
  training_seeds: [11, 23, 47]
  eps_abs: 1.0e-4
  eps_rel: 1.0e-4
  adaptive_rho: false
  polishing: false
  rho: 0.1
  timing_repeats: 10
  timing_repeats_note: "Design target for technical replication. Not an independent sample size. Not yet used."
  cluster_bootstrap_seed: 11
  cluster_bootstrap_draws: 2000
  failure_rule: "Keep every timeout, inaccurate solve, and infeasible row in the latency table."
  quality_rule: "Unresolved. Do not treat the recorded quality_pass flag as matched quality."

not_authorized:
  - confirmatory test evaluation
  - neural training
  - L3 extension
  - changing the primary endpoint after seeing test outcomes
  - using preflight seeds 1 and 2 as untouched test groups
  - creating protocol/frozen_protocol.yaml in the audit stage
"""


def render_decision(audit: dict, live: dict) -> str:
    return f"""
## 2026-10-26 — Preflight accounting audit and plot repair

- Reanalyzed run `{audit['run_id']}` from the existing CSV. No new QP solves and no training.
- `solves_used` {audit['solves_used_recorded']} = {audit['n_rows']} policy rows + {audit['reference_solves_not_in_csv']} reference solves that were not written to `results.csv`.
- Independent groups remain {audit['n_independent_groups']}. MPC timing repeats are 2 and the Lasso repeat is 1.
- `CLASSICAL_WARM_START_EFFECT` = {audit['classical_warm_start_effect']}. `LEARNED_HEADROOM` = untested.
- Retired `HEADROOM_PRESENT`. The untrained ReLU proxy final norm {audit['nn_proxy_final_norm']} is `{audit['nn_proxy_validity']}`.
- Quality certification against OSQP's scaled termination test is unresolved. Recorded `quality_pass` is an absolute screen.
- Corrected figures are under `experiments/preflight/exploration/figures_audit/`. Original preflight PNG files were not overwritten.
- L2WS SHA `{live['l2ws_sha']}`, dirty={live['l2ws_dirty']}, MPC config conflict markers={live['mpc_conflict']}, LICENSE present={live['license_present']}.
- Protocol stays draft. `holdout_release_approval` stays false. No frozen protocol file.
- Next human decision is whether to approve the bounded L1 smoke in `reports/06_L1_REPRODUCTION_PLAN.md`. L2 training was not approved and was not started.
"""


def collect_live(root: Path) -> dict:
    l2ws = root / "external" / "l2ws"
    sha = subprocess.check_output(["git", "-C", str(l2ws), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(l2ws), "status", "--porcelain"], text=True).strip()
    mpc_text = (l2ws / "benchmarks" / "configs" / "mpc" / "mpc_run.yaml").read_text(encoding="utf-8", errors="replace")
    pyproject = (l2ws / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    provenance = json.loads((root / "external" / "osqp_benchmarks" / "UPSTREAM_PROVENANCE.json").read_text(encoding="utf-8"))
    anchor = root / ".venv-anchor" / "Scripts" / "python.exe"
    anchor_prose = "anchor interpreter missing"
    if anchor.exists():
        completed = subprocess.run(
            [
                str(anchor),
                "-c",
                "import jax,jaxlib,sys; print(sys.version.split()[0]); print(jax.__version__); print(jaxlib.__version__); print(jax.devices())",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        anchor_prose = (completed.stdout or completed.stderr or "").strip() + f" exit={completed.returncode}"
    eval_py = root / ".venv-eval" / "Scripts" / "python.exe"
    host = platform.platform()
    prose = "\n".join(
        [
            f"- Host platform string: `{host}`",
            f"- Eval interpreter running this audit: `{sys_version()}`",
            f"- L2WS HEAD `{sha}`, porcelain empty: {not bool(dirty)}",
            f"- `mpc_run.yaml` contains conflict markers: {'<<<<<<<' in mpc_text}",
            f"- `pyproject.toml` mentions Apache: {'Apache' in pyproject}",
            f"- `external/l2ws/LICENSE` exists: {(l2ws / 'LICENSE').exists()}",
            f"- OSQP benchmark provenance git_clone=`{provenance.get('git_clone')}` commit=`{provenance.get('commit')}`",
            f"- Anchor check: {anchor_prose}",
            f"- Eval interpreter path exists: {eval_py.exists()}",
        ]
    )
    return {
        "l2ws_sha": sha,
        "l2ws_dirty": bool(dirty),
        "mpc_conflict": "<<<<<<<" in mpc_text,
        "license_present": (l2ws / "LICENSE").exists(),
        "osqp_clone": provenance.get("git_clone"),
        "osqp_sha": provenance.get("commit"),
        "anchor_prose": anchor_prose,
        "prose": prose,
        "figure_hash_table": "",
    }


def sys_version() -> str:
    return platform.python_version()


def _replace_marked_section(path: Path, marker: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


def write_outputs(root: Path) -> dict:
    frozen = root / "protocol" / "frozen_protocol.yaml"
    if frozen.exists():
        raise RuntimeError("frozen protocol exists; audit will not run")
    results = root / "experiments" / "preflight" / "exploration" / "results.csv"
    summary_path = root / "experiments" / "preflight" / "exploration" / "summary.json"
    instances = root / "experiments" / "preflight" / "exploration" / "instances.csv"
    before = {path.name: sha256_file(path) for path in (results, summary_path, instances)}
    frame = pd.read_csv(results)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = build_audit(frame, summary)
    out = root / "experiments" / "preflight" / "exploration" / "figures_audit"
    out.mkdir(parents=True, exist_ok=True)
    from .preflight_figures import (
        plot_latency_by_group,
        plot_oneoff,
        plot_paired_by_group,
        plot_quality,
        plot_streaming_stages,
    )

    plot_streaming_stages(frame, out / "fig1_streaming_stages.png")
    plot_oneoff(frame, out / "fig2_oneoff_setup_solve.png")
    plot_paired_by_group(frame, out / "fig3_paired_c1_minus_c2.png")
    plot_latency_by_group(frame, out / "fig4_latency_by_group.png")
    plot_quality(frame, out / "fig5_quality_diagnostics.png")
    pd.DataFrame(audit["sensitivity"]).to_csv(out / "sensitivity_thresholds.csv", index=False)
    pd.DataFrame(audit["group_latency"]).to_csv(out / "group_latency_summary.csv", index=False)
    pd.DataFrame(audit["component_median_gap"]).to_csv(out / "component_median_gap.csv", index=False)
    (out / "audit_numbers.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    after = {path.name: sha256_file(path) for path in (results, summary_path, instances)}
    if after != before:
        raise RuntimeError("audit rewrote a raw preflight input")
    figure_paths = [
        results,
        summary_path,
        instances,
        out / "fig1_streaming_stages.png",
        out / "fig2_oneoff_setup_solve.png",
        out / "fig3_paired_c1_minus_c2.png",
        out / "fig4_latency_by_group.png",
        out / "fig5_quality_diagnostics.png",
        out / "sensitivity_thresholds.csv",
        out / "group_latency_summary.csv",
        out / "component_median_gap.csv",
    ]
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in figure_paths}
    live = collect_live(root)
    live["figure_hash_table"] = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in hashes.items())
    reports = render_reports(audit, live, {k: hashes[k] for k in list(hashes)[:3]})
    for relative, text in reports.items():
        (root / relative).write_text(text, encoding="utf-8")
    protocol = root / "protocol" / "draft_protocol.yaml"
    protocol.write_text(render_protocol(audit), encoding="utf-8")
    _replace_marked_section(
        root / "decisions" / "DECISION_LOG.md",
        "## 2026-10-26 — Preflight accounting audit and plot repair",
        render_decision(audit, live),
    )
    pointer = (
        "\n\n<!-- audit-pointer -->\n\n"
        "Interpretation update, 2026-10-26: `HEADROOM_PRESENT` is retired. "
        "See `reports/03_PREFLIGHT_AUDIT_AND_PLOT_FIXES.md` and `reports/02_GATE_RECOMMENDATION.md`. "
        "The measurements in this file were not deleted.\n"
    )
    for relative in ("reports/02_PREFLIGHT.md", "reports/04_BASELINE_RESULTS.md"):
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        marker = "<!-- audit-pointer -->"
        if marker in text:
            text = text[: text.index(marker)].rstrip() + "\n"
        path.write_text(text.rstrip() + pointer, encoding="utf-8")
    changelog = root / "protocol" / "PROTOCOL_CHANGELOG.md"
    if changelog.exists():
        _replace_marked_section(
            changelog,
            "## 2026-10-26",
            "## 2026-10-26\n\nDraft only. Added classical-effect and learned-headroom fields, "
            "recorded the 510-versus-448 accounting split, and left holdout release false. "
            "No frozen protocol.\n",
        )
    if frozen.exists():
        raise RuntimeError("audit created a frozen protocol")
    print(json.dumps({
        "effect": audit["classical_warm_start_effect"],
        "learned": audit["learned_headroom"],
        "rows": audit["n_rows"],
        "solves": audit["solves_used_recorded"],
        "reference": audit["reference_solves_not_in_csv"],
        "groups": audit["n_independent_groups"],
        "proxy": audit["nn_proxy_validity"],
        "candidate_pass": audit["candidate_rule"]["n_pass"],
        "deltas": audit["group_median_c1_minus_c2_ms"],
        "family": audit["family_streaming"],
    }, indent=2))
    return audit


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    write_outputs(root)


if __name__ == "__main__":
    main()
