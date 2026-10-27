"""Write the preflight and gate reports from a summary payload and the CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_preflight_reports(project_root: Path, payload: dict, results_csv: Path) -> None:
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results_csv)
    lines = []
    lines.append("# Preflight — classical OSQP baselines")
    lines.append("")
    lines.append(f"- Run id: `{payload['run_id']}`")
    lines.append(f"- Host: `{payload['host']}`")
    lines.append(f"- Python: `{payload['python']}`")
    lines.append(f"- OSQP: `{payload['osqp_version']}`")
    lines.append(f"- Solves used / budget: {payload['solves_used']} / {payload['budget']}")
    lines.append(f"- Neural training: {payload['neural_training']}")
    lines.append(f"- QP hashes identical across methods: {payload['hash_identical_across_methods']}")
    lines.append(f"- Headroom label: **{payload['headroom']}**")
    lines.append(f"- Settings SHA: `{payload['settings_sha']}`")
    lines.append(f"- OMP_NUM_THREADS: `{payload['omp_num_threads']}`")
    lines.append("")
    lines.append("## Families")
    lines.append("")
    for fam in payload["families"]:
        lines.append(
            f"- {fam['family']} group `{fam['group_id']}` n={fam['n']} m={fam['m']} steps={fam['steps']}"
        )
    lines.append("")
    lines.append("## Overall method summary")
    lines.append("")
    lines.append("Streaming medians exclude step 0. C0 has no streaming total because each solve includes a fresh setup.")
    lines.append("")
    lines.append("| method | rows | groups | fail rate | quality pass | median stream ms | p95 stream ms | median iters | median one-off ms | median |f-fref| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["summaries"]:
        lines.append(
            "| {method} | {n_rows} | {n_groups} | {failure_rate} | {quality_pass_rate} | {median_streaming_ms} | {p95_streaming_ms} | {median_iters} | {median_oneoff_ms} | {median_objective_abs_gap} |".format(
                **{k: _fmt(row.get(k)) for k in row}
            )
        )
    lines.append("")
    lines.append("## By family")
    lines.append("")
    for family, rows in payload["by_family"].items():
        lines.append(f"### {family}")
        lines.append("")
        for row in rows:
            lines.append(
                f"- {row['method']}: median stream ms={_fmt(row.get('median_streaming_ms'))}, "
                f"p95={_fmt(row.get('p95_streaming_ms'))}, iters={_fmt(row.get('median_iters'))}, "
                f"solve wall ms={_fmt(row.get('median_solve_wall_ms'))}, "
                f"update ms={_fmt(row.get('median_update_ms'))}, "
                f"OSQP solve ms={_fmt(row.get('median_osqp_solve_ms'))}, "
                f"status={row.get('status_counts')}"
            )
        lines.append("")
    lines.append("## NN overhead")
    lines.append("")
    proxy = payload["nn_proxy"]
    lines.append(f"- Label: `{proxy['label']}`")
    lines.append(f"- Proxy: {proxy['proxy']}")
    lines.append(f"- Mean milliseconds: {_fmt(proxy['mean_ms'])}")
    lines.append("")
    lines.append("This number is not a measured trained network.")
    lines.append("")
    lines.append("## Cache check")
    lines.append("")
    cached = frame.loc[frame["method"] == "C1_CACHED_COLD"]
    if not cached.empty:
        lines.append(
            f"- C1 max setup_count within a (group, repeat) should be 1. Observed max setup_count={int(cached['setup_count'].max())}."
        )
        lines.append(
            f"- C1 warm_start_calls on the last step of each sequence: see results.csv. "
            f"Median warm_start_calls={float(cached['warm_start_calls'].median()):.1f}."
        )
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(f"- Result rows including failures: {len(frame)}")
    lines.append(f"- Failed rows: {int(frame['method_failed'].sum())}")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (reports / "02_PREFLIGHT.md").write_text(text, encoding="utf-8")
    (reports / "04_BASELINE_RESULTS.md").write_text(
        text.replace("# Preflight — classical OSQP baselines", "# Baseline results (exploratory preflight)", 1),
        encoding="utf-8",
    )
    gate = payload["headroom"]
    status = "NEGATIVE_FEASIBILITY" if gate == "NO_HEADROOM" else "NEEDS_HUMAN_APPROVAL"
    if gate == "INCONCLUSIVE":
        status = "NEEDS_HUMAN_APPROVAL"
    rec = [
        "# Gate recommendation",
        "",
        "```",
        "GATE: G1",
        f"STATUS: {status}",
        f"HEADROOM_LABEL: {gate}",
        f"Evidence: run {payload['run_id']}, OSQP {payload['osqp_version']}, solves {payload['solves_used']}",
        "Risks/unknowns: single-digit exploration groups; tolerances not frozen; L2WS state mapping unmapped",
        "Locked or pending choices: primary family, confirmatory seeds, eps, training budget",
        "Any protocol deviations: none on a confirmatory test (test partition was not opened)",
        "Next permitted action: read this draft; do not train",
        "Action requiring human signoff: freeze protocol and any training above the pilot cap",
        "```",
        "",
        "Primary question is unchanged.",
        "",
        f"Exploration seeds that must stay out of a future test set: `{json.dumps(payload['exploration_seeds_excluded_from_future_test'])}`",
        "",
        "Source dimensions that exist in `run_parametric_problems.py` and were not used, because the preflight starts at the smallest listed size:",
        "",
        f"`{json.dumps(payload['source_dimensions_not_used'])}`",
        "",
        "Switching to a larger listed source dimension is a protocol choice. It is not authorized by this preflight.",
        "",
    ]
    (reports / "02_GATE_RECOMMENDATION.md").write_text("\n".join(rec), encoding="utf-8")
