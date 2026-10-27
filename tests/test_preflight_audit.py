"""Accounting, pairing, step-0 exclusion, and protocol safety for the preflight audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qpwr.analysis.preflight_audit import (
    C0,
    C1,
    C2,
    C3,
    build_audit,
    classify_proxy,
    component_median_gap,
    independent_group_ids,
    paired_c1_minus_c2,
    render_protocol,
    streaming_rows,
)
from qpwr.analysis.preflight_figures import plot_paired_by_group

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    rows = []
    design = (("mpc/dim20/seed1", "mpc", 2), ("lasso/dim50/seed1", "lasso", 1))
    for group_id, family, repeats in design:
        for step in range(3):
            for repeat in range(repeats):
                for method, wall in ((C0, 20.0), (C1, 10.0), (C2, 4.0), (C3, 5.0)):
                    stages = {
                        "parameter_update_ns": 20_000.0,
                        "feature_ns": 0.0,
                        "lookup_ns": 0.0,
                        "infer_ns": 0.0,
                        "state_insert_ns": 1_000.0 if method in {C1, C3} else 0.0,
                        "solve_wall_ns": wall * 1e6,
                        "verify_ns": 1_000.0,
                        "fallback_ns": 0.0,
                    }
                    setup = 1_000_000.0 if step == 0 else 0.0
                    total = sum(stages.values())
                    rows.append(
                        {
                            "group_id": group_id,
                            "family": family,
                            "instance_id": f"{group_id}/step{step}",
                            "step": step,
                            "timing_repeat": repeat,
                            "method": method,
                            "qp_hash": f"hash-{group_id}-{step}",
                            "status": "solved",
                            "quality_pass": True,
                            "method_failed": False,
                            "timeout": False,
                            "objective": 10.0,
                            "reference_objective": 10.0,
                            "primal_residual": 1e-8,
                            "dual_residual": 1e-6,
                            "setup_ns": setup,
                            "oneoff_complete_ns": setup + total,
                            "total_streaming_ns": None if method == C0 else total,
                            "solver_settings_sha": "abc",
                            "osqp_version": "1.1.3",
                            **stages,
                        }
                    )
    return pd.DataFrame(rows)


def test_step0_is_excluded_from_streaming_rows():
    frame = _rows()
    streamed = streaming_rows(frame)
    assert (streamed["step"] > 0).all()
    assert (frame["step"] == 0).any()
    assert C0 not in set(streamed["method"])


def test_paired_join_requires_instance_step_and_hash():
    frame = _rows()
    paired = paired_c1_minus_c2(frame)
    assert (paired["step"] > 0).all()
    assert paired["instance_id"].is_unique is False or len(paired) > 0
    broken = frame.copy()
    mask = (broken["method"] == C2) & (broken["step"] == 1) & (broken["group_id"] == "mpc/dim20/seed1")
    broken.loc[mask, "qp_hash"] = "different-hash"
    broken_pairs = paired_c1_minus_c2(broken)
    assert len(broken_pairs) < len(paired)
    assert not (
        (broken_pairs["group_id"] == "mpc/dim20/seed1")
        & (broken_pairs["step"] == 1)
    ).any()


def test_timing_repeats_are_not_independent_groups():
    frame = _rows()
    assert independent_group_ids(frame) == ["lasso/dim50/seed1", "mpc/dim20/seed1"]
    mpc = frame.loc[frame["family"] == "mpc"]
    assert int(mpc["timing_repeat"].nunique()) == 2
    assert int(mpc["group_id"].nunique()) == 1


def test_component_medians_need_not_sum_to_the_median_total():
    rows = []
    for update, solve, total in ((1.0, 0.0, 1.0), (0.0, 100.0, 100.0), (0.0, 0.0, 0.0)):
        rows.append(
            {
                "family": "mpc",
                "method": C1,
                "step": 1,
                "total_streaming_ns": total,
                "parameter_update_ns": update,
                "feature_ns": 0.0,
                "lookup_ns": 0.0,
                "infer_ns": 0.0,
                "state_insert_ns": 0.0,
                "solve_wall_ns": solve,
                "verify_ns": 0.0,
                "fallback_ns": 0.0,
            }
        )
    gap = component_median_gap(pd.DataFrame(rows))
    assert len(gap) == 1
    assert gap[0]["sum_of_component_medians_ms"] != pytest.approx(gap[0]["median_total_ms"])


def test_exploding_relu_proxy_is_invalid():
    assert classify_proxy(2.221564248396294e25) == "INVALID_NUMERICAL_BLOWUP"
    assert classify_proxy(1.0) == "UNVALIDATED_ESTIMATE"


def test_protocol_text_stays_draft_and_sealed():
    audit = {
        "run_id": "x",
        "classical_warm_start_effect": "observed",
        "nn_proxy_validity": "INVALID_NUMERICAL_BLOWUP",
        "n_rows": 4,
        "solves_used_recorded": 5,
        "reference_solves_not_in_csv": 1,
        "n_independent_groups": 1,
        "osqp_version": "1.1.3",
        "family_streaming": [],
        "group_median_c1_minus_c2_ms": {"g": 1.25},
    }
    text = render_protocol(audit)
    assert "status: draft" in text
    assert "holdout_release_approval: false" in text
    assert "learned_headroom: untested" in text
    assert "quality_certification: unresolved" in text
    assert not (ROOT / "protocol" / "frozen_protocol.yaml").exists()


def test_group_effect_uses_group_medians():
    frame = _rows()
    summary = {
        "run_id": "synthetic",
        "solves_used": int(frame["instance_id"].nunique()) + len(frame),
        "nn_proxy": {"mean_ms": 0.01, "final_norm": 1e25},
        "settings_sha": "abc",
        "primal_tol": 1e-3,
        "stationarity_tol": 1e-2,
        "settings": {"eps_abs": 1e-4, "eps_rel": 1e-4},
    }
    audit = build_audit(frame, summary)
    assert audit["n_independent_groups"] == 2
    assert audit["reference_matches_one_per_instance"]
    assert audit["classical_warm_start_effect"] == "observed"
    assert audit["learned_headroom"] == "untested"
    assert audit["nn_proxy_validity"] == "INVALID_NUMERICAL_BLOWUP"
    assert audit["c0_streaming_all_null"]
    assert audit["unmatched_c1_streaming_rows"] == 0
    assert audit["n_failed"] == 0


def test_figures_accept_a_synthetic_frame(tmp_path: Path):
    plot_paired_by_group(_rows(), tmp_path / "paired.png")
    assert (tmp_path / "paired.png").stat().st_size > 0


def test_recorded_preflight_accounting():
    results = ROOT / "experiments" / "preflight" / "exploration" / "results.csv"
    summary_path = ROOT / "experiments" / "preflight" / "exploration" / "summary.json"
    if not results.exists():
        pytest.skip("exploration CSV is not in this checkout")
    frame = pd.read_csv(results)
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = build_audit(frame, payload)
    assert audit["n_rows"] == 448
    assert audit["n_instances"] == 62
    assert audit["n_independent_groups"] == 3
    assert payload["solves_used"] == 510
    assert audit["reference_solves_not_in_csv"] == 62
    assert audit["reference_matches_one_per_instance"]
    assert audit["hash_nunique_max"] == 1
    assert audit["n_failed"] == 0
    assert audit["c0_streaming_all_null"]
    assert audit["unmatched_c1_streaming_rows"] == 0
    repeats = frame.groupby("family")["timing_repeat"].nunique().to_dict()
    assert int(repeats["mpc"]) == 2
    assert int(repeats["lasso"]) == 1
    assert audit["classical_warm_start_effect"] in {"observed", "not observed", "inconclusive"}
    assert audit["learned_headroom"] == "untested"
