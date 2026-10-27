"""Classical preflight. No neural-network training."""

from __future__ import annotations

import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..analysis.aggregation import oracle_best_note, summarize_method
from ..analysis.failure import retain_failures
from ..analysis.figures import plot_paired_difference, plot_stage_bars
from ..baselines.cached_cold import solve_cached_cold
from ..baselines.previous import solve_auto_previous, solve_explicit_previous
from ..instance import qp_content_hash
from ..manifest import write_instance_manifest
from ..solver import DEFAULT_SETTINGS, SolveRecord, osqp, settings_sha, solve_fresh_cold
from ..sources import (
    LASSO_SOURCE_DIMENSIONS,
    MPC_SOURCE_DIMENSIONS,
    generate_lasso_group,
    generate_mpc_group,
)


def _record_row(rec: SolveRecord, *, repeat: int, order: int, settings: dict, reference_obj: float | None) -> dict:
    stages = rec.stages_ns
    streaming = rec.total_streaming_ns
    return {
        "run_id": None,
        "group_id": rec.group_id,
        "instance_id": rec.instance_id,
        "step": rec.step,
        "family": rec.family,
        "method": rec.method,
        "timing_repeat": repeat,
        "execution_order": order,
        "qp_hash": rec.qp_hash,
        "n": rec.n,
        "m": rec.m,
        "nnzP": rec.nnzP,
        "nnzA": rec.nnzA,
        "status": rec.status,
        "status_val": rec.status_val,
        "quality_pass": rec.quality_pass,
        "objective": rec.objective,
        "reference_objective": reference_obj,
        "primal_residual": rec.primal_residual,
        "dual_residual": rec.dual_residual,
        "iterations": rec.iterations,
        "setup_ns": stages.get("setup_ns", 0),
        "parameter_update_ns": stages.get("parameter_update_ns", 0),
        "feature_ns": stages.get("feature_ns", 0),
        "lookup_ns": stages.get("lookup_ns", 0),
        "infer_ns": stages.get("infer_ns", 0),
        "state_insert_ns": stages.get("state_insert_ns", 0),
        "solve_wall_ns": stages.get("solve_wall_ns", 0),
        "verify_ns": stages.get("verify_ns", 0),
        "fallback_ns": stages.get("fallback_ns", 0),
        "total_streaming_ns": streaming if streaming is not None else np.nan,
        "oneoff_complete_ns": rec.oneoff_complete_ns,
        "osqp_solve_ns": rec.osqp_solve_ns if rec.osqp_solve_ns is not None else np.nan,
        "osqp_update_ns": rec.osqp_update_ns if rec.osqp_update_ns is not None else np.nan,
        "osqp_setup_ns": rec.osqp_setup_ns if rec.osqp_setup_ns is not None else np.nan,
        "osqp_run_ns": rec.osqp_run_ns if rec.osqp_run_ns is not None else np.nan,
        "cache_policy": rec.cache_policy,
        "solver_id": rec.solver_id,
        "setup_count": rec.setup_count,
        "warm_start_calls": rec.warm_start_calls,
        "method_failed": rec.method_failed,
        "timeout": rec.timeout,
        "notes": rec.notes,
        "solver_settings_sha": settings_sha(settings),
        "osqp_version": osqp.__version__,
    }


def _nn_proxy_ms() -> dict:
    """A labeled estimate of dense inference cost. Not a trained model."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(256, 256))
    x = rng.normal(size=(256,))
    for _ in range(3):
        x = np.maximum(0.0, w @ x)
    t0 = time.perf_counter_ns()
    for _ in range(20):
        x = np.maximum(0.0, w @ x)
    elapsed = (time.perf_counter_ns() - t0) / 20.0 / 1e6
    final_norm = float(np.linalg.norm(x))
    # An unnormalized Gaussian matvec iterated through ReLU is not a model.
    # A huge final norm means the recurrence diverged; its elapsed time is not
    # an inference-cost measurement.
    validity = (
        "INVALID_NUMERICAL_BLOWUP"
        if (not np.isfinite(final_norm) or final_norm > 1.0e6)
        else "UNVALIDATED_ESTIMATE"
    )
    return {
        "label": "ESTIMATE_NOT_A_TRAINED_MODEL",
        "proxy": "20-run mean of an untrained 256-wide ReLU matvec on CPU",
        "mean_ms": float(elapsed),
        "final_norm": final_norm,
        "validity": validity,
        "research_use": "not evidence of trained-network inference cost",
    }


def _apply_policies(instances, settings, primal_tol, stationarity_tol, repeats, rng, budget, used):
    runners = [
        (
            "C0",
            lambda insts: [
                solve_fresh_cold(inst, settings, primal_tol=primal_tol, stationarity_tol=stationarity_tol)
                for inst in insts
            ],
        ),
        (
            "C1",
            lambda insts: solve_cached_cold(
                insts, settings, primal_tol=primal_tol, stationarity_tol=stationarity_tol
            ),
        ),
        (
            "C2",
            lambda insts: solve_auto_previous(
                insts, settings, primal_tol=primal_tol, stationarity_tol=stationarity_tol
            ),
        ),
        (
            "C3",
            lambda insts: solve_explicit_previous(
                insts, settings, primal_tol=primal_tol, stationarity_tol=stationarity_tol
            ),
        ),
    ]
    rows_src = []
    for repeat in range(repeats):
        order = list(rng.permutation(len(runners)))
        for position, idx in enumerate(order):
            cost = len(instances)
            if used + cost > budget:
                raise RuntimeError(f"preflight budget exceeded ({used}+{cost}>{budget})")
            recs = runners[idx][1](instances)
            used += cost
            rows_src.append((recs, repeat, position))
    return rows_src, used


def run_preflight(project_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    budget = int(config["max_total_solves"])
    if budget > 1000:
        raise RuntimeError("preflight cap is 1000 QP solves")
    run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(DEFAULT_SETTINGS)
    settings.update(config.get("osqp_settings") or {})
    primal_tol = float(config.get("primal_tol", 1e-3))
    stationarity_tol = float(config.get("stationarity_tol", 1e-2))
    rng = np.random.default_rng(int(config.get("order_seed", 20261023)))
    used = 0
    all_instances = []
    reference_obj: dict[str, float] = {}
    raw_rows: list[dict] = []
    families_meta = []

    mpc_cfg = config["mpc"]
    for seed in mpc_cfg["seeds"]:
        insts, refs = generate_mpc_group(
            project_root,
            dimension=int(mpc_cfg["dimension"]),
            seed=int(seed),
            n_steps=int(mpc_cfg["n_steps"]),
            reference_settings=config.get("reference_settings"),
        )
        used += len(refs)
        if used > budget:
            raise RuntimeError("reference rollout exceeded the preflight budget")
        for rec in refs:
            if rec.objective is not None:
                reference_obj[rec.instance_id] = rec.objective
        all_instances.extend(insts)
        packed, used = _apply_policies(
            insts,
            settings,
            primal_tol,
            stationarity_tol,
            int(mpc_cfg["repeats"]),
            rng,
            budget,
            used,
        )
        for recs, repeat, position in packed:
            for rec in recs:
                raw_rows.append(
                    _record_row(
                        rec,
                        repeat=repeat,
                        order=position,
                        settings=settings,
                        reference_obj=reference_obj.get(rec.instance_id),
                    )
                )
        families_meta.append(
            {
                "family": "mpc",
                "group_id": insts[0].group_id,
                "n": insts[0].n,
                "m": insts[0].m,
                "steps": len(insts),
                "seed": int(seed),
                "dimension": int(mpc_cfg["dimension"]),
            }
        )

    lasso_cfg = config["lasso"]
    for seed in lasso_cfg["seeds"]:
        insts = generate_lasso_group(
            project_root,
            dimension=int(lasso_cfg["dimension"]),
            seed=int(seed),
            n_steps=int(lasso_cfg["n_steps"]),
        )
        for inst in insts:
            if used + 1 > budget:
                raise RuntimeError("lasso reference exceeded the preflight budget")
            ref_settings = dict(settings)
            ref_settings.update(config.get("reference_settings") or {})
            ref = solve_fresh_cold(
                inst, ref_settings, primal_tol=primal_tol, stationarity_tol=stationarity_tol
            )
            used += 1
            if ref.objective is not None:
                reference_obj[inst.instance_id] = ref.objective
        all_instances.extend(insts)
        packed, used = _apply_policies(
            insts,
            settings,
            primal_tol,
            stationarity_tol,
            int(lasso_cfg["repeats"]),
            rng,
            budget,
            used,
        )
        for recs, repeat, position in packed:
            for rec in recs:
                raw_rows.append(
                    _record_row(
                        rec,
                        repeat=repeat,
                        order=position,
                        settings=settings,
                        reference_obj=reference_obj.get(rec.instance_id),
                    )
                )
        families_meta.append(
            {
                "family": "lasso",
                "group_id": insts[0].group_id,
                "n": insts[0].n,
                "m": insts[0].m,
                "steps": len(insts),
                "seed": int(seed),
                "dimension": int(lasso_cfg["dimension"]),
            }
        )

    frame = pd.DataFrame(raw_rows)
    frame["run_id"] = run_id
    frame = retain_failures(frame)
    # Identical QP hashes across methods.
    hash_nunique = frame.groupby("instance_id")["qp_hash"].nunique()
    if int(hash_nunique.max()) != 1:
        raise RuntimeError("QP hash differs across policies; the replay is not paired")
    manifest_sha = write_instance_manifest(
        output_dir / "instances.csv",
        all_instances,
        {inst.group_id: "exploration" for inst in all_instances},
    )
    frame.to_csv(output_dir / "results.csv", index=False)
    methods = [
        "C0_FRESH_COLD",
        "C1_CACHED_COLD",
        "C2_CACHED_AUTO_PREVIOUS",
        "C3_CACHED_EXPLICIT_PREVIOUS",
    ]
    summaries = [summarize_method(frame, m) for m in methods]
    by_family = {}
    for family, sub in frame.groupby("family"):
        by_family[str(family)] = [summarize_method(sub, m) for m in methods]
    proxy = _nn_proxy_ms()
    c1 = next(s for s in summaries if s["method"] == "C1_CACHED_COLD")
    c2 = next(s for s in summaries if s["method"] == "C2_CACHED_AUTO_PREVIOUS")
    m1 = c1.get("median_streaming_ms")
    m2 = c2.get("median_streaming_ms")
    if m1 is None or m2 is None:
        classical_effect = "inconclusive"
    elif m1 > m2:
        classical_effect = "observed"
    else:
        classical_effect = "not observed"
    learned_headroom = "untested"
    # HEADROOM_PRESENT compared C2 with an untrained matvec. That label is retired.
    headroom = "RETIRED_DO_NOT_USE"
    payload = {
        "run_id": run_id,
        "solves_used": used,
        "budget": budget,
        "osqp_version": osqp.__version__,
        "settings": settings,
        "settings_sha": settings_sha(settings),
        "primal_tol": primal_tol,
        "stationarity_tol": stationarity_tol,
        "families": families_meta,
        "summaries": summaries,
        "by_family": by_family,
        "nn_proxy": proxy,
        "headroom": headroom,
        "headroom_note": (
            "HEADROOM_PRESENT is retired. classical_warm_start_effect is a pooled "
            "descriptive label. learned_headroom stays untested until a trained model is evaluated."
        ),
        "classical_warm_start_effect": classical_effect,
        "learned_headroom": learned_headroom,
        "manifest_sha256": manifest_sha,
        "n_instances": len(all_instances),
        "n_rows": int(len(frame)),
        "hash_identical_across_methods": True,
        "neural_training": False,
        "host": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "source_dimensions_not_used": {
            "mpc": [d for d in MPC_SOURCE_DIMENSIONS if d != int(mpc_cfg["dimension"])],
            "lasso": [d for d in LASSO_SOURCE_DIMENSIONS if d != int(lasso_cfg["dimension"])],
        },
        "oracle_note": oracle_best_note(),
        "exploration_seeds_excluded_from_future_test": {
            "mpc": list(mpc_cfg["seeds"]),
            "lasso": list(lasso_cfg["seeds"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_stage_bars(
        summaries,
        output_dir / "stage_bars.png",
        f"Preflight median stage times ({run_id})",
    )
    plot_paired_difference(
        frame,
        output_dir / "c1_minus_c2.png",
        f"Paired C1-C2 streaming difference ({run_id})",
    )
    return payload
