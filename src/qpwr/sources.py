"""Load OSQP-benchmark generators without editing upstream files.

Upstream commit (API, 2026-10-23): 5c79e870c4bd697383f66f5dff26aea29dc1ebfa
The full git clone repeatedly failed on this host while transferring the
large Maros-Meszaros blobs. The parametric sources were fetched from that
commit via raw.githubusercontent.com and hashed. See UPSTREAM_PROVENANCE.json.

Independent groups use the seed argument that already exists on
ControlExample and LassoExample. The upstream parametric drivers call those
classes with the default seed=1 only. Passing other seeds does not change
the distribution, dimension, or sparsity recipe.

The frozen QP sequence is a reference rollout. Evaluated methods replay
those matrices and do not propagate their own MPC states.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as spa

from .instance import QPInstance
from .solver import DEFAULT_SETTINGS, solve_fresh_cold

OSQP_BENCHMARK_SHA = "5c79e870c4bd697383f66f5dff26aea29dc1ebfa"
L2WS_SHA = "14d59d03590ca9f34e8d6daa675002d8b4f1a769"

# Smallest dimensions named in upstream run_parametric_problems.py.
# Larger source dimensions exist (MPC 40/60/80, Lasso 100/150/200) and are
# not used unless a later approved protocol selects them for headroom.
MPC_SOURCE_DIMENSIONS = (20, 40, 60, 80)
LASSO_SOURCE_DIMENSIONS = (50, 100, 150, 200)


def benchmark_root(project_root: Path) -> Path:
    return project_root / "external" / "osqp_benchmarks"


def ensure_import_path(project_root: Path) -> None:
    root = str(benchmark_root(project_root))
    if root not in sys.path:
        sys.path.insert(0, root)


def load_provenance(project_root: Path) -> dict:
    path = benchmark_root(project_root) / "UPSTREAM_PROVENANCE.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_vec(v: np.ndarray) -> np.ndarray:
    return np.array(v, dtype=np.float64, copy=True)


def _snapshot_mpc(instance_obj, *, group_id: str, step: int, seed: int) -> QPInstance:
    qp = instance_obj.qp_problem
    params = {
        "dimension": int(instance_obj.nx),
        "nu": int(instance_obj.nu),
        "horizon": int(instance_obj.T),
        "x0": _copy_vec(instance_obj.x0).tolist(),
        "observable": "reference_rollout_x0",
    }
    return QPInstance(
        instance_id=f"{group_id}/step{step:04d}",
        group_id=group_id,
        family="mpc",
        source_repo_sha=OSQP_BENCHMARK_SHA,
        source_file="problem_classes/control.py",
        source_seed=int(seed),
        step=step,
        P_full=spa.csc_matrix(qp["P"], dtype=np.float64).copy(),
        q=_copy_vec(qp["q"]),
        A=spa.csc_matrix(qp["A"], dtype=np.float64).copy(),
        l=_copy_vec(qp["l"]),
        u=_copy_vec(qp["u"]),
        problem_params=params,
        metadata={
            "check_psd": True,
            "psd_tol": 1e-6,
            "symmetry_tol": 1e-6,
            "generator": "ControlExample",
        },
    )


def generate_mpc_group(
    project_root: Path,
    *,
    dimension: int,
    seed: int,
    n_steps: int,
    reference_settings: dict | None = None,
) -> tuple[list[QPInstance], list]:
    """Freeze one reference closed loop. n_steps cannot exceed the source default 100."""
    if dimension not in MPC_SOURCE_DIMENSIONS:
        raise ValueError(
            f"MPC dimension {dimension} is not in the upstream list {MPC_SOURCE_DIMENSIONS}"
        )
    if not 1 <= n_steps <= 100:
        raise ValueError("n_steps must be in 1..100, the upstream n_simulation cap")
    ensure_import_path(project_root)
    from problem_classes.control import ControlExample

    example = ControlExample(int(dimension), seed=int(seed))
    group_id = f"mpc/dim{int(dimension)}/seed{int(seed)}"
    settings = dict(DEFAULT_SETTINGS)
    settings.update(reference_settings or {})
    settings["eps_abs"] = float(settings.get("eps_abs", 1e-8))
    settings["eps_rel"] = float(settings.get("eps_rel", 1e-8))
    settings["max_iter"] = int(settings.get("max_iter", 20000))
    frozen: list[QPInstance] = []
    references = []
    for step in range(n_steps):
        frozen.append(_snapshot_mpc(example, group_id=group_id, step=step, seed=seed))
        rec = solve_fresh_cold(
            frozen[-1],
            settings,
            primal_tol=1e-3,
            stationarity_tol=1e-2,
        )
        if rec.x is None or rec.status not in {"solved", "solved inaccurate"}:
            raise RuntimeError(
                f"reference MPC rollout stopped at {group_id} step {step} status={rec.status}"
            )
        references.append(rec)
        nx = example.nx
        nu = example.nu
        u = rec.x[nx * (example.T + 1) : nx * (example.T + 1) + nu]
        x_next = np.asarray(example.A @ example.x0).reshape(-1) + np.asarray(example.B @ u).reshape(-1)
        example.update_x0(np.real(x_next))
    return frozen, references


def generate_lasso_group(
    project_root: Path,
    *,
    dimension: int,
    seed: int,
    n_steps: int,
    minimum_lambda_over_max: float = 0.01,
) -> list[QPInstance]:
    """One design matrix is one group. Lambda values stay inside the group."""
    if dimension not in LASSO_SOURCE_DIMENSIONS:
        raise ValueError(
            f"Lasso dimension {dimension} is not in the upstream list {LASSO_SOURCE_DIMENSIONS}"
        )
    if not 1 <= n_steps <= 100:
        raise ValueError("n_steps must be in 1..100, the upstream n_problems cap")
    ensure_import_path(project_root)
    from problem_classes.lasso import LassoExample

    example = LassoExample(int(dimension), seed=int(seed))
    lambdas = np.logspace(
        np.log10(minimum_lambda_over_max * example.lambda_max),
        np.log10(example.lambda_max),
        100,
    )[::-1]
    lambdas = lambdas[:n_steps]
    group_id = f"lasso/dim{int(dimension)}/seed{int(seed)}"
    frozen: list[QPInstance] = []
    for step, lam in enumerate(lambdas):
        example.update_lambda(float(lam))
        qp = example.qp_problem
        frozen.append(
            QPInstance(
                instance_id=f"{group_id}/step{step:04d}",
                group_id=group_id,
                family="lasso",
                source_repo_sha=OSQP_BENCHMARK_SHA,
                source_file="problem_classes/lasso.py",
                source_seed=int(seed),
                step=step,
                P_full=spa.csc_matrix(qp["P"], dtype=np.float64).copy(),
                q=_copy_vec(qp["q"]),
                A=spa.csc_matrix(qp["A"], dtype=np.float64).copy(),
                l=_copy_vec(qp["l"]),
                u=_copy_vec(qp["u"]),
                problem_params={
                    "dimension": int(dimension),
                    "lambda": float(lam),
                    "lambda_max": float(example.lambda_max),
                    "m_data": int(example.m),
                },
                metadata={
                    "check_psd": True,
                    "psd_tol": 1e-6,
                    "symmetry_tol": 1e-8,
                    "generator": "LassoExample",
                },
            )
        )
    return frozen
