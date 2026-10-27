"""Versioned artifacts for a future evaluation. This module does not solve QPs.

Contract version ``qpwr-eval-contract-2`` keeps matrices and vectors in
sidecar arrays. A result row stores hashes and scalar metrics, not the arrays.
Each JSON artifact also carries ``metadata_sha256`` over its claims, including
the array-file hash. A stale digest rejects an edit to horizon, seed, scaling,
symmetrization, or solver settings. The digest is stored beside those claims
and is not an external signature: rewriting a field and the digest together
is not detected. Version 1 sealed only the array file.

Offline metric evaluation is not part of streaming latency E1. E1 remains
parameter update, features, lookup or inference, state insertion, solve wall,
online verification, and fallback. Setup stays in E2. If a future method
verifies a solve online, that cost belongs in the existing ``verify_ns``
stage. ``quality_offline_ns`` is intentionally absent from E1.

A reference artifact is a tighter fresh solve used for the objective gap.
``common_kkt_certificate`` is false: a small gap does not show that the
candidate and the reference meet one KKT test.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as spa

from .instance import QPInstance, qp_content_hash
from .quality import (
    METRIC_IMPLEMENTATION_VERSION,
    QualityReport,
    evaluate_arrays,
)
from .timing import STREAMING_STAGES

CONTRACT_VERSION = "qpwr-eval-contract-2"
_SEAL_EXCLUDED = frozenset({"metadata_sha256", "json_path", "npz_path"})
OFFLINE_METRIC_STAGE = "quality_offline_ns"
REFERENCE_NOTE = (
    "The reference objective comes from a separate fresh solve. "
    "A small objective gap is not a shared KKT certificate."
)


class ContractError(ValueError):
    """Hash, schema, or availability failure. Not a large residual."""


def offline_metric_stage_is_outside_e1() -> bool:
    return OFFLINE_METRIC_STAGE not in STREAMING_STAGES


def _sha256_file(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _metadata_sha256(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key not in _SEAL_EXCLUDED}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    sealed = {key: value for key, value in payload.items() if key not in _SEAL_EXCLUDED}
    sealed["metadata_sha256"] = _metadata_sha256(sealed)
    return sealed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_seal(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("contract_version") != CONTRACT_VERSION:
        raise ContractError(f"unexpected contract version in {path.name}")
    stored = data.get("metadata_sha256")
    if not isinstance(stored, str) or stored != _metadata_sha256(data):
        raise ContractError("artifact metadata seal does not match")
    return data


def _instance_from_npz(path: Path, meta: dict[str, Any]) -> QPInstance:
    with np.load(path, allow_pickle=False) as blob:
        P = spa.csc_matrix(
            (blob["P_data"], blob["P_indices"], blob["P_indptr"]),
            shape=tuple(int(v) for v in blob["P_shape"]),
        )
        A = spa.csc_matrix(
            (blob["A_data"], blob["A_indices"], blob["A_indptr"]),
            shape=tuple(int(v) for v in blob["A_shape"]),
        )
        instance = QPInstance(
            instance_id=str(meta["instance_id"]),
            group_id=str(meta["group_id"]),
            family=str(meta["family"]),
            source_repo_sha=str(meta["source_repo_sha"]),
            source_file=str(meta["source_file"]),
            source_seed=None if meta["source_seed"] is None else int(meta["source_seed"]),
            step=int(meta["step"]),
            P_full=P,
            q=np.asarray(blob["q"]),
            A=A,
            l=np.asarray(blob["l"]),
            u=np.asarray(blob["u"]),
            metadata={"check_psd": True},
        )
    digest = qp_content_hash(instance)
    if digest != meta["qp_hash"]:
        raise ContractError("reloaded QP hash does not match the artifact metadata")
    return instance


def save_instance_artifact(
    directory: Path,
    instance: QPInstance,
    *,
    dimension: int | None = None,
    horizon_steps: int | None = None,
    scaling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one instance sidecar. The matrix is not copied into result rows."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"instance_{instance.instance_id.replace('/', '_')}"
    npz_path = directory / f"{stem}.npz"
    P = instance.P_full
    A = instance.A
    np.savez_compressed(
        npz_path,
        P_data=P.data,
        P_indices=P.indices,
        P_indptr=P.indptr,
        P_shape=np.array(P.shape),
        q=instance.q,
        A_data=A.data,
        A_indices=A.indices,
        A_indptr=A.indptr,
        A_shape=np.array(A.shape),
        l=instance.l,
        u=instance.u,
    )
    meta = {
        "artifact_role": "instance",
        "contract_version": CONTRACT_VERSION,
        "dimension": instance.n if dimension is None else dimension,
        "dtype": "float64",
        "dual_sign_convention": "OSQP: P_sym x + q + A.T @ y",
        "family": instance.family,
        "file_sha256": _sha256_file(npz_path),
        "group_id": instance.group_id,
        "horizon_steps": horizon_steps,
        "instance_id": instance.instance_id,
        "matrix_format": "csc",
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "objective_symmetrization": "(P + P.T) / 2",
        "qp_hash": qp_content_hash(instance),
        "scaling": scaling or {"recorded": False, "coordinates": "original"},
        "source_file": instance.source_file,
        "source_repo_sha": instance.source_repo_sha,
        "source_seed": instance.source_seed,
        "step": instance.step,
        "storage": "symmetric CSC P, CSC A, contiguous float64 vectors",
    }
    json_path = directory / f"{stem}.json"
    _write_json(json_path, meta)
    meta["json_path"] = str(json_path)
    meta["npz_path"] = str(npz_path)
    return meta


def load_instance_artifact(directory: Path, instance_id: str) -> tuple[QPInstance, dict[str, Any]]:
    stem = f"instance_{instance_id.replace('/', '_')}"
    json_path = directory / f"{stem}.json"
    npz_path = directory / f"{stem}.npz"
    meta = _read_json(json_path)
    if _sha256_file(npz_path) != meta["file_sha256"]:
        raise ContractError("instance array file hash does not match metadata")
    instance = _instance_from_npz(npz_path, meta)
    return instance, meta


def _vector_payload(path: Path, **vectors: np.ndarray) -> None:
    arrays = {key: np.ascontiguousarray(value, dtype=np.float64).reshape(-1) for key, value in vectors.items()}
    np.savez_compressed(path, **arrays)


def save_candidate_artifact(
    directory: Path,
    *,
    solution_id: str,
    qp_hash: str,
    x: np.ndarray,
    y: np.ndarray | None,
    status: str,
    settings_sha: str,
    iterations: int | None,
    report: QualityReport,
    reference_objective: float | None = None,
    reference_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"candidate_{solution_id}.npz"
    if y is None:
        _vector_payload(npz_path, x=np.asarray(x))
    else:
        _vector_payload(npz_path, x=np.asarray(x), y=np.asarray(y))
    meta = {
        "artifact_role": "candidate",
        "common_kkt_certificate": False,
        "contract_version": CONTRACT_VERSION,
        "file_sha256": _sha256_file(npz_path),
        "iterations": iterations,
        "metric_implementation_version": report.implementation_version,
        "metrics": report.as_dict(),
        "qp_hash": qp_hash,
        "reference_artifact_sha256": reference_artifact_sha256,
        "reference_note": REFERENCE_NOTE,
        "reference_objective": reference_objective,
        "settings_sha": settings_sha,
        "solution_id": solution_id,
        "status": status,
    }
    json_path = directory / f"candidate_{solution_id}.json"
    _write_json(json_path, meta)
    meta["json_path"] = str(json_path)
    meta["npz_path"] = str(npz_path)
    return meta


def load_candidate_artifact(directory: Path, solution_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = directory / f"candidate_{solution_id}.npz"
    meta = _read_json(directory / f"candidate_{solution_id}.json")
    if _sha256_file(npz_path) != meta["file_sha256"]:
        raise ContractError("candidate array file hash does not match metadata")
    if meta.get("common_kkt_certificate") is not False:
        raise ContractError("candidate artifact must not claim a shared KKT certificate")
    with np.load(npz_path, allow_pickle=False) as blob:
        arrays = {key: np.asarray(blob[key]) for key in blob.files}
    return arrays, meta


def save_reference_artifact(
    directory: Path,
    *,
    reference_id: str,
    qp_hash: str,
    x_ref: np.ndarray | None,
    y_ref: np.ndarray | None,
    objective: float,
    status: str,
    settings_sha: str,
    report: QualityReport | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"reference_{reference_id}.npz"
    vectors: dict[str, np.ndarray] = {}
    if x_ref is not None:
        vectors["x_ref"] = np.asarray(x_ref)
    if y_ref is not None:
        vectors["y_ref"] = np.asarray(y_ref)
    if not vectors:
        vectors["objective"] = np.array([objective], dtype=np.float64)
    _vector_payload(npz_path, **vectors)
    meta = {
        "artifact_role": "reference",
        "common_kkt_certificate": False,
        "contract_version": CONTRACT_VERSION,
        "file_sha256": _sha256_file(npz_path),
        "metric_implementation_version": None if report is None else report.implementation_version,
        "metrics": None if report is None else report.as_dict(),
        "objective": objective,
        "qp_hash": qp_hash,
        "reference_id": reference_id,
        "reference_note": REFERENCE_NOTE,
        "settings_sha": settings_sha,
        "status": status,
        "x_ref_stored": x_ref is not None,
        "y_ref_stored": y_ref is not None,
    }
    json_path = directory / f"reference_{reference_id}.json"
    _write_json(json_path, meta)
    meta["json_path"] = str(json_path)
    meta["npz_path"] = str(npz_path)
    return meta


def load_reference_artifact(directory: Path, reference_id: str) -> dict[str, Any]:
    npz_path = directory / f"reference_{reference_id}.npz"
    meta = _read_json(directory / f"reference_{reference_id}.json")
    if _sha256_file(npz_path) != meta["file_sha256"]:
        raise ContractError("reference array file hash does not match metadata")
    return meta


def evaluate_loaded_instance(
    instance: QPInstance,
    x: np.ndarray,
    y: np.ndarray | None,
    *,
    reference_objective: float | None = None,
    status: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> QualityReport:
    """Recompute metrics from a loaded instance. Still does not call a solver."""
    return evaluate_arrays(
        P=instance.P_full,
        q=instance.q,
        A=instance.A,
        l=instance.l,
        u=instance.u,
        x=x,
        y=y,
        reference_objective=reference_objective,
        status=status,
        thresholds=thresholds,
    )
