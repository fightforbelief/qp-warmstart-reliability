"""Manifests, hashes, and the holdout lock.

evaluate --partition test refuses to run unless the protocol is frozen
and a human set holdout_release_approval: true.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .instance import QPInstance, qp_content_hash


class HoldoutLocked(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_sha(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()


def assert_holdout_released(protocol: dict[str, Any], partition: str) -> None:
    if partition != "test":
        return
    if protocol.get("status") != "frozen":
        raise HoldoutLocked(
            "confirmatory test partition is sealed: protocol status is not 'frozen'"
        )
    if protocol.get("holdout_release_approval") is not True:
        raise HoldoutLocked(
            "confirmatory test partition is sealed: holdout_release_approval is not true"
        )


def write_instance_manifest(path: Path, instances: list[QPInstance], partition_of: dict[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance_id",
        "group_id",
        "family",
        "step",
        "partition",
        "source_repo_sha",
        "source_file",
        "source_seed",
        "n",
        "m",
        "nnzP",
        "nnzA",
        "qp_hash",
    ]
    rows = []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for inst in instances:
            row = {
                "instance_id": inst.instance_id,
                "group_id": inst.group_id,
                "family": inst.family,
                "step": inst.step,
                "partition": partition_of.get(inst.group_id, "exploration"),
                "source_repo_sha": inst.source_repo_sha,
                "source_file": inst.source_file,
                "source_seed": "" if inst.source_seed is None else inst.source_seed,
                "n": inst.n,
                "m": inst.m,
                "nnzP": inst.metadata.get("nnzP"),
                "nnzA": inst.metadata.get("nnzA"),
                "qp_hash": qp_content_hash(inst),
            }
            rows.append(row)
            w.writerow(row)
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def save_npz(path: Path, instance: QPInstance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    p = instance.P_triu
    np.savez_compressed(
        path,
        P_data=p.data,
        P_indices=p.indices,
        P_indptr=p.indptr,
        P_shape=np.array(p.shape),
        q=instance.q,
        A_data=instance.A.data,
        A_indices=instance.A.indices,
        A_indptr=instance.A.indptr,
        A_shape=np.array(instance.A.shape),
        l=instance.l,
        u=instance.u,
        instance_id=np.array(instance.instance_id),
        group_id=np.array(instance.group_id),
        qp_hash=np.array(qp_content_hash(instance)),
    )


FORBIDDEN_ONLINE_KEYS = frozenset(
    {
        "active_set",
        "optimal_x",
        "optimal_y",
        "reference_x",
        "reference_y",
        "current_solve_time",
        "current_iterations",
        "fastest_method",
        "oracle_best",
        "test_label",
    }
)


class LeakageError(RuntimeError):
    pass


def deployment_features(observables: dict[str, Any]) -> dict[str, Any]:
    """Features that are legal before the solve. Oracle keys are rejected."""
    bad = FORBIDDEN_ONLINE_KEYS.intersection(observables)
    if bad:
        raise LeakageError(f"oracle or post-solve keys are not deployment features: {sorted(bad)}")
    return dict(observables)


class TrainOnlyScaler:
    """Affine scaler fit on train rows only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._fit_ids: set[str] = set()

    def fit(self, rows: list[tuple[str, np.ndarray]]) -> "TrainOnlyScaler":
        if not rows:
            raise LeakageError("scaler fit received no train rows")
        parts = [np.ascontiguousarray(v, dtype=np.float64).reshape(-1) for _, v in rows]
        mat = np.vstack(parts)
        self.mean_ = mat.mean(axis=0)
        self.std_ = mat.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-12, 1.0, self.std_)
        self._fit_ids = {i for i, _ in rows}
        return self

    def transform(self, row_id: str, value: np.ndarray, partition: str) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise LeakageError("scaler used before train-only fit")
        if partition != "train" and row_id in self._fit_ids:
            raise LeakageError("non-train row id collides with a train fit id")
        if partition == "train" and row_id not in self._fit_ids:
            raise LeakageError("train transform id was not in the fit set")
        v = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
        return (v - self.mean_) / self.std_
