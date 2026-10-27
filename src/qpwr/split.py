"""Group-level splits. One trajectory or one design matrix is one group."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class GroupAssignment:
    group_id: str
    partition: str
    seed: int


def make_groups(
    group_ids: list[str],
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
) -> list[GroupAssignment]:
    ids = [str(g) for g in group_ids]
    if len(ids) != len(set(ids)):
        raise SplitError("duplicate group ids")
    need = n_train + n_val + n_test
    if min(n_train, n_val, n_test) < 0 or need != len(ids):
        raise SplitError(
            f"requested {n_train}+{n_val}+{n_test}={need} groups but received {len(ids)}"
        )
    rng = np.random.default_rng(seed)
    ordered = sorted(ids)
    perm = list(rng.permutation(np.array(ordered, dtype=object)))
    rows: list[GroupAssignment] = []
    cuts = [
        ("train", perm[:n_train]),
        ("val", perm[n_train : n_train + n_val]),
        ("test", perm[n_train + n_val :]),
    ]
    for partition, chunk in cuts:
        for gid in chunk:
            rows.append(GroupAssignment(str(gid), partition, seed))
    _assert_disjoint(rows)
    return rows


def _assert_disjoint(rows: list[GroupAssignment]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        if row.group_id in seen:
            raise SplitError(f"group {row.group_id} appears in {seen[row.group_id]} and {row.partition}")
        seen[row.group_id] = row.partition


def assignment_sha(rows: list[GroupAssignment]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.group_id):
        h.update(f"{row.group_id},{row.partition},{row.seed}\n".encode())
    return h.hexdigest()


def partition_map(rows: list[GroupAssignment]) -> dict[str, str]:
    return {r.group_id: r.partition for r in rows}
