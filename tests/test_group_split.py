import pytest

from qpwr.split import SplitError, assignment_sha, make_groups


def test_groups_are_disjoint_and_deterministic():
    ids = [f"traj-{i:02d}" for i in range(24)]
    a = make_groups(ids, seed=20261023, n_train=12, n_val=4, n_test=8)
    b = make_groups(list(reversed(ids)), seed=20261023, n_train=12, n_val=4, n_test=8)
    assert assignment_sha(a) == assignment_sha(b)
    parts = {}
    for row in a:
        assert row.group_id not in parts
        parts[row.group_id] = row.partition
    assert sum(p == "train" for p in parts.values()) == 12
    assert sum(p == "val" for p in parts.values()) == 4
    assert sum(p == "test" for p in parts.values()) == 8


def test_rejects_counts_that_do_not_cover_every_group():
    with pytest.raises(SplitError):
        make_groups(["a", "b"], seed=1, n_train=1, n_val=1, n_test=1)
