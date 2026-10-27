import numpy as np

from qpwr.analysis.bootstrap import cluster_bootstrap_ci, median_of_group_medians


def test_bootstrap_uses_groups_and_is_seed_stable():
    # Group medians are 50 and 10. Their median is 30.
    # The median of the five raw rows is 10, so a row-wise summary is different.
    values = np.array([0.0, 100.0, 10.0, 10.0, 10.0])
    groups = np.array(["a", "a", "b", "b", "b"])
    assert median_of_group_medians(values, groups) == 30.0
    assert float(np.median(values)) == 10.0
    # Repeating rows inside a group does not create new independent units.
    repeated_v = np.concatenate([values, values, values])
    repeated_g = np.concatenate([groups, groups, groups])
    assert median_of_group_medians(repeated_v, repeated_g) == 30.0
    first = cluster_bootstrap_ci(values, groups, seed=11, n_boot=300)
    second = cluster_bootstrap_ci(values, groups, seed=11, n_boot=300)
    assert first["estimate"] == 30.0
    assert first["n_groups"] == 2
    assert first["low"] == second["low"]
    assert first["high"] == second["high"]
    assert first["low"] <= first["estimate"] <= first["high"]
