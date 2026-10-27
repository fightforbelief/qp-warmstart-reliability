import numpy as np
import scipy.sparse as spa

from qpwr.instance import QPInstance, qp_content_hash
from qpwr.solver import DEFAULT_SETTINGS, solve_fresh_cold
from qpwr.baselines.previous import solve_auto_previous


def _inst(step, q):
    return QPInstance(
        instance_id=f"id{step}",
        group_id="g",
        family="lasso",
        source_repo_sha="none",
        source_file="unit",
        source_seed=1,
        step=step,
        P_full=spa.csc_matrix([[2.0]]),
        q=np.array([q], dtype=float),
        A=spa.eye(1, format="csc"),
        l=np.array([-5.0]),
        u=np.array([5.0]),
    )


def test_same_instance_hash_across_policies():
    seq = [_inst(0, -1.0), _inst(1, -2.0)]
    settings = dict(DEFAULT_SETTINGS)
    fresh = [solve_fresh_cold(i, settings, primal_tol=1e-5, stationarity_tol=1e-4) for i in seq]
    auto = solve_auto_previous(seq, settings, primal_tol=1e-5, stationarity_tol=1e-4)
    for a, b, inst in zip(fresh, auto, seq):
        assert a.qp_hash == b.qp_hash == qp_content_hash(inst)
