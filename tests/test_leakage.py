import numpy as np
import pytest

from qpwr.baselines.nearest import NearestNeighborIndex
from qpwr.learning.direct import DirectRegressor
from qpwr.learning.solver_aware import L2WSInspired
from qpwr.manifest import LeakageError, TrainOnlyScaler, deployment_features


def test_scaler_and_nearest_are_train_only():
    scaler = TrainOnlyScaler()
    scaler.fit([("t1", np.array([0.0, 2.0])), ("t2", np.array([2.0, 4.0]))])
    out = scaler.transform("t1", np.array([0.0, 2.0]), "train")
    assert out.shape == (2,)
    with pytest.raises(LeakageError):
        scaler.transform("t1", np.array([0.0, 2.0]), "test")
    index = NearestNeighborIndex()
    index.fit(
        [
            {"partition": "train", "instance_id": "t1", "feature": np.array([0.0]), "payload": {"x": 1}},
        ]
    )
    with pytest.raises(LeakageError):
        index.fit([{"partition": "test", "instance_id": "v", "feature": np.array([1.0]), "payload": {}}])
    assert index.query(np.array([0.1]), "heldout")["x"] == 1


def test_direct_regressor_rejects_test_rows_and_inspired_name():
    model = DirectRegressor()
    with pytest.raises(LeakageError):
        model.fit([{"partition": "test"}])
    model.fit([{"partition": "train", "oracle": False}])
    inspired = L2WSInspired()
    assert inspired.name == "C6_L2WS_INSPIRED"
    assert inspired.mapping_status == "UNMAPPED"
    with pytest.raises(NotImplementedError):
        inspired.fit([])


def test_deployment_features_reject_oracle_keys():
    deployment_features({"q_delta": 0.2, "previous_iterations": 10})
    with pytest.raises(LeakageError):
        deployment_features({"active_set": [1, 0], "q_delta": 0.2})
    with pytest.raises(LeakageError):
        deployment_features({"fastest_method": "C2"})
