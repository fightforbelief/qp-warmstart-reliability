from qpwr.analysis.aggregation import oracle_best_note
from qpwr.manifest import deployment_features


def test_oracle_best_is_labeled_and_not_a_feature():
    text = oracle_best_note()
    assert "oracle_only" in text
    import pytest

    from qpwr.manifest import LeakageError

    with pytest.raises(LeakageError):
        deployment_features({"oracle_best": "C2"})
