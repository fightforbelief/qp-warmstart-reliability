from qpwr.timing import streaming_total_ns


def test_streaming_total_is_the_sum_of_declared_stages_and_nonnegative():
    stages = {
        "parameter_update_ns": 10,
        "feature_ns": 0,
        "lookup_ns": 0,
        "infer_ns": 0,
        "state_insert_ns": 5,
        "solve_wall_ns": 100,
        "verify_ns": 2,
        "fallback_ns": 0,
        "setup_ns": 999,
    }
    total = streaming_total_ns(stages)
    assert total == 117
    assert total >= 0
    # Setup is outside the streaming estimand.
    assert "setup_ns" not in {
        "parameter_update_ns",
        "feature_ns",
        "lookup_ns",
        "infer_ns",
        "state_insert_ns",
        "solve_wall_ns",
        "verify_ns",
        "fallback_ns",
    }
