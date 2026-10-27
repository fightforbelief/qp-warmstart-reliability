import pandas as pd

from qpwr.analysis.failure import failure_rate, retain_failures


def test_failed_rows_are_kept():
    frame = pd.DataFrame(
        {
            "method": ["C2", "C2", "C2"],
            "method_failed": [False, True, False],
            "total_streaming_ns": [100, None, 120],
            "status": ["solved", "maximum iterations reached", "solved"],
        }
    )
    kept = retain_failures(frame)
    assert len(kept) == 3
    assert failure_rate(kept) == 1 / 3
    assert kept["status"].tolist()[1] == "maximum iterations reached"
