from pathlib import Path

import pytest
import yaml

from qpwr.manifest import HoldoutLocked, assert_holdout_released


def test_draft_protocol_seals_the_test_partition():
    protocol = yaml.safe_load(Path("protocol/draft_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["status"] != "frozen"
    assert protocol["holdout_release_approval"] is False
    with pytest.raises(HoldoutLocked):
        assert_holdout_released(protocol, "test")
    assert_holdout_released(protocol, "train")
