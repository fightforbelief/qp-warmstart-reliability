from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_l1_copy_stays_inside_the_approval_cap():
    setup = yaml.safe_load((ROOT / "configs" / "l1_partial" / "unconstrained_qp_setup.yaml").read_text(encoding="utf-8"))
    run = yaml.safe_load((ROOT / "configs" / "l1_partial" / "unconstrained_qp_run.yaml").read_text(encoding="utf-8"))
    assert setup["n_orig"] == 20
    assert setup["N_train"] == 100
    assert setup["N_test"] == 10
    assert run["N_train"] == 100
    assert run["N_test"] == 10
    assert int(run["nn_cfg"]["epochs"]) <= 5
    assert int(run["eval_unrolls"]) == 50
    executed = int(run["nn_cfg"]["epochs"] / run["epochs_jit"]) * int(run["epochs_jit"])
    assert 0 < executed <= 5


def test_confirmatory_protocol_remains_unfrozen():
    protocol = yaml.safe_load((ROOT / "protocol" / "draft_protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["status"] == "draft"
    assert protocol["holdout_release_approval"] is False
    assert not (ROOT / "protocol" / "frozen_protocol.yaml").exists()
