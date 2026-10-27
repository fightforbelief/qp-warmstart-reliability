"""Deterministic metric and contract tests. No solver is imported or called."""

import json

import numpy as np
import pytest
import scipy.sparse as spa

from qpwr.instance import QPInstance, objective_value, qp_content_hash
from qpwr.quality import (
    QualityInputError,
    criteria_failures,
    evaluate_arrays,
    historical_report_from_row,
    primal_violation,
    relative_objective_gap,
    stationarity_inf,
)
from qpwr.quality_contract import (
    OFFLINE_METRIC_STAGE,
    ContractError,
    _metadata_sha256,
    _sha256_file,
    evaluate_loaded_instance,
    load_candidate_artifact,
    load_instance_artifact,
    load_reference_artifact,
    offline_metric_stage_is_outside_e1,
    save_candidate_artifact,
    save_instance_artifact,
    save_reference_artifact,
)
from qpwr.timing import STREAMING_STAGES


def _box(P, q, l, u, A=None):
    P_mat = spa.csc_matrix(np.asarray(P, dtype=float))
    n = P_mat.shape[0]
    if A is None:
        A = spa.eye(n, format="csc")
    return QPInstance(
        instance_id="toy/box",
        group_id="toy-group",
        family="mpc",
        source_repo_sha="none",
        source_file="unit",
        source_seed=0,
        step=1,
        P_full=P_mat,
        q=np.array(q, dtype=float),
        A=A,
        l=np.array(l, dtype=float),
        u=np.array(u, dtype=float),
        metadata={"check_psd": True},
    )


def test_satisfies_both_finite_bounds():
    inst = _box([[1.0]], [0.0], [-1.0], [1.0])
    report = evaluate_arrays(P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([0.25]), y=np.array([0.0]))
    assert report.r_p.value == pytest.approx(0.0)
    assert report.n_finite_bound_rows.value == pytest.approx(1.0)


def test_lower_bound_violation_only():
    inst = _box([[1.0]], [0.0], [0.0], [np.inf])
    report = evaluate_arrays(P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([-1.5]), y=np.array([0.0]))
    assert report.r_p.value == pytest.approx(1.5)
    assert primal_violation(inst, np.array([-1.5])) == pytest.approx(1.5)


def test_upper_bound_violation_only():
    inst = _box([[1.0]], [0.0], [-np.inf], [1.0])
    report = evaluate_arrays(P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([3.0]), y=np.array([0.0]))
    assert report.r_p.value == pytest.approx(2.0)


def test_mixed_finite_and_infinite_bounds():
    A = spa.eye(2, format="csc")
    l = np.array([-np.inf, -1.0])
    u = np.array([0.5, np.inf])
    P = spa.eye(2, format="csc")
    q = np.zeros(2)
    x = np.array([1.5, -2.0])
    report = evaluate_arrays(P=P, q=q, A=A, l=l, u=u, x=x, y=np.zeros(2))
    assert report.r_p.value == pytest.approx(1.0)
    assert report.n_finite_bound_rows.value == pytest.approx(2.0)


def test_both_bounds_infinite_does_not_create_a_violation():
    inst = _box([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [-np.inf, -1.0], [np.inf, 1.0])
    x = np.array([50.0, 0.0])
    report = evaluate_arrays(P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=x, y=np.zeros(2))
    assert report.r_p.value == pytest.approx(0.0)
    assert report.n_finite_bound_rows.value == pytest.approx(1.0)


def test_no_finite_bound_rows_define_candidate_norm_as_zero():
    inst = _box([[1.0]], [0.0], [-np.inf], [np.inf])
    report = evaluate_arrays(
        P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([4.0]), y=np.array([0.0])
    )
    assert report.r_p.value == pytest.approx(0.0)
    assert report.r_p_norm_candidate.value == pytest.approx(0.0)
    assert report.ax_f_inf.value == pytest.approx(0.0)


def test_zero_stationarity_and_known_nonzero_stationarity():
    inst = _box([[2.0]], [0.0], [-5.0], [5.0])
    zero = evaluate_arrays(P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([0.0]), y=np.array([0.0]))
    assert zero.r_d.value == pytest.approx(0.0)
    assert zero.r_d_norm.value == pytest.approx(0.0)
    nonzero = evaluate_arrays(
        P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([1.5]), y=np.array([0.0])
    )
    # P_sym x + q + A'y = 3.
    assert nonzero.r_d.value == pytest.approx(3.0)
    assert nonzero.norm_px.value == pytest.approx(3.0)
    assert nonzero.norm_q.value == pytest.approx(0.0)
    assert nonzero.norm_aty.value == pytest.approx(0.0)
    assert nonzero.r_d_norm.value == pytest.approx(3.0 / 4.0)
    assert stationarity_inf(inst, np.array([1.5]), np.array([0.0])) == pytest.approx(3.0)


def test_known_objective_and_reference_gap():
    inst = _box([[1.0]], [-3.0], [-10.0], [10.0])
    x = np.array([3.0])
    assert objective_value(inst, x) == pytest.approx(-4.5)
    report = evaluate_arrays(
        P=inst.P_full,
        q=inst.q,
        A=inst.A,
        l=inst.l,
        u=inst.u,
        x=x,
        y=np.array([0.0]),
        reference_objective=-4.0,
    )
    assert report.objective.value == pytest.approx(-4.5)
    assert report.relative_gap.value == pytest.approx(relative_objective_gap(-4.5, -4.0))
    assert report.relative_gap.value == pytest.approx(0.5 / 5.0)


def test_nonsymmetric_p_uses_symmetric_stationarity():
    # Skew part changes P x but not the scalar 0.5 x' P_sym x.
    P = np.array([[0.0, 4.0], [0.0, 0.0]])
    q = np.zeros(2)
    A = np.zeros((1, 2))
    x = np.array([1.0, 0.0])
    report = evaluate_arrays(P=P, q=q, A=A, l=np.array([-np.inf]), u=np.array([np.inf]), x=x, y=np.array([0.0]))
    raw = P @ x
    assert np.linalg.norm(raw, ord=np.inf) == pytest.approx(0.0)
    assert report.r_d.value == pytest.approx(2.0)
    assert report.objective.value == pytest.approx(0.0)
    x2 = np.array([1.0, 1.0])
    report2 = evaluate_arrays(P=P, q=q, A=A, l=np.array([-np.inf]), u=np.array([np.inf]), x=x2, y=np.array([0.0]))
    P_sym = 0.5 * (P + P.T)
    assert report2.objective.value == pytest.approx(0.5 * float(x2 @ P_sym @ x2))


def test_incompatible_shapes_and_nonfinite_candidates_are_not_poor_residuals():
    P = np.eye(2)
    q = np.zeros(2)
    A = np.eye(2)
    with pytest.raises(QualityInputError):
        evaluate_arrays(P=P, q=q, A=A, l=np.zeros(2), u=np.ones(2), x=np.ones(3), y=np.zeros(2))
    with pytest.raises(QualityInputError):
        evaluate_arrays(P=P, q=q, A=A, l=np.zeros(2), u=np.ones(2), x=np.array([np.nan, 0.0]), y=np.zeros(2))
    with pytest.raises(QualityInputError):
        evaluate_arrays(P=P, q=q, A=A, l=np.zeros(2), u=np.ones(2), x=np.zeros(2), y=np.array([np.inf, 0.0]))


def test_candidate_normalized_primal_uses_only_finite_bound_rows():
    A = np.eye(2)
    l = np.array([0.0, -np.inf])
    u = np.array([np.inf, np.inf])
    x = np.array([-2.0, 9.0])
    report = evaluate_arrays(P=np.eye(2), q=np.zeros(2), A=A, l=l, u=u, x=x, y=np.zeros(2))
    assert report.r_p.value == pytest.approx(2.0)
    assert report.ax_f_inf.value == pytest.approx(2.0)
    assert report.r_p_norm_candidate.value == pytest.approx(2.0 / 3.0)


def test_thresholds_name_failures_without_selecting_a_project_rule():
    inst = _box([[1.0]], [0.0], [0.0], [1.0])
    report = evaluate_arrays(
        P=inst.P_full,
        q=inst.q,
        A=inst.A,
        l=inst.l,
        u=inst.u,
        x=np.array([2.0]),
        y=None,
        reference_objective=0.0,
    )
    failed = criteria_failures(report, status="solved inaccurate", primal_tol=0.1, stationarity_tol=0.1, require_solved=True)
    assert "status_not_exactly_solved" in failed
    assert "raw_primal" in failed
    assert "unavailable:raw_stationarity" in failed
    assert report.r_d.value is None


def test_historical_row_does_not_invent_normalized_residuals():
    row = {
        "primal_residual": 0.0002,
        "dual_residual": 0.002,
        "objective": 3.0,
        "reference_objective": 3.1,
        "quality_pass": True,
    }
    report = historical_report_from_row(row)
    assert report.input_status == "historical_scalars_only"
    assert report.r_p.value == pytest.approx(0.0002)
    assert report.relative_gap.value == pytest.approx(abs(3.0 - 3.1) / (1.0 + 3.1))
    assert report.r_d_norm.state == "unavailable"
    assert report.r_d_norm.value is None
    assert report.r_p_norm_candidate.value is None
    assert report.norm_px.value is None
    assert "not recomputed" in report.notes[0]


def test_artifact_round_trip_and_hash_verification(tmp_path):
    inst = _box([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0])
    meta = save_instance_artifact(tmp_path, inst, dimension=2, horizon_steps=25)
    loaded, loaded_meta = load_instance_artifact(tmp_path, inst.instance_id)
    assert loaded_meta["qp_hash"] == meta["qp_hash"]
    assert loaded_meta["horizon_steps"] == 25
    x = np.array([0.25, -0.5])
    y = np.zeros(2)
    report = evaluate_loaded_instance(loaded, x, y, reference_objective=0.1, status="solved")
    again = evaluate_loaded_instance(loaded, x, y, reference_objective=0.1, status="solved")
    assert report.as_dict() == again.as_dict()
    cand = save_candidate_artifact(
        tmp_path,
        solution_id="s1",
        qp_hash=meta["qp_hash"],
        x=x,
        y=y,
        status="solved",
        settings_sha="abc",
        iterations=7,
        report=report,
        reference_objective=0.1,
    )
    arrays, cand_meta = load_candidate_artifact(tmp_path, "s1")
    assert cand_meta["file_sha256"] == cand["file_sha256"]
    assert cand_meta["common_kkt_certificate"] is False
    assert np.allclose(arrays["x"], x)
    ref = save_reference_artifact(
        tmp_path,
        reference_id="r1",
        qp_hash=meta["qp_hash"],
        x_ref=np.zeros(2),
        y_ref=np.zeros(2),
        objective=0.1,
        status="solved",
        settings_sha="tighter",
        report=report,
    )
    assert ref["common_kkt_certificate"] is False
    npz = tmp_path / "instance_toy_box.npz"
    blob = npz.read_bytes()
    npz.write_bytes(blob + b"tamper")
    with pytest.raises(ContractError):
        load_instance_artifact(tmp_path, inst.instance_id)


def test_repeated_array_evaluation_is_identical():
    kwargs = dict(
        P=np.eye(1),
        q=np.array([-1.0]),
        A=np.ones((1, 1)),
        l=np.array([-2.0]),
        u=np.array([2.0]),
        x=np.array([0.5]),
        y=np.array([0.25]),
        reference_objective=1.0,
    )
    first = evaluate_arrays(**kwargs).as_dict()
    second = evaluate_arrays(**kwargs).as_dict()
    assert first == second


def test_current_primal_norm_can_hide_a_small_row_behind_a_large_feasible_row():
    """Locks the documented candidate. A large feasible Ax dominates ||(Ax)_F||."""
    report = evaluate_arrays(
        P=np.eye(2),
        q=np.zeros(2),
        A=np.eye(2),
        l=np.array([-1.0e9, 0.0]),
        u=np.array([1.0e9, 0.1]),
        x=np.array([1.0e9, 0.2]),
        y=np.zeros(2),
    )
    assert report.r_p.value == pytest.approx(0.1)
    assert report.ax_f_inf.value == pytest.approx(1.0e9)
    assert report.r_p_norm_candidate.value == pytest.approx(0.1 / (1.0 + 1.0e9))
    hidden = evaluate_arrays(
        P=np.eye(1),
        q=np.zeros(1),
        A=np.ones((1, 1)),
        l=np.array([1.0e6]),
        u=np.array([np.inf]),
        x=np.array([0.0]),
        y=np.zeros(1),
    )
    assert hidden.r_p.value == pytest.approx(1.0e6)
    assert hidden.ax_f_inf.value == pytest.approx(0.0)
    assert hidden.r_p_norm_candidate.value == pytest.approx(1.0e6)


def test_metadata_seal_rejects_a_horizon_or_settings_edit(tmp_path):
    import json

    inst = _box([[1.0]], [0.0], [-1.0], [1.0])
    meta = save_instance_artifact(tmp_path, inst, dimension=1, horizon_steps=25)
    path = tmp_path / "instance_toy_box.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["horizon_steps"] = 100
    payload["source_seed"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="metadata seal"):
        load_instance_artifact(tmp_path, inst.instance_id)
    report = evaluate_arrays(
        P=inst.P_full, q=inst.q, A=inst.A, l=inst.l, u=inst.u, x=np.array([0.0]), y=np.array([0.0])
    )
    save_candidate_artifact(
        tmp_path,
        solution_id="s-seal",
        qp_hash=meta["qp_hash"],
        x=np.array([0.0]),
        y=np.array([0.0]),
        status="solved",
        settings_sha="settings-a",
        iterations=3,
        report=report,
    )
    candidate = tmp_path / "candidate_s-seal.json"
    body = json.loads(candidate.read_text(encoding="utf-8"))
    body["settings_sha"] = "settings-b"
    candidate.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="metadata seal"):
        load_candidate_artifact(tmp_path, "s-seal")


def _replace_json(path, **changes):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_metadata_seal_rejects_each_covered_field(tmp_path):
    """One field at a time. A combined edit does not show which field is sealed."""
    inst = _box([[1.0]], [0.0], [-1.0], [1.0])
    save_instance_artifact(
        tmp_path,
        inst,
        dimension=1,
        horizon_steps=25,
        scaling={"recorded": False, "coordinates": "original"},
    )
    path = tmp_path / "instance_toy_box.json"
    original = path.read_text(encoding="utf-8")
    edits = {
        "instance_id": "toy/other",
        "group_id": "other-group",
        "family": "lasso",
        "dimension": 20,
        "horizon_steps": 100,
        "step": 9,
        "source_seed": 99,
        "qp_hash": "0" * 64,
        "file_sha256": "0" * 64,
        "matrix_format": "csr",
        "dtype": "float32",
        "objective_symmetrization": "P",
        "dual_sign_convention": "flipped",
        "scaling": {"recorded": True, "coordinates": "scaled"},
        "metric_implementation_version": "qpwr-quality-metrics-0",
    }
    for key, value in edits.items():
        _replace_json(path, **{key: value})
        with pytest.raises(ContractError, match="metadata seal"):
            load_instance_artifact(tmp_path, inst.instance_id)
        path.write_text(original, encoding="utf-8")
    payload = json.loads(original)
    payload["metadata_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="metadata seal"):
        load_instance_artifact(tmp_path, inst.instance_id)
    path.write_text(original, encoding="utf-8")
    loaded, meta = load_instance_artifact(tmp_path, inst.instance_id)
    assert meta["horizon_steps"] == 25
    assert meta["source_seed"] == 0
    assert meta["dtype"] == "float64"
    assert meta["matrix_format"] == "csc"
    assert np.allclose(loaded.P_full.toarray(), inst.P_full.toarray())
    assert np.allclose(loaded.A.toarray(), inst.A.toarray())
    assert np.allclose(loaded.q, inst.q)


def test_digest_rewritten_with_its_field_is_not_an_external_signature(tmp_path):
    inst = _box([[1.0]], [0.0], [-1.0], [1.0])
    save_instance_artifact(tmp_path, inst, dimension=1, horizon_steps=25)
    path = tmp_path / "instance_toy_box.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["horizon_steps"] = 100
    payload["metadata_sha256"] = _metadata_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, meta = load_instance_artifact(tmp_path, inst.instance_id)
    assert meta["horizon_steps"] == 100


def test_solution_qp_hash_and_reference_settings_are_sealed(tmp_path):
    first = _box([[1.0]], [0.0], [-1.0], [1.0])
    second = QPInstance(
        instance_id="toy/other",
        group_id="toy-group",
        family="mpc",
        source_repo_sha="none",
        source_file="unit",
        source_seed=0,
        step=2,
        P_full=spa.csc_matrix([[1.0]]),
        q=np.array([1.0]),
        A=spa.eye(1, format="csc"),
        l=np.array([-1.0]),
        u=np.array([1.0]),
        metadata={"check_psd": True},
    )
    meta_first = save_instance_artifact(tmp_path, first, dimension=1, horizon_steps=25)
    meta_second = save_instance_artifact(tmp_path, second, dimension=1, horizon_steps=25)
    assert meta_first["qp_hash"] != meta_second["qp_hash"]
    report = evaluate_arrays(
        P=first.P_full, q=first.q, A=first.A, l=first.l, u=first.u, x=np.zeros(1), y=np.zeros(1)
    )
    save_candidate_artifact(
        tmp_path,
        solution_id="s-link",
        qp_hash=meta_first["qp_hash"],
        x=np.zeros(1),
        y=np.zeros(1),
        status="solved",
        settings_sha="settings-a",
        iterations=1,
        report=report,
    )
    candidate = tmp_path / "candidate_s-link.json"
    original = candidate.read_text(encoding="utf-8")
    _replace_json(candidate, qp_hash=meta_second["qp_hash"])
    with pytest.raises(ContractError, match="metadata seal"):
        load_candidate_artifact(tmp_path, "s-link")
    candidate.write_text(original, encoding="utf-8")
    _, cand_meta = load_candidate_artifact(tmp_path, "s-link")
    assert cand_meta["qp_hash"] == meta_first["qp_hash"]
    save_reference_artifact(
        tmp_path,
        reference_id="r-link",
        qp_hash=meta_first["qp_hash"],
        x_ref=np.zeros(1),
        y_ref=np.zeros(1),
        objective=0.0,
        status="solved",
        settings_sha="settings-ref",
        report=report,
    )
    reference = tmp_path / "reference_r-link.json"
    _replace_json(reference, settings_sha="settings-other")
    with pytest.raises(ContractError, match="metadata seal"):
        load_reference_artifact(tmp_path, "r-link")


def test_matrix_value_change_fails_qp_hash_after_the_file_digest_is_refreshed(tmp_path):
    inst = _box([[2.0, 0.0], [0.0, 2.0]], [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0])
    save_instance_artifact(tmp_path, inst, dimension=2, horizon_steps=25)
    npz_path = tmp_path / "instance_toy_box.npz"
    json_path = tmp_path / "instance_toy_box.json"
    with np.load(npz_path) as blob:
        arrays = {key: np.array(blob[key]) for key in blob.files}
    arrays["q"] = arrays["q"].copy()
    arrays["q"][0] = 4.0
    np.savez_compressed(npz_path, **arrays)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["file_sha256"] = _sha256_file(npz_path)
    payload["metadata_sha256"] = _metadata_sha256(payload)
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="QP hash"):
        load_instance_artifact(tmp_path, inst.instance_id)


def test_duplicate_and_reordered_sparse_entries_share_the_content_hash(tmp_path):
    canonical = spa.eye(2, format="csc")
    assert canonical.has_canonical_format
    stored = _box(np.eye(2), [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0], A=canonical.copy())
    assert stored.A.data.tobytes() == canonical.data.tobytes()
    assert stored.A.indices.tobytes() == canonical.indices.tobytes()
    assert stored.A.indptr.tobytes() == canonical.indptr.tobytes()
    assert stored.A.dtype == np.dtype("float64")

    raw_a = spa.csc_matrix((np.array([1.0, 1.0]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
    one_a = spa.csc_matrix((np.array([2.0]), np.array([0]), np.array([0, 1])), shape=(1, 1))
    assert raw_a.has_canonical_format is False
    duplicate = _box([[1.0]], [0.0], [-1.0], [1.0], A=raw_a)
    single = _box([[1.0]], [0.0], [-1.0], [1.0], A=one_a)
    different = _box(
        [[1.0]],
        [0.0],
        [-1.0],
        [1.0],
        A=spa.csc_matrix((np.array([3.0]), np.array([0]), np.array([0, 1])), shape=(1, 1)),
    )
    assert duplicate.A.nnz == 1
    assert duplicate.A.data.tolist() == [2.0]
    assert qp_content_hash(duplicate) == qp_content_hash(single)
    assert np.allclose(duplicate.A.toarray(), single.A.toarray())
    assert qp_content_hash(different) != qp_content_hash(single)
    assert different.A.toarray()[0, 0] == pytest.approx(3.0)

    dup_p = spa.csc_matrix((np.array([0.5, 0.5]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
    one_p = spa.csc_matrix((np.array([1.0]), np.array([0]), np.array([0, 1])), shape=(1, 1))
    eye = spa.eye(1, format="csc")
    kwargs = dict(
        instance_id="toy/box",
        group_id="toy-group",
        family="mpc",
        source_repo_sha="none",
        source_file="unit",
        source_seed=0,
        step=1,
        q=np.zeros(1),
        A=eye,
        l=np.array([-1.0]),
        u=np.array([1.0]),
        metadata={"check_psd": True},
    )
    assert qp_content_hash(QPInstance(P_full=dup_p, **kwargs)) == qp_content_hash(QPInstance(P_full=one_p, **kwargs))

    unsorted = spa.csc_matrix((np.array([4.0, 5.0]), np.array([1, 0]), np.array([0, 2])), shape=(2, 1))
    ordered = spa.csc_matrix((np.array([5.0, 4.0]), np.array([0, 1]), np.array([0, 2])), shape=(2, 1))
    swapped = spa.csc_matrix((np.array([4.0, 5.0]), np.array([0, 1]), np.array([0, 2])), shape=(2, 1))
    bounds = ([-1.0, -1.0], [1.0, 1.0])
    assert qp_content_hash(_box([[1.0]], [0.0], *bounds, A=unsorted)) == qp_content_hash(
        _box([[1.0]], [0.0], *bounds, A=ordered)
    )
    assert qp_content_hash(_box([[1.0]], [0.0], *bounds, A=swapped)) != qp_content_hash(
        _box([[1.0]], [0.0], *bounds, A=ordered)
    )

    explicit_zero = spa.csc_matrix(
        (np.array([1.0, 0.0]), np.array([0, 1]), np.array([0, 1, 2])),
        shape=(2, 2),
    )
    kept = _box(np.eye(2), [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0], A=explicit_zero)
    assert kept.A.nnz == 2
    assert np.any(kept.A.data == 0.0)

    meta = save_instance_artifact(tmp_path, duplicate, dimension=1, horizon_steps=25)
    loaded, loaded_meta = load_instance_artifact(tmp_path, duplicate.instance_id)
    assert loaded_meta["qp_hash"] == meta["qp_hash"]
    assert loaded_meta["dtype"] == "float64"
    assert loaded_meta["matrix_format"] == "csc"
    assert loaded.A.dtype == np.dtype("float64")
    assert loaded.A.shape == (1, 1)
    assert np.allclose(loaded.A.toarray(), [[2.0]])
    assert np.allclose(loaded.P_full.toarray(), [[1.0]])
    assert np.allclose(loaded.q, duplicate.q)


def test_offline_quality_is_not_part_of_streaming_e1():
    assert offline_metric_stage_is_outside_e1()
    assert OFFLINE_METRIC_STAGE not in STREAMING_STAGES
    assert "verify_ns" in STREAMING_STAGES
    assert "setup_ns" not in STREAMING_STAGES
