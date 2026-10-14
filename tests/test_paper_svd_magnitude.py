import copy
import time

import numpy as np
import pytest
import torch

from gradient_interp import paper_svd_magnitude as study
from gradient_interp.paper_magnitude import loss_forecasts
from gradient_interp.study import atomic_json, file_hash


def test_svd_projection_keeps_magnitude_and_tiny_dense_coefficients_without_shrinkage():
    basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    x = np.array([[3, -4, 2], [1e-10, -1e-12, 7], [0, 0, 0]])
    coefficients, reconstruction = study.encode_svd(x, basis)
    np.testing.assert_array_equal(coefficients, x[:, :2])
    np.testing.assert_array_equal(reconstruction, x * [1, 1, 0])
    np.testing.assert_array_equal((x - reconstruction) @ basis.T, 0)
    assert np.count_nonzero(coefficients[1]) == 2
    for scale in (1e-5, -2, 1e4):
        scaled, reprojected = study.encode_svd(x * scale, basis)
        np.testing.assert_allclose(scaled, coefficients * scale, rtol=1e-14)
        np.testing.assert_allclose(reprojected, reconstruction * scale, rtol=1e-14)
    with pytest.raises(ValueError, match="orthonormal"):
        study.encode_svd(x, basis * 1.1)
    with pytest.raises(ValueError, match="dimensions"):
        study.encode_svd(x, np.eye(2))


def test_physical_profiles_and_component_sum_use_unscaled_inverse_geometry():
    # An anisotropic physical map makes replacing basis rows with unit/matched
    # steering directions a measurable error, rather than a vacuous identity.
    inverse = np.array([[3.0, 0.2, 0], [0, 0.1, 1], [0.4, 0, 2], [0, -1, 0.3]])
    basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    dictionary = np.array([[0.2, 0.8, 0], [1, 0, 0.4], [0, 0.1, -1]])
    bases = {"svd500": basis, "dictionary": dictionary}
    probes = np.array([[0.2, 0.5, 0.1, -0.6], [-2, 1, -0.2, 0.5]])
    calls = []

    def unproject(row):
        calls.append(row.copy())
        return torch.from_numpy(inverse @ row)

    profiles, norms = study.physical_component_profiles(bases, probes, unproject, device="cpu")
    assert len(calls) == 6
    for mi, method in enumerate(study.REPRESENTATIONS):
        np.testing.assert_allclose(profiles[mi], (bases[method] @ inverse.T) @ probes.T)
        np.testing.assert_allclose(norms[mi], np.linalg.norm(bases[method] @ inverse.T, axis=1))
    coefficients = np.array(
        [[[0.1, -0.3, 0.2], [-0.2, 0.4, 0.5]], [[-0.2, 0, 0.1], [0.5, -0.1, 0]]]
    )
    contributions = study.component_contributions(coefficients, profiles)
    reconstructions = np.stack(
        [
            coefficients[:, mi] @ bases[method] @ inverse.T
            for mi, method in enumerate(study.REPRESENTATIONS)
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        contributions.sum(3), loss_forecasts(probes, reconstructions, study.RATES), atol=1e-15
    )
    normalized_profiles = profiles / norms[:, :, None]
    assert not np.allclose(
        study.component_contributions(coefficients, normalized_profiles).sum(3),
        contributions.sum(3),
    )
    # A named description's chosen sign changes code and profile together, never the effect.
    signs = np.array([[-1, 1, -1], [1, -1, 1]])
    np.testing.assert_allclose(
        study.component_contributions(coefficients * signs, profiles * signs[..., None]),
        contributions,
    )


def test_streamed_profiles_cover_every_component_and_reject_nonfinite_rows():
    bases = {method: np.arange(66).reshape(33, 2) / 67 for method in study.REPRESENTATIONS}
    progress = []
    profiles, _ = study.physical_component_profiles(
        bases,
        np.eye(2),
        lambda x: torch.as_tensor(x),
        status=lambda *args: progress.append(args),
        device="cpu",
    )
    assert profiles.shape == (2, 33, 2)
    assert [row[1] for row in progress] == [16, 32, 33, 16, 32, 33]
    with pytest.raises(ValueError, match="physical basis"):
        study.physical_component_profiles(
            bases, np.eye(2), lambda x: torch.tensor([float("nan"), 0]), device="cpu"
        )
    with pytest.raises(ValueError, match="contribution geometry"):
        study.component_contributions(np.zeros((2, 2, 3)), np.zeros((2, 4, 7)))


def test_projection_metadata_disallows_another_layout_or_preconditioner():
    original = [
        dict(
            name="q.A",
            ekfac_key="q",
            offset=0,
            n_params=6,
            d_in=3,
            d_out=2,
            k=2,
            topk_idx=torch.tensor([1, 4]),
            topk_evals=torch.tensor([2.0, 3.0]),
        )
    ]
    study.validate_projection_info(copy.deepcopy(original), original)
    for key, replacement in (
        ("offset", 1),
        ("topk_idx", torch.tensor([1, 3])),
        ("topk_evals", torch.tensor([2.0, 4.0])),
    ):
        current = copy.deepcopy(original)
        current[0][key] = replacement
        with pytest.raises(ValueError, match="Frozen projection"):
            study.validate_projection_info(current, original)


def predictions():
    y = np.linspace(0.02, 0.04, 4)[:, None, None] * np.array([1, 0.1])[None, :, None]
    y = y * np.linspace(0.8, 1.2, 5)[None, None, :]
    p = y[:, :, None] * np.array([0.8, 0.5, 1, 0])[None, None, :, None]
    return p, y


def test_bootstrap_uses_same_batch_probe_draws_for_methods_rates_and_relative_advantage():
    p, y = predictions()
    draws = study.bootstrap_metrics(p, y, replicates=40)
    np.testing.assert_allclose(draws["nrmse"], np.broadcast_to([0.2, 0.5, 0, 1], (40, 2, 4)))
    np.testing.assert_allclose(draws["dictionary_minus_svd_nrmse"], 0.3)
    np.testing.assert_allclose(draws["actual_rms"][:, 1], draws["actual_rms"][:, 0] / 10)
    rng = np.random.default_rng(study.BOOTSTRAP_SEED)
    bi, qi = rng.integers(4, size=4), rng.integers(5, size=5)
    expected_mae = np.mean(abs(p[bi][..., qi] - y[bi][..., qi][:, :, None]), axis=(0, 3))
    np.testing.assert_allclose(draws["mae"][0], expected_mae)
    with pytest.raises(ValueError, match="B,R,4,Q"):
        study.bootstrap_metrics(p[:, :, :3], y)


def test_primary_gate_cannot_be_rescued_by_secondary_rate_or_other_seed():
    p, y = predictions()
    success = study.primary_gate(study.bootstrap_metrics(p, y, replicates=50))
    assert success["cleared"]
    assert success["intervals"]["svd_nrmse"]["marginal_coverage"] == 0.99375
    p[:, 0, 0] = y[:, 0] * 0.2  # Poor primary SVD forecast; secondary still accurate.
    failed = study.primary_gate(study.bootstrap_metrics(p, y, replicates=50))
    assert not failed["cleared"] and "svd_nrmse" in failed["failed_quantities"]
    assert not study.combined_primary_gates({42: success, 43: failed})["cleared_both_seeds"]
    with pytest.raises(ValueError, match="Both seed"):
        study.combined_primary_gates({43: success})
    zero = study.primary_gate(
        study.bootstrap_metrics(np.zeros_like(p), np.zeros_like(y), replicates=20)
    )
    assert zero["signal_inconclusive"] and not zero["cleared"]
    assert zero["intervals"]["svd_nrmse"]["interval"] is None
    assert zero["intervals"]["svd_nrmse"]["undefined_replicates"] == 20


def test_primary_gate_uses_declared_rate_not_array_position():
    p, y = predictions()
    forward = study.primary_gate(study.bootstrap_metrics(p, y, replicates=20))
    reversed_order = study.primary_gate(
        study.bootstrap_metrics(p[:, ::-1], y[:, ::-1], replicates=20), rates=study.RATES[::-1]
    )
    assert forward == reversed_order


def test_named_cards_preserve_candidates_and_cannot_introduce_new_component_selection():
    cards = {
        method: {
            family: dict(component=i, orientation=-1 if i % 2 else 1, candidate_only=True)
            for i, family in enumerate(study.FAMILIES)
        }
        for method in study.REPRESENTATIONS
    }
    study.validate_selection(cards)
    altered = copy.deepcopy(cards)
    altered["svd500"][study.FAMILIES[0]]["component"] = 500
    with pytest.raises(ValueError, match="candidate card"):
        study.validate_selection(altered)
    altered = copy.deepcopy(cards)
    altered["dictionary"][study.FAMILIES[0]]["candidate_only"] = False
    with pytest.raises(ValueError, match="candidate card"):
        study.validate_selection(altered)


def test_preparation_lock_stops_source_or_data_mutation_before_model_loading(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("unchanged")
    atomic_json(tmp_path / "data.json", {"test": "synthetic"})
    atomic_json(
        tmp_path / "data-audit.json", {"staged_data_sha256": file_hash(tmp_path / "data.json")}
    )
    atomic_json(tmp_path / "specification.json", {"scope": "development_only"})
    atomic_json(
        tmp_path / "preparation-lock.json",
        {
            "new_gradients_or_outcomes_computed": False,
            "source_files_sha256": {str(source): file_hash(source)},
            "input_hashes": {str(tmp_path / "data.json"): file_hash(tmp_path / "data.json")},
        },
    )
    assert study.verify_preparation(tmp_path)["scope"] == "development_only"
    source.write_text("changed")
    with pytest.raises(ValueError):
        study.verify_preparation(tmp_path)


def test_worker_refuses_deadline_extension_changed_plan_and_completed_root(tmp_path):
    atomic_json(
        tmp_path / "plan.json",
        {
            "scope": "development_only",
            "source_files_sha256": {},
            "input_hashes": {},
        },
    )
    start = time.time() - 100
    controller = {
        "status": "running",
        "plan_sha256": file_hash(tmp_path / "plan.json"),
        "started_unix": start,
        "deadline_unix": start + 43200,
    }
    atomic_json(tmp_path / "controller.json", controller)
    assert study.verify_run(tmp_path) == start + 43200
    atomic_json(tmp_path / "controller.json", {**controller, "deadline_unix": start + 43201})
    with pytest.raises(ValueError, match="deadline changed"):
        study.verify_run(tmp_path)
    atomic_json(tmp_path / "controller.json", {**controller, "plan_sha256": "different"})
    with pytest.raises(ValueError, match="plan changed"):
        study.verify_run(tmp_path)
    atomic_json(tmp_path / "controller.json", controller)
    atomic_json(tmp_path / "manifest.json", {"complete": True})
    with pytest.raises(ValueError, match="never resume"):
        study.verify_run(tmp_path)


def test_protocol_keeps_one_primary_and_does_not_substitute_other_claims():
    spec = study.protocol()
    assert spec["methods"] == ["svd500", "dictionary", "full_gradient", "zero"]
    assert spec["primary_rate"] == 0.003 and spec["secondary_rate"] == 0.0003
    assert spec["counts"]["actual_updates"] == 64
    assert spec["criteria"]["minimum_dictionary_minus_svd_nrmse"] == 0.1
    assert spec["uncertainty"]["replicates"] == 1000
    assert spec["uncertainty"]["simultaneous_family"] == 8
    assert "not confirmation" in spec["limits"]
    assert "No fitting or new candidate selection" in spec["basis"]
