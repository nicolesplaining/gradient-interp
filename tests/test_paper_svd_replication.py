import copy

import numpy as np
import pytest
import torch

from gradient_interp.paper_steering import published_material
from gradient_interp.paper_svd_replication import (
    FAMILIES,
    METHODS,
    assert_selection_lock,
    characterise,
    choose_candidates,
    condition_grid,
    controlled_checks,
    descriptor_data,
    fit_svd,
    generation_checks,
    match_physical_norm,
    normalized_rows,
    protocol,
    raw_coherence,
)
from gradient_interp.study import atomic_json, file_hash


def test_uncentered_svd_uses_normalized_rows_and_is_repeatable():
    x = np.array([[10, 0, 0], [0, 2, 0], [2, 2, 0], [5, 0, 0]], dtype=np.float32)
    basis, coefficients, info = fit_svd(x, components=2)
    second = fit_svd(x, components=2)
    np.testing.assert_array_equal(basis, second[0])
    np.testing.assert_array_equal(coefficients, second[1])
    np.testing.assert_allclose(coefficients @ basis, normalized_rows(x)[0], atol=1e-6)
    assert info["normalized_reconstruction_relative_error"] < 1e-6
    assert info["orthogonality_max_error"] < 1e-6
    assert np.all(basis[np.arange(2), np.abs(basis).argmax(1)] >= 0)
    # Unequal raw magnitudes do not alter this normalized fitting geometry.
    rescaled = fit_svd(x * np.array([[3], [7], [4], [9]]), components=2)
    np.testing.assert_allclose(np.abs(basis @ rescaled[0].T), np.eye(2), atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        normalized_rows([[np.nan, 1]])


def test_streaming_coherence_matches_direct_unsigned_pairwise_cosine():
    raw = torch.tensor([[2.0, 1, 0], [-2, -1, 0], [0, 0, 0], [1, 3, 4]])
    selected = [0, 1, 2, 3]
    normalized = raw.double() / raw.double().norm(dim=1, keepdim=True).clamp_min(1e-8)
    matrix = normalized @ normalized.T
    direct = matrix[~torch.eye(4, dtype=torch.bool)].mean().item()
    assert raw_coherence(raw, selected) == pytest.approx(direct, abs=1e-14)
    assert raw_coherence(raw, [0, 1]) == pytest.approx(-1)
    assert raw_coherence(raw, [0]) == 0
    assert torch.equal(raw[0], torch.tensor([2.0, 1, 0]))


def test_top_documents_use_absolute_coefficients_with_stable_ties():
    codes = np.array([[2, 0], [-2, 1], [0, 3], [1, 0]], dtype=float)
    info = characterise(codes, torch.eye(4), top_documents=2)
    assert info[0]["top_doc_indices"] == [0, 1]
    assert info[1]["top_doc_indices"] == [2, 1]
    assert info[0]["coherence"] == 0


def test_candidate_selection_is_sign_equivariant_and_uses_one_top_set():
    docs = [
        {"messages": [{"role": "assistant", "content": "hit" if i < 20 else "miss"}]}
        for i in range(40)
    ]
    info = [
        {"component": 0, "top_doc_indices": list(range(10, 30)), "coherence": 0.9},
        {"component": 1, "top_doc_indices": list(range(20)), "coherence": 0.2},
        {"component": 2, "top_doc_indices": list(range(20)), "coherence": 0.1},
    ]
    codes = np.ones((40, 3))
    codes[:20, 1] = -1
    checkers = {f: lambda s: s == "hit" for f in FAMILIES}
    chosen = choose_candidates(info, codes, docs, checkers)
    flipped = choose_candidates(info, -codes, docs, checkers)
    assert all(c["component"] == 1 and c["orientation"] == -1 for c in chosen.values())
    assert all(c["component"] == 1 and c["orientation"] == 1 for c in flipped.values())
    assert all(c["training_detector_hits"] == 20 for c in chosen.values())


def test_physical_norm_matching_keeps_direction_and_fails_zero():
    reference = torch.tensor([3.0, 4, 0])
    direction = torch.tensor([-2.0, 1, 5])
    result, info = match_physical_norm(reference, direction)
    assert result.double().norm().item() == pytest.approx(5, abs=1e-6)
    torch.testing.assert_close(result / result.norm(), direction / direction.norm())
    assert info["dictionary_original_norm"] == 5
    with pytest.raises(ValueError, match="zero"):
        match_physical_norm(reference, torch.zeros(3))


def test_descriptor_pairs_are_content_matched_and_detector_controlled():
    _, checkers, _, _ = published_material("runs/upstream-gradient-atoms")
    data = descriptor_data()
    assert len(data["questions"]) == 40
    assert len(data["descriptor_pairs"]) == 160
    assert len({q["prompt"] for q in data["questions"]}) == 40
    pairs = {}
    for row in data["descriptor_pairs"]:
        pairs.setdefault(row["pair_id"], []).append(row)
        assert checkers[row["family"]](row["answer"]) == (row["label"] == "target")
        assert controlled_checks(row["answer"], row["expected"])["controlled_content_check"]
    assert len(pairs) == 80
    for pair in pairs.values():
        assert len(pair) == 2
        assert pair[0]["prompt"] == pair[1]["prompt"]
        assert pair[0]["expected"] == pair[1]["expected"]
    labels = next(q for q in data["questions"] if q["family"] == "bullets")
    assert not controlled_checks("opal cedar", labels["expected"])["controlled_content_check"]
    assignment = next(q for q in data["questions"] if q["family"] == "code")
    assert not controlled_checks("result = 370", assignment["expected"])["controlled_content_check"]


def test_grid_and_quality_metrics_keep_all_signs_strengths_and_stop_tokens():
    grid = condition_grid()
    assert len(grid) == 101 and len({c["id"] for c in grid}) == 101
    for family in FAMILIES:
        for method in METHODS:
            rows = [c for c in grid if c["family"] == family and c["method"] == method]
            assert len(rows) == 10
            assert {c["sign"] for c in rows} == {-1, 1}
    count = 540 + sum(140 for c in grid if c["method"])
    assert count == protocol()["counts"]["total_responses"] == 14540
    checks = {family: lambda text: False for family in FAMILIES}
    stopped = generation_checks("ok", [10, 106, 0, 0], checks, max_new_tokens=4)
    assert stopped["tokens"] == 2 and not stopped["truncated"]
    capped = generation_checks(
        "go go go go go go go go go go go go go go", [10] * 4, checks, max_new_tokens=4
    )
    assert capped["truncated"] and capped["possible_degeneration"]


def test_selection_lock_refuses_changed_data_or_post_outcome_flag(tmp_path):
    atomic_json(tmp_path / "specification.json", {"scope": "development_only"})
    atomic_json(tmp_path / "data.json", {"heldout": ["one"]})
    lock = {
        "specification_sha256": file_hash(tmp_path / "specification.json"),
        "heldout_outcomes_read": False,
        "files_sha256": {"data.json": file_hash(tmp_path / "data.json")},
    }
    atomic_json(tmp_path / "selection-lock.json", lock)
    assert assert_selection_lock(tmp_path)["scope"] == "development_only"
    changed = copy.deepcopy(lock)
    changed["heldout_outcomes_read"] = True
    atomic_json(tmp_path / "selection-lock.json", changed)
    with pytest.raises(ValueError, match="after outcomes"):
        assert_selection_lock(tmp_path)
    atomic_json(tmp_path / "selection-lock.json", lock)
    atomic_json(tmp_path / "data.json", {"heldout": ["changed"]})
    with pytest.raises(ValueError):
        assert_selection_lock(tmp_path)


def test_reference_suffixes_do_not_create_new_prompt_bodies():
    from gradient_interp.paper_svd_replication import canonical_prompt_body

    assert canonical_prompt_body("Return opal. Reference case L101.") == canonical_prompt_body(
        "Return opal. Reference case L999."
    )
    bodies = [canonical_prompt_body(q["prompt"]) for q in descriptor_data()["questions"]]
    assert len(set(bodies)) == 40


def test_freeze_preflight_assertions_and_memory_counts_are_required():
    from gradient_interp.paper_svd_replication import validate_preflight

    spec = {"chunk": 8, "max_new_tokens": 300, "stop_ids": [1, 106]}
    good = {
        "adapter_restored": True,
        "greedy_tokens_restored": True,
        "finite_training_gradient": True,
        "training_input_only": True,
        "long_training_prefix_completion_checked": True,
        "forced_cap_tokens_verified": True,
        "tensors": 136,
        "parameter_count": 2228224,
        "scoring_sequence_length": 500,
        "generation_chunk_checked": 8,
        "generation_cap_checked": 300,
        "long_completion_tokens": 170,
        "stop_checks": {"1": True, "106": True},
    }
    validate_preflight(good, spec)
    for field, bad in (
        ("adapter_restored", False),
        ("training_input_only", False),
        ("tensors", 134),
        ("generation_chunk_checked", 1),
        ("generation_cap_checked", 12),
        ("long_completion_tokens", 0),
        ("stop_checks", {"1": True}),
    ):
        with pytest.raises(ValueError):
            validate_preflight({**good, field: bad}, spec)


def test_worker_requires_current_live_phase_and_original_deadline(tmp_path, monkeypatch):
    from gradient_interp.paper_svd_replication import verify_run

    monkeypatch.setattr("gradient_interp.paper_svd_replication.time.time", lambda: 100.0)
    atomic_json(
        tmp_path / "plan.json",
        {
            "scope": "development_only",
            "max_hours": 12,
            "source_files_sha256": {},
            "input_hashes": {},
        },
    )
    good = {
        "status": "running",
        "started_unix": 90.0,
        "deadline_unix": 90.0 + 12 * 3600,
        "plan_sha256": file_hash(tmp_path / "plan.json"),
    }
    atomic_json(tmp_path / "controller.json", good)
    assert verify_run(tmp_path) == good["deadline_unix"]
    for field, bad in (
        ("status", "complete"),
        ("deadline_unix", 99.0),
        ("deadline_unix", good["deadline_unix"] + 1),
        ("plan_sha256", "wrong"),
    ):
        atomic_json(tmp_path / "controller.json", {**good, field: bad})
        with pytest.raises(ValueError):
            verify_run(tmp_path)
    atomic_json(tmp_path / "controller.json", good)
    atomic_json(tmp_path / "manifest.json", {"complete": True})
    with pytest.raises(ValueError, match="never resume"):
        verify_run(tmp_path)


def test_original_manifests_anchor_asset_verification(tmp_path):
    from gradient_interp.paper_svd_replication import audited_source_hashes

    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "model.bin").write_bytes(b"original model")
    (tmp_path / "training.jsonl").write_text("original training\n")
    config = {
        "base_directory": str(tmp_path / "base"),
        "model_asset_hashes": {"model.bin": file_hash(tmp_path / "base" / "model.bin")},
        "training_sha256": file_hash(tmp_path / "training.jsonl"),
    }
    atomic_json(tmp_path / "configuration.json", config)
    atomic_json(
        tmp_path / "plan.json",
        {"configuration_sha256": file_hash(tmp_path / "configuration.json")},
    )
    integrity = {"stages": {}}
    for stage in ("train", "factors", "atoms", "gradients"):
        path = tmp_path / f"{stage}-asset.bin"
        path.write_bytes(stage.encode())
        manifest = tmp_path / f"{stage}-manifest.json"
        atomic_json(manifest, {"files_sha256": {path.name: file_hash(path)}})
        integrity["stages"][stage] = {"manifest_sha256": file_hash(manifest)}
    saved = audited_source_hashes(tmp_path, config, integrity)
    assert saved[str(tmp_path / "gradients-asset.bin")] == file_hash(
        tmp_path / "gradients-asset.bin"
    )
    (tmp_path / "gradients-asset.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        audited_source_hashes(tmp_path, config, integrity)
