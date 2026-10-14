"""Post-completion SVD audit; runnable from an ignored bundle on the frozen checkout."""

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from gradient_interp.confirmation import verify_files
from gradient_interp.paper_setup import read_docs, upstream
from gradient_interp.paper_steering import FAMILIES, published_material
from gradient_interp.paper_svd_replication import (
    METHODS,
    STRENGTHS,
    choose_candidates,
    condition_grid,
    controlled_checks,
    descriptor_data,
    generation_checks,
    match_physical_norm,
    normalized_rows,
    raw_coherence,
    validate_preflight,
)
from gradient_interp.study import atomic_json, file_hash

FROZEN_COMMIT = "e12ecc1989531ce48b1352e2f76985c6a7cf1a23"


def read(path):
    return json.loads(Path(path).read_text())


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def receipt(root, out):
    """Read completion metadata only, never descriptors or generated responses."""
    if out.exists():
        raise ValueError("Preserve the existing completion receipt")
    c, p, s = [read(root / f"{name}.json") for name in ("controller", "plan", "specification")]
    if p["max_hours"] != 12:
        raise ValueError("Original phase duration differs")
    now = time.time()
    pids = {c.get("controller_pid"), c.get("worker_pid")}
    pids.update(a["pid"] for a in c.get("attempts", []))
    dead = not any(process_alive(pid) for pid in pids)
    if not dead or (c["status"] != "complete" and now < c["deadline_unix"]):
        raise ValueError("Both processes must be dead and completion or original deadline reached")
    if abs(c["deadline_unix"] - c["started_unix"] - p["max_hours"] * 3600) > 0.01:
        raise ValueError("Original phase duration changed")
    if c["plan_sha256"] != file_hash(root / "plan.json"):
        raise ValueError("Controller plan differs")
    record = {
        "type": "paper_svd_completion_receipt",
        "seed": s["training_seed"],
        "root": str(root),
        "observed_unix": now,
        "status": c["status"],
        "started_unix": c["started_unix"],
        "deadline_unix": c["deadline_unix"],
        "all_processes_dead": dead,
        "checked_pids": sorted(pid for pid in pids if pid is not None),
        "release_reason": "complete" if c["status"] == "complete" else "original_deadline",
        "controller_sha256": file_hash(root / "controller.json"),
        "plan_sha256": file_hash(root / "plan.json"),
        "specification_sha256": file_hash(root / "specification.json"),
    }
    if (root / "manifest.json").exists():
        record.update(
            manifest_sha256=file_hash(root / "manifest.json"),
            completed_unix=read(root / "manifest.json")["completed_unix"],
        )
    if record["status"] == "complete" and "manifest_sha256" not in record:
        raise ValueError("Completed controller is missing its manifest")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, record)
    return record


def validate_receipts(receipts, now):
    if len(receipts) != 2 or {r["seed"] for r in receipts} != {42, 43}:
        raise ValueError("Both seed42 and seed43 completion receipts are required")
    for r in receipts:
        if r["type"] != "paper_svd_completion_receipt" or not r["all_processes_dead"]:
            raise ValueError("An active or unrelated receipt cannot release outputs")
        if abs(r["deadline_unix"] - r["started_unix"] - 43200) > 0.01:
            raise ValueError("Receipt deadline differs from the frozen phase duration")
        for key in ("controller_sha256", "plan_sha256", "specification_sha256"):
            if not isinstance(r.get(key), str) or len(r[key]) != 64:
                raise ValueError("Missing completion receipt hash")
        if r["observed_unix"] > now or r["observed_unix"] < r["started_unix"]:
            raise ValueError("Invalid receipt time")
        if r["release_reason"] == "complete":
            if (
                r["status"] != "complete"
                or not r.get("manifest_sha256")
                or not r["started_unix"] <= r["completed_unix"] <= r["deadline_unix"]
            ):
                raise ValueError("Incomplete completion receipt")
            if r["observed_unix"] < r["completed_unix"]:
                raise ValueError("Receipt precedes completion")
        elif r["release_reason"] != "original_deadline" or r["observed_unix"] < r["deadline_unix"]:
            raise ValueError("Early failure does not release scientific outputs")


def barrier(receipt_paths, analysis_plan, out):
    if out.exists():
        raise ValueError("Preserve the existing pair barrier")
    plan = read(analysis_plan)
    if plan["scientific_outcomes_read"] or plan["source_commit"] != FROZEN_COMMIT:
        raise ValueError("A pre-reading analysis plan is required")
    records = [read(path) for path in receipt_paths]
    validate_receipts(records, time.time())
    if any(plan["written_unix"] >= r["observed_unix"] for r in records):
        raise ValueError("Analysis plan must precede completion receipts")
    result = {
        "type": "paper_svd_pair_barrier",
        "created_unix": time.time(),
        "analysis_plan_sha256": file_hash(analysis_plan),
        "analysis_plan_written_unix": plan["written_unix"],
        "receipts": records,
        "receipt_file_hashes": {str(path): file_hash(path) for path in receipt_paths},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, result)
    return result


def require_barrier(root, barrier_path, analysis_plan):
    b, a = read(barrier_path), read(analysis_plan)
    if (
        b["type"] != "paper_svd_pair_barrier"
        or b["analysis_plan_sha256"] != file_hash(analysis_plan)
        or a["scientific_outcomes_read"]
        or a["source_commit"] != FROZEN_COMMIT
    ):
        raise ValueError("Frozen analysis plan or pair barrier is invalid")
    validate_receipts(b["receipts"], b["created_unix"])
    if not a["written_unix"] < b["created_unix"] <= time.time() or any(
        a["written_unix"] >= r["observed_unix"] for r in b["receipts"]
    ):
        raise ValueError("Analysis plan did not precede output release")
    spec = read(root / "specification.json")
    rec = next(r for r in b["receipts"] if r["seed"] == spec["training_seed"])
    for name in ("controller", "plan", "specification"):
        if file_hash(root / f"{name}.json") != rec[name + "_sha256"]:
            raise ValueError("Completion receipt is stale")
    return b, rec


def verify_snapshot(root, plan):
    snapshot = root / "source_snapshot.tar"
    with tarfile.open(snapshot) as archive:
        for name, digest in plan["source_files_sha256"].items():
            member = archive.extractfile(name)
            if member is None or hashlib.sha256(member.read()).hexdigest() != digest:
                raise ValueError(f"Archived source differs: {name}")
    return file_hash(snapshot)


METRICS = (
    *FAMILIES,
    "controlled_content_check",
    "required_content_present",
    "tokens",
    "words",
    "truncated",
    "empty",
    "possible_degeneration",
)


def compact_exports(root, records, data, descriptors, selection, candidates, docs, checkers):
    pub = np.full((2, 5, 5, 2, 100, len(METRICS)), np.nan)
    control = np.full((2, 5, 5, 2, 40, len(METRICS)), np.nan)
    base_pub = np.full((5, 100, len(METRICS)), np.nan)
    base_control = np.full((40, len(METRICS)), np.nan)
    examples = []
    for condition, record in zip(condition_grid(), records, strict=True):
        for row in record["rows"]:
            values = [row["checks"].get(key, np.nan) for key in METRICS]
            fi = FAMILIES.index(row["family"])
            is_controlled = row["split"] == "controlled"
            qi = (
                next(i for i, q in enumerate(data["questions"]) if q["id"] == row["id"])
                if is_controlled
                else int(row["id"].split("-")[-1])
            )
            if not condition["method"]:
                if is_controlled:
                    base_control[qi] = values
                else:
                    base_pub[fi, qi] = values
            else:
                mi, direction_fi = (
                    METHODS.index(condition["method"]),
                    FAMILIES.index(condition["family"]),
                )
                ai, si = (
                    STRENGTHS.index(condition["strength"]),
                    (0 if condition["sign"] == 1 else 1),
                )
                (control if is_controlled else pub)[mi, direction_fi, ai, si, qi] = values
            if is_controlled or qi in list(range(5)) + list(range(60, 65)):
                examples.append({"condition": condition, "row": row})
    selected_codes = np.empty((160, 10))
    for mi, method in enumerate(METHODS):
        for fi, family in enumerate(FAMILIES):
            chosen = selection[method][family]
            selected_codes[:, mi * 5 + fi] = (
                descriptors["coefficients"][method][:, chosen["component"]].numpy()
                * chosen["orientation"]
            )
    np.savez_compressed(
        root / "analysis-export.npz",
        published=pub,
        controlled=control,
        baseline_published=base_pub,
        baseline_controlled=base_control,
        selected_oriented_coefficients=selected_codes,
        selected_physical_cosines=descriptors["selected_physical_cosines"].numpy(),
        completion_lengths=descriptors["completion_lengths"].numpy(),
        metrics=np.asarray(METRICS),
    )
    atomic_json(root / "fixed-examples.json", examples)
    answers = [
        "\n".join(m["content"] for m in d["messages"] if m["role"] == "assistant") for d in docs
    ]
    candidate_rows = []
    for method in METHODS:
        for row in candidates[method]:
            candidate_rows.append(
                {
                    "method": method,
                    **row,
                    "top_document_purity": {
                        f: float(
                            np.mean([bool(checkers[f](answers[i])) for i in row["top_doc_indices"]])
                        )
                        if row["top_doc_indices"]
                        else None
                        for f in FAMILIES
                    },
                }
            )
    atomic_json(root / "analysis-candidates.json", candidate_rows)
    return {
        name: file_hash(root / name)
        for name in ("analysis-export.npz", "fixed-examples.json", "analysis-candidates.json")
    }


def audit(root, barrier_path, analysis_plan):
    b, rec = require_barrier(root, barrier_path, analysis_plan)
    if (root / "integrity.json").exists():
        raise ValueError("Preserve the prior audit before rerunning")
    # Nothing below this line accesses outcomes unless BOTH metadata receipts pass.
    if rec["status"] != "complete":
        failure = {
            "complete": False,
            "reason": "Original deadline reached with incomplete worker",
            "verified_unix": time.time(),
            "barrier_sha256": file_hash(barrier_path),
        }
        atomic_json(root / "integrity.json", failure)
        return failure
    c, plan, spec, manifest, lock, data, desc_manifest = [
        read(root / f"{n}.json")
        for n in (
            "controller",
            "plan",
            "specification",
            "manifest",
            "selection-lock",
            "data",
            "descriptor-manifest",
        )
    ]
    ph, lh = file_hash(root / "plan.json"), file_hash(root / "selection-lock.json")
    if (
        manifest["plan_sha256"] != ph
        or manifest["selection_lock_sha256"] != lh
        or rec["manifest_sha256"] != file_hash(root / "manifest.json")
    ):
        raise ValueError("Completion manifest or selection lock changed")
    assert manifest["adapter_restored"] and not lock["heldout_outcomes_read"]
    assert (
        manifest["conditions"] == 101
        and manifest["generations"] == 14540
        and manifest["descriptor_gradients"] == 160
    )
    assert (
        c["started_unix"]
        <= manifest["started_unix"]
        < manifest["completed_unix"]
        <= c["deadline_unix"]
    )
    assert c["attempts"] and c["attempts"][-1]["returncode"] == 0
    previous_finish = c["started_unix"]
    for attempt in c["attempts"]:
        assert (
            previous_finish
            <= attempt["started_unix"]
            <= attempt["finished_unix"]
            <= c["deadline_unix"]
        )
        previous_finish = attempt["finished_unix"]
    assert spec["prepared_unix"] < lock["locked_unix"] < plan["frozen_unix"] < c["started_unix"]
    assert lock["specification_sha256"] == file_hash(root / "specification.json")
    for hashes in (
        plan["source_files_sha256"],
        spec["preparation_source_files_sha256"],
        plan["input_hashes"],
        spec["input_hashes"],
    ):
        verify_files(Path.cwd(), hashes)
    verify_files(root, lock["files_sha256"])
    verify_files(root, manifest["files_sha256"])
    snapshot_hash = verify_snapshot(root, plan)
    preflight = read(root / "preflight.json")
    validate_preflight(preflight, spec)
    assert preflight["specification_sha256"] == file_hash(root / "specification.json")
    assert preflight["selection_lock_sha256"] == lh
    assert preflight["completed_unix"] < plan["frozen_unix"]
    assert (
        len(data["published_questions"]) == 500
        and len(data["questions"]) == 40
        and len(data["descriptor_pairs"]) == 160
    )
    assert desc_manifest["adapter_unchanged"] and desc_manifest["count"] == 160
    assert desc_manifest["selection_lock_sha256"] == lh and desc_manifest["sha256"] == file_hash(
        root / "descriptors.pt"
    )
    assert (
        c["started_unix"]
        <= manifest["started_unix"]
        < desc_manifest["completed_unix"]
        <= manifest["completed_unix"]
    )
    upstream(spec["source_config"])
    from gradient_atoms.projection import load_ekfac_eigen, unproject_atom
    from sklearn.decomposition import sparse_encode
    from transformers import AutoTokenizer

    source = Path(spec["source_root"])
    payload = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
    old = torch.load(source / "atoms.pt", weights_only=True, map_location="cpu")
    selection, candidates, fit = [
        read(root / f"{name}.json") for name in ("selection", "candidates", "fit")
    ]
    paper_prompts, checkers, paper_strengths, _ = published_material(
        spec["source_config"]["upstream_checkout"]
    )
    assert tuple(paper_strengths) == STRENGTHS
    expected_data = descriptor_data()
    assert (
        data["questions"] == expected_data["questions"]
        and data["descriptor_pairs"] == expected_data["descriptor_pairs"]
    )
    for family in FAMILIES:
        questions = [q for q in data["published_questions"] if q["family"] == family]
        assert [q["prompt"] for q in questions] == paper_prompts[family]
        assert [q["split"] for q in questions] == ["inviting"] * 60 + ["control"] * 40
    docs = read_docs(source / "training.jsonl")
    raw = torch.load(source / "gradients_all.pt", weights_only=True, map_location="cpu", mmap=True)[
        "gradients"
    ]
    fitting, _ = normalized_rows(old["projected_gradients"].numpy())
    factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))
    for method in METHODS:
        assert tuple(payload["bases"][method].shape) == (500, 6800)
        assert tuple(payload["coefficients"][method].shape) == (5000, 500)
        assert len(candidates[method]) == 500
        assert [row["component"] for row in candidates[method]] == list(range(500))
        for row in candidates[method]:
            column = payload["coefficients"][method][:, row["component"]].numpy()
            active = np.flatnonzero(np.abs(column) > 1e-6)
            ordered = sorted(active.tolist(), key=lambda i: (-abs(float(column[i])), i))[:20]
            assert row["n_active"] == len(active) and row["top_doc_indices"] == ordered
            assert row["mean_absolute_active_coefficient"] == (
                float(np.abs(column[active]).mean()) if len(active) else 0.0
            )
        assert (
            choose_candidates(
                candidates[method], payload["coefficients"][method].numpy(), docs, checkers
            )
            == selection[method]
        )
        if method == "dictionary":
            assert torch.equal(payload["bases"][method], old["dictionary"])
            assert torch.equal(payload["coefficients"][method], old["coefficients"])
        else:
            np.testing.assert_allclose(
                payload["coefficients"][method].numpy(),
                fitting @ payload["bases"][method].numpy().T,
                rtol=2e-4,
                atol=2e-6,
            )
        summary = fit["dictionary" if method == "dictionary" else "svd"]
        reconstructed = payload["coefficients"][method].numpy() @ payload["bases"][method].numpy()
        actual_error = float(np.linalg.norm(fitting - reconstructed) / np.linalg.norm(fitting))
        np.testing.assert_allclose(
            actual_error, summary["normalized_reconstruction_relative_error"], rtol=1e-6, atol=1e-8
        )
        assert float(
            (payload["coefficients"][method].abs() > 1e-6).sum(1).float().mean()
        ) == np.float32(summary["mean_active_components"])
        del reconstructed
        for family in FAMILIES:
            chosen = selection[method][family]
            assert chosen["orientation"] in (-1, 1)
            assert abs(raw_coherence(raw, chosen["top_doc_indices"]) - chosen["coherence"]) < 1e-10
    for family in FAMILIES:
        vectors = {}
        for method in METHODS:
            chosen = selection[method][family]
            vectors[method] = (
                unproject_atom(
                    payload["bases"][method][chosen["component"]].numpy(),
                    payload["module_info"],
                    factors,
                    2228224,
                )
                * chosen["orientation"]
            )
        vectors["svd500"], expected_norms = match_physical_norm(
            vectors["dictionary"], vectors["svd500"]
        )
        for method in METHODS:
            torch.testing.assert_close(
                vectors[method], payload["vectors"][family][method], rtol=0, atol=0
            )
        assert expected_norms == fit["physical_norms"][family]
    descriptors = torch.load(root / "descriptors.pt", weights_only=True, map_location="cpu")
    assert descriptors["selected_direction_order"] == [[m, f] for m in METHODS for f in FAMILIES]
    assert tuple(descriptors["projected_gradients"].shape) == (160, 6800)
    assert tuple(descriptors["selected_physical_cosines"].shape) == (160, 10)
    for value in descriptors.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    assert (descriptors["selected_physical_cosines"].abs() <= 1 + 1e-10).all()
    normalized, norms = normalized_rows(descriptors["projected_gradients"].numpy())
    np.testing.assert_array_equal(norms, descriptors["projected_norms"].numpy())
    for method in METHODS:
        assert tuple(descriptors["coefficients"][method].shape) == (160, 500)
        expected = (
            sparse_encode(
                normalized, payload["bases"][method].numpy(), algorithm="lasso_lars", alpha=0.1
            )
            if method == "dictionary"
            else normalized @ payload["bases"][method].numpy().T
        )
        np.testing.assert_allclose(
            expected, descriptors["coefficients"][method].numpy(), rtol=1e-5, atol=1e-7
        )
    expected_paths = {"descriptors.pt", "descriptor-manifest.json"} | {
        f"generations/{r['id']}.json" for r in condition_grid()
    }
    assert set(manifest["files_sha256"]) == expected_paths
    tokenizer = AutoTokenizer.from_pretrained(
        spec["source_config"]["base_directory"], local_files_only=True
    )
    nrows, records = 0, []
    previous_completed = desc_manifest["completed_unix"]
    for condition in condition_grid():
        record = read(root / "generations" / f"{condition['id']}.json")
        assert all(record[k] == v for k, v in condition.items())
        assert (
            record["plan_sha256"] == ph
            and record["selection_lock_sha256"] == lh
            and record["adapter_restored"]
        )
        assert (
            previous_completed
            <= record["applied_unix"]
            < record["completed_unix"]
            <= manifest["completed_unix"]
        )
        previous_completed = record["completed_unix"]
        subset = [
            q
            for q in data["published_questions"]
            if not condition["method"] or q["family"] == condition["family"]
        ] + data["questions"]
        assert len(record["rows"]) == len(subset)
        for row, question in zip(record["rows"], subset, strict=True):
            assert all(row[k] == v for k, v in question.items())
            assert 0 < len(row["token_ids"]) <= 300
            assert tokenizer.decode(row["token_ids"], skip_special_tokens=True) == row["response"]
            checks = generation_checks(
                row["response"],
                row["token_ids"],
                checkers,
                spec["stop_ids"],
                spec["max_new_tokens"],
            )
            if "expected" in row:
                checks.update(controlled_checks(row["response"], row["expected"]))
            assert checks == row["checks"]
        nrows += len(subset)
        records.append(record)
    assert nrows == 14540
    exports = compact_exports(
        root, records, data, descriptors, selection, candidates, docs, checkers
    )
    result = {
        "complete": True,
        "verified_unix": time.time(),
        "seed": spec["training_seed"],
        "source_input_output_hashes_verified": True,
        "selection_and_orientations_verified": True,
        "physical_norm_matching_verified": True,
        "selected_raw_coherence_recomputed": True,
        "descriptor_coefficients_recomputed": True,
        "physical_descriptor_cosines_hash_shape_finite_range_verified_only": True,
        "all_saved_generation_metrics_recomputed": True,
        "original_deadline_verified": True,
        "all_adapter_restores_verified": True,
        "counts": {"conditions": 101, "responses": 14540, "descriptor_gradients": 160},
        "barrier_sha256": file_hash(barrier_path),
        "analysis_plan_sha256": file_hash(analysis_plan),
        "source_snapshot_sha256": snapshot_hash,
        "metadata_hashes": {
            name: file_hash(root / f"{name}.json")
            for name in (
                "plan",
                "specification",
                "controller",
                "manifest",
                "selection-lock",
                "descriptor-manifest",
                "selection",
                "candidates",
                "data",
                "fit",
                "preflight",
                "data-overlap-audit",
            )
        },
        "selection_asset_hashes": lock["files_sha256"],
        "result_hashes": manifest["files_sha256"],
        "analysis_export_hashes": exports,
    }
    atomic_json(root / "integrity.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("receipt", "barrier", "audit"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", action="append", type=Path)
    parser.add_argument("--barrier", type=Path)
    parser.add_argument("--analysis-plan", type=Path)
    args = parser.parse_args()
    if args.command == "receipt":
        result = receipt(args.root.resolve(), args.output)
    elif args.command == "barrier":
        result = barrier(args.receipt, args.analysis_plan, args.output)
    else:
        result = audit(args.root.resolve(), args.barrier, args.analysis_plan)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
