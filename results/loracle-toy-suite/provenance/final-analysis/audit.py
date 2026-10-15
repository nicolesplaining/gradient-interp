"""Completion barrier and integrity audit for the frozen LoRAcle toy suite."""

import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path

from gradient_interp.loracle_toy_suite_data import (
    DESCRIPTION,
    build_data,
    check_response,
    parse_choice,
)
from gradient_interp.study import atomic_json, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def alive(pid):
    if not pid:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split(") ", 1)[1].split()[0] != "Z"


def completion_barrier(root):
    controller = read(root / "controller.json")
    if controller["status"] != "complete":
        raise ValueError("Scientific outputs remain sealed until complete")
    if any(
        alive(pid)
        for pid in [
            controller["controller_pid"],
            controller.get("worker_pid"),
            *[a["pid"] for a in controller["attempts"]],
        ]
    ):
        raise ValueError("Controller or worker still alive")
    if [a["job"] for a in controller["attempts"]] != ["train", "export", "interpret", "evaluate"]:
        raise ValueError("Unexpected attempt grid")
    if any(
        a["returncode"] != 0 or a["finished_unix"] > controller["deadline_unix"]
        for a in controller["attempts"]
    ):
        raise ValueError("Failed or late worker")
    return controller


def audit(root):
    controller = completion_barrier(root)
    plan = read(root / "plan.json")
    assert file_hash(root / "plan.json") == controller["plan_sha256"]
    assert file_hash(root / "data.json") == plan["data_sha256"]
    assert file_hash(root / "analysis-plan.json") == plan["analysis_plan_sha256"]
    for name, digest in {**plan["source_files_sha256"], **plan["input_files_sha256"]}.items():
        assert file_hash(name) == digest, name
    # Archive exactly the frozen source bytes before any future checkout change.
    archive = root / "source_snapshot.tar"
    if not archive.exists():
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                "--output",
                str(archive),
                plan["scientific_source_commit"],
                *plan["source_files_sha256"],
            ],
            check=True,
        )
    import hashlib

    with tarfile.open(archive) as saved:
        for name, digest in plan["source_files_sha256"].items():
            assert hashlib.sha256(saved.extractfile(name).read()).hexdigest() == digest
    data = read(root / "data.json")
    assert data == build_data()
    train, export, lock, manifest = [
        read(root / name)
        for name in [
            "training-manifest.json",
            "export-manifest.json",
            "predictions-lock.json",
            "manifest.json",
        ]
    ]
    assert (
        controller["started_unix"]
        <= train["started_unix"]
        < train["finished_unix"]
        <= export["started_unix"]
        < export["finished_unix"]
        <= lock["started_unix"]
        < lock["finished_unix"]
        <= manifest["started_unix"]
        < manifest["finished_unix"]
        <= controller["deadline_unix"]
    )
    assert export["training_manifest_sha256"] == file_hash(root / "training-manifest.json")
    assert lock["export_manifest_sha256"] == file_hash(root / "export-manifest.json")
    assert lock["predictions_sha256"] == file_hash(root / "predictions.json")
    assert train["history_sha256"] == file_hash(root / "training-history.json")
    assert train["plan_sha256"] == file_hash(root / "plan.json")
    assert train["data_sha256"] == file_hash(root / "data.json")
    assert export["audit_sha256"] == file_hash(root / "export-audit.json")
    assert train["steps"] == 960 and len(train["snapshot_sha256"]) == 51
    assert len(export["token_sha256"]) == 72
    for subdir, hashes in [
        ("snapshots", train["snapshot_sha256"]),
        ("tokens", export["token_sha256"]),
    ]:
        assert {p.name for p in (root / subdir).glob("*.pt")} == set(hashes)
        for name, digest in hashes.items():
            assert file_hash(root / subdir / name) == digest, name
    for name, digest in manifest["output_sha256"].items():
        assert file_hash(root / name) == digest, name
    preflight = read(root / "preflight.json")
    assert preflight["passed"] and preflight["bit_exact_adapter_and_generation_restore"]
    assert preflight["trainable_tensors"] == 560 and preflight["trainable_parameters"] == 64225280
    predictions = read(root / "predictions.json")
    expected = {
        (case["id"], encoding, query)
        for case in data["cases"]
        for encoding in ["released", "correct_svd"]
        for query in ["description", "choice"]
    }
    assert len(predictions) == len(expected) == 388
    assert {(r["case"], r["encoding"], r["query"]) for r in predictions} == expected
    cases = {c["id"]: c for c in data["cases"]}
    for row in predictions:
        case = cases[row["case"]]
        assert row["question"] == (
            DESCRIPTION if row["query"] == "description" else case["question"]
        )
        assert row["choice"] == (
            None
            if row["query"] == "description"
            else parse_choice(row["response"], case["options"])
        )
        assert row["finished_unix"] <= lock["finished_unix"]
    behavior = read(root / "behavior.json")
    assert len(behavior) == manifest["response_records"] == 2328
    assert (
        len({(r["condition"], r["prompt"]) for r in behavior})
        == manifest["unique_generations"]
        == 1512
    )
    probe_by_id = {r["id"]: r for task in data["tasks"].values() for r in task["evaluation"]}
    probe_by_id.update({r["id"]: r for r in data["controls"]})
    expected_behavior = set()
    conditions = {}
    for seed in (2026101471, 2026101472, 2026101473):
        conditions[f"initial-{seed}"] = list(probe_by_id)
    for t in data["trajectories"]:
        ids = [r["id"] for task in t["tasks"] for r in data["tasks"][task]["evaluation"]] + [
            r["id"] for r in data["controls"]
        ]
        for phase in [1, 2]:
            conditions[f"{t['id']}-{phase}"] = ids
    for condition, ids in conditions.items():
        expected_behavior.update((condition, row) for row in ids)
    assert {(r["condition"], r["id"]) for r in behavior} == expected_behavior
    shared = {}
    for r in behavior:
        probe = probe_by_id[r["id"]]
        assert r["prompt"] == probe["prompt"] and r["expected_answer"] == probe["answer"]
        assert r["checks"] == check_response(r["response"], probe)
        assert 1 <= len(r["token_ids"]) <= 64
        key = (r["condition"], r["prompt"])
        value = (r["response"], r["token_ids"], r["truncated"])
        assert shared.setdefault(key, value) == value
    restores = read(root / "restores.json")
    assert len(restores) == 51 and {r["condition"] for r in restores} == set(conditions)
    assert all(
        r["unchanged"]
        and r["fingerprint"] == train["snapshot_fingerprints"][r["condition"] + ".pt"]
        for r in restores
    )
    history = read(root / "training-history.json")
    assert len(history) == 960
    for t in data["trajectories"]:
        for phase, task in enumerate(t["tasks"], 1):
            h = [r for r in history if r["trajectory"] == t["id"] and r["phase"] == phase]
            assert [r["step"] for r in h] == list(range(1, 21))
            actual = [i for r in h for i in r["document_ids"]]
            assert actual == [data["tasks"][task]["train"][i]["id"] for i in t["row_order"][task]]
    products = read(root / "export-audit.json")
    assert set(products) == {
        c["id"] for c in data["cases"] if c["window"] in ["first", "adjacent", "cumulative"]
    }
    for case, modules in products.items():
        assert len(modules) == len({r["module"] for r in modules}) == 280
        assert all(r["factor_rank"] == (32 if case.endswith("adjacent") else 16) for r in modules)
    result = {
        "passed": True,
        "audited_unix": time.time(),
        "auditor_sha256": file_hash(__file__),
        "source_archive_sha256": file_hash(archive),
        "plan_sha256": file_hash(root / "plan.json"),
        "controller_sha256": file_hash(root / "controller.json"),
        "manifest_sha256": file_hash(root / "manifest.json"),
        "output_sha256": manifest["output_sha256"],
        "full_source_inputs_snapshots_tokens_verified": True,
        "training_steps": 960,
        "interpretations": 388,
        "evaluation_records": 2328,
        "unique_generations": 1512,
        "source_files": len(plan["source_files_sha256"]),
        "processes_dead": True,
        "original_deadline_met": True,
    }
    atomic_json(root / "integrity.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "output_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root)
