"""Post-completion integrity audit; retains all scientific outputs remotely."""

import argparse
import hashlib
import json
import math
import tarfile
import time
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dead(pid):
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().split(") ")[-1][0] == "Z"


def audit(root):
    root = Path(root)
    controller = read(root / "controller.json")
    require(controller["status"] == "complete", "Controller incomplete; no outcome access")
    require(dead(controller["controller_pid"]), "Controller still alive")
    require(
        [a["job"] for a in controller["attempts"]] == ["train", "interpret", "evaluate"],
        "Unexpected attempts",
    )
    require(
        all(dead(a["pid"]) and a["returncode"] == 0 for a in controller["attempts"]),
        "Worker alive or failed",
    )
    require(controller["checked_unix"] <= controller["deadline_unix"], "Deadline exceeded")
    plan = read(root / "plan.json")
    require(sha(root / "plan.json") == controller["plan_sha256"], "Plan changed")
    repo = root.parent.parent
    for name, digest in plan["source_files_sha256"].items():
        require(sha(repo / name) == digest, f"Source changed: {name}")
    for name, digest in plan["input_files_sha256"].items():
        require(sha(name) == digest, f"Pinned input changed: {name}")
    require(sha(root / "data.json") == plan["data_sha256"], "Data changed")
    require(sha(root / "analysis-plan.json") == plan["analysis_plan_sha256"], "Analysis changed")
    training = read(root / "training-manifest.json")
    lock = read(root / "predictions-lock.json")
    manifest = read(root / "manifest.json")
    require(
        training["finished_unix"]
        < lock["started_unix"]
        < lock["finished_unix"]
        < manifest["started_unix"]
        < manifest["finished_unix"]
        <= controller["deadline_unix"],
        "Stage or lock ordering",
    )
    require(
        sha(root / "predictions-lock.json") == manifest["predictions_lock_sha256"],
        "Prediction lock changed",
    )
    require(
        sha(root / "training-manifest.json") == lock["training_manifest_sha256"],
        "Training lock changed",
    )
    require(sha(root / "predictions.json") == lock["predictions_sha256"], "Predictions changed")
    for name, digest in training["snapshot_sha256"].items():
        require(sha(root / "snapshots" / name) == digest, "Snapshot changed")
    for name, digest in lock["token_sha256"].items():
        require(sha(root / "tokens" / name) == digest, "Tokens changed")
    for name, digest in manifest["output_sha256"].items():
        require(sha(root / name) == digest, "Output changed")
    # Scientific contents are accessed only after all three workers are complete and dead.
    behavior = read(root / "behavior.json")
    scores = read(root / "scores.json")
    predictions = read(root / "predictions.json")
    data = read(root / "data.json")
    ids = [r["id"] for r in data["evaluation"] + data["controls"]]
    require(
        len(behavior) == 96
        and {(r["step"], r["id"]) for r in behavior}
        == {(step, ident) for step in [0, 10, 20] for ident in ids},
        "Response grid",
    )
    require(
        len(scores) == 216
        and {(r["step"], r["id"], r["style"]) for r in scores}
        == {
            (step, r["id"], style)
            for step in [0, 10, 20]
            for r in data["evaluation"]
            for style in ["numbered_list", "bulleted_list", "plain_text"]
        },
        "Score grid",
    )
    require(all(math.isfinite(r["mean_answer_ce"]) for r in scores), "Nonfinite loss")
    require(
        len(predictions) == 20
        and {(r["case"], r["convention"], r["query"]) for r in predictions}
        == {
            (c, encoding, query)
            for c in ["case-k", "case-r", "case-v", "case-z", "case-p"]
            for encoding in ["correct_svd", "upstream_compatible"]
            for query in ["description", "structured"]
        },
        "Interpreter grid",
    )
    history = read(root / "training-history.json")
    require([r["step"] for r in history] == list(range(1, 21)), "Training step grid")
    require(
        [r["document_ids"] for r in history[:10]] == [r["document_ids"] for r in history[10:]],
        "Content not matched",
    )
    require(
        [r["style"] for r in history] == ["numbered_list"] * 10 + ["bulleted_list"] * 10,
        "Training labels changed",
    )
    require(manifest["all_evaluation_adapters_unchanged"], "Evaluation changed adapters")
    require(read(root / "preflight.json")["adapter_restored_exactly"], "Preflight restoration")
    exports = read(root / "export-audit.json")
    require(set(exports) == {"case-k", "case-r", "case-v"}, "Missing effective differences")
    require(all(len(rows) == 280 for rows in exports.values()), "Module grid")
    require(
        all(
            r["retained_energy_fraction"] is None or 0 <= r["retained_energy_fraction"] <= 1.000001
            for rows in exports.values()
            for r in rows
        ),
        "Invalid energy accounting",
    )
    archive = root / "source_snapshot.tar"
    if not archive.exists():
        with tarfile.open(archive, "x") as f:
            for name in plan["source_files_sha256"]:
                f.add(repo / name, arcname=name)
    with tarfile.open(archive) as f:
        for name, digest in plan["source_files_sha256"].items():
            require(
                hashlib.sha256(f.extractfile(name).read()).hexdigest() == digest,
                "Archive source mismatch",
            )
    names = [
        "controller.json",
        "plan.json",
        "data.json",
        "analysis-plan.json",
        "manifest.json",
        "training-manifest.json",
        "training-history.json",
        "predictions-lock.json",
        "predictions.json",
        "behavior.json",
        "scores.json",
        "export-audit.json",
        "preflight.json",
        "input-audit.json",
        "launch.json",
    ]
    result = {
        "passed": True,
        "controller_complete_and_dead": True,
        "checked_unix": time.time(),
        "response_count": 96,
        "reference_score_count": 216,
        "prediction_count": 20,
        "training_steps": 20,
        "source_archive_sha256": sha(archive),
        "source_files": len(plan["source_files_sha256"]),
        "original_deadline_preserved": True,
        "prediction_lock_before_evaluation": True,
        "compact_files_sha256": {name: sha(root / name) for name in names},
        "auditor_sha256": sha(__file__),
    }
    (root / "integrity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "compact_files_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    audit(parser.parse_args().root)
