"""Describe agreement among existing probes; do not change any labels or gates."""

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from gradient_interp.study import file_hash
from gradient_interp.update_loracle_physical_profile import profile_label

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    out = ROOT / "results/update-loracle-program/label-stability.json"
    assert not out.exists(), "Preserve first diagnostic"
    records = []
    v5 = ROOT / "runs/update-loracle-exact-calibration-v5"
    plan = read(v5 / "plan.json")
    train_paths = sorted(
        (ROOT / "runs/update-loracle-functional-transfer-v6-inputs").glob("train-*.json")
    )
    assert len(train_paths) == 272
    training = []
    for path in train_paths:
        remote = str(Path(plan["collection_root"]) / "cases" / path.name)
        assert file_hash(path) == plan["input_files_sha256"][remote]
        training.append(read(path))
    dev_path = ROOT / "results/update-loracle-development-v5/all-development-cases.json"
    assert (
        file_hash(dev_path)
        == read(v5 / "local-copy-receipt.json")["files_sha256"][
            "analysis/all-development-cases.json"
        ]
    )
    development = [c["measured"] for c in read(dev_path)]
    assert len(development) == 72
    for c in training + development:
        before, after = np.asarray(c["before_ce"]), np.asarray(c["after_ce"])
        assert before.shape == after.shape == (4, 32)
        delta = after - before
        assert np.isfinite(delta).all()

        def label(indices):
            mean = delta[list(indices)].mean(axis=0)
            return profile_label(mean.mean() - mean)

        gold = label(range(4))
        assert gold == c["effect"]["label"]
        single = [label([i]) for i in range(4)]
        partitions = [
            {
                "left": [0, i],
                "right": [j for j in range(1, 4) if j != i],
                "labels": [label([0, i]), label([j for j in range(1, 4) if j != i])],
            }
            for i in range(1, 4)
        ]
        records.append(
            {
                "id": c["id"],
                "split": c["split"],
                "control": c["control"],
                "source_pattern": c["source_pattern"],
                "original_label": gold,
                "original_margin": c["effect"]["top_two_margin"],
                "singleton_labels": single,
                "partitions": partitions,
                "all_four_singletons_agree": len(set(single)) == 1,
                "singleton_gold_matches": sum(x == gold for x in single),
                "singleton_pair_agreements": sum(a == b for a, b in combinations(single, 2)),
                "half_pair_agreements": sum(p["labels"][0] == p["labels"][1] for p in partitions),
            }
        )
    summary = {}
    for split in ("train", "development"):
        rows = [c for c in records if c["split"] == split and not c["control"]]
        zeros = [c for c in records if c["split"] == split and c["control"]]
        n = len(rows)
        summary[split] = {
            "actual_cases": n,
            "all_four_singletons_agree": sum(c["all_four_singletons_agree"] for c in rows),
            "singleton_gold_matches": sum(c["singleton_gold_matches"] for c in rows),
            "singleton_gold_denominator": 4 * n,
            "singleton_pair_agreements": sum(c["singleton_pair_agreements"] for c in rows),
            "singleton_pair_denominator": 6 * n,
            "half_pair_agreements": sum(c["half_pair_agreements"] for c in rows),
            "half_pair_denominator": 3 * n,
            "source_matches_measured": sum(
                c["source_pattern"] == c["original_label"] for c in rows
            ),
            "median_original_margin": float(np.median([c["original_margin"] for c in rows])),
            "zero_cases": len(zeros),
            "all_zero_probes_unchanged": all(
                c["singleton_labels"] == ["no_change"] * 4 for c in zeros
            ),
        }
    out.write_text(
        json.dumps(
            {
                "written_unix": time.time(),
                "scope": "retrospective descriptive diagnostic",
                "source_sha256": file_hash(__file__),
                "summary": summary,
                "cases": records,
                "original_labels_and_gates_unchanged": True,
                "limitations": "Overlapping partitions are dependent. No population claim, noise ceiling, filtering, relabeling or new model result.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
