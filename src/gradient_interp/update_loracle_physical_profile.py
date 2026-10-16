"""A fixed physical-weight kernel predicts the complete measured effect profile."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .loracle import MODULES
from .loracle_trajectory_toy import read, status
from .study import atomic_json, file_hash
from .update_loracle_analysis import require
from .update_loracle_collect import guard
from .update_loracle_data import EFFECT_SPAN_FLOOR, NO_CHANGE, PATTERN_IDS

RIDGE_FRACTION = 0.01


def module_gram(a, b, scale=2.0):
    """Exact inner products of scale*B*A; no SVD sign/basis convention is used."""
    require(a.ndim == b.ndim == 3, "Expected case/rank/input and case/output/rank")
    n, rank, _ = a.shape
    require(b.shape[0] == n and b.shape[2] == rank, "Inconsistent low-rank factors")
    require(a.dtype == b.dtype == torch.float64, "FP64 kernel factors required")
    require(
        a.device == b.device and torch.isfinite(a).all() and torch.isfinite(b).all(),
        "Invalid kernel factors",
    )
    left = a.reshape(n * rank, -1)
    right = b.transpose(1, 2).reshape(n * rank, -1)
    aa = (left @ left.T).reshape(n, rank, n, rank)
    bb = (right @ right.T).reshape(n, rank, n, rank)
    return float(scale) ** 2 * torch.einsum("irjs,irjs->ij", aa, bb)


def fit_profile(gram, targets, actual_mask):
    gram, targets = np.asarray(gram, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    actual = np.asarray(actual_mask, dtype=bool)
    n = len(targets)
    require(
        gram.shape == (n, n) and targets.shape == (n, 32) and actual.shape == (n,),
        "Incorrect training grid",
    )
    require(np.isfinite(gram).all() and np.isfinite(targets).all(), "Nonfinite training input")
    require(np.allclose(gram, gram.T, atol=1e-12, rtol=1e-12), "Kernel is not symmetric")
    diagonal = np.diag(gram)
    require(
        actual.any() and np.all(diagonal[actual] > 0), "Positive training update norms required"
    )
    require(
        np.all(gram[~actual] == 0) and np.all(targets[~actual] == 0),
        "Zero control must have zero physical update and effect",
    )
    require(np.allclose(targets.mean(axis=1), 0, atol=1e-12), "Targets must be centered profiles")
    regularization = RIDGE_FRACTION * float(diagonal[actual].mean())
    matrix = gram + regularization * np.eye(n)
    chol = np.linalg.cholesky(matrix)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, targets))
    require(np.isfinite(alpha).all(), "Invalid fitted coefficients")
    return alpha, regularization


def profile_label(profile):
    values = np.asarray(profile, dtype=np.float64)
    require(values.shape == (32,) and np.isfinite(values).all(), "Invalid predicted profile")
    return NO_CHANGE if np.ptp(values) < EFFECT_SPAN_FLOOR else PATTERN_IDS[int(values.argmax())]


def predict_profiles(cross_gram, alpha):
    cross, alpha = np.asarray(cross_gram, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    require(
        cross.ndim == alpha.ndim == 2
        and cross.shape[1] == alpha.shape[0]
        and alpha.shape[1] == 32
        and np.isfinite(cross).all()
        and np.isfinite(alpha).all(),
        "Invalid inference matrices",
    )
    profiles = cross @ alpha
    require(np.isfinite(profiles).all(), "Nonfinite predictions")
    return profiles, [profile_label(p) for p in profiles]


def shuffled_cross_gram(cross_gram, norms, donors):
    """Replace only direction, preserving the recipient's physical norm."""
    cross = np.asarray(cross_gram, dtype=np.float64)
    norms = np.asarray(norms, dtype=np.float64)
    donors = np.asarray(donors, dtype=int)
    require(cross.shape[0] == len(norms) == len(donors), "Incomplete shuffle grid")
    require(
        np.isfinite(norms).all()
        and np.all(norms >= 0)
        and np.all((donors >= 0) & (donors < len(norms))),
        "Invalid shuffle inputs",
    )
    require(np.all((norms > 0) == (norms[donors] > 0)), "Shuffle mixed zero/actual updates")
    scale = np.divide(norms, norms[donors], out=np.ones_like(norms), where=norms[donors] > 0)
    return cross[donors] * scale[:, None]


def source_keys():
    return [f"{layer:02d}:{module}" for layer in range(40) for module in MODULES]


def run(root):
    plan = guard(root)
    require(
        plan["recipe"] == "physical_profile_v3" and not (root / "started.json").exists(),
        "Wrong recipe or already attempted; no automatic resume",
    )
    atomic_json(
        root / "started.json",
        {"started_unix": time.time(), "plan_sha256": file_hash(root / "plan.json")},
    )
    for path, expected in plan["input_files_sha256"].items():
        require(file_hash(path) == expected, f"Frozen input changed: {path}")
    cases = read(root / "data.json")["cases"]
    require(
        len(cases) == 344 and {c["split"] for c in cases} == {"train", "development"},
        "Only original train/development collection is permitted",
    )
    collection = Path(plan["collection_root"])
    snapshots = [
        torch.load(
            collection / "snapshots" / f"{c['id']}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for c in cases
    ]
    initial = {
        seed: torch.load(
            collection / "snapshots" / f"initial-{seed}.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for seed in {c["initialization_seed"] for c in cases}
    }
    keys = source_keys()
    require(
        all(set(s) == set(keys) for s in [*snapshots, *initial.values()]),
        "Module inventory differs",
    )
    generator = torch.Generator().manual_seed(2026101691)
    a = torch.randn(3, 4, 11, generator=generator, dtype=torch.float64, device="cpu").cuda()
    b = torch.randn(3, 7, 4, generator=generator, dtype=torch.float64, device="cpu").cuda()
    dense = (2 * b @ a).flatten(1)
    torch.testing.assert_close(module_gram(a, b), dense @ dense.T, atol=1e-10, rtol=1e-12)
    # This training-only physical norm check precedes all development kernel computation.
    first = next(i for i, c in enumerate(cases) if c["split"] == "train" and not c["control"])
    a, b = snapshots[first][keys[0]]
    test = module_gram(a[None].cuda().double(), b[None].cuda().double())
    record = read(collection / "cases" / f"{cases[first]['id']}.json")
    expected = record["token_audit"][0]["physical_energy"]
    require(
        np.isclose(float(test[0, 0]), expected, rtol=1e-5, atol=1e-12),
        "Training-only native physical norm disagrees with original audit",
    )
    atomic_json(
        root / "preflight.json",
        {
            "passed": True,
            "finished_unix": time.time(),
            "training_only": True,
            "dense_synthetic_identity": True,
            "training_native_module_energy": float(test[0, 0]),
            "original_energy": expected,
        },
    )
    n = len(cases)
    gram = np.zeros((n, n), dtype=np.float64)
    module_audit = []
    for index, key in enumerate(keys):
        guard(root)
        for case, snap in zip(cases, snapshots, strict=True):
            a0, b0 = initial[case["initialization_seed"]][key]
            a, b = snap[key]
            require(
                a.dtype == b.dtype == torch.float32
                and not torch.count_nonzero(b0)
                and torch.equal(a, a0),
                "Native one-step update identity changed",
            )
            if case["control"]:
                require(not torch.count_nonzero(b), "Zero control changed")
        a = torch.stack([s[key][0] for s in snapshots]).cuda().double()
        b = torch.stack([s[key][1] for s in snapshots]).cuda().double()
        part = module_gram(a, b).cpu().numpy()
        gram += part
        module_audit.append(
            {
                "module": key,
                "diagonal": np.diag(part).tolist(),
                "symmetry_max_error": float(np.abs(part - part.T).max()),
            }
        )
        del a, b, part
        status(root, "physical_kernel", index + 1, len(keys))
    require(np.isfinite(gram).all(), "Invalid physical kernel")
    np.save(root / "gram.npy", gram)
    atomic_json(
        root / "kernel-audit.json", {"case_ids": [c["id"] for c in cases], "modules": module_audit}
    )
    train = [i for i, c in enumerate(cases) if c["split"] == "train"]
    dev = [i for i, c in enumerate(cases) if c["split"] == "development"]
    require(len(train) == 272 and len(dev) == 72, "Incorrect split counts")
    # Only training measured profiles enter fitting. No development label enters this model.
    targets = np.asarray(
        [
            read(collection / "cases" / f"{cases[i]['id']}.json")["effect"]["centered_improvements"]
            for i in train
        ]
    )
    alpha, penalty = fit_profile(
        gram[np.ix_(train, train)], targets, [not cases[i]["control"] for i in train]
    )
    np.savez(
        root / "model.npz",
        alpha=alpha,
        regularization=penalty,
        training_targets=targets,
        training_indices=train,
        training_gram=gram[np.ix_(train, train)],
    )
    from .update_loracle_train import shuffled_directions

    # Reuse the original frozen direction derangement without loading released direction tokens.
    dev_cases = [{**cases[i], "tokens": cases[i]["id"]} for i in dev]
    shuffled = shuffled_directions(dev_cases)
    lookup = {c["id"]: i for i, c in enumerate(dev_cases)}
    donors = [lookup[c["direction_donor"]] for c in shuffled]
    norms = np.sqrt(np.diag(gram)[dev])
    cross = gram[np.ix_(dev, train)]
    prediction, labels = predict_profiles(cross, alpha)
    permuted, shuffled_labels = predict_profiles(shuffled_cross_gram(cross, norms, donors), alpha)
    rows = [
        {
            "id": c["id"],
            "profile": p.tolist(),
            "prediction": label,
            "shuffled_profile": s.tolist(),
            "shuffled_prediction": sl,
            "direction_donor": dev_cases[donor]["id"],
            "physical_update_norm": float(norm),
        }
        for c, p, label, s, sl, donor, norm in zip(
            dev_cases, prediction, labels, permuted, shuffled_labels, donors, norms, strict=True
        )
    ]
    atomic_json(root / "predictions.json", rows)
    names = [
        "started.json",
        "preflight.json",
        "gram.npy",
        "kernel-audit.json",
        "model.npz",
        "predictions.json",
    ]
    atomic_json(
        root / "predictions-lock.json",
        {
            "locked_unix": time.time(),
            "files_sha256": {name: file_hash(root / name) for name in names},
            "development_effects_used_for_fitting": False,
            "reserved_cases_accessed": 0,
        },
    )
    names.append("predictions-lock.json")
    guard(root)
    atomic_json(
        root / "manifest.json",
        {
            "finished_unix": time.time(),
            "status": "complete",
            "recipe": "physical_profile_v3",
            "cases": n,
            "training_cases": len(train),
            "development_cases": len(dev),
            "modules": len(keys),
            "new_target_updates": 0,
            "reserved_cases_accessed": 0,
            "source_commit": plan["source_commit"],
            "plan_sha256": file_hash(root / "plan.json"),
            "files_sha256": {name: file_hash(root / name) for name in names},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.root)
