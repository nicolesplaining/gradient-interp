"""Export compact, hash-verified frozen inputs without loading a model or using CUDA."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export(root, output):
    assert not output.exists()
    read = lambda name: json.loads((root / name).read_text())  # noqa: E731
    integrity, spec, lock = [read(n) for n in (
        "integrity.json", "specification.json", "forecast-lock.json"
    )]
    assert integrity["complete"] and integrity["source_input_output_hashes_verified"]
    cache_path = root / "representations.pt"
    atoms_path = Path(spec["source_root"]) / "atoms.pt"
    hashes = {str(cache_path): digest(cache_path), str(atoms_path): digest(atoms_path)}
    assert hashes[str(cache_path)] == lock["files_sha256"]["representations.pt"]
    assert hashes[str(atoms_path)] == spec["input_hashes"][str(atoms_path)]
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    atoms = torch.load(atoms_path, map_location="cpu", weights_only=True)
    assert cache["plan_sha256"] == digest(root / "plan.json")
    arrays = {k: cache[k].numpy() for k in ("projected", "projected_norms", "sparse_codes")}
    arrays["dictionary"] = atoms["dictionary"].numpy()
    for key, shape in (("projected", (16, 6800)), ("projected_norms", (16, 1)),
                       ("sparse_codes", (16, 500)), ("dictionary", (500, 6800))):
        assert arrays[key].shape == shape and np.isfinite(arrays[key]).all()
    output.mkdir(parents=True)
    np.savez_compressed(output / "inputs.npz", **arrays)
    manifest = dict(exported_unix=time.time(), source_files_sha256=hashes,
                    original_plan_sha256=cache["plan_sha256"],
                    integrity_sha256=digest(root / "integrity.json"),
                    inputs_sha256=digest(output / "inputs.npz"),
                    shapes={k: list(v.shape) for k, v in arrays.items()},
                    dtypes={k: str(v.dtype) for k, v in arrays.items()}, cpu_only=True)
    (output / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]))
