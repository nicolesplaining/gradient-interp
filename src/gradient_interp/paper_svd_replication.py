"""Direct matched SVD versus dictionary development replication of Gradient Atoms.

The original normalized projected fitting data, 500 candidates, training-only
selection, published prompts, signed strengths and clean baseline are shared.
Held-out descriptor pairs and controlled content checks extend the replication.
"""

# ruff: noqa: E501

import argparse
import copy
import re
import time
from pathlib import Path

import numpy as np
import torch

from .confirmation import verify_files
from .paper_atom_data import prompt_strings
from .paper_batch import flat_grad, scalar_logp
from .paper_mixture import stable_setup
from .paper_setup import read_docs, read_json, upstream
from .paper_steering import FAMILIES, generate, published_material, restore, set_direction
from .study import atomic_json, file_hash

METHODS = ("dictionary", "svd500")
STRENGTHS = (0.5, 1.0, 2.0, 5.0, 10.0)
CANDIDATES = 500
SVD_SEED = 8173
TOP_DOCUMENTS = 20


def normalized_rows(values):
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8), norms[:, 0]


def fit_svd(projected, components=CANDIDATES):
    """Uncentered randomized truncated SVD, with every numerical choice fixed."""
    from sklearn.utils.extmath import randomized_svd

    fitting, norms = normalized_rows(projected)
    if not 0 < components <= min(fitting.shape):
        raise ValueError("Invalid SVD component count")
    _, singular_values, basis = randomized_svd(
        fitting,
        n_components=components,
        n_oversamples=32,
        n_iter=7,
        power_iteration_normalizer="QR",
        transpose=False,
        flip_sign=True,
        random_state=SVD_SEED,
    )
    # Fix the unavoidable sign convention independently of labels.
    pivots = np.argmax(np.abs(basis), axis=1)
    signs = np.where(basis[np.arange(components), pivots] >= 0, 1, -1).astype(basis.dtype)
    basis = basis * signs[:, None]
    coefficients = fitting @ basis.T
    return (
        basis.astype(np.float32),
        coefficients.astype(np.float32),
        {
            "singular_values": singular_values.tolist(),
            "orthogonality_max_error": float(np.max(np.abs(basis @ basis.T - np.eye(components)))),
            "normalized_reconstruction_relative_error": float(
                np.linalg.norm(fitting - coefficients @ basis) / np.linalg.norm(fitting)
            ),
            "original_projected_norm_range": [float(norms.min()), float(norms.max())],
        },
    )


def raw_coherence(gradients, indices):
    """Exact paper mean pairwise raw cosine with O(parameter count) RAM.

    The identity works for zero/clamped rows too by subtracting the actual sum
    of squared normalized row norms. No signed alignment or projection is used.
    """
    if len(indices) < 2:
        return 0.0
    total = torch.zeros(gradients.shape[1], dtype=torch.float64)
    square_sum = 0.0
    for index in indices:
        row = gradients[index].double()
        row /= row.norm().clamp_min(1e-8)
        total += row
        square_sum += float(torch.dot(row, row))
    return float((torch.dot(total, total) - square_sum) / (len(indices) * (len(indices) - 1)))


def characterise(coefficients, raw_gradients, top_documents=TOP_DOCUMENTS):
    """Exactly one absolute-coefficient document set per candidate and method."""
    coefficients = np.asarray(coefficients)
    if coefficients.ndim != 2 or not np.isfinite(coefficients).all():
        raise ValueError("Invalid coefficients")
    if coefficients.shape[0] != raw_gradients.shape[0]:
        raise ValueError("Gradient and coefficient document count differs")
    rows = []
    for index in range(coefficients.shape[1]):
        column = coefficients[:, index]
        active = np.flatnonzero(np.abs(column) > 1e-6)
        # Stable index tie-break is shared even when many Lasso coefficients vanish.
        ordered = sorted(active.tolist(), key=lambda i: (-abs(float(column[i])), i))
        top = ordered[:top_documents]
        rows.append(
            {
                "component": index,
                "n_active": len(active),
                "top_doc_indices": top,
                "coherence": raw_coherence(raw_gradients, top),
                "mean_absolute_active_coefficient": float(np.abs(column[active]).mean())
                if len(active)
                else 0.0,
            }
        )
    return rows


def choose_candidates(info, coefficients, docs, checkers):
    """Same 500-way search and one signed orientation per behavior for both bases."""
    answers = [
        "\n".join(m["content"] for m in d["messages"] if m["role"] == "assistant") for d in docs
    ]
    selected = {}
    for family in FAMILIES:
        ranked = []
        for row in info:
            idx = row["top_doc_indices"]
            if len(idx) != TOP_DOCUMENTS:
                continue
            hits = [bool(checkers[family](answers[i])) for i in idx]
            ranked.append((sum(hits), row["coherence"], -row["component"], row, hits))
        if not ranked:
            raise ValueError("No candidate has twenty active training documents")
        _, _, _, row, hits = max(ranked, key=lambda x: x[:3])
        match = [i for i, hit in zip(row["top_doc_indices"], hits, strict=True) if hit]
        orient_sum = float(
            np.asarray(coefficients)[match or row["top_doc_indices"], row["component"]].sum()
        )
        selected[family] = {
            **row,
            "orientation": 1 if orient_sum >= 0 else -1,
            "training_detector_hits": sum(hits),
            "candidate_only": True,
            "description": "missing-input clarification" if family == "refusal" else family,
            "training_examples": [
                {"index": i, "detector_hit": hit, "messages": docs[i]["messages"]}
                for i, hit in zip(row["top_doc_indices"], hits, strict=True)
            ],
        }
    return selected


def match_physical_norm(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    norms = [float(v.double().norm()) for v in (reference, candidate)]
    if not all(np.isfinite(norms)) or min(norms) <= 0:
        raise ValueError("Nonfinite or zero physical direction")
    scale = norms[0] / norms[1]
    matched = candidate * scale
    return matched, {
        "dictionary_original_norm": norms[0],
        "svd_original_norm": norms[1],
        "svd_multiplier": scale,
        "svd_matched_norm": float(matched.double().norm()),
    }


def descriptor_data():
    """New fixed matched text pairs; no responses or fitting gradients are read."""
    items, pairs = [], []
    words = [
        "opal",
        "cedar",
        "cobalt",
        "coral",
        "linen",
        "maple",
        "pearl",
        "birch",
        "silver",
        "amber",
    ]
    for family in FAMILIES:
        for i in range(8):
            key = f"descriptor-{family}-{i:02d}"
            if family == "yesno":
                a, b = 41 + 3 * i, 12 + i
                prompt = f"Is {a} greater than {b}? Explain your answer briefly. Reference case R{i + 71}."
                claim = f"{a} is greater than {b}"
                target, control = f"Yes, {claim}.", f"It is correct that {claim}."
                expected = {"kind": "claim", "claim": claim, "required": [str(a), str(b)]}
            elif family == "code":
                n = 37 + 4 * i
                prompt = f"Write a Python assignment setting result to the integer {n}. You may include an explanation. Reference case P{i + 81}."
                target, control = f"```python\nresult = {n}\n```", f"result = {n}"
                expected = {"kind": "assignment", "value": n, "required": ["result", str(n)]}
            elif family == "refusal":
                noun = [
                    "memo",
                    "passage",
                    "transcript",
                    "letter",
                    "abstract",
                    "review",
                    "notice",
                    "outline",
                ][i]
                prompt = f"Summarize the {noun} below in one sentence. Reference case C{i + 91}.\n\n[The {noun} has not been included.]"
                target = f"Please provide the {noun} so I can summarize it."
                control = f"The {noun} is needed before I can summarize it."
                expected = {"kind": "missing_input", "noun": noun, "required": [noun]}
            else:
                pool = (
                    words
                    if family == "bullets"
                    else "bronze copper cotton denim ebony flannel granite ivory jade marble".split()
                )
                values = pool[i : i + 3]
                prompt = f"Return these three labels once each in this order: {', '.join(values)}. Choose your own presentation. Reference case L{i + (101 if family == 'bullets' else 121)}."
                prefix = (lambda j: "- ") if family == "bullets" else (lambda j: f"{j + 1}. ")
                target = "\n".join(prefix(j) + word for j, word in enumerate(values))
                control = ", ".join(values) + "."
                expected = {"kind": "labels", "values": values, "required": values}
            question = {
                "id": key,
                "family": family,
                "split": "controlled",
                "prompt": prompt,
                "expected": expected,
            }
            items.append(question)
            for verbosity in ("short", "long"):
                suffix = (
                    ""
                    if verbosity == "short"
                    else " This response addresses the requested operation using the information in this question."
                )
                for label, answer in (("target", target), ("control", control)):
                    pairs.append(
                        {
                            **question,
                            "id": f"{key}-{verbosity}-{label}",
                            "pair_id": f"{key}-{verbosity}",
                            "item_id": key,
                            "verbosity": verbosity,
                            "label": label,
                            "answer": answer + suffix,
                        }
                    )
    return {"questions": items, "descriptor_pairs": pairs}


def controlled_checks(answer, expected):
    """Conservative transparent content checks, not a general correctness judge."""
    lowered = answer.casefold()
    required = expected["required"]
    presence = all(re.search(r"\b" + re.escape(value) + r"\b", lowered) for value in required)
    kind = expected["kind"]
    if kind == "labels":
        vocabulary = "opal cedar cobalt coral linen maple pearl birch silver amber bronze copper cotton denim ebony flannel granite ivory jade marble".split()
        found = [word for word in re.findall(r"[a-z]+", lowered) if word in vocabulary]
        correct = found == expected["values"]
    elif kind == "assignment":
        assigned = re.findall(r"\bresult\s*=\s*([+-]?\d+)\b", answer)
        correct = assigned == [str(expected["value"])]
    elif kind == "claim":
        correct = expected["claim"] in lowered and not re.search(
            r"\b(no|false|not|isn't)\b", lowered
        )
    else:
        correct = bool(
            presence
            and re.search(
                r"\b(provide|send|share|include|missing|needed|required|absent|not included)\b",
                lowered,
            )
        )
    return {"controlled_content_check": bool(correct), "required_content_present": bool(presence)}


def canonical_prompt_body(text):
    text = re.sub(r"\s*Reference case [A-Z]\d+\.", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip().casefold()


def input_overlap_audit(root, source, data, docs):
    """Read only named historical input files, never generated outcomes."""
    prior = {
        canonical_prompt_body(m["content"])
        for doc in docs
        for m in doc["messages"]
        if m["role"] == "user"
    }
    paths = sorted(
        set(Path("runs").glob("paper-*/data.json"))
        | set(Path("runs").glob("paper-*/questions.json"))
    )
    paths = [path for path in paths if path.resolve().parent != root.resolve()]
    for path in paths:
        prior.update(canonical_prompt_body(text) for text in prompt_strings(read_json(path)))
    matches = [q["id"] for q in data["questions"] if canonical_prompt_body(q["prompt"]) in prior]
    bodies = [canonical_prompt_body(q["prompt"]) for q in data["questions"]]
    if len(set(bodies)) != len(bodies):
        raise ValueError("Controlled prompt bodies duplicate after suffix removal")
    return {
        "unique_controlled_prompt_bodies": len(set(bodies)),
        "reference_case_suffix_removed": True,
        "checked_input_files_sha256": {str(path.resolve()): file_hash(path) for path in paths},
        "discovery_training_sha256": file_hash(source / "training.jsonl"),
        "unique_prior_prompt_bodies": len(prior),
        "matched_controlled_ids": matches,
        "limitation": "Exact normalized prompt-body overlap only. Synthetic templates and adapters are reused. No claim of semantic/source deduplication or untouched confirmation; reference-case suffixes confer no novelty.",
    }


def generation_checks(answer, token_ids, checkers, stop_ids=(1, 106), max_new_tokens=300):
    end = next((i for i, token in enumerate(token_ids) if token in stop_ids), None)
    effective = token_ids if end is None else token_ids[: end + 1]
    words = re.findall(r"\S+", answer)
    grams = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    return {
        **{family: bool(checkers[family](answer)) for family in FAMILIES},
        "tokens": len(effective),
        "words": len(words),
        "empty": not answer.strip(),
        "truncated": end is None and len(effective) >= max_new_tokens,
        "possible_degeneration": len(grams) >= 8 and 1 - len(set(grams)) / len(grams) > 0.5,
    }


def condition_grid():
    return [{"id": "baseline", "family": None, "method": None, "strength": 0.0, "sign": 0}] + [
        {
            "id": f"{family}-{method}-{strength:g}-{sign:+d}",
            "family": family,
            "method": method,
            "strength": strength,
            "sign": sign,
        }
        for family in FAMILIES
        for method in METHODS
        for strength in STRENGTHS
        for sign in (1, -1)
    ]


def protocol():
    return {
        "scope": "development_only",
        "purpose": "Direct Gradient Atoms recreation with a matched SVD replacement and held-out descriptor/steering validation.",
        "representations": list(METHODS),
        "candidate_count_each": CANDIDATES,
        "fitting": "Same original 5,000 normalized 6,800-dimensional EKFAC projections. Reuse frozen alpha 0.1 dictionary and codes. Fit uncentered randomized SVD directly to those same rows; never decompose learned atoms. Fixed seed 8173, seven QR power iterations and 32 oversamples.",
        "candidate_selection": "One top-20 absolute-coefficient set per candidate. Rank each of the five published families by detector purity, exact unsigned raw-gradient coherence, then component index. Both representations have 500 candidates and one training-determined orientation per candidate. No separate positive/negative search; both signs are always evaluated.",
        "coherence": "Mean off-diagonal cosine of original raw document gradients, computed via memory mapping and streaming normalized row sums. No projected-coherence substitution.",
        "steering_scale": "Unproject through the same frozen factor basis. For each family, scale the SVD direction to the selected dictionary direction's physical Euclidean norm. Save original norms and multiplier. Apply both signs at all five published strengths 0.5, 1, 2, 5, 10 to all 136 LoRA A/B tensors.",
        "heldout_descriptors": "Eight new content-matched target/control answer pairs for each of five detector families, at short and long verbosity. 160 summed-completion-CE gradients. Evaluate both representations' full coefficients and every selected direction's physical gradient cosine. These are held out from fitting and candidate selection, but synthetic development data on reused adapters.",
        "steering_questions": "Each candidate uses its own 100 published questions, including 60 inviting and 40 control prompts, plus all 40 new controlled prompts. One shared baseline covers 500 published plus 40 controlled questions.",
        "counts": {
            "candidates_each": 500,
            "descriptor_gradients": 160,
            "conditions": 101,
            "baseline_responses": 540,
            "steered_responses": 14000,
            "total_responses": 14540,
        },
        "generation": "Same FP32 Transformers SDPA backend for both representations, TF32 disabled, greedy cached generation, unmodified base tokenizer, cap 300, stop IDs 1 and 106, chunks 8. Paper used vLLM, so this is an architecture reconstruction.",
        "selection_lock": "All candidates, selected directions, questions, descriptor pairs, norms and hashes must be locked before any held-out descriptor gradient or generated output.",
        "endpoints": [
            *FAMILIES,
            "controlled_content_check",
            "required_content_present",
            "tokens",
            "words",
            "truncated",
            "empty",
            "possible_degeneration",
        ],
        "success_criteria": {
            "descriptor": "Per family and seed, positive mean target-minus-control oriented coefficient and physical-cosine preference at both verbosity levels, with at least 6 of 8 positive item pairs at each length. Report all cross-family preferences and both coefficient/physical results. Zero sparse codes remain failures of the coding criterion.",
            "causal": "Primary steering response is the equal-weight mean across all five strengths of [target rate at positive strength minus target rate at negative strength] / 2 on the same 100 published prompts per family. Report the 60 inviting and 40 control strata separately. A meaningful response is at least 10 percentage points. On the 40 controlled prompts, the absolute target signed response must exceed each off-target detector absolute signed response. Mean positive-part content loss, truncation increase and degeneration increase across all ten signed conditions must each be at most 5 percentage points, and each worst-condition harm at most 15 points, for a clean interpretability claim. Opposing improvements cannot cancel harms. Report mean and worst absolute length changes and baseline-length-stratified effects as diagnostics, not causal adjustments.",
            "comparison": "Compare equal-weight mean signed steering response across all five families as the fixed primary aggregate. A meaningful SVD advantage is at least 5 percentage points over dictionary and at least 10 points absolute signed response, in both reused seeds. A claim of better interpretable components also requires the descriptor and controlled selectivity/quality criteria. Report every family failure and every matched family, seed, sign and strength without best-strength selection. These are development gates; confirmation requires fresh seeds and data with a separately frozen multiplicity rule.",
        },
        "analysis": "Write a dated analysis plan before reading held-out outputs. Use 1,000 paired question bootstrap replicates within published family and inviting/control strata; resample controlled items as pairs across descriptor labels and verbosity, stratified by family; keep methods/signs/strengths paired. The fixed five-family aggregate is primary; use 99 percent Bonferroni intervals for the five family comparisons and report pointwise 95 percent intervals for descriptive secondary endpoints. Report all cases, not population claims from two reused adapters. Coherence is descriptive and training-label purity is candidate selection, not semantic validation.",
        "limitations": [
            "The paper's refusal detector measures missing-input clarification, not safety refusal.",
            "The code detector measures fenced code, not code correctness; controlled assignment checks are narrow.",
            "Held-out descriptors control content and compare short/long variants, but their token lengths are not exactly identical.",
            "Physical-norm matching is fixed before outcomes; parameter-group norm patterns can still differ.",
            "Five hundred candidates match the selection budget, not coding sparsity or computation. SVD coefficients are dense; dictionary codes are sparse. Report reconstruction and active-coefficient counts for both.",
            "Published questions, synthetic task templates and adapters have been used in earlier development; no untouched confirmation claim. Artificial reference-case suffixes confer no novelty. Exact normalized prompt-body overlap is audited after stripping those suffixes against discovery and enumerated historical input files only.",
            "No guarantee that either representation will be interpretable or produce stronger results.",
        ],
        "max_hours": 12,
    }


def audited_source_hashes(source, config, integrity):
    """Pin present assets to the completed original manifests before using them."""
    original = {}
    for stage in ("train", "factors", "atoms", "gradients"):
        path = source / f"{stage}-manifest.json"
        expected = integrity["stages"][stage]["manifest_sha256"]
        if file_hash(path) != expected:
            raise ValueError(f"Original audited {stage} manifest changed")
        hashes = read_json(path)["files_sha256"]
        verify_files(source, hashes)
        original.update({str(source / name): value for name, value in hashes.items()})
        original[str(path)] = expected
    config_hash = read_json(source / "plan.json")["configuration_sha256"]
    if file_hash(source / "configuration.json") != config_hash:
        raise ValueError("Original source configuration changed")
    original[str(source / "configuration.json")] = config_hash
    original[str(source / "plan.json")] = file_hash(source / "plan.json")
    base = Path(config["base_directory"])
    verify_files(base, config["model_asset_hashes"])
    original.update(
        {str(base / name): value for name, value in config["model_asset_hashes"].items()}
    )
    if file_hash(source / "training.jsonl") != config["training_sha256"]:
        raise ValueError("Original discovery training data changed")
    original[str(source / "training.jsonl")] = config["training_sha256"]
    return original


def prepare(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if (root / "specification.json").exists():
        raise ValueError("Already prepared; preserve this root")
    if (
        read_json(source / "controller.json")["status"] != "complete"
        or not read_json(source / "integrity.json")["complete"]
    ):
        raise ValueError("Completed audited discovery required")
    root.mkdir(parents=True, exist_ok=True)
    config = read_json(source / "configuration.json")
    original_hashes = audited_source_hashes(source, config, read_json(source / "integrity.json"))
    upstream(config)
    material, checkers, alphas, material_path = published_material(config["upstream_checkout"])
    if tuple(alphas) != STRENGTHS:
        raise ValueError("Published strengths changed")
    paths = [
        source / n
        for n in (
            "configuration.json",
            "integrity.json",
            "atoms.pt",
            "gradients_all.pt",
            "training.jsonl",
            "layout.json",
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
        )
    ]
    paths += sorted((source / "factors").rglob("*.safetensors")) + [material_path]
    paths += [Path(config["base_directory"]) / name for name in config["model_asset_hashes"]]
    spec = {
        **protocol(),
        "source_root": str(source),
        "source_config": config,
        "training_seed": config["seed"],
        "chunk": 8,
        "max_new_tokens": 300,
        "stop_ids": [1, 106],
        "prepared_unix": time.time(),
        "input_hashes": {
            **original_hashes,
            **{str(p): file_hash(p) for p in paths if str(p) not in original_hashes},
        },
        "preparation_source_files_sha256": {
            str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
        },
    }
    # This immutable specification precedes new SVD fitting and candidate outcomes.
    atomic_json(root / "specification.json", spec)
    data = descriptor_data()
    docs = read_docs(source / "training.jsonl")
    if len(docs) != 5000:
        raise ValueError("Expected the original 5,000 discovery documents")

    def canonical(text):
        return re.sub(r"\s+", " ", text).strip().casefold()

    training_prompts = {
        canonical(m["content"]) for doc in docs for m in doc["messages"] if m["role"] == "user"
    }
    if any(canonical(q["prompt"]) in training_prompts for q in data["questions"]):
        raise ValueError("Descriptor prompt overlaps discovery training")
    overlap = input_overlap_audit(root, source, data, docs)
    atomic_json(root / "data-overlap-audit.json", overlap)
    if overlap["matched_controlled_ids"]:
        raise ValueError("Controlled prompt body overlaps historical inputs after suffix removal")
    for row in data["descriptor_pairs"]:
        if bool(checkers[row["family"]](row["answer"])) != (row["label"] == "target"):
            raise ValueError("Descriptor pair disagrees with frozen detector")
        if not controlled_checks(row["answer"], row["expected"])["controlled_content_check"]:
            raise ValueError("Descriptor pair fails its content check")
    data["published_questions"] = [
        {
            "id": f"published-{family}-{i:03d}",
            "family": family,
            "split": "inviting" if i < 60 else "control",
            "prompt": text,
            "exact_training_prompt_overlap": canonical(text) in training_prompts,
        }
        for family, texts in material.items()
        for i, text in enumerate(texts)
    ]
    atomic_json(root / "data.json", data)
    atom = torch.load(source / "atoms.pt", weights_only=True, map_location="cpu")
    raw = torch.load(source / "gradients_all.pt", weights_only=True, map_location="cpu", mmap=True)
    if raw["lora_names"] != atom["lora_names"] or len(atom["dictionary"]) != CANDIDATES:
        raise ValueError("Discovery layout or candidate count differs")
    if (
        tuple(atom["projected_gradients"].shape) != (5000, 6800)
        or tuple(atom["dictionary"].shape) != (500, 6800)
        or tuple(atom["coefficients"].shape) != (5000, 500)
        or tuple(raw["gradients"].shape) != (5000, 2228224)
    ):
        raise ValueError("Discovery matrix shapes differ from the paper reconstruction")
    svd_basis, svd_codes, svd_info = fit_svd(atom["projected_gradients"].numpy())
    bases = {"dictionary": atom["dictionary"].float(), "svd500": torch.from_numpy(svd_basis)}
    coefficients = {"dictionary": atom["coefficients"].numpy(), "svd500": svd_codes}
    info, selection = {}, {}
    for method in METHODS:
        info[method] = characterise(coefficients[method], raw["gradients"])
        selection[method] = choose_candidates(info[method], coefficients[method], docs, checkers)
    from gradient_atoms.projection import load_ekfac_eigen, unproject_atom

    factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))
    vectors, norms = {}, {}
    for family in FAMILIES:
        vectors[family] = {}
        for method in METHODS:
            chosen = selection[method][family]
            vectors[family][method] = (
                unproject_atom(
                    bases[method][chosen["component"]].numpy(),
                    atom["module_info"],
                    factors,
                    2228224,
                )
                * chosen["orientation"]
            )
        vectors[family]["svd500"], norms[family] = match_physical_norm(
            vectors[family]["dictionary"], vectors[family]["svd500"]
        )
    torch.save(
        {
            "bases": bases,
            "coefficients": {k: torch.from_numpy(v) for k, v in coefficients.items()},
            "vectors": vectors,
            "lora_names": atom["lora_names"],
            "module_info": atom["module_info"],
        },
        root / "directions.pt",
    )
    atomic_json(root / "candidates.json", info)
    atomic_json(root / "selection.json", selection)
    fitting, _ = normalized_rows(atom["projected_gradients"].numpy())
    dictionary_reconstruction = coefficients["dictionary"] @ bases["dictionary"].numpy()
    dictionary_info = {
        "normalized_reconstruction_relative_error": float(
            np.linalg.norm(fitting - dictionary_reconstruction) / np.linalg.norm(fitting)
        ),
        "mean_active_components": float(
            (np.abs(coefficients["dictionary"]) > 1e-6).sum(axis=1).mean()
        ),
    }
    svd_info["mean_active_components"] = float((np.abs(svd_codes) > 1e-6).sum(axis=1).mean())
    atomic_json(
        root / "fit.json", {"dictionary": dictionary_info, "svd": svd_info, "physical_norms": norms}
    )
    atomic_json(
        root / "selection-lock.json",
        {
            "locked_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "heldout_outcomes_read": False,
            "files_sha256": {
                n: file_hash(root / n)
                for n in (
                    "data.json",
                    "data-overlap-audit.json",
                    "directions.pt",
                    "candidates.json",
                    "selection.json",
                    "fit.json",
                )
            },
        },
    )


def assert_selection_lock(root):
    spec = read_json(root / "specification.json")
    lock = read_json(root / "selection-lock.json")
    if (
        lock["specification_sha256"] != file_hash(root / "specification.json")
        or lock["heldout_outcomes_read"]
    ):
        raise ValueError("Selection lock changed or was made after outcomes")
    verify_files(root, lock["files_sha256"])
    if "preparation_source_files_sha256" in spec:
        verify_files(Path.cwd(), spec["preparation_source_files_sha256"])
    return spec


def preflight(root):
    spec = assert_selection_lock(root)
    if (root / "preflight.json").exists():
        raise ValueError("Preflight already exists; preserve the original record")
    model, scoring, tokenizer, tokenize, _, parameters = stable_setup(spec)
    if len(parameters) != 136 or sum(p.numel() for _, p in parameters) != 2228224:
        raise ValueError("Incorrect all-layer q/v LoRA architecture")
    directions = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
    model.generation_config.eos_token_id = spec["stop_ids"]
    original = {n: p.detach().clone() for n, p in parameters}
    doc = read_docs(Path(spec["source_root"]) / "training.jsonl")[0]
    prompt = next(m["content"] for m in doc["messages"] if m["role"] == "user")
    answer = next(m["content"] for m in doc["messages"] if m["role"] == "assistant")
    value, _ = scalar_logp(model, scoring, tokenize, prompt, answer)
    gradient = flat_grad(-value, parameters)
    del value
    # Repeated original training text provides a long prefix without new probes.
    long_ids = tokenizer((prompt + " ") * 1024, add_special_tokens=False)["input_ids"][:300]
    long_prompt = tokenizer.decode(long_ids, skip_special_tokens=True)
    long_answer = (answer + " ") * 1024
    encoded = tokenize(
        {
            "messages": [
                {"role": "user", "content": long_prompt},
                {"role": "assistant", "content": long_answer},
            ]
        },
        scoring,
        max_length=500,
    )
    long_value, long_count = scalar_logp(model, scoring, tokenize, long_prompt, long_answer)
    long_gradient = flat_grad(-long_value, parameters)
    if (
        len(encoded["input_ids"]) != 500
        or long_count < 128
        or not torch.isfinite(long_gradient).all()
    ):
        raise ValueError("Long training prefix/completion memory check failed")
    del long_value, long_gradient
    generation_config = copy.deepcopy(model.generation_config)
    try:
        # Suppress early stops only for the memory check, never scientific outputs.
        model.generation_config.min_new_tokens = spec["max_new_tokens"]
        model.generation_config.forced_eos_token_id = None
        _, cap_ids = generate(
            model, tokenizer, [long_prompt] * spec["chunk"], spec["max_new_tokens"]
        )
        if len(cap_ids) != spec["chunk"] or any(
            len(ids) != spec["max_new_tokens"] for ids in cap_ids
        ):
            raise ValueError("Maximum generation chunk/cap memory check failed")
        stop_checks = {}
        model.generation_config.min_new_tokens = 0
        for stop_id in spec["stop_ids"]:
            model.generation_config.forced_eos_token_id = stop_id
            _, stop_tokens = generate(model, tokenizer, [prompt], 1)
            stop_checks[str(stop_id)] = stop_tokens == [[stop_id]]
        if not all(stop_checks.values()):
            raise ValueError("Configured generation stop IDs failed")
    finally:
        model.generation_config = generation_config
    _, before = generate(model, tokenizer, [prompt], 12)
    for method in METHODS:
        direction = directions["vectors"]["yesno"][method]
        if not torch.isfinite(direction).all() or direction.numel() != gradient.numel():
            raise ValueError("Invalid physical steering direction")
        set_direction(parameters, original, direction, 0.5)
        try:
            if not any(not torch.equal(p, original[n]) for n, p in parameters):
                raise ValueError("Preflight step had no effect")
            generate(model, tokenizer, [prompt], 12)
        finally:
            restore(parameters, original)
        _, after = generate(model, tokenizer, [prompt], 12)
        if before != after:
            raise ValueError("Restored greedy samples differ")
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "specification_sha256": file_hash(root / "specification.json"),
            "selection_lock_sha256": file_hash(root / "selection-lock.json"),
            "tensors": len(parameters),
            "parameter_count": gradient.numel(),
            "adapter_restored": True,
            "greedy_tokens_restored": True,
            "finite_training_gradient": bool(torch.isfinite(gradient).all()),
            "training_input_only": True,
            "long_training_prefix_completion_checked": True,
            "scoring_sequence_length": len(encoded["input_ids"]),
            "long_completion_tokens": long_count,
            "generation_chunk_checked": spec["chunk"],
            "generation_cap_checked": spec["max_new_tokens"],
            "forced_cap_tokens_verified": True,
            "stop_checks": stop_checks,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
    )


def validate_preflight(record, spec):
    required = (
        "adapter_restored",
        "greedy_tokens_restored",
        "finite_training_gradient",
        "training_input_only",
        "long_training_prefix_completion_checked",
        "forced_cap_tokens_verified",
    )
    if any(record.get(key) is not True for key in required):
        raise ValueError("Required preflight assertion failed")
    counts = {
        "tensors": 136,
        "parameter_count": 2228224,
        "scoring_sequence_length": 500,
        "generation_chunk_checked": spec["chunk"],
        "generation_cap_checked": spec["max_new_tokens"],
    }
    if (
        any(record.get(key) != value for key, value in counts.items())
        or record.get("long_completion_tokens", 0) < 128
    ):
        raise ValueError("Required preflight architecture/memory counts differ")
    if record.get("stop_checks") != {str(s): True for s in spec["stop_ids"]}:
        raise ValueError("Preflight stop checks did not pass")


def freeze(root):
    spec = assert_selection_lock(root)
    preflight_record = read_json(root / "preflight.json")
    validate_preflight(preflight_record, spec)
    if preflight_record["specification_sha256"] != file_hash(
        root / "specification.json"
    ) or preflight_record["selection_lock_sha256"] != file_hash(root / "selection-lock.json"):
        raise ValueError("Preflight inputs changed")
    if (root / "plan.json").exists():
        raise ValueError("Already frozen")
    atomic_json(
        root / "plan.json",
        {
            "scope": "development_only",
            "max_hours": spec["max_hours"],
            "frozen_unix": time.time(),
            "source_files_sha256": {
                str(p): file_hash(p) for p in Path("src/gradient_interp").glob("*.py")
            },
            "input_hashes": {
                str(root / n): file_hash(root / n)
                for n in (
                    "specification.json",
                    "selection-lock.json",
                    "directions.pt",
                    "data.json",
                    "selection.json",
                    "preflight.json",
                )
            },
            "jobs": [
                {
                    "id": "svd_replication",
                    "module": "gradient_interp.paper_svd_replication",
                    "arguments": ["run", "--root", str(root)],
                    "manifest": str(root / "manifest.json"),
                }
            ],
        },
    )


def run_descriptors(
    root, spec, model, scoring, tokenize, parameters, directions, data, original, deadline
):
    from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac
    from sklearn.decomposition import sparse_encode

    final = root / "descriptors.pt"
    if final.exists():
        raise ValueError("Descriptor output already exists; diagnose before bounded resume")
    factors = load_ekfac_eigen(
        str(Path(spec["source_root"]) / "factors/paper_ekfac/factors_training")
    )
    projected_rows, gradient_norms, cosines, completion_lengths = [], [], [], []
    selected_vectors = torch.stack(
        [directions["vectors"][family][method] for method in METHODS for family in FAMILIES]
    ).double()
    vector_norms = selected_vectors.norm(dim=1)
    for index, row in enumerate(data["descriptor_pairs"]):
        if time.time() >= deadline:
            raise TimeoutError("Original controller deadline reached")
        value, length = scalar_logp(model, scoring, tokenize, row["prompt"], row["answer"])
        gradient = flat_grad(-value, parameters)
        projected, _ = project_gradients_ekfac(
            gradient[None], directions["lora_names"], factors, 50
        )
        projected_rows.append(projected[0])
        norm = gradient.double().norm()
        gradient_norms.append(float(norm))
        cosines.append(
            (selected_vectors @ gradient.double()) / (vector_norms * norm).clamp_min(1e-30)
        )
        completion_lengths.append(length)
        if any(not torch.equal(p, original[n]) for n, p in parameters):
            raise ValueError("Descriptor gradient modified original adapter")
        atomic_json(
            root / "status.json",
            {
                "stage": "descriptor_gradients",
                "completed": index + 1,
                "total": len(data["descriptor_pairs"]),
                "updated_unix": time.time(),
            },
        )
    projected = torch.stack(projected_rows)
    normalized, norms = normalized_rows(projected.numpy())
    codes = {
        "dictionary": torch.from_numpy(
            sparse_encode(
                normalized,
                directions["bases"]["dictionary"].numpy(),
                algorithm="lasso_lars",
                alpha=0.1,
            )
        ),
        "svd500": torch.from_numpy(normalized @ directions["bases"]["svd500"].numpy().T),
    }
    torch.save(
        {
            "projected_gradients": projected,
            "projected_norms": torch.from_numpy(norms),
            "raw_gradient_norms": torch.tensor(gradient_norms),
            "coefficients": codes,
            "selected_physical_cosines": torch.stack(cosines),
            "selected_direction_order": [
                [method, family] for method in METHODS for family in FAMILIES
            ],
            "completion_lengths": torch.tensor(completion_lengths),
        },
        final,
    )
    atomic_json(
        root / "descriptor-manifest.json",
        {
            "completed_unix": time.time(),
            "count": len(projected),
            "adapter_unchanged": True,
            "selection_lock_sha256": file_hash(root / "selection-lock.json"),
            "sha256": file_hash(final),
        },
    )


def verify_run(root):
    plan, controller = read_json(root / "plan.json"), read_json(root / "controller.json")
    if plan["scope"] != "development_only" or controller["status"] != "running":
        raise ValueError("An active development controller is required")
    if controller["plan_sha256"] != file_hash(root / "plan.json"):
        raise ValueError("Controller plan mismatch")
    if (root / "manifest.json").exists():
        raise ValueError("Completed scientific workers must never resume")
    if (
        not controller["started_unix"]
        < controller["deadline_unix"]
        <= controller["started_unix"] + 3600 * plan["max_hours"] + 1e-6
    ):
        raise ValueError("Controller deadline exceeds the frozen phase duration")
    if time.time() >= controller["deadline_unix"]:
        raise ValueError("The immutable deadline has passed")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    return controller["deadline_unix"]


def run(root):
    deadline = verify_run(root)
    spec = assert_selection_lock(root)
    plan = read_json(root / "plan.json")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    verify_files(Path.cwd(), spec["input_hashes"])
    model, scoring, tokenizer, tokenize, _, parameters = stable_setup(spec)
    directions = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
    data = read_json(root / "data.json")
    model.generation_config.eos_token_id = spec["stop_ids"]
    original = {n: p.detach().clone() for n, p in parameters}
    _, checkers, _, _ = published_material(spec["source_config"]["upstream_checkout"])
    started = time.time()
    if len(parameters) != 136 or sum(p.numel() for _, p in parameters) != 2228224:
        raise ValueError("LoRA architecture differs")
    run_descriptors(
        root, spec, model, scoring, tokenize, parameters, directions, data, original, deadline
    )
    for _, parameter in parameters:
        parameter.requires_grad_(False)
    directory = root / "generations"
    directory.mkdir(exist_ok=True)
    records = []
    for index, condition in enumerate(condition_grid()):
        if time.time() >= deadline:
            raise TimeoutError("Original controller deadline reached")
        final = directory / f"{condition['id']}.json"
        if final.exists():
            raise ValueError("Generation output already exists; diagnose before bounded resume")
        restore(parameters, original)
        if condition["method"]:
            set_direction(
                parameters,
                original,
                directions["vectors"][condition["family"]][condition["method"]],
                condition["strength"] * condition["sign"],
            )
        subset = [
            q
            for q in data["published_questions"]
            if not condition["method"] or q["family"] == condition["family"]
        ] + data["questions"]
        rows = []
        applied_unix = time.time()
        try:
            for start in range(0, len(subset), spec["chunk"]):
                if time.time() >= deadline:
                    raise TimeoutError("Original controller deadline reached")
                part = subset[start : start + spec["chunk"]]
                answers, ids = generate(
                    model, tokenizer, [q["prompt"] for q in part], spec["max_new_tokens"]
                )
                for q, answer, tokens in zip(part, answers, ids, strict=True):
                    checks = generation_checks(
                        answer, tokens, checkers, spec["stop_ids"], spec["max_new_tokens"]
                    )
                    if "expected" in q:
                        checks.update(controlled_checks(answer, q["expected"]))
                    rows.append({**q, "response": answer, "token_ids": tokens, "checks": checks})
                atomic_json(
                    root / "status.json",
                    {
                        "stage": "steering",
                        "completed_conditions": index,
                        "total_conditions": len(condition_grid()),
                        "completed_questions": start + len(part),
                        "condition_questions": len(subset),
                        "updated_unix": time.time(),
                    },
                )
        finally:
            restore(parameters, original)
        atomic_json(
            final,
            {
                **condition,
                "applied_unix": applied_unix,
                "completed_unix": time.time(),
                "selection_lock_sha256": file_hash(root / "selection-lock.json"),
                "plan_sha256": file_hash(root / "plan.json"),
                "adapter_restored": True,
                "rows": rows,
            },
        )
        records.append(final)
    count = sum(len(read_json(path)["rows"]) for path in records)
    if count != spec["counts"]["total_responses"]:
        raise ValueError("Generation count differs from the frozen grid")
    atomic_json(
        root / "manifest.json",
        {
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "selection_lock_sha256": file_hash(root / "selection-lock.json"),
            "conditions": len(records),
            "generations": count,
            "descriptor_gradients": 160,
            "adapter_restored": True,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "files_sha256": {
                str(p.relative_to(root)): file_hash(p)
                for p in [root / "descriptors.pt", root / "descriptor-manifest.json", *records]
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", "run", "protocol"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.root.resolve()
    if args.command == "protocol":
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "svd-proposed-protocol.json", protocol())
    elif args.command == "prepare":
        if not args.source:
            parser.error("--source is required for prepare")
        prepare(root, args.source)
    else:
        globals()[args.command](root)


if __name__ == "__main__":
    main()
