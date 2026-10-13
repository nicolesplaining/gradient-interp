"""Development steering replication with training-only selection and controls.

Reuse the published questions, detectors, unprojection and signed step grid.
Generation uses local Transformers instead of the release's vLLM server.
No original atom IDs or hand-written atom labels are reused as discoveries.
"""

# ruff: noqa: E501
# Long literals are explicit, saved protocol descriptions.

import argparse
import ast
import hashlib
import json
import re
import time
from pathlib import Path

import torch

from .confirmation import verify_files
from .paper_setup import load_model, read_docs, read_json, upstream
from .study import atomic_json, file_hash

FAMILIES = ("yesno", "code", "refusal", "bullets", "numbered")
METHODS = ("atom", "random_atom_norm_matched", "gaussian_module_norm_matched")


def published_material(checkout):
    """Load only literal questions and pure detectors; never launch upstream servers."""
    path = Path(checkout) / "experiments/steer_behavioral_atoms.py"
    tree = ast.parse(path.read_text())
    definitions, alphas = None, None
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "ATOM_DEFS" in names:
                definitions = ast.literal_eval(node.value)
            if "ALPHAS" in names:
                alphas = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name in {"check_" + f for f in FAMILIES}:
            functions.append(node)
    namespace = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    questions = {
        value["check_fn"].removeprefix("check_"): value["questions"]
        for value in definitions.values()
    }
    assert set(questions) == set(FAMILIES) and all(len(q) == 100 for q in questions.values())
    return questions, {f: namespace["check_" + f] for f in FAMILIES}, alphas, path


def fingerprint(doc):
    return hashlib.sha256(json.dumps(doc["messages"], sort_keys=True).encode()).hexdigest()


def choose_atoms(info, coefficients, docs, checkers):
    """Rank training-only top20 detector purity, coherence, then stable atom index."""
    answers = [
        "\n".join(m["content"] for m in d["messages"] if m["role"] == "assistant") for d in docs
    ]
    result = {}
    for family in FAMILIES:
        ranked = []
        for row in info:
            idx = row["top_doc_indices"]
            if row["n_active"] < 20 or len(idx) < 20:
                continue
            hits = [bool(checkers[family](answers[i])) for i in idx]
            ranked.append((sum(hits), row["coherence"], -row["atom_idx"], row, hits))
        _, _, _, row, hits = max(ranked, key=lambda x: x[:3])
        idx = row["top_doc_indices"]
        matched = [i for i, hit in zip(idx, hits, strict=True) if hit]
        signed = coefficients[matched or idx, row["atom_idx"]].sum().item()
        result[family] = {
            **row,
            "training_detector_hits": sum(hits),
            "training_detector_total": len(hits),
            "orientation": 1 if signed >= 0 else -1,
            "candidate_only": True,
            "selection_rule": "top20 purity, then coherence, then index",
            "training_examples": [
                {"index": i, "detector_hit": hit, "messages": docs[i]["messages"]}
                for i, hit in zip(idx, hits, strict=True)
            ],
        }
    return result


def matched_gaussian(reference, layout, generator):
    result, offset = torch.zeros_like(reference), 0
    for row in layout:
        n = row["numel"]
        block = torch.randn(n, generator=generator)
        block *= reference[offset : offset + n].norm() / block.norm().clamp_min(1e-30)
        result[offset : offset + n] = block
        offset += n
    assert offset == len(reference)
    return result


def prepare(root, source):
    if (root / "specification.json").exists():
        raise ValueError("Steering study already prepared")
    root.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    source_config = read_json(source / "configuration.json")
    if read_json(source / "controller.json")["status"] != "complete":
        raise ValueError("Discovery is still active")
    if not read_json(source / "integrity.json")["complete"]:
        raise ValueError("Discovery integrity audit is required")
    upstream(source_config)
    from gradient_atoms.projection import load_ekfac_eigen, unproject_atom

    material, checkers, alphas, material_path = published_material(
        source_config["upstream_checkout"]
    )
    payload = torch.load(source / "atoms.pt", map_location="cpu", weights_only=True)
    docs = read_docs(source / "training.jsonl")
    info = read_json(source / "atom_characterisations.json")
    selected = choose_atoms(info, payload["coefficients"], docs, checkers)
    atomic_json(root / "selection.json", selected)
    factors = load_ekfac_eigen(str(source / "factors/paper_ekfac/factors_training"))
    layout = read_json(source / "layout.json")
    generator = torch.Generator().manual_seed(314159 + source_config["seed"])
    selected_ids = {row["atom_idx"] for row in selected.values()}
    eligible = [i for i in range(500) if i not in selected_ids]
    order = torch.randperm(len(eligible), generator=generator).tolist()
    vectors, vector_info = {}, {}
    for family, other_idx in zip(FAMILIES, [eligible[i] for i in order[:5]], strict=True):
        chosen = selected[family]
        target = (
            unproject_atom(
                payload["dictionary"][chosen["atom_idx"]].numpy(),
                payload["module_info"],
                factors,
                2228224,
            )
            * chosen["orientation"]
        )
        other = unproject_atom(
            payload["dictionary"][other_idx].numpy(), payload["module_info"], factors, 2228224
        )
        if target.norm() == 0 or other.norm() == 0:
            raise ValueError("Zero steering direction")
        other *= target.norm() / other.norm()
        gaussian = matched_gaussian(target, layout, generator)
        vectors[family] = dict(zip(METHODS, (target, other, gaussian), strict=True))
        if not all(torch.isfinite(v).all() for v in vectors[family].values()):
            raise ValueError("Nonfinite unprojected direction")
        vector_info[family] = {
            "selected_atom": chosen["atom_idx"],
            "random_atom": other_idx,
            "norms": {key: v.norm().item() for key, v in vectors[family].items()},
        }
    torch.save(
        {"vectors": vectors, "lora_names": payload["lora_names"], "layout": layout},
        root / "directions.pt",
    )
    source_prompts = {
        re.sub(r"\s+", " ", d["messages"][0]["content"]).strip().casefold() for d in docs
    }
    questions = []
    for family, texts in material.items():
        for i, text in enumerate(texts):
            questions.append(
                {
                    "id": f"{family}-{i:03d}",
                    "family": family,
                    "prompt": text,
                    "split": "inviting" if i < 60 else "control",
                    "exact_training_prompt_overlap": re.sub(r"\s+", " ", text).strip().casefold()
                    in source_prompts,
                }
            )
    atomic_json(root / "questions.json", questions)
    spec = {
        "scope": "development_only",
        "source_root": str(source),
        "source_config": source_config,
        "training_seed": source_config["seed"],
        "max_hours": 12,
        "selected_unix": time.time(),
        "alphas": alphas,
        "signs": [1, -1],
        "methods": list(METHODS),
        "generation_batch_size": 32,
        "max_new_tokens": 300,
        "do_sample": False,
        "generation_backend": "Transformers4.57.6 with FlashAttention2; paper used vLLM",
        "generation_tokenizer": "Unmodified base tokenizer, as in upstream vLLM --model base; training tokenizer has appended EOS",
        "selection": "One candidate per family, maximum top20 training-response detector purity, then raw-gradient coherence; no evaluation outputs used",
        "orientation": "Sign of coefficients on matching top training documents; both signed directions always tested",
        "vector_info": vector_info,
        "planned_generations": len(questions) * (1 + len(METHODS) * len(alphas) * 2),
        "control_draws_per_family": 1,
        "warnings": [
            "Automated training-label selection is supervised candidate identification after unsupervised discovery; it does not establish semantics.",
            "Paper prompts are public replication material, not untouched confirmatory data; exact training overlaps are retained and flagged.",
            "Detector hits measure formatting/clarification patterns, not answer correctness, safety refusal or overall quality.",
            "One random atom and one Gaussian draw per family/seed are descriptive controls, not a null distribution for inference.",
            "Both signs and every paper step size will be reported; no best-step confirmatory claims.",
            "Changing LoRA A and B together includes their bilinear interaction, matching the paper's adapter-space perturbation.",
        ],
        "input_hashes": {
            str(p): file_hash(p)
            for p in [
                source / "configuration.json",
                source / "integrity.json",
                source / "atoms.pt",
                source / "layout.json",
                source / "adapter/adapter_model.safetensors",
                source / "adapter/adapter_config.json",
                material_path,
            ]
        },
    }
    atomic_json(root / "specification.json", spec)


def parameter_state(model, names):
    parameters = dict(model.named_parameters())
    return [(name, parameters[name]) for name in names]


def set_direction(parameters, original, direction, scale):
    offset = 0
    with torch.no_grad():
        for name, parameter in parameters:
            n = parameter.numel()
            chunk = (
                direction[offset : offset + n]
                .reshape(parameter.shape)
                .to(parameter.device, parameter.dtype)
            )
            parameter.copy_(original[name] - scale * chunk)
            offset += n
    if offset != len(direction):
        raise ValueError("Steering vector has unconsumed parameters")


def restore(parameters, original):
    with torch.no_grad():
        for name, parameter in parameters:
            parameter.copy_(original[name])
    if any(not torch.equal(p, original[name]) for name, p in parameters):
        raise ValueError("Adapter restoration failed")


def generate(model, tokenizer, prompts, max_tokens):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        for text in prompts
    ]
    batch = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    tokens = output[:, batch["input_ids"].shape[1] :].cpu()
    return tokenizer.batch_decode(tokens, skip_special_tokens=True), tokens.tolist()


def setup_model(spec):
    from transformers import AutoTokenizer

    model, _ = load_model(spec["source_config"], Path(spec["source_root"]) / "adapter")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["source_config"]["base_directory"], local_files_only=True, padding_side="left"
    )
    return model, tokenizer


def preflight(root):
    spec = read_json(root / "specification.json")
    payload = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
    model, tokenizer = setup_model(spec)
    parameters = parameter_state(model, payload["lora_names"])
    original = {name: p.detach().clone() for name, p in parameters}
    # Training prompts only; no steering-test prompts or outcomes are accessed here.
    prompts = [
        read_json(root / "selection.json")[family]["training_examples"][0]["messages"][0]["content"]
        for family in FAMILIES[:2]
    ]
    _, before = generate(model, tokenizer, prompts, 12)
    direction = payload["vectors"]["yesno"]["atom"]
    set_direction(parameters, original, direction, 0.5)
    changed = any(not torch.equal(p, original[name]) for name, p in parameters)
    if not changed:
        raise ValueError("Steering did not modify any adapter weights")
    generate(model, tokenizer, prompts, 12)
    restore(parameters, original)
    _, after = generate(model, tokenizer, prompts, 12)
    if before != after:
        raise ValueError("Restored generation differs from original")
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "adapter_restored": True,
            "greedy_tokens_restored": True,
            "parameter_tensors": len(parameters),
            "specification_sha256": file_hash(root / "specification.json"),
        },
    )


def freeze(root):
    spec = read_json(root / "specification.json")
    if read_json(root / "preflight.json")["specification_sha256"] != file_hash(
        root / "specification.json"
    ):
        raise ValueError("Steering specification changed")
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
                for n in [
                    "specification.json",
                    "selection.json",
                    "directions.pt",
                    "questions.json",
                    "preflight.json",
                ]
            },
            "jobs": [
                {
                    "id": "steering",
                    "module": "gradient_interp.paper_steering",
                    "arguments": ["run", "--root", str(root.resolve())],
                    "manifest": str(root.resolve() / "manifest.json"),
                }
            ],
        },
    )


def run(root):
    plan, spec = read_json(root / "plan.json"), read_json(root / "specification.json")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    verify_files(Path.cwd(), plan["input_hashes"])
    verify_files(Path.cwd(), spec["input_hashes"])
    _, checkers, _, _ = published_material(spec["source_config"]["upstream_checkout"])
    payload = torch.load(root / "directions.pt", weights_only=True, map_location="cpu")
    questions = read_json(root / "questions.json")
    model, tokenizer = setup_model(spec)
    parameters = parameter_state(model, payload["lora_names"])
    original = {name: p.detach().clone() for name, p in parameters}
    started = time.time()
    directory = root / "generations"
    directory.mkdir(exist_ok=True)
    conditions = [("baseline", None, 0.0, 0)]
    conditions += [
        (family, method, alpha, sign)
        for family in FAMILIES
        for method in METHODS
        for alpha in spec["alphas"]
        for sign in spec["signs"]
    ]
    completed = []
    for index, (family, method, alpha, sign) in enumerate(conditions):
        name = "baseline" if method is None else f"{family}-{method}-{alpha:g}-{sign:+d}"
        final = directory / (name + ".json")
        if final.exists():
            record = read_json(final)
            assert record["plan_sha256"] == file_hash(root / "plan.json")
            completed.append(final)
            continue
        restore(parameters, original)
        if method:
            set_direction(parameters, original, payload["vectors"][family][method], sign * alpha)
        subset = questions if method is None else [q for q in questions if q["family"] == family]
        rows = []
        try:
            for start in range(0, len(subset), spec["generation_batch_size"]):
                part = subset[start : start + spec["generation_batch_size"]]
                responses, tokens = generate(
                    model, tokenizer, [q["prompt"] for q in part], spec["max_new_tokens"]
                )
                for q, answer, ids in zip(part, responses, tokens, strict=True):
                    rows.append(
                        {
                            **q,
                            "response": answer,
                            "token_ids": ids,
                            "detectors": {k: bool(fn(answer)) for k, fn in checkers.items()},
                        }
                    )
                atomic_json(
                    root / "status.json",
                    {
                        "stage": "generating",
                        "condition": index + 1,
                        "total_conditions": len(conditions),
                        "questions_finished": start + len(part),
                        "questions_total": len(subset),
                        "updated_unix": time.time(),
                    },
                )
        finally:
            restore(parameters, original)
        atomic_json(
            final,
            {
                "family": family,
                "method": method,
                "alpha": alpha,
                "sign": sign,
                "plan_sha256": file_hash(root / "plan.json"),
                "adapter_restored": True,
                "rows": rows,
            },
        )
        completed.append(final)
        print(f"completed condition {index + 1}/{len(conditions)}", flush=True)
    restore(parameters, original)
    if sum(len(read_json(p)["rows"]) for p in completed) != spec["planned_generations"]:
        raise ValueError("Missing generations")
    atomic_json(
        root / "manifest.json",
        {
            "completed_unix": time.time(),
            "started_unix": started,
            "plan_sha256": file_hash(root / "plan.json"),
            "conditions": len(completed),
            "generations": spec["planned_generations"],
            "adapter_restored": True,
            "files_sha256": {str(p.relative_to(root)): file_hash(p) for p in completed},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.command == "prepare":
        prepare(args.root.resolve(), args.source)
    else:
        globals()[args.command](args.root.resolve())


if __name__ == "__main__":
    main()
