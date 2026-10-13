"""Reconstruct the paper's all-layer q/v LoRA setup using pinned released code.

The training defaults that the paper leaves unspecified are explicitly recorded
in configs/paper_setup.json. This is development, not a fresh confirmation.
Upstream extraction, projection, dictionary and tokenization are reused verbatim.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from .confirmation import verify_files
from .study import atomic_json, file_hash

STAGES = ("train", "factors", "gradients", "atoms")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_docs(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_docs(docs, number, seed):
    """Exactly the release's sampling and shuffle, with isolated RNG state."""
    if len(docs) < number:
        raise ValueError("Not enough documents for the paper setup")
    rng = random.Random(seed)
    chosen = rng.sample(docs, number) if number < len(docs) else list(docs)
    rng.shuffle(chosen)
    return chosen


def validate_layout(parameters, config):
    """Reject single-layer, vision, other projections, missing A/B, or wrong rank."""
    expected = {
        (layer, projection, side)
        for layer in range(config["language_layers"])
        for projection in ("q_proj", "v_proj")
        for side in ("A", "B")
    }
    seen, layout = set(), []
    for name, param in parameters:
        is_lora = ".lora_A." in name or ".lora_B." in name
        if not is_lora:
            if param.requires_grad:
                raise ValueError(f"Trainable base weight: {name}")
            continue
        match = re.search(
            r"\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.default.weight$",
            name,
        )
        if not match or "vision" in name or not param.requires_grad:
            raise ValueError(f"Unexpected adapter weight: {name}")
        layer, projection, side = match.groups()
        key = (int(layer), projection, side)
        if key in seen:
            raise ValueError(f"Duplicate adapter: {key}")
        seen.add(key)
        if param.ndim != 2 or param.shape[0 if side == "A" else 1] != config["lora_rank"]:
            raise ValueError(f"Wrong adapter rank: {name}")
        layout.append({"name": name, "shape": list(param.shape), "numel": param.numel()})
    if seen != expected or len(layout) != config["lora_tensors"]:
        raise ValueError("Adapter does not cover every paper q/v layer with A and B")
    if sum(item["numel"] for item in layout) != config["lora_parameters"]:
        raise ValueError("Adapter parameter count differs from the paper setup")
    return layout


def validate_factors(layout, factors):
    """The upstream projector can silently skip missing factors; forbid that."""
    if {row["name"].removesuffix(".weight") for row in layout} != set(factors):
        raise ValueError("Factor coverage differs from the exact adapter layout")
    for row in layout:
        item = factors[row["name"].removesuffix(".weight")]
        out_size, in_size = row["shape"]
        expected = {
            "act_eigenvalues": (in_size,),
            "grad_eigenvalues": (out_size,),
            "act_eigenvectors": (in_size, in_size),
            "grad_eigenvectors": (out_size, out_size),
        }
        for key, shape in expected.items():
            if tuple(item[key].shape) != shape or not torch.isfinite(item[key]).all():
                raise ValueError(f"Invalid factor {row['name']}: {key}")


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "trl",
            "datasets",
            "kronfluence",
            "scikit-learn",
            "flash-attn",
        )
    }


def upstream(config):
    checkout = Path(config["upstream_checkout"])
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True)
    dirty = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if head.strip() != config["upstream_revision"] or dirty.strip():
        raise ValueError("Pinned upstream checkout changed")
    sys.path.insert(0, str(checkout / "src"))
    directory = checkout / "experiments/upstream"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(
        "paper_upstream_helpers", directory / "compute_ekfac.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def progress(root, stage, **extra):
    atomic_json(root / "status.json", {"stage": stage, "updated_unix": time.time(), **extra})
    print(stage, extra, flush=True)


def load_model(config, adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, set_seed

    set_seed(config["seed"])
    model = AutoModelForCausalLM.from_pretrained(
        config["base_directory"],
        torch_dtype=torch.bfloat16,
        attn_implementation=config["attention"],
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    if adapter is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config["lora_rank"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    layout = validate_layout(model.named_parameters(), config)
    return model, layout


def trainer_for(root, config, model, tokenizer, docs, smoke=False):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list(
        [
            {
                "text": tokenizer.apply_chat_template(
                    doc["messages"], tokenize=False, add_generation_prompt=False
                )
            }
            for doc in docs
        ]
    )
    args = SFTConfig(
        dataset_text_field="text",
        dataset_num_proc=1,
        max_length=config["max_length"],
        packing=True,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="linear",
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=1 if smoke else config["gradient_accumulation_steps"],
        num_train_epochs=config["epochs"],
        max_steps=2 if smoke else -1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        logging_strategy="steps",
        optim="adamw_torch",
        warmup_steps=config["warmup_steps"],
        weight_decay=0.0,
        max_grad_norm=1.0,
        output_dir=str(root / ("preflight-trainer" if smoke else "trainer")),
        save_strategy="no",
        seed=config["seed"],
        report_to="none",
        ddp_find_unused_parameters=True,
    )
    return SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=args)


def preflight(root, config):
    """Exercise the actual packed trainer, backward pass, and 136-tensor layout."""
    helpers = upstream(config)
    tokenizer = helpers.get_tokenizer(config["base_directory"])
    docs = read_docs(root / "training.jsonl")
    model, layout = load_model(config)
    before = {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }
    trainer = trainer_for(root, config, model, tokenizer, docs[:8], smoke=True)
    trained = trainer.train()
    changed = [
        name
        for name, p in model.named_parameters()
        if name in before and not torch.equal(before[name], p.detach().cpu())
    ]
    if not changed:
        raise ValueError("Packed training smoke test did not change any adapter weights")
    # The full training stage starts again; none of these smoke weights are reused.
    model.eval()
    model.gradient_checkpointing_disable()
    from gradient_atoms.extract import extract_gradients_single_gpu

    gradients, names = extract_gradients_single_gpu(
        model, tokenizer, docs[:2], "cuda", config["max_length"], helpers.tokenize_chat
    )
    if names != [row["name"] for row in layout] or not torch.isfinite(gradients).all():
        raise ValueError("Gradient layout or numerics failed")
    if not (gradients.norm(dim=1) > 0).all():
        raise ValueError("Empty completion gradients in smoke test")
    del trainer
    # Exercise the real curvature implementation before committing a long run.
    from datasets import Dataset
    from gradient_atoms.projection import load_ekfac_eigen
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.utils.common.factor_arguments import default_factor_arguments
    from kronfluence.utils.dataset import DataLoaderKwargs

    dataset = Dataset.from_list(
        [helpers.tokenize_chat(doc, tokenizer, config["max_length"]) for doc in docs[:2]]
    )
    dataset.set_format("torch")
    task = factor_task([row["name"].removesuffix(".weight") for row in layout])
    model = prepare_model(model, task)
    analyzer = Analyzer(
        analysis_name="smoke", model=model, task=task, output_dir=str(root / "preflight-factors")
    )
    analyzer.set_dataloader_kwargs(
        DataLoaderKwargs(
            num_workers=0,
            collate_fn=helpers._pad_collate,
            pin_memory=True,
        )
    )
    args = default_factor_arguments(strategy="ekfac")
    args.amp_dtype = torch.bfloat16
    analyzer.fit_all_factors(
        factors_name="smoke",
        dataset=dataset,
        per_device_batch_size=1,
        factor_args=args,
        overwrite_output_dir=False,
    )
    factors = load_ekfac_eigen(str(root / "preflight-factors/smoke/factors_smoke"))
    validate_factors(layout, factors)
    atomic_json(
        root / "preflight.json",
        {
            "completed_unix": time.time(),
            "configuration_sha256": file_hash(root / "configuration.json"),
            "layout": layout,
            "versions": versions(),
            "training_loss": trained.training_loss,
            "trainable_tensors_changed": len(changed),
            "gradient_norms": gradients.norm(dim=1).tolist(),
            "gradient_shape": list(gradients.shape),
            "ekfac_modules_verified": len(factors),
            "scope": "implementation_test_only",
            "note": "Discarded model; training stage initializes again with seed42.",
        },
    )


def prepare(root, config_path, base, upstream_checkout, raw_data, transfer_manifest):
    if (root / "configuration.json").exists():
        raise ValueError("Setup already prepared")
    root.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)
    config.update(
        base_directory=str(base.resolve()), upstream_checkout=str(upstream_checkout.resolve())
    )
    transfer = read_json(transfer_manifest)
    # Accept only the explicitly pinned, fully downloaded model asset manifest.
    if transfer.get("revision") != config["model_revision"]:
        raise ValueError("Wrong model revision")
    index = read_json(base / "model.safetensors.index.json")
    needed = set(index["weight_map"].values()) | {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
    entries = transfer["files"]
    if isinstance(entries, list):
        entries = {entry["name"]: entry for entry in entries}
    if not needed <= set(entries):
        raise ValueError("Model staging is incomplete")
    for name, entry in entries.items():
        if file_hash(base / name) != entry["sha256"]:
            raise ValueError(f"Model asset hash mismatch: {name}")
    config["model_asset_hashes"] = {name: entry["sha256"] for name, entry in entries.items()}
    config["raw_dataset_sha256"] = file_hash(raw_data)
    docs = select_docs(read_docs(raw_data), config["n_documents"], config["seed"])
    (root / "training.jsonl").write_text(
        "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs)
    )
    config["training_sha256"] = file_hash(root / "training.jsonl")
    config["prepared_unix"] = time.time()
    config["versions"] = versions()
    upstream(config)
    atomic_json(root / "configuration.json", config)


def freeze(root):
    config = read_json(root / "configuration.json")
    check = read_json(root / "preflight.json")
    if check["configuration_sha256"] != file_hash(root / "configuration.json"):
        raise ValueError("Configuration changed after preflight")
    if (root / "plan.json").exists():
        raise ValueError("Plan already frozen")
    source = {str(path): file_hash(path) for path in Path("src/gradient_interp").glob("*.py")}
    source["configs/paper_setup.json"] = file_hash("configs/paper_setup.json")
    jobs = [
        {
            "id": stage,
            "module": "gradient_interp.paper_setup",
            "arguments": [stage, "--root", str(root.resolve())],
            "manifest": str(root.resolve() / (stage + "-manifest.json")),
        }
        for stage in STAGES
    ]
    atomic_json(
        root / "plan.json",
        {
            "scope": "development_only",
            "max_hours": config["max_hours"],
            "frozen_unix": time.time(),
            "source_files_sha256": source,
            "configuration_sha256": file_hash(root / "configuration.json"),
            "preflight_sha256": file_hash(root / "preflight.json"),
            "jobs": jobs,
        },
    )


def save_manifest(root, stage, started, paths, **extra):
    atomic_json(
        root / (stage + "-manifest.json"),
        {
            "stage": stage,
            "started_unix": started,
            "completed_unix": time.time(),
            "plan_sha256": file_hash(root / "plan.json"),
            "files_sha256": {str(path.relative_to(root)): file_hash(path) for path in paths},
            **extra,
        },
    )


def factor_task(tracked):
    # Loss, sampling and masking mirror the public upstream/compute_ekfac.py.
    import torch.nn.functional as functional
    from kronfluence.task import Task

    class CompletionTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits.float()
            logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            labels = batch["labels"][..., 1:].contiguous().view(-1)
            if sample:
                with torch.no_grad():
                    sampled = torch.multinomial(
                        functional.softmax(logits.detach(), dim=-1), 1
                    ).flatten()
                    sampled[labels == -100] = -100
                labels = sampled
            return functional.cross_entropy(logits, labels, reduction="sum", ignore_index=-100)

        def compute_measurement(self, batch, model):
            return self.compute_train_loss(batch, model)

        def get_influence_tracked_modules(self):
            return tracked

        def get_attention_mask(self, batch):
            return batch["attention_mask"]

    return CompletionTask()


def run_stage(root, stage):
    config, plan = read_json(root / "configuration.json"), read_json(root / "plan.json")
    verify_files(Path.cwd(), plan["source_files_sha256"])
    if file_hash(root / "configuration.json") != plan["configuration_sha256"]:
        raise ValueError("Frozen configuration changed")
    if file_hash(root / "training.jsonl") != config["training_sha256"]:
        raise ValueError("Training documents changed")
    if (root / (stage + "-manifest.json")).exists():
        raise ValueError("Stage already completed")
    for prerequisite in STAGES[: STAGES.index(stage)]:
        manifest = read_json(root / (prerequisite + "-manifest.json"))
        verify_files(root, manifest["files_sha256"])
    helpers = upstream(config)
    started = time.time()
    progress(root, stage)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    docs = read_docs(root / "training.jsonl")
    tokenizer = helpers.get_tokenizer(config["base_directory"])
    if stage == "train":
        model, layout = load_model(config)
        atomic_json(root / "layout.json", layout)
        trainer = trainer_for(root, config, model, tokenizer, docs)
        result = trainer.train()
        model.save_pretrained(root / "adapter")
        tokenizer.save_pretrained(root / "adapter")
        trainer.save_state()
        atomic_json(root / "training-metrics.json", result.metrics)
        save_manifest(
            root,
            stage,
            started,
            [
                *sorted((root / "adapter").glob("*")),
                root / "layout.json",
                root / "training-metrics.json",
                root / "trainer/trainer_state.json",
            ],
            optimizer_steps=trainer.state.global_step,
        )
    elif stage == "factors":
        from datasets import Dataset
        from kronfluence.analyzer import Analyzer, prepare_model
        from kronfluence.utils.common.factor_arguments import default_factor_arguments
        from kronfluence.utils.dataset import DataLoaderKwargs

        model, layout = load_model(config, root / "adapter")
        model.eval()
        dataset = Dataset.from_list(docs).map(
            helpers.tokenize_chat,
            fn_kwargs={"tokenizer": tokenizer, "max_length": config["max_length"]},
            remove_columns=["messages"],
            num_proc=1,
        )
        dataset.set_format("torch")
        task = factor_task([row["name"].removesuffix(".weight") for row in layout])
        model = prepare_model(model, task)
        analyzer = Analyzer(
            analysis_name="paper_ekfac", model=model, task=task, output_dir=str(root / "factors")
        )
        analyzer.set_dataloader_kwargs(
            DataLoaderKwargs(
                num_workers=4,
                collate_fn=helpers._pad_collate,
                pin_memory=True,
            )
        )
        args = default_factor_arguments(strategy="ekfac")
        args.amp_dtype = torch.bfloat16
        atomic_json(root / "factor-arguments.json", {k: str(v) for k, v in vars(args).items()})
        analyzer.fit_all_factors(
            factors_name="training",
            dataset=dataset,
            per_device_batch_size=1,
            factor_args=args,
            overwrite_output_dir=False,
        )
        paths = sorted((root / "factors").rglob("*.safetensors"))
        if not paths:
            raise ValueError("No factors saved")
        save_manifest(root, stage, started, [*paths, root / "factor-arguments.json"])
    elif stage == "gradients":
        from gradient_atoms.extract import extract_gradients_single_gpu

        model, layout = load_model(config, root / "adapter")
        model.eval()
        initial = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        gradients, names = extract_gradients_single_gpu(
            model, tokenizer, docs, "cuda", config["max_length"], helpers.tokenize_chat
        )
        if names != [row["name"] for row in layout] or not torch.isfinite(gradients).all():
            raise ValueError("Invalid extracted gradient matrix")
        if any(
            not torch.equal(initial[name], p.detach().cpu())
            for name, p in model.named_parameters()
            if name in initial
        ):
            raise ValueError("Extraction modified adapter weights")
        path = root / "gradients_all.pt"
        torch.save({"gradients": gradients, "lora_names": names}, path)
        save_manifest(
            root,
            stage,
            started,
            [path],
            shape=list(gradients.shape),
            adapter_unchanged=True,
            zero_rows=int((gradients.norm(dim=1) == 0).sum()),
        )
    else:
        from gradient_atoms.dictionary import characterise_atoms, run_dictionary_learning
        from gradient_atoms.projection import load_ekfac_eigen, project_gradients_ekfac

        payload = torch.load(
            root / "gradients_all.pt", map_location="cpu", weights_only=True, mmap=True
        )
        gradients, names = payload["gradients"], payload["lora_names"]
        factors = load_ekfac_eigen(str(root / "factors/paper_ekfac/factors_training"))
        validate_factors(read_json(root / "layout.json"), factors)
        projected, module_info = project_gradients_ekfac(
            gradients, names, factors, config["top_k_eigen"]
        )
        if projected.shape != (
            config["n_documents"],
            config["lora_tensors"] * config["top_k_eigen"],
        ):
            raise ValueError("Projection dimension differs from the paper")
        dictionary, coefficients, norms = run_dictionary_learning(
            projected,
            config["n_atoms"],
            config["dictionary_alpha"],
            config["dictionary_batch_size"],
            config["dictionary_max_iter"],
            config["seed"],
        )
        info = characterise_atoms(
            dictionary,
            coefficients,
            gradients,
            docs,
            config["n_atoms"],
            config["coherence_top_documents"],
        )
        torch.save(
            {
                "dictionary": torch.from_numpy(dictionary),
                "coefficients": torch.from_numpy(coefficients),
                "grad_norms": torch.from_numpy(norms),
                "module_info": module_info,
                "lora_names": names,
                "projected_gradients": projected,
            },
            root / "atoms.pt",
        )
        atomic_json(root / "atom_characterisations.json", info)
        save_manifest(
            root,
            stage,
            started,
            [root / "atoms.pt", root / "atom_characterisations.json"],
            projected_shape=list(projected.shape),
            n_atoms=len(info),
        )
    progress(root, stage + "_complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "freeze", *STAGES))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_setup.json"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--raw-data", type=Path)
    parser.add_argument("--transfer-manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "prepare":
        prepare(root, args.config, args.base, args.upstream, args.raw_data, args.transfer_manifest)
    elif args.command == "preflight":
        preflight(root, read_json(root / "configuration.json"))
    elif args.command == "freeze":
        freeze(root)
    else:
        run_stage(root, args.command)


if __name__ == "__main__":
    main()
