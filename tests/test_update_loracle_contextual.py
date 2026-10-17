"""Prompt-only input, matched controls and the unchanged numerical training recipe."""

import ast
import copy
import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from gradient_interp import update_loracle_contextual as f
from gradient_interp import update_loracle_contextual_analysis as analysis
from gradient_interp import update_loracle_factorized as previous
from gradient_interp.study import atomic_json, file_hash
from gradient_interp.update_loracle_data import LABELS, NO_CHANGE


def cases(n_actual=256, n_zero=16):
    labels = [v for v in LABELS if v != NO_CHANGE]
    return [
        {
            "id": str(i),
            "tokens": f"tokens-{i}",
            "physical_update_norm": 1.0 if i < n_actual else 0.0,
            "label": labels[i % 32] if i < n_actual else NO_CHANGE,
            "control": i >= n_actual,
            "probes": [
                {"prompt": f'Query {i}-{j}: labels "a", b, c.\nReturn a list.'} for j in range(4)
            ],
        }
        for i in range(n_actual + n_zero)
    ]


def test_context_reads_only_four_prompt_texts_and_keeps_the_original_question():
    class Allowlist(dict):
        def __getitem__(self, key):
            assert key in ("id", "tokens", "physical_update_norm", "probes", "prompt")
            return super().__getitem__(key)

    case = cases(1, 0)[0]
    case.update(label="PRIVATE", source_pattern="PRIVATE", batch="PRIVATE", losses="PRIVATE")
    case["probes"] = [
        Allowlist({**probe, "values": "PRIVATE", "answer": "PRIVATE"}) for probe in case["probes"]
    ]
    rows = f.question_views(Allowlist(case))
    old = previous.question_views(case)
    for row, before in zip(rows, old, strict=True):
        assert "PRIVATE" not in str(row)
        assert row["text"].endswith(before["text"])
        payload = row["text"].split("Evaluation prompts as a JSON list:\n")[1].split("\n\n")[0]
        assert json.loads(payload) == [p["prompt"] for p in case["probes"]]
        assert row["probe_prompts"] == json.loads(payload)
        assert {k: v for k, v in row.items() if k not in ("text", "probe_prompts")} == {
            k: v for k, v in before.items() if k != "text"
        }
    altered = copy.deepcopy(case)
    altered["label"] = "another outcome"
    assert f.question_views(altered) == rows


@pytest.mark.parametrize("mutation", ["missing", "short", "duplicate", "blank", "not_string"])
def test_invalid_query_context_cannot_silently_fall_back(mutation):
    case = cases(1, 0)[0]
    if mutation == "missing":
        del case["probes"]
    elif mutation == "short":
        case["probes"].pop()
    elif mutation == "duplicate":
        case["probes"][3] = case["probes"][0]
    elif mutation == "blank":
        case["probes"][0]["prompt"] = " "
    else:
        case["probes"][0]["prompt"] = 123
    with pytest.raises((KeyError, ValueError)):
        f.question_views(case)


def test_weights_targets_and_order_equal_original_recipe():
    data = cases()
    now, balance = f.training_instances(data)
    old, old_balance = previous.training_instances(data)
    assert balance == old_balance
    assert len(now) == 1040
    remove = ("text", "probe_prompts")
    assert [{k: v for k, v in r.items() if k not in remove} for r in now] == [
        {k: v for k, v in r.items() if k not in remove} for r in old
    ]
    # Compare the AST of unchanged numerical helpers, not only sample outputs.
    for name in (
        "target_components",
        "join_components",
        "option_map",
        "question",
        "training_instances",
        "training_majorities",
        "score",
    ):
        assert ast.dump(ast.parse(inspect.getsource(getattr(f, name)))) == ast.dump(
            ast.parse(inspect.getsource(getattr(previous, name)))
        )


def test_direction_shuffle_keeps_recipient_prompts_options_and_norm():
    data = cases(8, 2)
    shuffled = f.shuffled_directions(data)
    by_id = {c["id"]: c for c in data}
    for original, control in zip(data, shuffled, strict=True):
        assert control["tokens"] == by_id[control["direction_donor"]]["tokens"]
        assert (control["direction_donor"] == original["id"]) == original["control"]
        a = f.frozen_question_inputs([original])
        b = f.frozen_question_inputs([control])
        assert a == b


def test_original_audit_join_strips_every_probe_field_except_text(tmp_path, monkeypatch):
    data = cases(2, 0)
    for c in data:
        c["split"] = "development"
        for p in c["probes"]:
            p.update(values=["SECRET"], reference="SECRET")
    atomic_json(tmp_path / "data.json", {"cases": data})
    calls = []

    def verified(collection, split):
        calls.append((collection, split))
        return [{k: v for k, v in c.items() if k != "probes"} for c in data]

    monkeypatch.setattr(f, "original_verified_cases", verified)
    result = f.verified_cases(tmp_path, "development")
    assert calls == [(tmp_path, "development")]
    assert all(set(p) == {"prompt"} for c in result for p in c["probes"])
    assert "SECRET" not in str(f.frozen_question_inputs(result))
    atomic_json(tmp_path / "data.json", {"cases": data[:1]})
    with pytest.raises(ValueError, match="grid changed"):
        f.verified_cases(tmp_path, "development")
    calls.clear()
    with pytest.raises(ValueError, match="reserved"):
        f.verified_cases(tmp_path, "reserved")
    assert not calls


def test_forward_records_actual_question_and_constrains_valid_options(monkeypatch):
    row = next(r for r in f.training_instances(cases())[0] if r["head"] == "case")
    seen = []

    def forward(handle, tokens, text, target, scale):
        seen.append((text, target, scale))
        scores = [0.0] * 33
        scores[1], scores[25] = 2, 100
        return {"choice_logits": scores, "loss": 2.0, "hook_calls": 2, "input_tokens": 5100}

    monkeypatch.setattr(f, "choice_forward", forward)
    result = f.forward(None, None, row, train=True, loss_scale=0.3)
    assert result["prediction"] == row["options"]["B"]
    assert result["question_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest()
    assert seen == [(row["text"], row["target_option"], 0.3)]
    bad = {**row, "target_option": 25}
    with pytest.raises(ValueError, match="displayed answer"):
        f.forward(None, None, bad, train=True)


def test_actual_780_step_loop_on_tiny_cpu_model_preserves_context_and_frozen_parameters(
    tmp_path, monkeypatch
):
    """Exercise real optimizer, accumulation and epoch saves using synthetic logits."""
    root, collection, prior = tmp_path / "run", tmp_path / "collection", tmp_path / "prior"
    root.mkdir()
    collection.mkdir()
    prior.mkdir()
    train, dev = cases(), cases(64, 8)
    token_path = tmp_path / "synthetic.pt"
    torch.save(torch.ones(2), token_path)
    for c in train + dev:
        c["tokens"] = token_path
    atomic_json(collection / "integrity.json", {"passed": True})
    atomic_json(prior / "simple-references.json", {"development": [{"id": c["id"]} for c in dev]})
    plan = {
        "recipe": "contextual_factorized_v7",
        "epochs": 3,
        "effective_batch": 4,
        "learning_rate": 0.00003,
        "training_seed": f.TRAINING_SEED,
        "input_files_sha256": {},
        "collection_root": str(collection),
        "collection_integrity_sha256": file_hash(collection / "integrity.json"),
        "prior_training_root": str(prior),
        "source_commit": "synthetic",
    }
    atomic_json(root / "plan.json", plan)
    atomic_json(root / "question-inputs.json", f.frozen_question_inputs(train + dev))
    model = torch.nn.Module()
    model.register_parameter("adapter", torch.nn.Parameter(torch.linspace(-0.1, 0.1, 33)))
    model.register_parameter("base", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    before, frozen = model.adapter.detach().clone(), model.base.detach().clone()
    monkeypatch.setattr(f, "load_interpreter", lambda: SimpleNamespace(model=model))
    monkeypatch.setattr(f, "guard", lambda _: plan)
    monkeypatch.setattr(f, "verified_cases", lambda _, split: train if split == "train" else dev)
    monkeypatch.setattr(f, "preflight", lambda *args: None)
    monkeypatch.setattr(f, "status", lambda *args: None)

    def save(model, path):
        path.mkdir()
        torch.save(model.adapter.detach(), path / "synthetic-adapter.pt")

    monkeypatch.setattr(f, "save_adapter", save)
    observed = []

    def fake_forward(handle, tokens, text, target, scale):
        assert "Evaluation prompts as a JSON list:" in text
        logits = handle.model.adapter
        loss = None
        if target is not None:
            objective = torch.nn.functional.cross_entropy(logits[None, :], torch.tensor([target]))
            loss = float(objective.detach())
            (objective * scale).backward()
        observed.append((target, scale, hashlib.sha256(text.encode()).hexdigest()))
        return {
            "choice_logits": logits.detach().tolist(),
            "loss": loss,
            "hook_calls": 2 if target is not None else 1,
            "input_tokens": 5100,
        }

    monkeypatch.setattr(f, "choice_forward", fake_forward)
    f.train(root)
    assert not torch.equal(before, model.adapter)
    assert torch.equal(frozen, model.base)
    assert model.base.grad is None
    history = f.read(root / "training-history.json")
    instances, _ = f.training_instances(train)
    analysis.check_history(history, instances)
    # Independent calculation uses the original recipe's targets and one batched CE.
    reference = torch.nn.Parameter(before.clone())
    optimizer = torch.optim.AdamW([reference], lr=0.00003, weight_decay=0.01, foreach=False)
    old_instances, _ = previous.training_instances(train)
    by_id = {row["id"]: row for row in old_instances}
    for step in history:
        optimizer.zero_grad(set_to_none=True)
        group = [by_id[name] for name in step["instances"]]
        targets = torch.tensor([row["target_option"] for row in group])
        weights = torch.tensor([row["weight"] for row in group])
        losses = torch.nn.functional.cross_entropy(
            reference[None, :].expand(4, -1), targets, reduction="none"
        )
        (losses * weights).mean().backward()
        torch.nn.utils.clip_grad_norm_([reference], 1.0, error_if_nonfinite=True)
        optimizer.param_groups[0]["lr"] = f.learning_rate(step["step"], 780, 39, 0.00003)
        optimizer.step()
    torch.testing.assert_close(model.adapter, reference, atol=1e-7, rtol=1e-6)
    assert len([v for v in observed if v[0] is not None]) == 3120
    assert len([v for v in observed if v[0] is None]) == 2304
    for filename in ["released-development.json", "released-shuffled-development.json"]:
        analysis.check_predictions(f.read(root / filename), dev)
    for epoch in range(1, 4):
        for filename in ["development.json", "shuffled-development.json"]:
            analysis.check_predictions(f.read(root / "epochs" / str(epoch) / filename), dev)
    manifest = f.read(root / "manifest.json")
    assert manifest["optimizer_steps"] == 780
    assert manifest["frozen_base_unchanged"] is True
    assert manifest["query_context_used_for_all_methods"] is True
    for name, expected in manifest["files_sha256"].items():
        assert file_hash(root / name) == expected
    damaged = copy.deepcopy(history)
    damaged[0]["question_sha256"][0] = "wrong"
    with pytest.raises(ValueError, match="context differs"):
        analysis.check_history(damaged, instances)
    predictions = f.read(root / "released-development.json")
    predictions[0]["heads"]["format"]["question_sha256"] = "wrong"
    with pytest.raises(ValueError, match="context hash"):
        analysis.check_predictions(predictions, dev)
    with pytest.raises(ValueError, match="already attempted"):
        f.train(root)
