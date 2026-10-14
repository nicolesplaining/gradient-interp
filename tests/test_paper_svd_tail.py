import ast
import copy
import importlib.util
import inspect
import time
from pathlib import Path

import pytest

from gradient_interp import paper_svd_replication as worker
from gradient_interp.study import atomic_json, file_hash


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


TAIL = load_script("run_paper_svd_tail")
AUDIT = load_script("audit_paper_svd")


def preflight():
    return {
        "completed_unix": 150,
        "adapter_restored": True,
        "greedy_tokens_restored": True,
        "finite_training_gradient": True,
        "training_input_only": True,
        "long_training_prefix_completion_checked": True,
        "forced_cap_tokens_verified": True,
        "tensors": 136,
        "parameter_count": 2228224,
        "scoring_sequence_length": 500,
        "generation_chunk_checked": 8,
        "generation_cap_checked": 300,
        "long_completion_tokens": 170,
        "stop_checks": {"1": True, "106": True},
    }


def assignment(seed, root="/original", output="/tail"):
    return {
        "seed": seed,
        "original_root": root,
        "tail_root": output,
        "original_started_unix": 90,
        "original_deadline_unix": 43290,
        "prefix_indices": list(range(68)),
        "tail_indices": list(range(68, 101)),
        **{
            k: "a" * 64
            for k in (
                "plan_sha256",
                "specification_sha256",
                "selection_lock_sha256",
                "controller_at_amendment_sha256",
            )
        },
        "runtime_asset_hashes": {},
    }


def amendment(items=None):
    return {
        "type": "paper_svd_parallel_execution_amendment",
        "frozen_unix": 100,
        "frozen_source_commit": TAIL.FROZEN_COMMIT,
        "scientific_choices_unchanged": True,
        "scientific_outcomes_read": False,
        "execution_host": TAIL.AUTHORIZED_HOST,
        "helper_sha256": file_hash(Path(TAIL.__file__)),
        "assignments": items or [assignment(42), assignment(43)],
    }


def test_fixed_assignment_has_no_overlap_gap_or_deadline_extension():
    good = amendment()
    assert TAIL.validate_assignment(good, 42)["tail_indices"] == list(range(68, 101))
    for changes in (
        {"tail_indices": list(range(67, 101))},
        {"prefix_indices": list(range(67))},
        {"tail_indices": list(range(68, 100))},
        {"original_deadline_unix": 50000},
    ):
        bad = copy.deepcopy(good)
        bad["assignments"][0].update(changes)
        with pytest.raises(ValueError):
            TAIL.validate_assignment(bad, 42)
    for changes in (
        {"execution_host": "unauthorized"},
        {"scientific_outcomes_read": True},
        {"scientific_choices_unchanged": False},
    ):
        with pytest.raises(ValueError):
            TAIL.validate_assignment({**good, **changes}, 42)


def test_scientific_generation_block_and_record_are_ast_identical_to_original():
    original = ast.parse(inspect.getsource(worker.run)).body[0]
    loop = next(
        n
        for n in original.body
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and ast.unparse(n.target) == "(index, condition)"
    )
    helper = ast.parse(inspect.getsource(TAIL.execute_condition)).body[0]

    def block(nodes):
        start = next(
            i
            for i, n in enumerate(nodes)
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "restore"
        )
        finish = next(i for i in range(start, len(nodes)) if isinstance(nodes[i], ast.Try))
        return ast.dump(ast.Module(body=nodes[start : finish + 1], type_ignores=[]))

    assert block(loop.body) == block(helper.body)

    def saved_record(nodes):
        return next(
            n
            for n in nodes
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "atomic_json"
            and ast.unparse(n.value.args[0]) == "final"
        )

    assert ast.dump(saved_record(loop.body)) == ast.dump(saved_record(helper.body))


def input_fixture(tmp_path):
    root, output, source, base = [tmp_path / p for p in ("original", "tail", "source", "base")]
    root.mkdir()
    (source / "adapter").mkdir(parents=True)
    base.mkdir()
    paths = [
        source / "atoms.pt",
        source / "adapter/adapter_model.safetensors",
        source / "adapter/adapter_config.json",
        base / "config.json",
    ]
    for path in paths:
        path.write_text("synthetic frozen runtime asset")
    assets = {str(path): file_hash(path) for path in paths}
    spec = {
        "training_seed": 42,
        "source_root": str(source),
        "source_config": {
            "base_directory": str(base),
            "model_asset_hashes": {"config.json": assets[str(base / "config.json")]},
        },
        "input_hashes": assets,
        "chunk": 8,
        "max_new_tokens": 300,
        "stop_ids": [1, 106],
    }
    atomic_json(root / "specification.json", spec)
    atomic_json(root / "plan.json", {"source_files_sha256": {}, "max_hours": 12})
    atomic_json(root / "data.json", {"synthetic": True})
    (root / "directions.pt").write_text("synthetic directions")
    atomic_json(
        root / "selection-lock.json",
        {
            "heldout_outcomes_read": False,
            "specification_sha256": file_hash(root / "specification.json"),
            "files_sha256": {
                name: file_hash(root / name) for name in ("data.json", "directions.pt")
            },
        },
    )
    item = assignment(42, str(root), str(output))
    item.update(
        plan_sha256=file_hash(root / "plan.json"),
        specification_sha256=file_hash(root / "specification.json"),
        selection_lock_sha256=file_hash(root / "selection-lock.json"),
        runtime_asset_hashes=assets,
    )
    amended = amendment([item, assignment(43)])
    ap, pp = tmp_path / "amendment.json", tmp_path / "preflight.json"
    atomic_json(ap, amended)
    atomic_json(
        pp,
        {
            **preflight(),
            "execution_host": TAIL.AUTHORIZED_HOST,
            "amendment_sha256": file_hash(ap),
            "helper_sha256": amended["helper_sha256"],
            "specification_sha256": item["specification_sha256"],
            "selection_lock_sha256": item["selection_lock_sha256"],
        },
    )
    return root, output, ap, pp


def test_input_guards_need_frozen_assets_new_preflight_and_no_prior_tail(tmp_path, monkeypatch):
    inputs = input_fixture(tmp_path)
    monkeypatch.setattr(TAIL.time, "time", lambda: 200)
    assert TAIL.validate_inputs(*inputs)[1]["seed"] == 42
    assert not (inputs[0] / "descriptors.pt").exists()
    pf = TAIL.read(inputs[3])
    atomic_json(inputs[3], {**pf, "completed_unix": 99})
    with pytest.raises(ValueError, match="preflight or original deadline"):
        TAIL.validate_inputs(*inputs)
    atomic_json(inputs[3], pf)
    inputs[1].mkdir()
    atomic_json(inputs[1] / "tail-controller.json", {"status": "failed"})
    with pytest.raises(ValueError, match="no automatic resume"):
        TAIL.validate_inputs(*inputs)
    (inputs[1] / "tail-controller.json").unlink()
    (inputs[0] / "directions.pt").write_text("changed")
    with pytest.raises(ValueError, match="input bank changed"):
        TAIL.validate_inputs(*inputs)


def condition_arguments(tmp_path):
    atomic_json(tmp_path / "plan.json", {"synthetic": True})
    atomic_json(tmp_path / "selection-lock.json", {"synthetic": True})
    directory = tmp_path / "generations"
    directory.mkdir()
    condition = worker.condition_grid()[68]
    data = {
        "published_questions": [
            {"id": f"q{i}", "family": condition["family"], "prompt": "synthetic"}
            for i in range(100)
        ],
        "questions": [{"id": f"control{i}", "prompt": "synthetic"} for i in range(40)],
    }
    directions = {"vectors": {condition["family"]: {condition["method"]: "synthetic vector"}}}
    return [
        tmp_path,
        {"chunk": 8, "max_new_tokens": 300, "stop_ids": [1, 106]},
        None,
        None,
        [],
        {},
        directions,
        data,
        {f: lambda _: False for f in worker.FAMILIES},
        condition,
        68,
        time.time() + 300,
        directory,
    ]


def test_suffix_condition_keeps_original_schema_count_and_restores(tmp_path, monkeypatch):
    arguments = condition_arguments(tmp_path)
    restored, calls = [], []
    monkeypatch.setattr(TAIL, "restore", lambda *args: restored.append(True))
    monkeypatch.setattr(TAIL, "set_direction", lambda *args: calls.append(args[-1]))
    monkeypatch.setattr(
        TAIL,
        "generate",
        lambda model, tok, prompts, cap: (["ok"] * len(prompts), [[7, 1]] * len(prompts)),
    )
    path, count = TAIL.execute_condition(*arguments)
    record = TAIL.read(path)
    assert count == len(record["rows"]) == 140 and len(restored) == 2
    assert calls == [arguments[9]["strength"] * arguments[9]["sign"]]
    assert set(record) == {
        *arguments[9],
        "applied_unix",
        "completed_unix",
        "selection_lock_sha256",
        "plan_sha256",
        "adapter_restored",
        "rows",
    }
    with pytest.raises(ValueError, match="already exists"):
        TAIL.execute_condition(*arguments)


def test_generation_exception_restores_and_does_not_commit_partial_condition(tmp_path, monkeypatch):
    arguments = condition_arguments(tmp_path)
    restored = []
    monkeypatch.setattr(TAIL, "restore", lambda *args: restored.append(True))
    monkeypatch.setattr(TAIL, "set_direction", lambda *args: None)

    def fail(*args):
        raise RuntimeError("synthetic generation failure")

    monkeypatch.setattr(TAIL, "generate", fail)
    with pytest.raises(RuntimeError, match="synthetic generation failure"):
        TAIL.execute_condition(*arguments)
    assert len(restored) == 2 and not list(arguments[-1].glob("*.json"))


def test_tail_receipt_requires_process_death_and_original_deadline(tmp_path, monkeypatch):
    atomic_json(
        tmp_path / "tail-controller.json",
        {"status": "complete", "controller_pid": 99, "worker_pid": None, "deadline_unix": 200},
    )
    atomic_json(
        tmp_path / "tail-manifest.json",
        {"seed": 42, "completed_unix": 150, "amendment_sha256": "a", "helper_sha256": "b"},
    )
    monkeypatch.setattr(TAIL.os, "kill", lambda *args: None)
    with pytest.raises(ValueError, match="still alive"):
        TAIL.receipt(tmp_path, tmp_path / "receipt.json")

    def dead(*args):
        raise ProcessLookupError

    monkeypatch.setattr(TAIL.os, "kill", dead)
    assert TAIL.receipt(tmp_path, tmp_path / "receipt.json")["all_processes_dead"]


def amended_metadata_fixture(root):
    plan = {"source_files_sha256": {}}
    spec = {
        "training_seed": 42,
        "input_hashes": {"synthetic asset": "b" * 64},
        "chunk": 8,
        "max_new_tokens": 300,
        "stop_ids": [1, 106],
    }
    initial = {
        "status": "running",
        "started_unix": 90,
        "deadline_unix": 43290,
        "attempts": [{"pid": 17, "started_unix": 91}],
    }
    prefix = {
        **initial,
        "status": "interrupted",
        "attempts": [{"pid": 17, "started_unix": 91, "finished_unix": 220, "returncode": -2}],
    }
    for name, value in (
        ("plan", plan),
        ("specification", spec),
        ("selection-lock", {}),
        ("controller-at-amendment", initial),
        ("prefix-controller", prefix),
    ):
        atomic_json(root / f"{name}.json", value)
    (root / "tail-source.py").write_text("# synthetic preserved helper source\n")
    item = assignment(42)
    item.update(
        prefix_indices=list(range(61)),
        tail_indices=list(range(61, 101)),
        plan_sha256=file_hash(root / "plan.json"),
        specification_sha256=file_hash(root / "specification.json"),
        selection_lock_sha256=file_hash(root / "selection-lock.json"),
        controller_at_amendment_sha256=file_hash(root / "controller-at-amendment.json"),
        runtime_asset_hashes=spec["input_hashes"],
    )
    other = assignment(43)
    other.update(prefix_indices=list(range(81)), tail_indices=list(range(81, 101)))
    amended = amendment([item, other])
    amended["helper_sha256"] = file_hash(root / "tail-source.py")
    atomic_json(root / "execution-amendment.json", amended)
    ah = file_hash(root / "execution-amendment.json")
    common = {
        "seed": 42,
        "execution_host": TAIL.AUTHORIZED_HOST,
        "amendment_sha256": ah,
        "helper_sha256": amended["helper_sha256"],
    }
    pf = {
        **preflight(),
        **common,
        "specification_sha256": item["specification_sha256"],
        "selection_lock_sha256": item["selection_lock_sha256"],
    }
    atomic_json(root / "tail-preflight.json", pf)
    tc = {
        **common,
        "status": "complete",
        "started_unix": 160,
        "deadline_unix": 43290,
        "assigned_indices": item["tail_indices"],
    }
    atomic_json(root / "tail-controller.json", tc)
    grid = worker.condition_grid()
    prefix_hashes = {f"generations/{grid[i]['id']}.json": "1" * 64 for i in item["prefix_indices"]}
    tail_hashes = {f"generations/{grid[i]['id']}.json": "2" * 64 for i in item["tail_indices"]}
    tm = {
        **common,
        "type": "paper_svd_tail_manifest",
        "started_unix": 170,
        "completed_unix": 250,
        "deadline_unix": 43290,
        "original_started_unix": 90,
        "adapter_restored": True,
        "preflight_sha256": file_hash(root / "tail-preflight.json"),
        "source_files_sha256": plan["source_files_sha256"],
        "runtime_asset_hashes": item["runtime_asset_hashes"],
        "assigned_indices": item["tail_indices"],
        "assigned_ids": [grid[i]["id"] for i in item["tail_indices"]],
        "conditions": 40,
        "generations": 5600,
        "files_sha256": tail_hashes,
        **{
            key: item[key]
            for key in ("plan_sha256", "specification_sha256", "selection_lock_sha256")
        },
    }
    atomic_json(root / "tail-manifest.json", tm)
    tr = {
        **common,
        "type": "paper_svd_tail_completion_receipt",
        "all_processes_dead": True,
        "completed_unix": 250,
        "observed_unix": 270,
        "manifest_sha256": file_hash(root / "tail-manifest.json"),
        "controller_sha256": file_hash(root / "tail-controller.json"),
    }
    atomic_json(root / "tail-completion-receipt.json", tr)
    pr = {
        "seed": 42,
        "all_processes_dead": True,
        "observed_unix": 240,
        "controller_sha256": file_hash(root / "prefix-controller.json"),
        "amendment_sha256": ah,
        "completed_prefix_indices": item["prefix_indices"],
        "generation_files_sha256": prefix_hashes,
    }
    atomic_json(root / "prefix-completion-receipt.json", pr)
    names = (
        "execution-amendment.json",
        "controller-at-amendment.json",
        "prefix-controller.json",
        "prefix-completion-receipt.json",
        "tail-controller.json",
        "tail-manifest.json",
        "tail-completion-receipt.json",
        "tail-preflight.json",
        "tail-source.py",
    )
    atomic_json(
        root / "execution-finalization.json",
        {
            "type": "paper_svd_execution_finalization",
            "seed": 42,
            "assembled_unix": 280,
            "amendment_sha256": ah,
            "all_processes_dead": True,
            "prefix_indices": item["prefix_indices"],
            "tail_indices": item["tail_indices"],
            "prefix_generation_hashes": prefix_hashes,
            "tail_generation_hashes": tail_hashes,
            "provenance_hashes": {name: file_hash(root / name) for name in names},
        },
    )
    controller = {**prefix, "status": "complete", "execution_amendment_sha256": ah}
    manifest = {"execution_amendment_sha256": ah, "files_sha256": {**prefix_hashes, **tail_hashes}}
    return controller, plan, spec, manifest


def test_amendment_audit_preserves_interrupted_attempt_and_disjoint_complete_grid(tmp_path):
    records = amended_metadata_fixture(tmp_path)
    validated = AUDIT.validate_execution_amendment(tmp_path, *records)
    assert validated["prefix_indices"] == list(range(61))
    assert validated["tail_indices"] == list(range(61, 101))
    assert records[0]["attempts"][-1]["returncode"] == -2
    prefix = TAIL.read(tmp_path / "prefix-completion-receipt.json")
    duplicate = f"generations/{worker.condition_grid()[61]['id']}.json"
    prefix["generation_files_sha256"][duplicate] = "2" * 64
    atomic_json(tmp_path / "prefix-completion-receipt.json", prefix)
    with pytest.raises(AssertionError):
        AUDIT.validate_execution_amendment(tmp_path, *records)


def test_amendment_audit_rejects_fake_successful_model_attempt_and_late_tail(tmp_path):
    records = amended_metadata_fixture(tmp_path)
    changed = copy.deepcopy(records)
    changed[0]["attempts"].append(
        {"pid": 999, "started_unix": 250, "finished_unix": 280, "returncode": 0}
    )
    with pytest.raises(AssertionError):
        AUDIT.validate_execution_amendment(tmp_path, *changed)
    tail = TAIL.read(tmp_path / "tail-manifest.json")
    tail["completed_unix"] = 50000
    atomic_json(tmp_path / "tail-manifest.json", tail)
    with pytest.raises(AssertionError):
        AUDIT.validate_execution_amendment(tmp_path, *records)
