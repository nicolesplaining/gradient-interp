import fcntl
import importlib.util
import signal
from pathlib import Path

import pytest

from gradient_interp.paper_svd_replication import condition_grid
from gradient_interp.study import atomic_json, file_hash


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FINAL = load_script("finalize_paper_svd_parallel")
WATCH = load_script("watch_paper_svd_prefix")
AUDIT = load_script("audit_paper_svd")


def fixture(root, watcher=False):
    original = {
        "status": "running",
        "started_unix": 90,
        "deadline_unix": 43290,
        "controller_pid": 18,
        "worker_pid": 17,
        "attempts": [{"pid": 17, "started_unix": 91}],
    }
    atomic_json(root / "preflight.json", {"peak_gpu_bytes": 40})
    plan = {
        "scope": "development_only",
        "max_hours": 12,
        "source_files_sha256": {},
        "input_hashes": {str(root / "preflight.json"): file_hash(root / "preflight.json")},
    }
    atomic_json(root / "plan.json", plan)
    original["plan_sha256"] = file_hash(root / "plan.json")
    spec = {
        "training_seed": 42,
        "input_hashes": {},
        "chunk": 8,
        "max_new_tokens": 300,
        "stop_ids": [1, 106],
        "counts": {"conditions": 101, "total_responses": 14540, "descriptor_gradients": 160},
    }
    atomic_json(root / "specification.json", spec)
    atomic_json(root / "data.json", {"synthetic": True})
    (root / "directions.pt").write_bytes(b"not a tensor")
    atomic_json(
        root / "selection-lock.json",
        {
            "heldout_outcomes_read": False,
            "specification_sha256": file_hash(root / "specification.json"),
            "files_sha256": {n: file_hash(root / n) for n in ("directions.pt", "data.json")},
        },
    )
    atomic_json(root / "controller-at-amendment.json", original)
    atomic_json(root / "controller.json", original)
    (root / "tail-source.py").write_text("# synthetic preserved helper\n")
    assignment = {
        "seed": 42,
        "original_root": str(root),
        "prefix_indices": list(range(61)),
        "tail_indices": list(range(61, 101)),
        "original_started_unix": 90,
        "original_deadline_unix": 43290,
        "runtime_asset_hashes": {},
        **{
            key: file_hash(root / name)
            for key, name in (
                ("plan_sha256", "plan.json"),
                ("specification_sha256", "specification.json"),
                ("selection_lock_sha256", "selection-lock.json"),
                ("controller_at_amendment_sha256", "controller-at-amendment.json"),
            )
        },
    }
    amendment = {
        "type": "paper_svd_parallel_execution_amendment",
        "frozen_source_commit": FINAL.FROZEN_COMMIT,
        "execution_host": FINAL.HOST,
        "scientific_choices_unchanged": True,
        "scientific_outcomes_read": False,
        "frozen_unix": 100,
        "helper_sha256": file_hash(root / "tail-source.py"),
        "finalizer_sha256": file_hash(Path(FINAL.__file__)),
        "prefix_watcher_sha256": file_hash(Path(WATCH.__file__)),
        "assignments": [
            assignment,
            {
                **assignment,
                "seed": 43,
                "prefix_indices": list(range(81)),
                "tail_indices": list(range(81, 101)),
            },
        ],
    }
    atomic_json(root / "execution-amendment.json", amendment)
    ah = file_hash(root / "execution-amendment.json")
    (root / "generations").mkdir()
    hashes = {}
    for index, condition in enumerate(condition_grid()):
        if watcher and index > 60:
            continue
        path = f"generations/{condition['id']}.json"
        (root / path).write_bytes(b"opaque output, deliberately not JSON\xff")
        hashes[path] = file_hash(root / path)
    if watcher:
        return amendment
    prefix = {
        **original,
        "status": "interrupted",
        "worker_pid": None,
        "attempts": [{"pid": 17, "started_unix": 91, "finished_unix": 220, "returncode": -2}],
    }
    atomic_json(root / "prefix-controller.json", prefix)
    atomic_json(root / "controller.json", prefix)
    grid = condition_grid()
    prefix_hashes = {
        f"generations/{grid[i]['id']}.json": hashes[f"generations/{grid[i]['id']}.json"]
        for i in assignment["prefix_indices"]
    }
    tail_hashes = {k: v for k, v in hashes.items() if k not in prefix_hashes}
    atomic_json(
        root / "prefix-completion-receipt.json",
        {
            "type": "paper_svd_prefix_completion_receipt",
            "seed": 42,
            "all_processes_dead": True,
            "observed_unix": 240,
            "controller_sha256": file_hash(root / "prefix-controller.json"),
            "amendment_sha256": ah,
            "completed_prefix_indices": assignment["prefix_indices"],
            "checked_pids": [17, 18],
            "generation_files_sha256": prefix_hashes,
            "scientific_output_contents_parsed": False,
            "watcher_sha256": amendment["prefix_watcher_sha256"],
        },
    )
    common = {
        "seed": 42,
        "execution_host": FINAL.HOST,
        "amendment_sha256": ah,
        "helper_sha256": amendment["helper_sha256"],
    }
    preflight = {
        **common,
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
        "specification_sha256": assignment["specification_sha256"],
        "selection_lock_sha256": assignment["selection_lock_sha256"],
    }
    atomic_json(root / "tail-preflight.json", preflight)
    atomic_json(
        root / "tail-controller.json",
        {
            **common,
            "status": "complete",
            "started_unix": 160,
            "deadline_unix": 43290,
            "assigned_indices": assignment["tail_indices"],
            "controller_pid": 22,
            "worker_pid": None,
        },
    )
    tm = {
        **common,
        "type": "paper_svd_tail_manifest",
        "started_unix": 170,
        "completed_unix": 250,
        "deadline_unix": 43290,
        "original_started_unix": 90,
        "adapter_restored": True,
        "preflight_sha256": file_hash(root / "tail-preflight.json"),
        "peak_gpu_bytes": 50,
        "source_files_sha256": {},
        "runtime_asset_hashes": {},
        "assigned_indices": assignment["tail_indices"],
        "assigned_ids": [grid[i]["id"] for i in assignment["tail_indices"]],
        "conditions": 40,
        "generations": 5600,
        "files_sha256": tail_hashes,
        **{
            k: assignment[k]
            for k in ("plan_sha256", "specification_sha256", "selection_lock_sha256")
        },
    }
    atomic_json(root / "tail-manifest.json", tm)
    atomic_json(
        root / "tail-completion-receipt.json",
        {
            **common,
            "type": "paper_svd_tail_completion_receipt",
            "all_processes_dead": True,
            "completed_unix": 250,
            "observed_unix": 270,
            "checked_pids": [22],
            "manifest_sha256": file_hash(root / "tail-manifest.json"),
            "controller_sha256": file_hash(root / "tail-controller.json"),
        },
    )
    (root / "descriptors.pt").write_bytes(b"opaque descriptor bytes\xff")
    atomic_json(
        root / "descriptor-manifest.json",
        {
            "count": 160,
            "adapter_unchanged": True,
            "selection_lock_sha256": assignment["selection_lock_sha256"],
            "completed_unix": 95,
            "sha256": file_hash(root / "descriptors.pt"),
        },
    )
    return amendment


def test_metadata_finalization_matches_amendment_auditor_without_parsing_outputs(
    tmp_path, monkeypatch
):
    fixture(tmp_path)
    previous = FINAL.read(tmp_path / "controller.json")
    original_read = Path.read_text

    def protected_read(path, *args, **kwargs):
        assert path.parent.name != "generations" and path.name != "descriptors.pt"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", protected_read)
    manifest, finalization, controller = FINAL.prepare_finalization(tmp_path, 280, lambda _: False)
    assert not (tmp_path / "manifest.json").exists()
    assert controller["attempts"] == previous["attempts"]
    assert controller["started_unix"] == previous["started_unix"]
    assert controller["deadline_unix"] == previous["deadline_unix"]
    assert (manifest["conditions"], manifest["generations"], manifest["descriptor_gradients"]) == (
        101,
        14540,
        160,
    )
    assert manifest["peak_gpu_bytes"] == 50 and len(manifest["files_sha256"]) == 103
    atomic_json(tmp_path / "execution-finalization.json", finalization)
    result = AUDIT.validate_execution_amendment(
        tmp_path,
        controller,
        FINAL.read(tmp_path / "plan.json"),
        FINAL.read(tmp_path / "specification.json"),
        manifest,
    )
    assert len(result["tail_indices"]) == 40


def test_finalizer_refuses_active_original_process_and_deadline_extension(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="still alive"):
        FINAL.prepare_finalization(tmp_path, 280, lambda pid: pid == 17)
    with pytest.raises(ValueError, match="deadline"):
        FINAL.prepare_finalization(tmp_path, 50000, lambda _: False)
    assert not (tmp_path / "manifest.json").exists()


def test_finalizer_rejects_changed_generation_or_descriptor_hash(tmp_path):
    fixture(tmp_path)
    path = next((tmp_path / "generations").glob("*.json"))
    old = path.read_bytes()
    path.write_bytes(b"tampered output")
    with pytest.raises(ValueError, match="generation changed"):
        FINAL.prepare_finalization(tmp_path, 280, lambda _: False)
    path.write_bytes(old)
    (tmp_path / "descriptors.pt").write_bytes(b"tampered descriptor")
    with pytest.raises(ValueError, match="Descriptor tensor hash"):
        FINAL.prepare_finalization(tmp_path, 280, lambda _: False)


def test_finalizer_rejects_overlapping_inventory_and_preserves_failure(tmp_path, monkeypatch):
    fixture(tmp_path)
    pr = FINAL.read(tmp_path / "prefix-completion-receipt.json")
    tail_name = f"generations/{condition_grid()[61]['id']}.json"
    pr["generation_files_sha256"][tail_name] = file_hash(tmp_path / tail_name)
    atomic_json(tmp_path / "prefix-completion-receipt.json", pr)
    with pytest.raises(ValueError, match="overlaps or is incomplete"):
        FINAL.prepare_finalization(tmp_path, 280, lambda _: False)
    original = (tmp_path / "controller.json").read_bytes()
    monkeypatch.setattr(
        FINAL, "run", lambda _: (_ for _ in ()).throw(ValueError("synthetic failure"))
    )
    with pytest.raises(ValueError, match="synthetic failure"):
        FINAL.guarded_run(tmp_path)
    assert (tmp_path / "controller.json").read_bytes() == original
    assert len(list(tmp_path.glob("finalization-failure-*.json"))) == 1


def mock_watcher_processes(root, monkeypatch, supported=True):
    state = {"alive": True, "signals": []}
    monkeypatch.setattr(WATCH.time, "time", lambda: 150)
    monkeypatch.setattr(WATCH, "alive", lambda _: state["alive"])
    monkeypatch.setattr(
        WATCH,
        "process_identity",
        lambda pid, module, root: {"pid": pid, "started_ticks": "123", "module": module},
    )

    def masks(pid):
        if not supported:
            raise ValueError("Original controller cannot handle SIGINT through Python")
        return {"SigIgn": 0, "SigBlk": 0, "SigCgt": 2}

    monkeypatch.setattr(WATCH, "sigint_state", masks)

    def interrupt(pid, number):
        assert pid == 18 and number == signal.SIGINT
        state["signals"].append(number)
        c = WATCH.read(root / "controller.json")
        c.update(status="interrupted", worker_pid=None)
        c["attempts"][-1].update(finished_unix=150, returncode=-2)
        atomic_json(root / "controller.json", c)
        state["alive"] = False

    monkeypatch.setattr(WATCH.os, "kill", interrupt)
    return state


def test_watcher_handles_already_completed_boundary_with_real_history_shape(tmp_path, monkeypatch):
    fixture(tmp_path, watcher=True)
    state = mock_watcher_processes(tmp_path, monkeypatch)
    WATCH.run(tmp_path, tmp_path / "execution-amendment.json")
    assert state["signals"] == [signal.SIGINT]
    receipt = WATCH.read(tmp_path / "prefix-completion-receipt.json")
    assert receipt["all_processes_dead"] and len(receipt["generation_files_sha256"]) == 61
    assert receipt["scientific_output_contents_parsed"] is False
    assert WATCH.read(tmp_path / "prefix-controller.json")["attempts"][-1]["returncode"] == -2


def test_watcher_refuses_extra_original_file_without_signaling_and_records_failure(
    tmp_path, monkeypatch
):
    fixture(tmp_path, watcher=True)
    state = mock_watcher_processes(tmp_path, monkeypatch)
    (tmp_path / "generations" / f"{condition_grid()[61]['id']}.json").write_text("opaque")
    original = (tmp_path / "controller.json").read_bytes()
    with pytest.raises(ValueError, match="ownership boundary"):
        WATCH.guarded_run(tmp_path, tmp_path / "execution-amendment.json")
    assert not state["signals"] and (tmp_path / "controller.json").read_bytes() == original
    assert len(list(tmp_path.glob("prefix-watcher-failure-*.json"))) == 1


def test_watcher_refuses_ignored_sigint_or_completed_controller(tmp_path, monkeypatch):
    fixture(tmp_path, watcher=True)
    state = mock_watcher_processes(tmp_path, monkeypatch, supported=False)
    with pytest.raises(ValueError, match="cannot handle SIGINT"):
        WATCH.run(tmp_path, tmp_path / "execution-amendment.json")
    assert not state["signals"]
    controller = WATCH.read(tmp_path / "controller.json")
    controller["status"] = "complete"
    atomic_json(tmp_path / "controller.json", controller)
    with pytest.raises(ValueError, match="not the frozen running process"):
        WATCH.run(tmp_path, tmp_path / "execution-amendment.json")


def test_finalizer_acquires_original_controller_lock_and_preserves_attempts_on_commit(
    tmp_path, monkeypatch
):
    fixture(tmp_path)
    with (tmp_path / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            FINAL.run(tmp_path)
    previous = FINAL.read(tmp_path / "controller.json")

    def dead(*args):
        raise ProcessLookupError

    monkeypatch.setattr(FINAL.os, "kill", dead)
    monkeypatch.setattr(FINAL.time, "time", lambda: 280)
    assert FINAL.run(tmp_path)["scientific_output_contents_parsed"] is False
    complete = FINAL.read(tmp_path / "controller.json")
    assert complete["status"] == "complete" and complete["attempts"] == previous["attempts"]
    assert complete["execution_finalization_sha256"] == file_hash(
        tmp_path / "execution-finalization.json"
    )
    with pytest.raises(ValueError, match="Preserve existing"):
        FINAL.run(tmp_path)
