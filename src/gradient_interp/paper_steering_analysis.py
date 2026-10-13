"""Descriptive paired analysis of both completed paper steering reproductions."""

# ruff: noqa: E501

import argparse
import csv
import html
import itertools
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from .paper_setup import read_json
from .paper_steering import FAMILIES, METHODS, published_material
from .study import atomic_json, file_hash

LABELS = {
    "yesno": "Yes/no openings",
    "code": "Fenced code",
    "refusal": "Missing-input requests",
    "bullets": "Bulleted lists",
    "numbered": "Numbered lists",
}
METHOD_LABELS = dict(
    zip(
        METHODS,
        ("Selected atom", "Random atom, equal norm", "Gaussian, equal block norms"),
        strict=True,
    )
)
SUBSETS = ("all", "inviting", "control", "exclude_exact_training_overlap")


def quality(row, cap=300):
    ids = row["token_ids"]
    end = next((i for i, token in enumerate(ids) if token in (1, 106)), None)
    tokens = ids[:end] if end is not None else ids
    tokens = [t for t in tokens if t != 0]
    longest = max((sum(1 for _ in group) for _, group in itertools.groupby(tokens)), default=0)
    grams = Counter(tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3)))
    return {
        "tokens": len(tokens),
        "truncated": end is None and len(tokens) >= cap,
        "empty": not row["response"].strip(),
        "possible_degeneration": not row["response"].strip()
        or longest >= 10
        or max(grams.values(), default=0) >= 10,
    }


def aligned_rows(rows, reference):
    mapping = {r["id"]: r for r in rows}
    if len(mapping) != len(rows) or set(mapping) != {r["id"] for r in reference}:
        raise ValueError("Unpaired or duplicate question IDs")
    return [mapping[r["id"]] for r in reference]


def subset_rows(rows, subset):
    if subset == "all":
        return rows
    if subset == "exclude_exact_training_overlap":
        return [r for r in rows if not r["exact_training_prompt_overlap"]]
    return [r for r in rows if r["split"] == subset]


def paired_stats(differences, bootstrap):
    differences = np.asarray(differences, dtype=float)
    return {
        "mean": float(differences.mean()),
        "ci95": np.quantile(differences[bootstrap].mean(1), [0.025, 0.975]).tolist(),
    }


def load_complete(root, checkers):
    state = read_json(root / "controller.json")
    if state["status"] != "complete":
        raise ValueError("Both runs must complete before outcome analysis")
    audit = read_json(root / "integrity.json")
    manifest = read_json(root / "manifest.json")
    spec = read_json(root / "specification.json")
    if not audit["complete"] or not audit["source_input_output_hashes_verified"]:
        raise ValueError("Remote integrity audit required")
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("specification.json", "specification_sha256"),
        ("plan.json", "plan_sha256"),
    ):
        if file_hash(root / name) != audit[key]:
            raise ValueError("Local provenance differs from remote audit")
    if manifest["completed_unix"] > state["deadline_unix"] or not manifest["adapter_restored"]:
        raise ValueError("Deadline or restoration check failed")
    questions = read_json(root / "questions.json")
    records = {}
    total = 0
    for filename, expected in manifest["files_sha256"].items():
        path = root / filename
        if file_hash(path) != expected:
            raise ValueError(f"Generation hash mismatch: {filename}")
        record = read_json(path)
        if not record["adapter_restored"] or record["plan_sha256"] != audit["plan_sha256"]:
            raise ValueError("Invalid condition provenance")
        key = (record["family"], record["method"], record["alpha"], record["sign"])
        if key in records:
            raise ValueError("Duplicate condition")
        expected_questions = (
            questions
            if record["method"] is None
            else [q for q in questions if q["family"] == record["family"]]
        )
        rows = aligned_rows(record["rows"], expected_questions)
        for row, question in zip(rows, expected_questions, strict=True):
            if any(row[k] != value for k, value in question.items()):
                raise ValueError("Question metadata changed")
            if row["detectors"] != {k: bool(fn(row["response"])) for k, fn in checkers.items()}:
                raise ValueError("Stored detector scores do not reproduce")
            row["quality"] = quality(row, spec["max_new_tokens"])
        record["rows"] = rows
        records[key] = record
        total += len(rows)
    expected_keys = {("baseline", None, 0.0, 0)} | set(
        itertools.product(FAMILIES, METHODS, spec["alphas"], spec["signs"])
    )
    if set(records) != expected_keys or total != spec["planned_generations"]:
        raise ValueError("Incomplete condition grid")
    return spec, records


def summarize(root, spec, records):
    rng = np.random.default_rng(8173)
    baseline = records[("baseline", None, 0.0, 0)]["rows"]
    metrics, comparisons, baselines, examples = [], [], [], []
    seed = spec["training_seed"]
    for family in FAMILIES:
        base = [r for r in baseline if r["family"] == family]
        for subset in SUBSETS:
            ref = subset_rows(base, subset)
            bootstrap = rng.integers(0, len(ref), (2000, len(ref)))
            base_values = np.array(
                [[r["detectors"][f] for f in FAMILIES] for r in ref], dtype=float
            )
            baselines.append(
                {
                    "seed": seed,
                    "family": family,
                    "subset": subset,
                    "n": len(ref),
                    "rates": dict(zip(FAMILIES, base_values.mean(0).tolist(), strict=True)),
                    "quality": {
                        key: float(np.mean([r["quality"][key] for r in ref]))
                        for key in ref[0]["quality"]
                    },
                }
            )
            for alpha, sign in itertools.product(spec["alphas"], spec["signs"]):
                values = {}
                for method in METHODS:
                    rows = aligned_rows(
                        subset_rows(records[(family, method, alpha, sign)]["rows"], subset), ref
                    )
                    v = np.array([[r["detectors"][f] for f in FAMILIES] for r in rows], dtype=float)
                    values[method] = v
                    metrics.append(
                        {
                            "seed": seed,
                            "family": family,
                            "subset": subset,
                            "n": len(ref),
                            "method": method,
                            "alpha": alpha,
                            "sign": sign,
                            "signed_step": alpha * sign,
                            "detectors": {
                                f: {
                                    "rate": float(v[:, i].mean()),
                                    "baseline": float(base_values[:, i].mean()),
                                    "change": paired_stats(v[:, i] - base_values[:, i], bootstrap),
                                }
                                for i, f in enumerate(FAMILIES)
                            },
                            "quality": {
                                key: float(np.mean([r["quality"][key] for r in rows]))
                                for key in rows[0]["quality"]
                            },
                            "response_changed_fraction": float(
                                np.mean(
                                    [
                                        r["response"] != b["response"]
                                        for r, b in zip(rows, ref, strict=True)
                                    ]
                                )
                            ),
                        }
                    )
                for control in METHODS[1:]:
                    comparisons.append(
                        {
                            "seed": seed,
                            "family": family,
                            "subset": subset,
                            "n": len(ref),
                            "alpha": alpha,
                            "sign": sign,
                            "signed_step": alpha * sign,
                            "control": control,
                            "differences": {
                                f: paired_stats(
                                    values["atom"][:, i] - values[control][:, i], bootstrap
                                )
                                for i, f in enumerate(FAMILIES)
                            },
                        }
                    )
        chosen = {f"{family}-{i:03d}" for i in (0, 1, 60, 61)}
        for q in base:
            if q["id"] not in chosen:
                continue
            conditions = []
            for method, alpha, sign in itertools.product(METHODS, (1, 5), spec["signs"]):
                row = next(
                    r for r in records[(family, method, alpha, sign)]["rows"] if r["id"] == q["id"]
                )
                conditions.append(
                    {
                        "method": method,
                        "signed_step": alpha * sign,
                        "response": row["response"],
                        "detectors": row["detectors"],
                        "quality": row["quality"],
                    }
                )
            examples.append(
                {
                    "seed": seed,
                    "family": family,
                    "id": q["id"],
                    "prompt": q["prompt"],
                    "split": q["split"],
                    "baseline": {
                        "response": q["response"],
                        "detectors": q["detectors"],
                        "quality": q["quality"],
                    },
                    "conditions": conditions,
                }
            )
    return {
        "seed": seed,
        "root": str(root),
        "baselines": baselines,
        "metrics": metrics,
        "comparisons": comparisons,
        "examples": examples,
    }


def write_report(data, output):
    import matplotlib.pyplot as plt

    all_metrics = [r for s in data for r in s["metrics"]]
    all_comparisons = [r for s in data for r in s["comparisons"]]
    all_baselines = [r for s in data for r in s["baselines"]]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for subset in ("all", "inviting", "control"):
        fig, axes = plt.subplots(
            5, 2, figsize=(12, 16), sharex=True, sharey=True, constrained_layout=True
        )
        for col, study in enumerate(data):
            for i, family in enumerate(FAMILIES):
                ax = axes[i, col]
                base = next(
                    b for b in study["baselines"] if b["family"] == family and b["subset"] == subset
                )["rates"][family]
                for method, color in zip(METHODS, ("#21618c", "#bd642d", "#6b7075"), strict=True):
                    rows = sorted(
                        [
                            r
                            for r in study["metrics"]
                            if r["family"] == family
                            and r["subset"] == subset
                            and r["method"] == method
                        ],
                        key=lambda r: r["signed_step"],
                    )
                    points = sorted(
                        [(r["signed_step"], r["detectors"][family]["rate"]) for r in rows]
                        + [(0, base)]
                    )
                    ax.plot(
                        [p[0] for p in points],
                        [100 * p[1] for p in points],
                        color=color,
                        marker=".",
                        label=METHOD_LABELS[method],
                    )
                ax.axvline(0, color="#999", linewidth=0.6)
                ax.set(
                    title=f"{LABELS[family]}, seed {study['seed']}",
                    ylim=(-3, 103),
                    xticks=[-10, -5, 0, 5, 10],
                )
                if col == 0:
                    ax.set_ylabel("Detector hit rate, %")
                if i == 4:
                    ax.set_xlabel("Signed step s, adapter weights = original − s × direction")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False
        )
        fig.text(
            0.5,
            1.04,
            f"LoRA behavior steering, {subset} prompts\nAll steps and both seeds, descriptive development reproduction",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        for ext in ("png", "pdf"):
            fig.savefig(output / f"curves-{subset}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, study in enumerate(data):
        bullet = sorted(
            [
                r
                for r in study["metrics"]
                if r["family"] == "bullets" and r["subset"] == "all" and r["method"] == "atom"
            ],
            key=lambda r: r["signed_step"],
        )
        axes[0, col].plot(
            [r["signed_step"] for r in bullet],
            [100 * r["detectors"]["bullets"]["rate"] for r in bullet],
            marker=".",
            label="Bullet detector",
            color="#21618c",
        )
        axes[0, col].plot(
            [r["signed_step"] for r in bullet],
            [100 * r["quality"]["truncated"] for r in bullet],
            marker=".",
            label="Truncated at 300 tokens",
            color="#bd642d",
        )
        axes[0, col].plot(
            [r["signed_step"] for r in bullet],
            [100 * r["quality"]["empty"] for r in bullet],
            marker=".",
            label="Empty response",
            color="#6b7075",
        )
        axes[0, col].set(
            title=f"Bullet atom, seed {study['seed']}", ylabel="Answers, %", ylim=(-3, 103)
        )
        axes[0, col].legend(fontsize=9, frameon=False)
        for method, color in zip(METHODS, ("#21618c", "#bd642d", "#6b7075"), strict=True):
            rows = sorted(
                [
                    r
                    for r in study["metrics"]
                    if r["family"] == "numbered" and r["subset"] == "all" and r["method"] == method
                ],
                key=lambda r: r["signed_step"],
            )
            axes[1, col].plot(
                [r["signed_step"] for r in rows],
                [r["quality"]["tokens"] for r in rows],
                marker=".",
                color=color,
                label=METHOD_LABELS[method],
            )
        axes[1, col].set(
            title=f"Numbered-list intervention, seed {study['seed']}",
            ylabel="Mean answer tokens",
            xlabel="Signed step s",
            ylim=(0, 310),
        )
        axes[1, col].legend(fontsize=9, frameon=False)
    fig.suptitle("Formatting changes are entangled with answer length", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"quality.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    with (output / "all-conditions.csv").open("w") as f:
        fields = [
            "seed",
            "family",
            "subset",
            "n",
            "method",
            "signed_step",
            "detector",
            "baseline",
            "rate",
            "change",
            "ci_low",
            "ci_high",
            "mean_tokens",
            "truncated_fraction",
            "possible_degeneration_fraction",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_metrics:
            for detector, d in r["detectors"].items():
                writer.writerow(
                    {
                        **{k: r[k] for k in fields[:6]},
                        "detector": detector,
                        "baseline": d["baseline"],
                        "rate": d["rate"],
                        "change": d["change"]["mean"],
                        "ci_low": d["change"]["ci95"][0],
                        "ci_high": d["change"]["ci95"][1],
                        "mean_tokens": r["quality"]["tokens"],
                        "truncated_fraction": r["quality"]["truncated"],
                        "possible_degeneration_fraction": r["quality"]["possible_degeneration"],
                    }
                )
    table = ""
    for r in sorted(
        [r for r in all_metrics if r["subset"] == "all" and r["method"] == "atom"],
        key=lambda r: (r["family"], r["seed"], r["signed_step"]),
    ):
        d = r["detectors"][r["family"]]
        contrasts = [
            next(
                c
                for c in all_comparisons
                if all(c[k] == r[k] for k in ("seed", "family", "subset", "signed_step"))
                and c["control"] == control
            )["differences"][r["family"]]
            for control in METHODS[1:]
        ]
        table += f"<tr><td>{r['seed']}</td><td>{LABELS[r['family']]}</td><td>{r['signed_step']:g}</td><td>{d['baseline']:.0%}</td><td>{d['rate']:.0%}</td><td>{100 * d['change']['mean']:+.0f}</td>"
        for c in contrasts:
            table += f"<td>{100 * c['mean']:+.0f} [{100 * c['ci95'][0]:+.0f}, {100 * c['ci95'][1]:+.0f}]</td>"
        table += f"<td>{r['quality']['truncated']:.0%}</td><td>{r['quality']['possible_degeneration']:.0%}</td></tr>"
    baseline_table = "".join(
        f"<tr><td>{r['seed']}</td><td>{LABELS[r['family']]}</td><td>{r['subset']}</td><td>{r['n']}</td><td>{r['rates'][r['family']]:.0%}</td><td>{r['quality']['truncated']:.0%}</td></tr>"
        for r in all_baselines
        if r["subset"] != "exclude_exact_training_overlap"
    )
    examples = [e for s in data for e in s["examples"]]
    atomic_json(output / "fixed-examples.json", examples)
    example_html = ""
    for e in examples:
        example_html += f"<details><summary>Seed {e['seed']} · {e['id']} · {e['split']}: {html.escape(e['prompt'])}</summary><h4>Baseline</h4><pre>{html.escape(e['baseline']['response'])}</pre>"
        for c in e["conditions"]:
            example_html += f"<h4>{METHOD_LABELS[c['method']]}, s={c['signed_step']:g}</h4><pre>{html.escape(c['response'])}</pre>"
        example_html += "</details>"
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>LoRA steering reproduction</title><style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#1c2833}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:35px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;padding:7px;border-bottom:1px solid #ddd}}img{{width:100%}}.note{{background:#f1f4f6;padding:18px}}a{{color:#21618c}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:12px}}summary{{cursor:pointer;padding:12px}}.scroll{{overflow:auto}}</style></head><body>
<h1>LoRA steering reproduction</h1><p>Both runs completed all 151 conditions and 15,500 generations. Their source, input and output hashes passed verification, every condition restored the adapter, and both finished before their frozen deadlines. This report retains all five behaviors, all three methods, both signs, and every published step size.</p>
<p class="note">This is a development reproduction of <a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a>. It tests generated response patterns after direct atom interventions. It does not yet predict the effect of a training batch, validate answer correctness, or establish a new research result.</p>
<h2>What we learned</h2><p>The atoms change response patterns, but many large shifts come with empty, repetitive or truncated answers. Numbered-list suppression is the clearest candidate for further study. At signed step -1, numbered answers fall from 57% in both baselines to 13% and 12%. The corresponding random atoms leave 46% and 53%, and the Gaussian controls leave 57% and 56%. No answers at that selected-atom step are empty, truncated or flagged for repetition. However, mean length also drops from 51.5 and 53.5 tokens to 18.1 and 17.5. These observations do not isolate formatting from verbosity.</p>
<p>At step +1, bullet detector hits rise from 27% and 37% to 86% and 85%, but 90% and 93% of answers hit the 300-token cap. Missing-input requests rise from 20% and 21% to 52% and 55%, with 14% and 51% truncation. Yes/no enhancement does not reproduce across both seeds. Code generation shows no consistent improvement. These are descriptive readings of the complete curves, not newly designated primary endpoints.</p>
<p>In the fixed computer-setup example, suppressing the numbered atom changes a numbered checklist into a sentence on both seeds, while the random controls retain lists. That illustrates the candidate effect; it is not a correctness evaluation. The next test needs to hold answer content fixed and measure format and length separately.</p><img src="quality.png" alt="Bullet changes coincide with truncation; numbered interventions change answer length">
<h2>What was tested</h2><ol><li>Use two reconstructed Gemma-3 4B adapters, rank-8 LoRA on q/v across 34 language layers.</li><li>Choose one candidate per behavior using only its top 20 training responses. Break purity ties by gradient coherence. Determine orientation from signed training coefficients.</li><li>Apply each candidate in both directions at steps 0.5, 1, 2, 5 and 10. Compare a random atom of equal total norm and a Gaussian direction with equal norm in each LoRA parameter block.</li><li>Generate greedy answers to the 100 published questions per family, 60 inviting and 40 controls. Apply the five published regex detectors to every answer.</li></ol>
<p>The same 500 public questions are reused across the two seeds. One exact training-prompt overlap is flagged. We report an overlap-excluded sensitivity in the downloadable data. Missing-input requests mean phrases such as asking the user to provide a text. They are not general safety refusals. Off-target detector rates are measured only on the current intervention family's questions.</p>
<h2>All results</h2><p>The horizontal axis is signed step s, with weights changed to original minus s times the oriented direction. Zero is the unchanged adapter. Every plotted point is retained; no best step was selected for a confirmatory claim.</p><img src="curves-all.png" alt="All target detector curves for both seeds and all controls">
<details><summary>Questions inviting each behavior</summary><img src="curves-inviting.png" alt="Inviting subset curves"></details><details><summary>Control questions</summary><img src="curves-control.png" alt="Control subset curves"></details>
<h2>Baseline behavior</h2><table><tr><th>Seed</th><th>Behavior</th><th>Subset</th><th>N</th><th>Hit rate</th><th>Truncated</th></tr>{baseline_table}</table>
<h2>Every selected-atom condition</h2><p>Changes and atom-minus-control differences are percentage points. Brackets give pointwise 95% paired prompt bootstrap intervals, 2,000 resamples. They condition on these two seeds and these particular random directions. They are descriptive, unadjusted for the many comparisons, and not evidence of population-level significance. A negative difference can be useful for suppression; both signs are shown.</p><div class="scroll"><table><tr><th>Seed</th><th>Behavior</th><th>Step</th><th>Base</th><th>Atom</th><th>Change</th><th>Minus random atom [CI]</th><th>Minus Gaussian [CI]</th><th>Truncated</th><th>Repetition/empty flag</th></tr>{table}</table></div>
<p>The repetition diagnostic flags empty responses, ten identical consecutive tokens, or a four-token sequence appearing at least ten times. It is a heuristic, not a quality judgment. Truncation means reaching the 300-token limit without EOS or end-of-turn. Both diagnostics and all off-target detector changes are available for every method and subset in the data.</p>
<h2>Fixed examples</h2><p>These are questions 0, 1, 60 and 61 in every family, for both seeds. Their identities and displayed steps 1 and 5 were saved before reading outcomes. Baselines and both random controls are included, regardless of whether an answer looks good.</p>{example_html}
<h2>Limits and next question</h2><p>There are only two seeds and one draw of each control per family. Hardware and training seed differ together. Public prompts are development material, and regex format hits are not correctness or validated general atom semantics. The original checkpoint was unavailable, so training was reconstructed from the paper's architecture and released defaults. Generation uses Transformers with FlashAttention rather than the paper's vLLM backend. LoRA A and B change together, including their bilinear interaction.</p><p>The next research question is whether a batch's LoRA update predicts these behavioral changes prospectively, against direct-gradient and other strong baselines. Steering effects alone do not answer that question.</p>
<h2>Data, code and provenance</h2><p><a href="all-conditions.csv">Every endpoint and condition, CSV</a> · <a href="summary.json">All rates, paired intervals and control comparisons</a> · <a href="fixed-examples.json">Fixed examples</a> · <a href="analysis-plan.json">Analysis plan saved before outcome inspection</a> · <a href="seed42/integrity.json">Seed42 integrity</a> · <a href="seed43/integrity.json">Seed43 integrity</a></p><p><a href="https://arxiv.org/abs/2603.14665">Paper</a> · <a href="https://github.com/jrosseruk/gradient_atoms/blob/a83858557b411249097861f25d57eb13020eb334/experiments/steer_behavioral_atoms.py">Pinned original questions and detectors</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/043443c/src/gradient_interp/paper_steering.py">Frozen generation code</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/main/src/gradient_interp/paper_steering_analysis.py">Analysis code</a></p></body></html>"""
    (output / "report.html").write_text(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-plan", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=Path("runs/upstream-gradient-atoms"))
    args = parser.parse_args()
    # Check both controllers before opening even one condition file.
    if any(read_json(root / "controller.json")["status"] != "complete" for root in args.roots):
        raise ValueError("Both runs must complete before analysis")
    if not args.analysis_plan.exists():
        raise ValueError("Save the descriptive analysis plan before inspecting outcomes")
    _, checkers, _, _ = published_material(args.upstream)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.analysis_plan, args.output / "analysis-plan.json")
    data = []
    for root in args.roots:
        spec, records = load_complete(root, checkers)
        data.append(summarize(root, spec, records))
        out = args.output / f"seed{spec['training_seed']}"
        out.mkdir(exist_ok=True)
        for name in (
            "controller.json",
            "manifest.json",
            "plan.json",
            "specification.json",
            "integrity.json",
            "selection.json",
            "preflight.json",
            "questions.json",
        ):
            shutil.copy2(root / name, out / name)
    atomic_json(
        args.output / "summary.json",
        [{k: v for k, v in s.items() if k != "examples"} for s in data],
    )
    write_report(data, args.output)
    print(
        json.dumps(
            {
                "seeds": [s["seed"] for s in data],
                "conditions": 302,
                "generations": 31000,
                "report": str(args.output / "report.html"),
            }
        )
    )


if __name__ == "__main__":
    main()
