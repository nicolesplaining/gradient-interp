"""Descriptive integrity-checked reports for completed paper-setup reconstructions."""

# ruff: noqa: E501
# Report prose and explicit source links are kept as readable HTML literals.

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from .paper_setup import read_docs, read_json
from .paper_steering import fingerprint
from .study import atomic_archive, atomic_json, file_hash


def extract(root):
    if read_json(root / "controller.json")["status"] != "complete":
        raise ValueError("Discovery is still active")
    audit = read_json(root / "integrity.json")
    if not audit["complete"]:
        raise ValueError("Integrity audit required")
    torch.set_num_threads(8)
    docs = read_docs(root / "training.jsonl")
    payload = torch.load(root / "atoms.pt", map_location="cpu", weights_only=True)
    raw = torch.load(root / "gradients_all.pt", map_location="cpu", weights_only=True, mmap=True)[
        "gradients"
    ]
    info = read_json(root / "atom_characterisations.json")
    ids = [fingerprint(d) for d in docs]
    prompt_ids = [d["messages"][0]["content"] for d in docs]
    enriched = []
    for row in info:
        seen, keep = set(), []
        for i in row["top_doc_indices"]:
            if ids[i] not in seen:
                seen.add(ids[i])
                keep.append(i)
        if len(keep) >= 2:
            g = raw[keep]
            g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cosine = g @ g.T
            mask = ~torch.eye(len(g), dtype=torch.bool)
            coherence = cosine[mask].mean().item()
        else:
            coherence = None
        enriched.append(
            {**row, "unique_top_documents": len(keep), "top20_deduplicated_coherence": coherence}
        )
    coefficients = payload["coefficients"].double()
    dictionary = payload["dictionary"].double()
    projected = payload["projected_gradients"].double()
    normalized = projected / projected.norm(dim=1, keepdim=True).clamp_min(1e-8)
    residual = coefficients @ dictionary - normalized
    error = residual.square().sum().item() / normalized.square().sum().item()
    config = read_json(root / "configuration.json")
    summary = {
        "seed": config["seed"],
        "scope": "in_sample_development_reconstruction",
        "n_documents": len(docs),
        "unique_documents": len(set(ids)),
        "unique_exact_prompts": len(set(prompt_ids)),
        "n_atoms": len(info),
        "coherence_gt_05": sum(row["coherence"] > 0.5 for row in info),
        "coherence_gt_01": sum(row["coherence"] > 0.1 for row in info),
        "deduplicated_coherence_gt_05": sum(
            row["top20_deduplicated_coherence"] is not None
            and row["top20_deduplicated_coherence"] > 0.5
            for row in enriched
        ),
        "deduplication_note": "Remove exact document duplicates from each original top20 without refilling; no change to fitted model/dictionary",
        "normalized_training_squared_relative_error": error,
        "active_coefficients_mean": (coefficients.abs() > 1e-6).sum(1).double().mean().item(),
        "active_coefficients_median": (coefficients.abs() > 1e-6).sum(1).double().median().item(),
        "runtime_minutes": audit["controller_runtime_seconds"] / 60,
        "gradient_shape": audit["gradient_shape"],
        "dictionary_shape": audit["dictionary_shape"],
        "atom_file_sha256": file_hash(root / "atoms.pt"),
        "atoms": enriched,
    }
    atomic_json(root / "discovery-summary.json", summary)
    # Only ~10 MB uncompressed; raw gradients and eigenvectors stay on the host.
    atomic_archive(
        root / "activation-patterns.npz",
        coefficients=coefficients.float().numpy(),
        document_ids=np.array(ids),
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "atoms"}, indent=2))


def aggregate_patterns(path):
    with np.load(path) as data:
        ids, inverse, counts = np.unique(
            data["document_ids"], return_inverse=True, return_counts=True
        )
        result = np.zeros((len(ids), data["coefficients"].shape[1]), dtype=np.float64)
        np.add.at(result, inverse, data["coefficients"])
        result /= counts[:, None]
    return ids, result


def report(first, second, output):
    import matplotlib.pyplot as plt
    from scipy.optimize import linear_sum_assignment

    output.mkdir(parents=True, exist_ok=True)
    summaries = [read_json(p / "discovery-summary.json") for p in (first, second)]
    ids_a, a = aggregate_patterns(first / "activation-patterns.npz")
    ids_b, b = aggregate_patterns(second / "activation-patterns.npz")
    if not np.array_equal(ids_a, ids_b):
        raise ValueError("Replications do not use the same unique documents")
    a -= a.mean(0)
    b -= b.mean(0)
    a /= np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=0), 1e-12)
    correlations = np.clip(a.T @ b, -1, 1)
    left, right = linear_sum_assignment(-np.abs(correlations))
    matches = [
        {
            "seed42_atom": int(i),
            "seed43_atom": int(j),
            "activation_correlation": float(correlations[i, j]),
            "seed42_coherence": summaries[0]["atoms"][i]["coherence"],
            "seed43_coherence": summaries[1]["atoms"][j]["coherence"],
        }
        for i, j in zip(left, right, strict=True)
    ]
    atomic_json(
        output / "cross-seed-matches.json",
        {
            "scope": "descriptive in-sample matching",
            "unique_documents": len(ids_a),
            "method": "One-to-one Hungarian matching on absolute centered activation correlation; exact duplicates averaged; arbitrary signs allowed",
            "median_absolute_correlation": float(np.median(np.abs(correlations[left, right]))),
            "matches": matches,
        },
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for summary, color in zip(summaries, ["#21618c", "#bd642d"], strict=True):
        values = np.sort([row["coherence"] for row in summary["atoms"]])[::-1]
        axes[0].plot(
            np.arange(1, 501),
            values,
            color=color,
            label=f"Seed {summary['seed']}: {summary['coherence_gt_05']} above 0.5",
        )
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Atom rank by coherence",
        ylabel="Raw-gradient coherence",
        title="Coherence in both reconstructions",
    )
    axes[0].legend(frameon=False, fontsize=9)
    values = np.sort(np.abs(correlations[left, right]))
    axes[1].plot(values, np.arange(1, 501) / 500, color="#283747")
    axes[1].set(
        xlabel="Absolute activation correlation",
        ylabel="Fraction of matched atoms",
        xlim=(0, 1),
        ylim=(0, 1),
        title="Matching on the same fitting documents",
    )
    fig.suptitle(
        "Gemma-3 4B IT, rank-8 LoRA across all 34 layers\nDescriptive fitting-data evidence; behavior tests are still required",
        fontsize=12,
    )
    for ext in ("png", "pdf"):
        fig.savefig(output / f"discovery.{ext}", dpi=180)
    plt.close(fig)
    table = "".join(
        f"<tr><td>{s['seed']}</td><td>{s['coherence_gt_05']}</td><td>{s['coherence_gt_01']}</td><td>{s['deduplicated_coherence_gt_05']}</td><td>{s['active_coefficients_mean']:.1f}</td><td>{s['normalized_training_squared_relative_error']:.3f}</td><td>{s['runtime_minutes']:.1f}</td></tr>"
        for s in summaries
    )
    top = ""
    for s in summaries:
        rows = sorted(s["atoms"], key=lambda row: -row["coherence"])[:10]
        top += f"<h2>Seed {s['seed']}: top atoms</h2><table><tr><th>Atom</th><th>Coherence</th><th>Unique top documents</th><th>Frequent response words, not validated labels</th></tr>"
        for row in rows:
            top += f"<tr><td>{row['atom_idx']}</td><td>{row['coherence']:.3f}</td><td>{row['unique_top_documents']}/20</td><td>{html.escape(', '.join(row['keywords'][:8]))}</td></tr>"
        top += "</table>"
    body = f"""<!doctype html><html><head><meta charset="utf-8"><title>Gradient Atoms setup reconstruction</title>
<style>body{{max-width:1050px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#1c2833}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}img{{width:100%}}.note{{background:#f3f5f6;padding:18px}}a{{color:#21618c}}</style></head><body>
<h1>Both LoRA reconstructions completed</h1><p>The two runs found coherent gradient clusters. These are measurements on the same 5,000 fitting documents. They do not yet show that an atom changes the intended behavior, generalizes to new examples, or predicts how much a batch will change behavior.</p>
<p class="note">The original rank-8 checkpoint was not publicly located. We reconstructed the paper's architecture and analysis using its released code and remaining training defaults. This is not an exact checkpoint replication. Seed and GPU hardware both differ between our two runs.</p>
<h2>What was run</h2><ol><li>Train Gemma-3 4B IT with rank-8 LoRA on q_proj and v_proj in all 34 language layers: 136 A/B tensors and 2,228,224 parameters.</li><li>Fit EKFAC on all 5,000 released training documents, then extract their summed completion-only gradients.</li><li>Use the released projection: 50 components per tensor, 6,800 coordinates total. Normalize every projected document gradient.</li><li>Fit 500 dictionary atoms with sparsity penalty 0.1. Measure pairwise raw-gradient similarity among each atom's top 20 activating documents.</li></ol>
<p>Every stage completed before its frozen deadline. File hashes, source hashes at completion, gradient and dictionary shapes, and unchanged adapter weights during gradient extraction were verified. Heavy outputs remain on the originating machines.</p>
<img src="discovery.png" alt="Atom coherence ranks and cross-seed activation correspondence">
<h2>Results</h2><table><tr><th>Seed</th><th>Coherence &gt;0.5</th><th>&gt;0.1</th><th>&gt;0.5 after exact dedup</th><th>Mean active atoms/doc</th><th>Squared reconstruction error</th><th>Minutes</th></tr>{table}</table>
<p>There are {summaries[0]["unique_documents"]:,} unique complete documents and {summaries[0]["unique_exact_prompts"]:,} unique exact prompts. The deduplication check removes repeated complete documents from each original top-20 set without refilling it. Reconstruction error is measured on normalized fitting gradients; it is not prediction error.</p>
<p>For context, the paper reported 5 atoms above coherence 0.5 and 43 above 0.1. Larger counts in these reconstructions are not evidence of improvement because the original trained checkpoint and its full training metadata were unavailable. <a href="https://arxiv.org/html/2603.14665v2#S4.SS2">Paper results</a>.</p>
<p>The median absolute activation correlation under one-to-one matching is {np.median(np.abs(correlations[left, right])):.3f}. This uses the same unique fitting documents in both runs and allows arbitrary atom signs. Matching optimizes this statistic, so the correlations are optimistically selected. They are a descriptive check, not held-out semantic validation or uncertainty over a population of seeds.</p>
{top}<h2>What remains</h2><p>Test the five published behavior families on generated answers, using newly selected atom IDs, both signs, every published step size, and random controls matched in size. Candidate selection uses training responses only and is frozen before generation. Then test prospective batch forecasts on separate data. No behavior-steering or batch-forecast result is claimed in this report.</p>
<h2>Code and provenance</h2><p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms paper</a> · <a href="https://github.com/jrosseruk/gradient_atoms/tree/a83858557b411249097861f25d57eb13020eb334">Pinned author code</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/5553bc7/src/gradient_interp/paper_setup.py">Our reconstruction code</a> · <a href="https://github.com/nicolesplaining/gradient-interp/blob/5553bc7/src/gradient_interp/paper_steering.py">Steering protocol</a></p>
<p>Seed 42: <a href="seed42/configuration.json">configuration</a>, <a href="seed42/plan.json">frozen plan</a>, <a href="seed42/integrity.json">integrity audit</a>, <a href="seed42/discovery-summary.json">all atom statistics</a>. Seed 43: <a href="seed43/configuration.json">configuration</a>, <a href="seed43/plan.json">frozen plan</a>, <a href="seed43/integrity.json">integrity audit</a>, <a href="seed43/discovery-summary.json">all atom statistics</a>. <a href="cross-seed-matches.json">All cross-seed matches</a>.</p>
<p>The released training config defaults to rank32 and seven projection types; the paper specifies rank8 and q/v only. We used the paper's rank/modules. Alpha64, dropout0.1, learning rate0.0003, three epochs, length500, effective batch64, warmup5 and packed SFT follow the remaining code defaults. The released projector uses Kronecker eigenvalue products rather than the corrected EKFAC lambda matrix. These choices and model/data revisions are saved in the per-seed configuration files.</p>
</body></html>"""
    (output / "report.html").write_text(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("extract", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.root)
    else:
        report(args.first, args.second, args.output)


if __name__ == "__main__":
    main()
