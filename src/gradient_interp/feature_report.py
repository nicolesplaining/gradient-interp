"""Reproduce figures and a readable report from the completed development pilot."""

# ruff: noqa: E501
# Long lines below are literal HTML and source links.

import argparse
import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .study import atomic_json, file_hash

LABELS = {
    "full_gradient": "Full gradient",
    "svd_128": "SVD128",
    "sae_paired": "Paired SAE",
    "sae_independent": "Independent SAE",
    "sae_permuted": "Shuffled SAE",
    "untrained_sae": "Untrained SAE",
    "training_mean": "Training mean",
    "sae_features_only": "SAE without offset",
    "sae_offset_only": "SAE offset only",
    "svd_32": "SVD32",
}


def report(pilot, audit, output):
    pilot, audit, output = Path(pilot), Path(audit), Path(output)
    if any(
        json.loads((p / "controller.json").read_text())["status"] != "complete"
        for p in (pilot, audit)
    ):
        raise ValueError("Report requires completed phases")
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((pilot / "summary.json").read_text())
    controls = json.loads((audit / "controls/summary.json").read_text())
    scale = json.loads((audit / "loracle-scale/summary.json").read_text())
    jobs = list(summary["results"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = [
        "full_gradient",
        "svd_128",
        "sae_paired",
        "sae_independent",
        "untrained_sae",
        "training_mean",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    for ax, job in zip(axes, jobs, strict=True):
        data = controls["results"][job]["rates"]["0.001"]
        y = np.arange(len(methods))
        ax.barh(
            y - 0.17,
            [data[m]["fixed_error"] for m in methods],
            height=0.32,
            color="#21618c",
            label="Same questions",
        )
        ax.barh(
            y + 0.17,
            [data[m]["transfer_error"] for m in methods],
            height=0.32,
            color="#dc7633",
            label="Independent questions",
        )
        ax.set_yticks(y, [LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1)
        ax.set_title(job.replace("-seed606", ""))
        ax.set_xlabel("Relative error, lower is better; zero forecast = 1")
        for i, m in enumerate(methods):
            for shift, key in ((-0.17, "fixed_error"), (0.17, "transfer_error")):
                ax.text(
                    data[m][key] + 0.012, i + shift, f"{data[m][key]:.3f}", va="center", fontsize=8
                )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    fig.suptitle(
        "Small-step development results\nSingle layer; rate 0.001; 16 batches; one seed per model",
        fontsize=13,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"small-step.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), constrained_layout=True)
    colors = ["#17202a", "#21618c", "#b03a2e", "#7d3c98", "#7f8c8d"]
    plotted = ["full_gradient", "svd_128", "sae_paired", "sae_independent", "sae_permuted"]
    for row, job in enumerate(jobs):
        rates = summary["results"][job]["rates"]
        x = [float(r) for r in rates]
        for col, (metric, title) in enumerate(
            (
                ("fixed_category_contrasts", "Same questions"),
                ("transfer_category_contrasts", "Independent questions"),
            )
        ):
            ax = axes[row, col]
            for method, color in zip(plotted, colors, strict=True):
                ax.plot(
                    x,
                    [r[method][metric]["relative_l2_error"] for r in rates.values()],
                    "o-",
                    color=color,
                    label=LABELS[method],
                    markersize=4,
                )
            ax.set_xscale("log")
            ax.set_xticks(x, [str(v) for v in x])
            ax.axhline(1, color="gray", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1.16 if row == 0 else 3.7)
            ax.set_title(f"{job.replace('-seed606', '')}: {title}")
            ax.set_xlabel("Step size")
            ax.set_ylabel("Relative forecast error")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False
    )
    fig.suptitle(
        "The 7B SAE advantage does not persist across step sizes\nDevelopment data; no confidence intervals from a single seed",
        fontsize=13,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"all-rates.{extension}", dpi=180)
    plt.close(fig)
    storage = []
    for job in jobs:
        metadata = json.loads((pilot / "jobs" / job / "forecast_metadata.json").read_text())
        storage.append({"job": job, **metadata["storage"][0]})
    atomic_json(output / "storage.json", storage)
    paths = {
        "pilot-summary.json": pilot / "summary.json",
        "pilot-plan.json": pilot / "plan.json",
        "pilot-freeze.json": pilot / "freeze.json",
        "pilot-integrity.json": pilot / "integrity-check.json",
        "audit-integrity.json": audit / "integrity-check.json",
        "controls-summary.json": audit / "controls/summary.json",
        "loracle-summary.json": audit / "loracle-scale/summary.json",
        "loracle-manifest.json": audit / "loracle-scale/manifest.json",
        "loracle-generations.json": audit / "loracle-scale/generations.json",
        "loracle-hook-results.json": audit / "loracle-scale/hook-results.json",
        "audit-plan.json": audit / "plan.json",
        "audit-controller.json": audit / "controller.json",
    }
    for name, source in paths.items():
        shutil.copyfile(source, output / name)
    rows = []
    for job in jobs:
        for rate, methods in controls["results"][job]["rates"].items():
            for method, metrics in methods.items():
                cv = metrics["scalar_calibration"]
                values = [
                    job.replace("Qwen2.5-", "").replace("-seed606", ""),
                    rate,
                    LABELS[method],
                    f"{metrics['fixed_error']:.4f}",
                    f"{metrics['transfer_error']:.4f}",
                    f"{cv['unscaled_error_same_folds']:.4f}",
                    f"{cv['scaled_error']:.4f}",
                ]
                rows.append(
                    "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>"
                )
    storage_rows = "".join(
        f"<tr><td>{s['job'].replace('-seed606', '')}</td>"
        + "".join(
            f"<td>{s[k] / 2**20:.1f}</td>"
            for k in ("raw_weight_bytes", "sae_context_bytes", "svd32_context_bytes")
        )
        + "</tr>"
        for s in storage
    )
    body = """<!doctype html><html><meta charset="utf-8"><title>Gradient feature pilot</title>
<style>body{font:16px/1.55 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#18232e}h1,h2{line-height:1.2}h2{margin-top:36px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #d7dfe5;padding:8px;text-align:left}.note{background:#f1f5f8;padding:16px;border-left:4px solid #21618c}a{color:#21618c}</style>
<h1>What the new gradient-feature pilot tells us</h1>
<p>Completed 13 October 2026 UTC. This is exploratory development, with one seed for each of two models.</p>
<p class="note">The 7B SAE beats SVD128 at the smallest step, but loses at larger steps and on 3B.
It uses more storage than the full gradient. The evidence does not establish useful compression,
interpretable features, or a reliable behavior-change predictor.</p>
<h2>What we tested</h2>
<ol><li>Captured gradients at one middle-layer MLP output weight in Qwen2.5-3B and 7B.</li>
<li>Trained a TopK residual-gradient SAE on 192 documents, with 64 held out. Used twice the hidden width in features, k=32, and 15 epochs.</li>
<li>Kept each feature's association with the token inputs. Compared independent averages and shuffled associations, full gradients, and SVD32/128.</li>
<li>Locked forecasts before applying 48 updates per model. Measured reference-answer log-score changes across eight task categories.</li>
<li>Compared forecasts on the same questions with transfer from a separate question set. Both sets and the update batches came from earlier development work.</li>
<li>After completion, tested average-gradient, offset-only, feature-only, untrained-SAE, and scalar-calibration explanations using cached data.</li></ol>
<p>Keeping input-feature associations follows the standard linear-layer gradient outer-product identity. That identity is not a research discovery.</p>
<h2>Where the SAE helped</h2>
<img src="small-step.png" alt="Errors for six methods on 3B and 7B at the smallest step">
<p>At rate 0.001, the paired SAE's 7B error was 0.255 on the same questions versus 0.346 for SVD128.
Independent-question error was 0.527 versus 0.580. The full gradient was much better on the same questions, at 0.042.
On 3B the SAE lost to SVD128, 0.467 versus 0.203. Average gradients and untrained features did not explain the 7B advantage.</p>
<p>Removing the learned SAE offset improved 7B same-question error to 0.220, but worsened transfer to 0.605.
This distinction matters: good reconstruction and good transfer are different objectives.</p>
<h2>Where it failed</h2><img src="all-rates.png" alt="All prespecified step sizes and both question conditions">
<p>At rate 0.01, 7B SAE same-question error rose to 0.392, versus 0.177 for SVD128.
At 0.1, first-order full-gradient error exceeded 2.8. Shrinking forecasts can help when the linear approximation breaks down.
The follow-up scalar control fits one nonnegative multiplier on other batches and other evaluation questions.
On 7B at 0.1 it reduced full-gradient transfer error from 2.840 to 0.685 on those cross-validation folds.
Those folds have only four evaluation questions per category, so their numbers must not be compared directly with the eight-question pilot totals.</p>
<p>Held-out raw-gradient reconstruction FVU was 1.111 for 3B and 1.117 for 7B. Values above one mean the SAE did worse than the training-set mean on that reconstruction metric.</p>
<h2>Storage does not support a compression claim</h2>
<table><tr><th>Model</th><th>Full weight update, MiB</th><th>SAE contexts, MiB</th><th>SVD32 contexts, MiB</th></tr>STORAGE</table>
<p>These counts exclude shared dictionaries and model weights. Retaining an input vector for each feature costs about twice as much as storing the full weight update. Scalar feature totals alone discard the input associations.</p>
<h2>A concrete limitation of the public LoRAcle</h2>
<p>The pinned AO checkpoint normalizes each update vector before injecting it. Across 60 cached updates and both token conventions, all 120 hook cases were bit-identical under positive scaling by 0.25, 1, and 4.
Eighteen generations covering three updates and both conventions also stayed identical within each scaling comparison.</p>
<p>This is an implementation audit, not a new forecasting result. It applies to this norm-match, identity-encoder configuration, above the norm-clamp floor and with fixed rank and prompt layout.
A magnitude predictor using this checkpoint would need another magnitude input. It does not say that every possible LoRAcle is scale-invariant.</p>
<h2>All follow-up numbers</h2>
<p>Relative L2 errors, lower is better. Scalar calibration is exploratory cross-validation holding out both connected update batches and alternating evaluation-question halves. No tests or confidence intervals are claimed.</p>
<table><tr><th>Model</th><th>Rate</th><th>Method</th><th>Same questions</th><th>Transfer</th><th>CV unscaled</th><th>CV scaled</th></tr>ROWS</table>
<h2>What follows</h2>
<p>The next fixed development pilot separates JSON formatting from factual correctness on synthetic supplied-record questions. It retains full-gradient and SVD baselines, tests new prompt templates, and records generated answers.
Known labels define the endpoints; they do not establish SAE feature semantics. Confirmation would require an independently frozen study on untouched data.</p>
<h2>Code, prior work, and integrity</h2>
<p><a href="https://github.com/nicolesplaining/gradient-interp/blob/6f03bf6/src/gradient_interp/feature_pilot.py">Frozen pilot code</a>;
<a href="https://github.com/nicolesplaining/gradient-interp/blob/48723f5/src/gradient_interp/feature_controls.py">follow-up controls</a>;
<a href="https://github.com/nicolesplaining/gradient-interp/blob/48723f5/src/gradient_interp/loracle_scale_audit.py">LoRAcle audit</a>.</p>
<p><a href="https://arxiv.org/abs/2603.14665">Gradient Atoms</a> learns sparse gradient directions and evaluates steering.
<a href="https://arxiv.org/abs/2406.04093">Scaling and evaluating sparse autoencoders</a> supplies the TopK SAE baseline.
The <a href="https://drive.google.com/file/d/1xsXOTgHNovlkEHpqsROaiGsr8Qk3jnyd/view">supplied gradient-SAE report</a> motivates gradient features; this small pilot is not a reproduction of its architecture or scale.
<a href="https://aclanthology.org/2023.acl-long.643/">Contrastive Error Attribution</a> is prior work for contrast-based influence, so a paired-gradient baseline is not itself novel.</p>
<p>The <a href="https://github.com/ceselder/loracles/blob/be71fa947a037f90d3d179a16cbd2ebc5d273a66/src/train_loracle/inference.py">actual LoRAcle injection code</a> explains the scaling result.
All completed pilot forecasts, outcomes, frozen source hashes, and checkpoint restoration passed verification. Both follow-up jobs finished before their fixed deadline.
Raw caches remain on the A100. Compact data, integrity records, and figure PDFs are saved beside this report.</p>
</html>"""
    (output / "report.html").write_text(
        body.replace("STORAGE", storage_rows).replace("ROWS", "".join(rows))
    )
    atomic_json(
        output / "provenance.json",
        {
            "pilot": str(pilot.resolve()),
            "audit": str(audit.resolve()),
            "report_source_sha256": file_hash(__file__),
            "input_files_sha256": {name: file_hash(path) for name, path in paths.items()},
            "loracle_all_hook_scalings_bit_identical": scale["all_hook_scalings_bit_identical"],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=Path("runs/feature-pilot"))
    parser.add_argument("--audit", type=Path, default=Path("runs/feature-audit"))
    parser.add_argument("--output", type=Path, default=Path("results/feature-pilot"))
    args = parser.parse_args()
    report(args.pilot, args.audit, args.output)


if __name__ == "__main__":
    main()
