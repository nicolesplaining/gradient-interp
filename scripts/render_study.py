"""Render the completed, read-only study analysis as a standalone HTML report."""
# ruff: noqa: E501

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

LABELS = {
    "dictionary": "Sparse dictionary",
    "dictionary_debiased": "Dictionary, refitted coefficients",
    "svd": "SVD projection",
    "full_gradient": "Full gradient",
    "dictionary_transfer": "Dictionary, separate probes",
    "full_gradient_transfer": "Full gradient, separate probes",
}
COLORS = ["#bb4e24", "#d4a72c", "#25789c", "#333333"]


def title(key):
    size = "0.5B" if "0.5B" in key else "3B"
    return size + (" · initial" if key.endswith("-0") else " · after 64 training steps")


def pct(value):
    return f"{value * 100:.1f}%"


def interval(value):
    return f"{pct(value['relative_l2_error'])} [{pct(value['relative_error_95ci'][0])}, {pct(value['relative_error_95ci'][1])}]"


def table(headers, rows):
    return (
        "<div class='table'><table><thead><tr>"
        + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def render(root):
    a = json.loads((root / "analysis.json").read_text())
    out = root / "report"
    out.mkdir(exist_ok=True)
    keys = sorted(a["methods"], key=lambda k: ("3B" in k, k.endswith("-64")))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        data = base64.b64encode((out / f"{name}.png").read_bytes()).decode()
        return f"<img src='data:image/png;base64,{data}' alt='{name.replace('_', ' ')}'>"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    methods = ["dictionary", "dictionary_debiased", "svd", "full_gradient"]
    for ax, key in zip(axes.flat, keys):
        for method, color, marker in zip(methods, COLORS, ["o", "s", "^", "x"]):
            rows = a["methods"][key]
            rates = sorted(map(float, rows))
            y = np.array([rows[str(rate)][method]["relative_l2_error"] for rate in rates])
            ci = np.array([rows[str(rate)][method]["relative_error_95ci"] for rate in rates])
            ax.plot(rates, y, marker=marker, color=color, label=LABELS[method], linewidth=1.6)
            ax.fill_between(rates, ci[:, 0], ci[:, 1], color=color, alpha=0.09)
        ax.axhline(1, color="#777777", linestyle=":", linewidth=1)
        ax.set(xscale="log", yscale="log", title=title(key), xlabel="SGD learning rate")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Relative prediction error")
    axes[1, 0].set_ylabel("Relative prediction error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Forecasting batch effects on fixed probes", fontsize=16)
    fig.text(
        0.5,
        0.945,
        "Shading: 95% crossed seed/batch bootstrap. Dotted line: zero-change prediction.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    forecast_plot = save(fig, "forecast_error")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["dictionary", "dictionary_transfer", "full_gradient", "full_gradient_transfer"]
    for i, name in enumerate(names):
        y = [a["methods"][k]["0.0001"][name]["relative_l2_error"] for k in keys]
        axes[0].bar(np.arange(4) + (i - 1.5) * 0.19, y, width=0.18, label=LABELS[name])
    axes[0].axhline(1, color="gray", linestyle=":")
    axes[0].set_xticks(range(4), [title(k).replace(" · ", "\n") for k in keys], fontsize=8)
    axes[0].set_title("Transfer to separate probe documents")
    axes[0].set_ylabel("Relative prediction error at learning rate 0.0001")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, frameon=False)
    for i, name in enumerate(("atom_removed", "random_removed", "rescaled_full")):
        values = [
            a["groups"][k]["0.0001"]["controls"][name]["target_to_off_target_ratio"] for k in keys
        ]
        y = np.array([v["value"] for v in values])
        ci = np.array([v["ci95"] for v in values])
        axes[1].errorbar(
            np.arange(4) + (i - 1) * 0.18,
            y,
            yerr=[y - ci[:, 0], ci[:, 1] - y],
            fmt="o",
            capsize=3,
            label=name.replace("_", " "),
        )
    axes[1].set_xticks(range(4), [title(k).replace(" · ", "\n") for k in keys], fontsize=8)
    axes[1].set_title("Does atom removal isolate the target category?")
    axes[1].set_ylabel("Mean absolute target change / mean off-target RMS")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.text(
        0.5,
        -0.025,
        "Learning rate 0.0001. Selectivity error bars: 95% crossed seed/batch bootstrap.",
        ha="center",
        fontsize=9,
    )
    transfer_plot = save(fig, "transfer_and_selectivity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, size in enumerate(("0.5B", "3B")):
        for j, name in enumerate(("gradient", "activation")):
            values = [v["metrics"][name]["raw_fvu"] for k, v in a["sae"].items() if size in k]
            x = i + (j - 0.5) * 0.3
            axes[0].scatter(
                np.full(3, x) + np.array([-0.035, 0, 0.035]),
                values,
                color=COLORS[j * 2],
                s=42,
                label=name if i == 0 else None,
            )
            axes[0].plot([x - 0.09, x + 0.09], [np.mean(values)] * 2, color=COLORS[j * 2])
    axes[0].axhline(1, color="gray", linestyle=":")
    axes[0].set(yscale="log", ylabel="Held-out raw FVU", title="Residual SAE reconstruction")
    axes[0].set_xticks([0, 1], ["0.5B", "3B"])
    axes[0].legend(frameon=False)
    counts = a["loracle"]["correct_counts"]
    axes[1].bar(range(4), list(counts.values()), color=[COLORS[0], COLORS[1], COLORS[2], "#999999"])
    axes[1].set_xticks(range(4), ["Correct\nSVD", "Upstream\nencoding", "Shuffled", "Zero"])
    axes[1].set(
        ylabel="Correct categories / 30 pure batches",
        ylim=(0, 30),
        title="Public LoRAcle checkpoint",
    )
    for i, count in enumerate(counts.values()):
        axes[1].text(i, count + 0.6, str(count), ha="center")
    fig.tight_layout()
    fig.text(
        0.5,
        -0.025,
        "SAE dots: three seeds; bars: seed means. FVU=1 is the training-mean baseline. LoRAcle: one interpreter checkpoint.",
        ha="center",
        fontsize=9,
    )
    secondary_plot = save(fig, "sae_and_loracle")

    body = [
        "<h1>What does a batch update change?</h1>",
        "<p class='subtitle'>Completed A100 study · 12 forecasting runs · 6 SAE comparisons · 60 LoRAcle batches</p>",
        f"<p>The queue finished in <strong>{a['wall_clock_hours']:.2f} hours</strong>, within the 24-hour budget. "
        "All forecast hashes and checkpoint-restoration checks passed. The GPU is idle. "
        "There are 2,160 batch–learning-rate cases, each measured under five interventions.</p>",
        "<p class='lead'>Gradient dictionaries recover recognizable task patterns, but this experiment does not establish "
        "a reliable predictor of how much a batch changes behavior on new documents. "
        "SVD retains more forecasting information in the stable regimes. Atom removal has no clear selectivity advantage "
        "over rescaling the update. The larger fine-tuned model exposes substantial checkpoint and step-size sensitivity.</p>",
        "<h2>1. Batch-effect forecasting</h2>",
        "<p>The target is the change in mean reference-answer log probability for each of ten categories. "
        "It is not generated-answer accuracy. Fixed-probe methods use those evaluation probes' derivatives before any update. "
        "Relative L2 error is the norm of prediction error divided by the norm of observed changes, pooled over seeds, "
        "batches and category coordinates. A prediction of zero change has 100% error. Lower is better.</p>",
        table(
            ["Checkpoint", "Full gradient", "SVD", "Dictionary", "Refitted dictionary"],
            [
                [title(k)]
                + [
                    interval(a["methods"][k]["0.0001"][m])
                    for m in ("full_gradient", "svd", "dictionary", "dictionary_debiased")
                ]
                for k in keys
            ],
        ),
        "<p class='caption'>Learning rate 0.0001. Brackets are 95% crossed seed/batch bootstrap intervals, 1,000 replicates. "
        "Only three model seeds were tested; these intervals should not be read as precise population guarantees.</p>",
        forecast_plot,
        "<p>For 0.5B, the dictionary adds substantial error beyond the SVD projection at both checkpoints. "
        "Refitting coefficients on the chosen sparse support reduces shrinkage error, without matching SVD. "
        "For initial 3B, SVD and the full-gradient baseline are close at the smallest step.</p>",
        "<p>The fine-tuned 3B aggregate is dominated by seed 47. At the smallest step, full-gradient error is "
        "13.4%, 2.0%, and 127.1% for seeds 11, 29, and 47. The Classify probe category accounts for 97.8% "
        "of pooled squared error, and one batch accounts for 73.9%. The dictionary's lower aggregate error here "
        "is consistent with shrinkage damping a large linear overprediction; it is not evidence of a general semantic advantage. "
        "At learning rate 0.01, pooled full-gradient error reaches 1,951%. The derivative approximation has a limited useful range.</p>",
        "<h2>2. Transfer and atom removal</h2>",
        transfer_plot,
        "<p>Separate-probe transfer calibrates category effects on different documents from those used to measure outcomes. "
        "Dictionary transfer error at the smallest step is 75.2%, 95.4%, 118.6%, and 107.9% for the four panels above. "
        "The last two are worse than predicting zero change. Having a recognizable category label does not make "
        "the magnitude of an update transfer reliably across documents.</p>",
        "<p>We remove the atom with the largest absolute predicted contribution to the target category on calibration probes. "
        "Controls subtract an isotropic random component with the same norm, or scale the full update to match "
        "the remaining update's norm. Atom removal produces larger effects than the random control, but also changes "
        "other categories. Its average target-share advantage over rescaling has a 95% interval containing zero "
        "in every model/checkpoint/step-size condition. This does not establish selective removal of a semantic behavior.</p>",
        table(
            [
                "Checkpoint",
                "Seen-family dictionary error",
                "Unseen-family dictionary error",
                "Mixed-batch error",
            ],
            [
                [title(k)]
                + [
                    pct(
                        a["groups"][k]["0.0001"]["family_splits"][f]["dictionary"][
                            "relative_l2_error"
                        ]
                    )
                    for f in ("seen", "unseen_family", "mixed")
                ]
                for k in keys
            ],
        ),
        "<p class='caption'>Coding and Classify are absent from dictionary fitting and development batches, but have "
        "calibration probes. The unseen-family subset contains only six pure batches; 24 pure batches are seen-family, "
        "and 30 are mixed. This is limited family holdout, not arbitrary new-behavior discovery.</p>",
        "<h2>3. What the patterns look like</h2>",
        "<p>In fine-tuned 3B seed 11, atom 3 has 20 Open QA examples among its top 20, including short "
        "questions about a monarch, a national park and a river. Atom 0 has 17 Brainstorm examples among its top 20, "
        "including business names, garden advice and product suggestions. These are plausible task patterns. "
        "They are examples selected by raw-gradient coherence, not a blinded interpretability evaluation. "
        "Selection chose 256 or 512 atoms; this does not estimate the true number of learned concepts.</p>",
        "<h2>4. SAE and LoRAcle comparisons</h2>",
        secondary_plot,
        "<p>The matched TopK SAEs use 2,048 features and k=32 at one middle residual layer, with "
        "a document-level 4,000/1,000 train/validation split and whitening fitted on training documents only. "
        "Gradient SAE raw FVU ranges from 3.20 to 7.76 for 0.5B and 2.98 to 13.54 for 3B. "
        "Every gradient SAE is worse than the training-mean baseline, FVU=1. Activation SAE FVU is 0.63–0.70 "
        "and 0.83–0.97 respectively. Many gradient features are inactive on validation. This configuration failed "
        "to generalize; it does not rule out better normalization or SAE training. No SAE behavior predictor was trained.</p>",
        "<p>The pretrained LoRAcle answers which task category a single batch update came from. "
        "Correct SVD and upstream-compatible encoding each get 13/30 pure batches right, shuffled inputs get "
        "11/30, and zero inputs get 3/30. The paired improvement over shuffling is only two batches. "
        "Both versions strongly favor Generation. This is weak evidence for source-task decoding, and no evidence "
        "for predicting behavioral effect size. It uses Qwen3-14B, rank-16 LoRA across seven module types, "
        "so it is not a matched head-to-head comparison with the smaller forecasting models.</p>",
        "<h2>5. Generated continuations</h2>",
        "<p>At the prespecified largest step, full updates change 229/1,200 saved continuations; atom removal "
        "changes 124/1,200 relative to the full update. Prompts are shared across batches, so these are "
        "descriptive counts, not independent accuracy trials. Outputs are limited to 48 new tokens. "
        "Some differences are wording changes, others change an answer or introduce repetition. They were not scored for correctness.</p>",
    ]
    example = a["generations"]["Qwen2.5-3B-seed11-warm64"]["examples"][1]
    body.append(
        "<details><summary>Example: a Brainstorm update changes a classification continuation</summary>"
    )
    body.append(
        "<p>3B, seed 11, after 64 training steps. Batch test-Brainstorm-16, learning rate 0.01. "
        "This illustrative example was inspected after the run.</p>"
    )
    for field in ("prompt", "baseline", "full", "atom_removed"):
        body.append(
            f"<p><strong>{field.replace('_', ' ').capitalize()}</strong></p><pre>{html.escape(example[field])}</pre>"
        )
    body += [
        "</details><h2>6. Protocol, checks and limits</h2>",
        "<p>Qwen2.5-0.5B and 3B; seeds 11, 29, 47; initial LoRA and checkpoints after 64 AdamW steps; "
        "rank-4 q/v adapters; one stateless SGD update per test batch. Each dictionary fits 5,000 No Robots document gradients "
        "through a rank-128 uncentered SVD with RMS whitening. Lasso-LARS dictionaries have 128, 256 or 512 atoms, "
        "with alpha 0.1 and up to 40 fitting epochs. Size selection uses development derivatives, not final outcomes. "
        "The alpha 0.03 candidate was dropped for runtime before any final-test forecasts or outcomes existed.</p>",
        "<p>There are 60 disjoint-source test batches of size 1, 4 or 16; 70 calibration and 70 evaluation probes "
        "have distinct source documents. All forecasts were saved and hashed before test interventions. "
        "The five measured updates are full, dictionary reconstruction, atom removed, random removed and rescaled full. "
        "Sixteen prespecified atom directions were checked by central finite differences at two radii. "
        "Most checks have less than 0.5% pooled derivative discrepancy; fine-tuned 3B seeds 11 and 47 "
        "have larger discrepancies, reducing from 8.1% to 4.9% and 30.7% to 20.3% when the radius is halved. "
        "This supports caution about local linearity, without identifying the sole cause of forecast errors.</p>",
        "<p>Limits: one dataset; short answer prefixes; ten coarse categories with as few as two probes; "
        "only three seeds; one-step LoRA SGD; no multi-step Adam forecast; no blinded semantic evaluation; "
        "no trained meta-model. The strongest predictor uses evaluation-probe gradients, so it is a known-probe "
        "first-order baseline rather than an open-ended interpretation of what the batch teaches.</p>",
        "<h2>7. What this adds and what to try next</h2>",
        "<p>The extension beyond Gradient Atoms is a locked batch-level forecasting benchmark with magnitude, "
        "held-out probe transfer and intervention controls. It uses SVD in place of the paper's EKFAC space, "
        "different models and a public dataset, so these results do not refute the paper's EKFAC steering results. "
        "The next useful tests are a step-size or curvature check before forecasting, a matched EKFAC projection, "
        "and a behavior-aware reconstruction objective evaluated on genuinely new probes. "
        "For LoRAcle, stronger input controls and an effect-size target are prerequisites for using it as a predictor. "
        "For gradient SAEs, fix held-out reconstruction before treating features as explanations.</p>",
        "<h2>Sources and reproducibility</h2><ul>",
        "<li><a href='https://arxiv.org/abs/2603.14665'>Gradient Atoms paper</a> and "
        "<a href='https://github.com/jrosseruk/gradient_atoms/tree/a83858557b411249097861f25d57eb13020eb334'>pinned implementation</a>: normalized sparse dictionary estimator and coherence scoring.</li>",
        "<li><a href='https://arxiv.org/abs/2406.04093'>Scaling and evaluating sparse autoencoders</a> and "
        "<a href='https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942'>TopK SAE reference</a>: architecture reference; our training recipe is simplified.</li>",
        "<li><a href='https://github.com/ceselder/loracles/tree/be71fa947a037f90d3d179a16cbd2ebc5d273a66'>Pinned LoRAcles code</a> and "
        "<a href='https://huggingface.co/ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120'>public interpreter checkpoint</a>.</li>",
        "<li><a href='https://huggingface.co/datasets/HuggingFaceH4/no_robots'>No Robots</a>, Rajani et al., 2023, CC-BY-NC-4.0. "
        "Dataset revision e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b. Prompts preserved; completions capped at 128 tokens.</li>",
        "<li><a href='https://github.com/nicolesplaining/gradient-interp/tree/1e6c877fd17ea55973de495cf1dba89cfff9a6c5'>Experiment code used for the run</a>. "
        "The analysis module and this renderer reproduce the tables and plots from saved outputs.</li></ul>",
        "<p>Machine-readable results: <a href='../analysis.json'>analysis.json</a>, "
        "<a href='../aggregate.json'>aggregate.json</a>, <a href='../plan.json'>frozen plan</a>, "
        "<a href='../integrity-check.json'>integrity checks</a>. Raw gradients and model artifacts remain on the A100. "
        "Plot PNGs and vector PDFs sit alongside this standalone report.</p>",
    ]
    style = """
    body{font:16px/1.65 system-ui,sans-serif;color:#182f38;background:#fafbf9;max-width:1120px;margin:40px auto;padding:0 30px}
    h1{font-size:40px;line-height:1.15;letter-spacing:-1px;margin-bottom:10px}h2{font-size:24px;margin-top:44px}
    .subtitle,.caption{color:#586970}.caption{font-size:13px}.lead{font-size:20px;border-left:4px solid #bb4e24;padding:15px 22px;background:white}
    img{width:100%;height:auto;margin:18px 0;background:white}a{color:#176e91}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
    td,th{padding:12px 10px;text-align:left;border-bottom:1px solid #dce3e4}th{background:#eaf0f0}pre{white-space:pre-wrap;background:#edf2f2;padding:16px}
    details{background:white;padding:20px}summary{cursor:pointer}li{margin-bottom:12px}@media print{body{margin:10px;font-size:11px}h2{break-after:avoid}img,table{break-inside:avoid}}
    """
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    document += (
        "<title>Gradient update study results</title><style>"
        + style
        + "</style><body>"
        + "\n".join(body)
        + "</body></html>"
    )
    (out / "results.html").write_text(document)
    print(out / "results.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
