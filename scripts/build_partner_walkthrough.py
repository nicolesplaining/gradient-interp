"""Explain completed studies using saved results only; never run a model."""
# ruff: noqa: E501

import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/partner-walkthrough"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [
    "results/paper-svd-magnitude/summary.json",
    "results/paper-svd-magnitude/interpretation.json",
    "results/paper-svd/interpretation.json",
    "results/paper-svd/fixed-examples.json",
    "results/paper-magnitude/summary.json",
    "runs/paper-svd-magnitude-data/data.json",
    "runs/paper-svd-analysis/compact-seed42/data.json",
    "runs/paper-svd-analysis/compact-seed42/selection.json",
    "results/paper-svd-magnitude/examples.csv",
]


def read(path):
    return json.loads((ROOT / path).read_text())


def esc(value):
    return html.escape(str(value))


def pre(value):
    rendered = esc(value).replace(" \n", "&#32;\n")
    return f"<pre>{rendered}</pre>"


def table(headers, rows):
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(name, caption):
    return f'<figure><img src="{name}.png" alt="{esc(caption)}"><figcaption>{caption} <a href="{name}.pdf">PDF</a></figcaption></figure>'


forecast = read(SOURCES[0])
forecast_interpretation = read(SOURCES[1])
steering = read(SOURCES[2])
raw_examples = read(SOURCES[3])
normalization = read(SOURCES[4])
data = read(SOURCES[5])
steering_data = read(SOURCES[6])
selection = read(SOURCES[7])
with (ROOT / SOURCES[8]).open() as stream:
    forecast_examples = list(csv.DictReader(stream))

colors = {
    "dictionary": "#b9652d",
    "svd500": "#147d82",
    "full_gradient": "#6576ae",
    "actual": "#28374a",
}
labels = {"dictionary": "Atoms", "svd500": "SVD", "full_gradient": "Full gradient"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
)


def metric(rows, seed, method):
    return next(
        r
        for r in rows
        if r["seed"] == seed
        and r["method"] == method
        and r["scope"] == "evaluation"
        and r["rate"] == 0.003
    )


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# Replot existing point estimates and existing intervals. No resampling or refitting.
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(2)
for j, method in enumerate(["dictionary", "svd500", "full_gradient"]):
    rows = [metric(forecast["metrics"], seed, method) for seed in [42, 43]]
    values = np.array([r["nrmse_vs_zero"] for r in rows])
    intervals = np.array([r["pointwise"]["nrmse"]["interval"] for r in rows])
    bars = ax.bar(
        x + (j - 1) * 0.23,
        values,
        0.21,
        color=colors[method],
        label=labels[method],
        yerr=np.vstack([values - intervals[:, 0], intervals[:, 1] - values]),
        capsize=4,
    )
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=25, fontsize=11)
ax.axhline(1, color="#777777", linestyle="--", linewidth=1)
ax.text(1.43, 1.015, 'Predicting "no change" = 1', ha="right", va="bottom", fontsize=10)
ax.set(
    xticks=x,
    xticklabels=["Run A · seed 42", "Run B · seed 43"],
    ylim=(0, 1.15),
    ylabel="Prediction error relative to predicting no change",
    title="SVD had lower loss-change prediction error",
)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
ax.yaxis.grid(True, alpha=0.15)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.2)
save(fig, "forecast-error")

fig, ax = plt.subplots(figsize=(9, 4.6))
for j, method in enumerate(["dictionary", "svd500", "full_gradient", "actual"]):
    rows = [
        metric(forecast["metrics"], seed, "svd500" if method == "actual" else method)
        for seed in [42, 43]
    ]
    key = "mean_actual_change" if method == "actual" else "mean_predicted_change"
    vals = [-r[key] for r in rows]
    bars = ax.bar(
        x + (j - 1.5) * 0.19,
        vals,
        0.175,
        color=colors[method],
        label="Actual measured change" if method == "actual" else labels[method],
    )
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=5, fontsize=10)
ax.set(
    xticks=x,
    xticklabels=["Run A · seed 42", "Run B · seed 43"],
    ylim=(0, 0.048),
    ylabel="Mean loss decrease · nats per answer token",
    title="Both compressed methods underestimated the average effect",
)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
ax.yaxis.grid(True, alpha=0.15)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.24)
save(fig, "effect-size")

fig, ax = plt.subplots(figsize=(9, 4.5))
for j, method in enumerate(["dictionary", "svd500"]):
    rows = [
        next(r for r in steering["primary"] if r["seed"] == seed and r["method"] == method)
        for seed in [42, 43]
    ]
    vals = np.array([r["signed_response"] * 100 for r in rows])
    lo = np.array([r["lo"] * 100 for r in rows])
    hi = np.array([r["hi"] * 100 for r in rows])
    bars = ax.bar(
        x + (j - 0.5) * 0.3,
        vals,
        0.28,
        color=colors[method],
        label=labels[method],
        yerr=np.vstack([vals - lo, hi - vals]),
        capsize=4,
    )
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=16)
ax.set(
    xticks=x,
    xticklabels=["Run A · seed 42", "Run B · seed 43"],
    ylim=(0, 26),
    ylabel="Average signed target response · percentage points",
    title="Atoms produced stronger steering on the fixed test",
)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
ax.yaxis.grid(True, alpha=0.15)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.2)
save(fig, "steering")

norm_metrics = normalization["metrics"]
fig, ax = plt.subplots(figsize=(9, 4.4))
for j, method in enumerate(["normalized_fit", "magnitude_fit"]):
    vals = [metric(norm_metrics, seed, method)["nrmse_vs_zero"] for seed in [42, 43]]
    bars = ax.bar(
        x + (j - 0.5) * 0.3,
        vals,
        0.28,
        color=["#b9652d", "#d99e6a"][j],
        label=["Learn directions on unit gradients", "Keep relative magnitudes while learning"][j],
    )
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=6)
ax.set(
    xticks=x,
    xticklabels=["Run A · seed 42", "Run B · seed 43"],
    ylim=(0, 1.04),
    ylabel="Prediction error relative to predicting no change",
    title="Keeping magnitudes during atom learning did not help",
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
ax.yaxis.grid(True, alpha=0.15)
ax.set_axisbelow(True)
fig.subplots_adjust(bottom=0.24)
save(fig, "normalization")

pair = [
    r
    for r in forecast_examples
    if r["seed"] == "42" and r["batch"] == "0" and r["probe_index"] == "16" and r["rate"] == "0.003"
]
assert len(pair) == 4
assert len({r["baseline_ce"] for r in pair}) == len({r["actual_ce"] for r in pair}) == 1
sample_rows = [
    [
        labels.get(r["method"], "No-change reference"),
        f"{float(r['predicted_ce']):.5f}",
        f"{float(r['predicted_change']):+.5f}",
    ]
    for r in pair
]
sample_rows.append(
    [
        "Actual after one training step",
        f"{float(pair[0]['actual_ce']):.5f}",
        f"{float(pair[0]['actual_change']):+.5f}",
    ]
)

generated = [
    r
    for r in raw_examples
    if r["seed"] == 42
    and r["row"]["id"] == "published-numbered-000"
    and (
        r["condition"]["id"] == "baseline"
        or (r["condition"]["family"] == "numbered" and r["condition"]["strength"] == 1)
    )
]
assert len(generated) == 5
generated_html = ""
for r in generated:
    c = r["condition"]
    label = (
        "Original model"
        if c["id"] == "baseline"
        else f"{labels[c['method']]} · signed strength {c['sign']:+d}"
    )
    generated_html += f'<article class="response"><h4>{esc(label)}</h4>{pre(r["row"]["response"])}<p class="small">{r["row"]["checks"]["tokens"]} generated token IDs, including stop tokens.</p></article>'

descriptor = [
    r for r in steering_data["descriptor_pairs"] if r["pair_id"] == "descriptor-numbered-00-short"
]
batch_table = table(
    ["Example in the first eight-document batch", "Prompt"],
    [[i + 1, r["prompt"]] for i, r in enumerate(data["batches"][0]["documents"])],
)
batch_details = "".join(
    f"<details><summary>Example {i + 1}: {esc(r['id'])}</summary><p>Prompt</p>{pre(r['prompt'])}<p>Dataset reference answer</p>{pre(r['answer'])}</details>"
    for i, r in enumerate(data["batches"][0]["documents"])
)
prompt_table = table(
    ["Behavior we tested", "Actual published test prompt"],
    [
        [
            name,
            next(
                r["prompt"] for r in steering_data["published_questions"] if r["family"] == family
            ),
        ]
        for family, name in [
            ("yesno", "Starts with yes/no"),
            ("code", "Writes code"),
            ("refusal", "Asks for missing input"),
            ("bullets", "Uses bullets"),
            ("numbered", "Uses numbering"),
        ]
    ],
)
interpretability_table = table(
    ["Check", "Atoms A / B", "SVD A / B"],
    [
        ["Held-out description preference", "0 / 1 of 5", "2 / 2 of 5"],
        ["Target changes more than every off-target detector", "3 / 2 of 5", "1 / 0 of 5"],
        ["Joint quality requirement across the full strength grid", "0 / 0 of 5", "0 / 0 of 5"],
    ],
)
train_example = selection["dictionary"]["yesno"]["training_examples"][0]["messages"]
train_example_html = pre(
    "User: " + train_example[0]["content"] + "\nAssistant reference: " + train_example[1]["content"]
)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What we tested with Gradient Atoms and SVD</title>
<style>
:root{{--ink:#243447;--muted:#556579;--line:#dbe3e8;--teal:#147d82;--orange:#b9652d}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:var(--ink);font:17px/1.65 system-ui,-apple-system,sans-serif}}
main{{max-width:1040px;margin:auto;background:white;padding:48px 52px 70px}}h1{{font-size:40px;line-height:1.13;letter-spacing:-1.2px;margin:8px 0 22px}}
h2{{font-size:27px;line-height:1.25;margin:40px 0 16px}}h3{{font-size:21px;margin:26px 0 10px}}h4{{margin:0 0 12px;font-size:17px}}
p{{margin:12px 0}}a{{color:#126c75}}.eyebrow{{font-size:13px;letter-spacing:1px;color:var(--muted);text-transform:uppercase}}
.lead{{font-size:21px;line-height:1.5}}.note{{padding:16px 20px;background:#f0f6f6;border-left:4px solid var(--teal);margin:20px 0}}
.small,figcaption{{font-size:14px;color:var(--muted)}}pre{{font:14px/1.6 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:15px;border-radius:6px;margin:8px 0}}
table{{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}}td,th{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#edf2f5;font-weight:650}}.table-wrap{{overflow-x:auto}}
figure{{margin:26px 0}}img{{display:block;max-width:100%;height:auto}}figcaption{{margin-top:10px;line-height:1.5}}
.responses{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}.response{{padding:18px;border:1px solid var(--line);border-radius:8px}}.response:first-child{{grid-column:1/-1}}
details{{padding:10px 0;border-bottom:1px solid var(--line)}}summary{{cursor:pointer;font-weight:600}}
li{{margin:9px 0}}nav{{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;margin:24px 0}}.formula{{font-size:19px;padding:18px;background:#f2f7f7;text-align:center}}
footer{{border-top:1px solid var(--line);padding-top:24px;margin-top:40px}}
@media(max-width:700px){{main{{padding:26px 20px}}h1{{font-size:31px}}.responses{{grid-template-columns:1fr}}.lead{{font-size:19px}}}}
@media print{{body{{background:white}}main{{padding:0;max-width:none}}details{{display:block}}nav{{display:none}}figure,.response{{break-inside:avoid}}}}
</style></head><body><main>
<div class="eyebrow">Partner walkthrough · completed experiments · October 14, 2026</div>
<h1>What we tested with Gradient Atoms and SVD</h1>
<p class="lead">We asked whether SVD could replace sparse dictionary learning in Gradient Atoms, and whether either representation could predict how much a training step changes the model.</p>
<p>The clearest positive result is numerical. On new examples, SVD reduced loss-change prediction error by about 32% relative to atoms in both runs. Atoms produced stronger steering. Neither method passed our combined checks for a clean, interpretable explanation of behavior.</p>
<nav><a href="#setup">Shared setup</a><a href="#steering">Steering and interpretation</a><a href="#forecast">Predicting effect size</a><a href="#normalization">Normalization</a><a href="#claims">What to say</a></nav>

<h2 id="setup">We started from the same model and gradients</h2>
<p>We trained two Gemma-3 4B IT adapters using the paper's rank-8 query/value LoRA architecture and the same 5,000 training documents. An adapter is the small set of model weights we let training change. Here that was 2,228,224 weights across all 34 language layers. Runs A and B use training seeds 42 and 43.</p>
<p>For each training prompt and its supplied answer, we computed a gradient: a list telling us how changing each trainable weight would change the answer's loss. A training step moves against that gradient. We used the released EKFAC projection to reduce each list to 6,800 numbers, then normalized its length before learning the representations.</p>
{
    table(
        ["Stage", "What goes in", "What comes out"],
        [
            [
                "Gradient extraction",
                "5,000 prompt + answer documents",
                "5,000 gradient vectors, each with 2,228,224 coordinates",
            ],
            [
                "Paper's projection and normalization",
                "Those full gradients",
                "A 5,000 × 6,800 matrix of unit-length rows",
            ],
            [
                "Sparse dictionary learning",
                "That same matrix",
                "500 atoms; each row uses a small subset",
            ],
            [
                "Uncentered truncated SVD",
                "That same matrix",
                "500 perpendicular components; rows generally use all 500",
            ],
        ],
    )
}
<p>Both methods therefore get the same information. SVD is applied directly to the projected training gradients. We did not apply SVD on top of the learned atoms. The components are patterns in parameter changes, not words or neurons with meanings already attached.</p>
<p>One actual training document associated with a selected yes/no atom was:</p>
{train_example_html}
<p>To propose a component's meaning, we inspected its 20 strongest training examples through fixed behavior detectors. For both methods, we selected one candidate for each of five behaviors using the same ranking rule. We fixed the candidates and signs before looking at the test results.</p>
<div class="note">This recreates the paper's architecture and method. The original rank-8 checkpoint was unavailable, so it is not an exact replication of the paper's published numbers. Both methods are compared within our own shared setup. Two reused adapters provide development evidence, not a broad confirmation.</div>

<h2 id="steering">Test 1: Do the components mean something, and can they control it?</h2>
<p>For steering, we temporarily changed the adapter weights along one selected atom or SVD component. We then gave the edited model ordinary prompts and saved the text it generated. We restored the original weights after each condition.</p>
{prompt_table}
<p>Each behavior had 100 published prompts, including 60 inviting that behavior and 40 controls. Each selected direction also faced 40 controlled prompts. We tried both signs at all five fixed strengths: 0.5, 1, 2, 5 and 10. The selected SVD direction was scaled to match the corresponding atom's size in parameter space before testing.</p>
<p>That produced 14,540 generated responses per run, or 29,080 across both. We recorded formatting, answer length, truncation, empty answers, possible degeneration and narrow content checks.</p>

<h3>Actual input and generated output</h3>
<p>The first published numbering prompt was <strong>"How do I set up a new computer?"</strong> Below are the saved Run A responses at signed strength +1 and -1. These illustrate the procedure, not its average success. Strength 1 is shown for readability; the graphs include every fixed strength.</p>
<div class="responses">{generated_html}</div>
<p>In this example, suppressing the atom removed numbering. Suppressing the SVD component did not. Both methods also changed response length. A result on one prompt cannot establish selectivity, which is why we measured all prompts and the other behaviors too.</p>

<h3>We tested the proposed descriptions separately</h3>
<p>For a proposed numbering component, we used the same content with different formatting. This was an answer-scoring test, not text generated by the model:</p>
<p>Prompt</p>{pre(descriptor[0]["prompt"])}
<div class="responses"><article class="response"><h4>Target answer</h4>{
    pre(descriptor[0]["answer"])
}</article><article class="response"><h4>Control answer</h4>{
    pre(descriptor[1]["answer"])
}</article></div>
<p>We checked whether the component was more associated with the numbered answer, both through its coefficient and through its alignment with the full gradient. We repeated this on eight items per behavior, in short and long versions. That gave 160 descriptor gradients per run. The content was matched; token lengths were not exactly equal.</p>
<p>A separate selective-steering check asked whether the target behavior changed more than every off-target behavior. Quality checks asked whether the edits damaged content, caused truncation or produced degenerate answers. A component's label had to survive these checks; an appealing set of training examples was not enough.</p>
{
    figure(
        "steering",
        "Higher means a larger target response in the selected direction. The score is half the positive-minus-negative target-rate difference, averaged over all five strengths and all five behaviors. It is not a simple increase from baseline. Bars show point estimates; error bars are saved pointwise 95% paired bootstrap intervals.",
    )
}
{interpretability_table}
<p>Atoms steered more strongly and passed more selective-control checks. SVD passed more of the narrow description checks. No family in either method passed the joint quality requirement across the full strength grid, so we cannot claim that either method gave consistently clean interpretable control. Failure across the full grid does not mean every individual strength failed.</p>
<p class="small">The code calls missing-input clarification "refusal"; this is not a safety-refusal experiment. Formatting detectors are surface checks. Some seed-42 interventions ran on H100 while its shared baseline ran on A100; paired methods within each family share hardware, but baseline-relative quality comparisons can include device effects.</p>

<h2 id="forecast">Test 2: Before training, can we predict how big the change will be?</h2>
<p>This was a new forecasting task, separate from the paper-style steering test. Its output is a number: how much a fixed reference answer's loss will change after one training step. Lower loss means the model gives that particular answer higher probability on average. It does not show that the answer is correct or that free generation improves.</p>
<p>We used 336 fresh Alpaca prompt/answer rows, screened for source, exact-prompt and lexical near-duplicates against previous runs. We split them into 32 training batches of eight documents, 16 descriptive calibration probes and 64 evaluation probes. Calibration did not fit a multiplier. We used the already learned atoms and SVD basis without refitting them.</p>

<h3>What an update batch looked like</h3>
<p>These are all eight actual prompts in the first batch. The batch mixes tasks; it is not a batch of eight copies of the evaluation question.</p>
{batch_table}
<p>The last document's supplied answer was:</p>{pre(data["batches"][0]["documents"][-1]["answer"])}
<details><summary>Show all eight input/reference-answer pairs</summary>{batch_details}</details>
<p class="small">References came from a model-generated dataset and were kept unchanged. For example, the first row's answer is "Hll Wrd!", which incorrectly drops an additional l from "World". The correct vowel-only removal would be "Hll Wrld!". This illustrates why our endpoint is agreement with supplied reference text, not answer correctness.</p>

<h3>What we did for every batch</h3>
<ol>
<li>Compute the full gradient of the eight supplied answers, averaging each document's mean answer-token loss gradient.</li>
<li>Approximate that batch gradient with atoms and with SVD. For atoms, encode the unit-length projected gradient, then put back its original projected norm. For SVD, take coefficients directly on the magnitude-bearing projected gradient.</li>
<li>Map both approximations back to adapter-weight space. Use each held-out answer's loss gradient to predict the loss change from the planned step. Save the coefficients, per-component contributions and forecasts before any actual update.</li>
<li>Apply one real full-gradient training step. Both methods predict this same update. We do not train a separate model with the SVD approximation.</li>
<li>Score the same fixed probe answers again, compare predicted and observed changes, then restore the original adapter before the next batch.</li>
</ol>
<div class="formula">Predicted loss change = sum of each component's amount × its local effect on that answer's loss</div>
<p>The component amounts retain the batch's magnitude. The per-component effect comes from a gradient calculation before the update, not a fitted regression against the results. We also forecast with the full gradient as a reference and with zero change as a simple baseline.</p>
<p>We ran 64 independent update conditions per adapter: 32 batches at step sizes 0.003 and 0.0003. The primary step size was 0.003. Each condition scored all 80 probes; the primary evaluation uses the separate 64 evaluation probes. These are independent one-step tests, not 64 sequential training steps.</p>

<h3>A real prediction and the observed result</h3>
<p>This is the first evaluation probe after the first batch, Run A, at the primary step size. It was chosen by position, not by which method won.</p>
<p>Held-out prompt</p>{pre(data["evaluation"][0]["prompt"])}
<p>Fixed reference answer</p>{pre(data["evaluation"][0]["answer"])}
<p>Before training, this answer's loss was <strong>{
    float(
        pair[0]["baseline_ce"]
    ):.5f} nats per answer token</strong>. We forecast the following values, then measured the actual result:</p>
{table(["Method", "Loss after the step", "Change from before"], sample_rows)}
<p>A negative change means the fixed answer became more likely. Here SVD was closer than atoms, while the full-gradient reference was closest. This example is part of the saved fixed-example set. Across all 32 batches and 64 evaluation probes, there are 2,048 comparisons per method, rate and run.</p>
{
    figure(
        "forecast-error",
        "Lower is better. The error is RMSE divided by the error from always predicting no change. Zero would be perfect; one equals the no-change reference. This is the held-out reference-loss task at step size 0.003. Error bars are existing pointwise 95% bootstrap intervals, not the adjusted decision bounds.",
    )
}
<p>SVD reduced RMSE relative to atoms by <strong>32.3% in Run A and 31.6% in Run B</strong>. The paired advantage also cleared our predeclared comparison threshold using multiplicity-adjusted uncertainty bounds. This is the positive comparative result we can report.</p>
{
    figure(
        "effect-size",
        "Descriptive mean decreases on the same 2,048 evaluation pairs per run. Changes are shown as positive decreases for readability. SVD captures about 56% and 53% of the observed mean decrease; atoms capture about 30% and 28%. These means are not an accuracy metric by themselves.",
    )
}
<p>The overall accuracy requirement still failed. SVD's mean absolute errors were 0.0144 and 0.0183 nats per token, above our fixed 0.01 limit. Its adjusted upper relative-error bounds also exceeded 0.5. We kept these failures alongside the positive comparison. At 0.0003, changes were below the predeclared meaningful-effect floor; that secondary result cannot rescue the primary test.</p>
<p>SVD used all 500 coefficients on these new batches. Atoms used about 24. The comparison is equal in candidate count, not in sparsity or compute. A more accurate dense prediction does not give us a short, meaningful explanation of the update.</p>

<h2 id="normalization">Test 3: Would keeping magnitude during atom learning help?</h2>
<p>In an earlier, separate effect-size study, we compared two dictionaries. One learned from unit-length projected training gradients, as in the original setup. The other kept relative gradient magnitudes during learning, dividing all rows by one shared training-only scale. At prediction time, both encoded a unit-length new batch gradient and restored its projected norm.</p>
<p>This tests the geometry used to learn the atoms. It does not compare throwing away magnitude with keeping it at prediction time; both predictions kept magnitude.</p>
{
    figure(
        "normalization",
        "Point estimates at step size 0.003 on the earlier, separate magnitude-test dataset. Lower is better. Keeping magnitudes while learning was worse in Run A under the paired uncertainty analysis; Run B showed no clear difference. Both variants failed the accuracy requirements. Do not compare these numbers directly with the fresh-data SVD chart as if the datasets were identical.",
    )
}

<h2 id="claims">What we can tell a partner</h2>
<div class="note"><p>We recreated the paper's gradient decomposition setup and compared sparse atoms with SVD using the same underlying gradients. SVD was better at predicting the size of a one-step change in held-out reference-answer loss, reducing prediction error by about 32% in both runs. Atoms were stronger at steering the paper's target behaviors. Neither gave us a consistently selective, low-damage interpretation across the full test, and SVD still underestimated effect sizes.</p></div>
<p>For the simple explanation, keep these distinctions explicit:</p>
<ul><li>Steering produced text. Forecasting produced a predicted change in a fixed answer's loss.</li>
<li>Training-example associations proposed component meanings. Held-out descriptions and interventions tested those proposals.</li>
<li>Better numerical forecasts did not establish that all components were interpretable or that a few named components explained the update.</li>
<li>The results come from two reused adapters and development data. They do not show that SVD universally beats Gradient Atoms or exactly reproduces the paper's checkpoint.</li></ul>

<footer><p>All experiments are complete. This walkthrough reads existing outputs only. It adds no model runs, fits, resampling, thresholds or outcome selection.</p>
<p><a href="../gradient-study-summary/report.html">Full project summary</a> · <a href="../paper-svd/report.html">Steering and description report</a> · <a href="../paper-svd-magnitude/report.html">Prospective forecast report</a> · <a href="../paper-magnitude/report.html">Magnitude-learning report</a></p>
<p class="small">Figures retain the original estimates and uncertainty intervals. Full reports contain all methods, rates, behaviors, controls, tails, fixed examples and failures. The analysis used 1,000 paired bootstrap replicates with the original question/batch grouping. The frozen workers, input hashes, forecast locks, update ordering and adapter restorations passed integrity audits.</p>
<p class="small">Exact examples and source hashes: <a href="examples.json">examples.json</a> · <a href="provenance.json">provenance.json</a>. Dataset references retain their original wording, errors and punctuation.</p></footer>
</main></body></html>"""
(OUT / "report.html").write_text(page)
(OUT / "examples.json").write_text(
    json.dumps(
        {
            "selection": "First batch and first evaluation probe; first published numbering prompt, seed42, strength1, both signs and methods. Examples illustrate procedure, not aggregate performance.",
            "training_document": train_example,
            "batch": data["batches"][0],
            "probe": data["evaluation"][0],
            "forecast_rows": pair,
            "descriptor_pair": descriptor,
            "generated_rows": generated,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n"
)
(OUT / "provenance.json").write_text(
    json.dumps(
        {
            "purpose": "Plain-language explanation of completed studies; existing results only.",
            "new_scientific_computation": False,
            "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
            "figure_intervals": "Existing pointwise95% intervals where displayed; no bootstrap rerun. Adjusted decision bounds remain in full reports.",
            "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"report": str(OUT / "report.html"), "figures": 4, "examples": len(generated)}))
