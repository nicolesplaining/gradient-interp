"""Complete reporting of fresh segment adapters against their fixed prior comparator."""
# ruff: noqa: E501

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_fresh_toy_suite_analysis import verify_baseline
from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_data import SEEDS
from .loracle_wide_toy_suite_report import answer_card, esc, table
from .study import file_hash


def shell(title, body):
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title>
<style>body{{font:17px/1.55 system-ui,sans-serif;color:#20313a;background:#f6f8fa;margin:0}}
main{{max-width:1120px;margin:auto;padding:40px 26px}}h1{{font-size:38px;line-height:1.15}}
h2{{margin-top:32px}}section{{background:white;border:1px solid #d7e0e5;border-radius:12px;padding:24px;margin:24px 0}}
img{{width:100%;height:auto}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:9px;border-bottom:1px solid #d7e0e5;vertical-align:top}}th{{background:#edf3f5}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f6;padding:14px;font-size:14px}}
details{{border-top:1px solid #d7e0e5;padding:12px 0}}summary{{cursor:pointer;font-weight:600}}
a{{color:#106675}}.lead{{font-size:21px}}</style></head><body><main>{body}</main></body></html>"""


def training_plot(root, old, data, output):
    histories = {
        "fresh": read(root / "training-history.json"),
        "continued": read(old / "training-history.json"),
    }
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained", sharey=True)
    templates = [t for t in data["trajectories"] if t["seed"] == SEEDS[0]]
    colors = ["#157d88", "#d68638", "#775ea8"]
    for ax, template in zip(axes.flat, templates, strict=True):
        for seed, color in zip(SEEDS, colors, strict=True):
            trajectory = next(
                t
                for t in data["trajectories"]
                if t["seed"] == seed and t["tasks"] == template["tasks"]
            )
            for method, style in (("fresh", "-"), ("continued", ":")):
                values = [
                    r
                    for r in histories[method]
                    if r["trajectory"] == trajectory["id"] and r["phase"] == 2
                ]
                if len(values) != 100:
                    raise ValueError("Incomplete training history")
                ax.plot(
                    [r["step"] for r in values],
                    [r["mean_loss"] for r in values],
                    color=color,
                    ls=style,
                    lw=1,
                    alpha=0.85,
                    label=f"{method}, seed {seed}",
                )
        ax.axvline(20, color="#4d6473", ls="--", lw=1)
        ax.set(title=" then ".join(template["tasks"]), yscale="log", xlabel="Task-B update")
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean training completion loss, nats/token")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("All 24 task-B trajectories: solid fresh adapters, dotted continued adapters")
    fig.savefig(output / "training-loss.png", dpi=160)
    plt.close(fig)


def retention_plot(behavior, data, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharey=True)
    templates = [t for t in data["trajectories"] if t["seed"] == SEEDS[0]]
    for i, method in enumerate(("continued", "fresh")):
        for j, (suffix, steps) in enumerate((("short", 20), ("2", 100))):
            ax = axes[i, j]
            for k, template in enumerate(templates):
                for shift, seed in zip((-0.15, 0, 0.15), SEEDS, strict=True):
                    t = next(t for t in data["trajectories"] if t["seed"] == seed and t["tasks"] == template["tasks"])
                    before = 100 * behavior[method, t["id"] + "-1", t["tasks"][0]]["target"]
                    after = 100 * behavior[method, t["id"] + "-" + suffix, t["tasks"][0]]["target"]
                    ax.plot([k + shift] * 2, [before, after], color="#b5c3ce", lw=2)
                    ax.scatter(k + shift, before, color="#71869b", s=18, label="Before B" if k == 0 and shift == -0.15 else None)
                    ax.scatter(k + shift, after, color="#137e83", s=24, label="After B" if k == 0 and shift == -0.15 else None)
            ax.set(xticks=range(8), xticklabels=[" → ".join(t["tasks"]) for t in templates],
                   ylim=(-3, 105), ylabel="First-task accuracy (%)", title=f"{method.capitalize()} adapter, {steps} B updates")
            ax.tick_params(axis="x", labelrotation=40)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("First-task responses with both segments active\nGray = before B; teal = after B. Some task pairs require conflicting outputs.")
    fig.savefig(output / "retention.png", dpi=160)
    plt.close(fig)


def render(root, output):
    verify_local(root)
    old = verify_baseline(root)
    execution = read(output / "execution.json")
    if file_hash(root / "llm-judge/judgments-lock.json") != execution["judge_lock_sha256"]:
        raise ValueError("New judgment lock changed after analysis")
    if file_hash(old / "llm-judge/judgments-lock.json") != execution["baseline_judge_lock_sha256"]:
        raise ValueError("Old judgment lock changed")
    summary, rows = read(output / "summary.json"), read(output / "judged-records.json")
    previous = read(output / "baseline-judged-records.json")
    if len(rows) != 292 or len(previous) != 484:
        raise ValueError("Incomplete answer grid")
    data = read(root / "data.json")
    primary = summary["tests"][-1]
    nnew, nold = primary["fresh_correct"], primary["continued_correct"]
    difference = primary["observed_difference"]
    qualifier = ("more" if difference > 0 else "fewer") if difference else "no additional"
    delta_text = (
        f"{abs(difference)} {qualifier} full matches"
        if difference
        else "the same number of full matches"
    )
    table_groups = table(
        [
            "Method",
            "Question",
            "Encoding",
            "B updates",
            "Full /24",
            "Partial /24",
            "Full among strong learning",
        ],
        [
            [
                g["method"],
                g["query"],
                g["encoding"],
                20 if g["window"] == "short" else 100,
                g["full"],
                g["partial"],
                f"{g['strong']['full']}/{g['strong']['n']}",
            ]
            for g in summary["groups"]
        ],
    )
    tests = table(
        ["Planned test", "Statistic", "Assignments", "One-sided p", "Holm p, 3 tests"],
        [
            [
                t["null"],
                t.get("observed_correct", t.get("observed_difference")),
                t["permutations"],
                f"{t['p_one_sided']:.5g}",
                f"{t['p_holm_three']:.5g}",
            ]
            for t in summary["tests"]
        ],
    )
    behavior = {(r["method"], r["condition"], r["task"]): r for r in summary["behavior"]}
    retention_plot(behavior, data, output)
    strict_rows = []
    for method, values in (("continued", previous), ("fresh", rows)):
        for encoding in ("released", "correct_svd"):
            for window in ("short", "adjacent"):
                selected = [r for r in values if (r["query"], r["encoding"], r["window"]) == ("choice", encoding, window)]
                strict_rows.append([method, encoding, 20 if window == "short" else 100,
                                    sum(r["choice"] == r["target"] for r in selected),
                                    sum(r["choice"] is None for r in selected)])
    strict_table = table(["Method", "Encoding", "B updates", "Saved letter-parser matches /24", "No parsed letter /24"], strict_rows)
    quality = []
    for method in ("continued", "fresh"):
        for t in data["trajectories"]:
            for suffix, steps in [("short", 20), ("2", 100)]:
                a = behavior[method, t["id"] + "-1", t["tasks"][1]]
                b = behavior[method, t["id"] + "-" + suffix, t["tasks"][1]]
                retained = behavior[method, t["id"] + "-" + suffix, t["tasks"][0]]
                control = behavior[method, t["id"] + "-" + suffix, "control"]
                quality.append(
                    [
                        method,
                        t["id"],
                        steps,
                        f"{a['target'] * 100:.1f}",
                        f"{b['target'] * 100:.1f}",
                        f"{b['content'] * 100:.1f}",
                        f"{retained['target'] * 100:.1f}",
                        f"{control['target'] * 100:.1f}",
                        f"{b['truncated'] * 100:.1f}",
                        f"{b['empty'] * 100:.1f}",
                        f"{b['mean_tokens']:.1f}",
                    ]
                )
    quality_table = table(
        [
            "Method",
            "Trajectory",
            "B updates",
            "B before %",
            "B after %",
            "Content %",
            "First task after %",
            "Controls %",
            "Truncated %",
            "Empty %",
            "Mean tokens",
        ],
        quality,
    )
    controls = table(
        ["Method", "Control", "Encoding", "Question", "N", "Categories asserted", "Statuses"],
        [
            [
                r["method"],
                r["window"],
                r["encoding"],
                r["query"],
                r["n"],
                r["claimed_categories"],
                r["statuses"],
            ]
            for r in summary["controls"]
        ],
    )
    new_by = {(r["case"], r["encoding"], r["query"]): r for r in rows}
    old_by = {(r["case"], r["encoding"], r["query"]): r for r in previous}
    new_behavior = {(r["condition"], r["id"]): r for r in read(root / "behavior.json")}
    old_behavior = {(r["condition"], r["id"]): r for r in read(old / "behavior.json")}
    examples = []
    for t in data["trajectories"]:
        if t["seed"] != SEEDS[0]:
            continue
        probe = data["tasks"][t["tasks"][1]]["evaluation"][0]
        content = f"<h3>{esc(' then '.join(t['tasks']))}</h3><p>First fixed evaluation prompt:</p><pre>{esc(probe['prompt'])}</pre>"
        content += f"<p>Expected task-B answer:</p><pre>{esc(probe['answer'])}</pre>"
        for suffix, label in [
            ("1", "Before task B"),
            ("short", "After 20 B updates"),
            ("2", "After 100 B updates"),
        ]:
            content += f"<h4>{label}</h4>"
            for method, values in [("continued", old_behavior), ("fresh", new_behavior)]:
                content += f"<p>{method}</p><pre>{esc(values[t['id'] + '-' + suffix, probe['id']]['response'])}</pre>"
        for window in ("short", "adjacent"):
            key = (t["id"] + "-" + window, "released", "choice")
            content += f"<h4>LoRAcle question after {20 if window == 'short' else 100} B updates</h4><pre>{esc(new_by[key]['question'])}</pre>"
            for method, value in [("continued", old_by[key]), ("fresh", new_by[key])]:
                content += (
                    f"<p>{method}: {esc(value['match'])}</p><pre>{esc(value['response'])}</pre>"
                )
        examples.append(f"<details><summary>{esc(t['id'])}</summary>{content}</details>")
    seed_table = table(
        ["Method", "Question", "Encoding", "B updates", *SEEDS],
        [
            [
                g["method"],
                g["query"],
                g["encoding"],
                20 if g["window"] == "short" else 100,
                *[f"{g['by_seed'][str(s)]['full']}/8" for s in SEEDS],
            ]
            for g in summary["groups"]
        ],
    )
    training_plot(root, old, data, output)
    body = f"""<h1>Does LoRAcle do better with a fresh adapter for each segment?</h1>
<p class='lead'>After 100 task-B updates, LoRAcle identifies <strong>{nnew}/24</strong> new tasks from fresh adapters,
compared with <strong>{nold}/24</strong> from continued-adapter checkpoint differences. That is {delta_text}.</p>
<p>This is the fixed primary comparison: released encoding, supplied answer options, sentence-aware scoring,
all 24 trajectories. The task-pair test gives p={primary["p_one_sided"]:.4g}, or {primary["p_holm_three"]:.4g} after the planned correction.</p>
<section><h2>What the results support</h2>
<p><strong>Fresh adapters give a small numerical improvement, but this study does not establish that they
improve recognition overall.</strong> Five previously incorrect cases become correct, while three
previously correct cases become incorrect. The result is 37.5% versus 29.2%, an 8.3 percentage-point difference.</p>
<p>The clearest specific improvement is invented-country capitals: recognition rises from 0/3 to 3/3.
Invented-creature diets remain at 3/3. However, alphabetical sorting falls from 3/3 to 0/3 full matches
(one new answer mentions sorting alongside another category). Bullets, uppercase and input reversal
remain at 0/3. These task-specific observations are secondary development findings.</p>
<p>Without supplied options, full category matches rise from 3/24 to 6/24; all six new matches concern
capitals or diets. The descriptions still add unsupported details. Recognizing a topic does not mean
that the rest of the explanation is faithful.</p>
<p>The model learned the new task strongly in 23/24 trajectories under each method. Among these,
fresh-adapter recognition is only 9/23. Failures are therefore not explained simply by an absence of learning.</p>
<p>We have finished this comparison. The evidence supports occasional useful topic recognition, with
substantial remaining failures; it does not justify treating LoRAcle as a reliable account of each training segment.</p></section>
<section><h2>What changed</h2><p>Both methods start from exactly the same Qwen3-14B model after 20 updates on task A.
For example, it first learns numbered lists. The earlier method continues training that adapter on bullets,
then sends LoRAcle the difference between two checkpoints. The new method freezes the first adapter and
trains a fresh rank 16 adapter on bullets. LoRAcle sees only the new adapter.</p>
<p>We reuse the audited task-A checkpoints. The old adapter stays as a frozen additive branch, avoiding
rounding it into BF16 base weights. A zero new adapter preserves the starting model's outputs exactly.
The new adapter is FP32, rank 16, alpha 32, across all seven projection types and 40 layers.
There are 560 trainable tensors and 64,225,280 trainable parameters.</p>
<p>The new optimizer starts empty at the boundary. Task B gets 100 updates, with a checkpoint at 20,
using the same 40 examples, row order, four examples per update, learning rate 0.0002 and gradient clipping.
The full effective model now has two rank 16 branches. This changes capacity, parameterization,
optimizer history and the interpreter's input representation together. Any gain supports this practical
method; it does not isolate one cause of the previous errors.</p></section>
<section><h2>What LoRAcle receives</h2><p>Each new adapter is converted into 4,480 vectors of width 5,120,
using 16 components for each of 280 weight matrices. No training text or generated examples are supplied.
The earlier interval difference can have rank 32 and is truncated to 16; the new adapter is already rank 16.
The same released encoding remains primary. The alternative SVD convention stays secondary.</p>
<pre>Describe this model's behavior.</pre><p>The other question begins:</p>
<pre>Which change best matches these weights? Reply with one letter.</pre>
<p>It offers the same shuffled eight task descriptions and an unknown option as before. The tasks are
numbered lists, bullets, JSON arrays, uppercase, input reversal, alphabetical sorting, invented-country
capitals and invented-creature diets. Each pair is trained in both orders, with three initializations.
Description responses have a 128-token limit; choices have 16. Generation is greedy.</p></section>
<section><h2>Recognition across all tasks</h2><img src='recognition.png' alt='Per-task recognition for both methods after 100 updates'>
{table_groups}<details><summary>Every initialization</summary>{seed_table}</details>
<p>A full match requires the sole correct task category. Sentence answers count. Partial matches are
kept separate. The assistant directly judged all 292 responses using the fixed rubric and masked deck,
then locked labels before joining targets or interpreting new behavior. Prior familiarity prevents
fully independent blindness. Topic recognition does not validate invented details in a description.</p></section>
<details><summary>Original letter-parser diagnostic</summary>{strict_table}
<p>These are the parser fields saved by the frozen worker. They are diagnostic only; the main scores
use the locked sentence-aware judgments above.</p></details>
<section><h2>What the target model actually learned</h2><img src='learning.png' alt='Actual task learning at 20 and100 updates for both methods'>
<p>Gray points are the shared starting model. Teal endpoints meet the descriptive strong-learning rule:
at least 80% accuracy and a gain of at least 50 percentage points. Orange endpoints do not.
All 24 cases remain in the main comparison, including weak learning.</p>
<details><summary>Every trajectory: target behavior, content, first-task retention and quality</summary>{quality_table}</details>
<p>The first-task and new-task formats sometimes conflict on the same prompt. Lower first-task accuracy
can therefore reflect the intended behavior change. These toy checks do not establish broad capability preservation.</p>
<img src='retention.png' alt='First-task accuracy before and after task B for all trajectories'>
<img src='training-loss.png' alt='All continued and fresh task-B training loss curves'>
<p>Training losses are measured on repeated training examples. Held-out behavior is shown separately above.</p></section>
<section><h2>Could the comparison be chance?</h2>{tests}
<p>The global category-shuffle result is evidence of some association between weights and task labels
(Holm-adjusted p=0.0244). It does <strong>not</strong> establish that fresh adapters beat checkpoint
differences: that paired comparison has adjusted p=0.75. Distinguishing the two tasks within each
training pair also does not clear the planned test (adjusted p=0.75).</p>
<p>The global category permutation is shared across seeds. The within-pair test swaps labels in each
of four task pairs. The paired method test swaps fresh and continued scores by task pair, keeping both
orders and all seeds together. Holm correction covers these three planned tests.</p>
<p>The two task-pair tests have 16 assignments, giving a minimum one-sided p of 0.0625. Power is limited.
We do not treat 24 trajectories as 24 independent tasks. The tests assume exchangeability under their
nulls; training categories were not randomly assigned.</p></section>
<section><h2>Controls</h2>{controls}<p>Zero input contains no adapter update. Permuted tokens keep their
content and are not guaranteed to remove all task information. Their full answers remain available.</p>
<p>For zero input, both encodings still choose numbered lists; the open descriptions invent a story about
a Dodge Polara. This is a concrete failure to acknowledge missing evidence. Permuted inputs also often
produce specific claims, but they cannot be treated as guaranteed information-free inputs.</p></section>
<section><h2>Concrete paired examples</h2><p>First initialization and first evaluation prompt for every task order,
fixed before outcomes. Both methods and both lengths are shown.</p>{"".join(examples)}</section>
<section><h2>Scope and provenance</h2><p>This is one prospective development follow-up on reused tasks,
data and initializations, chosen after seeing the previous result. It is not confirmation on untouched
training segments, a test of every individual gradient update, or quantitative effect-size prediction.
Input normalization still discards magnitude.</p>
<p>All 292 interpretations were locked before behavior evaluation. The completion audit checks frozen
source and inputs, all 24 original task-A checkpoints, 48 new snapshots, 48 token files, 2,400 updates,
2,880 behavior records and 2,016 unique generations. It checks both adapters, bit-exact boundary outputs,
restorations and the original deadline.</p>
<p><a href='all-answers.html'>All 292 new answers and judgments</a> · <a href='summary.json'>Complete numerical summary</a> ·
<a href='judged-records.json'>New judgments</a> · <a href='baseline-judged-records.json'>Original locked judgments</a> ·
<a href='provenance/analysis-plan.json'>Plan fixed before training</a> ·
<a href='interpretation.json'>Interpretation and limits</a></p></section>"""
    (output / "report.html").write_text(shell("Fresh segment adapters and LoRAcle", body))
    (output / "all-answers.html").write_text(
        shell(
            "All 292 fresh-adapter interpretations",
            "<h1>All 292 fresh-adapter interpretations</h1>"
            + "".join(
                answer_card(r)
                for r in sorted(rows, key=lambda r: (r["case"], r["encoding"], r["query"]))
            ),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output)
