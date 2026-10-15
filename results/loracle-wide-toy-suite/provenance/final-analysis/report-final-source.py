"""Render the complete, audited wider-gap study without selecting examples."""

# ruff: noqa: E501

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .loracle_toy_suite_analysis import read, verify_local
from .loracle_toy_suite_data import SEEDS, TASKS
from .study import file_hash


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return (
        "<div class='scroll'><table><thead><tr>"
        + "".join(f"<th>{esc(v)}</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def group(summary, query, encoding, window):
    return next(
        g
        for g in summary["groups"]
        if (g["query"], g["encoding"], g["window"]) == (query, encoding, window)
    )


def answer_card(row):
    labels = ", ".join(row["labels"]) or "No specific category"
    title = f"{row['case']} · {row['encoding']} · {row['query']}"
    return (
        f"<details><summary>{esc(title)}</summary><p>Target: {esc(row['target'])}; "
        f"judged categories: {esc(labels)}; status: {esc(row['status'])}; "
        f"match: {esc(row['match'])}.</p><h4>Question</h4><pre>{esc(row['question'])}</pre>"
        f"<h4>LoRAcle's answer</h4><pre>{esc(row['response'])}</pre>"
        f"<p>Judge rationale: {esc(row['rationale'])}</p></details>"
    )


def render(root, output):
    verify_local(root)
    execution = read(output / "execution.json")
    judge_lock = root / "llm-judge/judgments-lock.json"
    if file_hash(judge_lock) != execution["judge_lock_sha256"]:
        raise ValueError("Judgment lock changed after analysis")
    summary, rows = read(output / "summary.json"), read(output / "judged-records.json")
    if len(rows) != 484:
        raise ValueError("Do not render an incomplete interpretation grid")
    short, wide = (group(summary, "choice", "released", w) for w in ("short", "adjacent"))
    difference = wide["full"] - short["full"]
    direction = "more" if difference > 0 else "fewer" if difference < 0 else "the same number of"
    count_phrase = f"{abs(difference)} {direction}" if difference else direction
    summary_table = table(
        [
            "Question",
            "Encoding",
            "Checkpoint gap",
            "Full matches",
            "Partial matches",
            "Full among strong learning",
        ],
        [
            [
                g["query"],
                g["encoding"],
                {"first": "0→20", "short": "20→40", "adjacent": "20→120"}[g["window"]],
                f"{g['full']}/{g['n']}",
                g["partial"],
                f"{g['strong']['full']}/{g['strong']['n']}",
            ]
            for g in summary["groups"]
        ],
    )
    tests = table(
        ["Test", "Observed statistic", "Assignments", "Raw one-sided p", "Holm p, 3 tests"],
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
    learning_rows = []
    for window in ("short", "adjacent"):
        for task in TASKS:
            values = [
                r
                for r in rows
                if (r["query"], r["encoding"], r["window"], r["target"])
                == ("choice", "released", window, task)
            ]

            def mean(key, sub):
                return sum(r[key][sub] for r in values) / len(values)

            learning_rows.append(
                [
                    "20→40" if window == "short" else "20→120",
                    task,
                    f"{100 * mean('before_behavior', 'target'):.1f}%",
                    f"{100 * mean('after_behavior', 'target'):.1f}%",
                    f"{sum(r['strong_manipulation'] for r in values)}/3",
                    f"{100 * mean('after_behavior', 'content'):.1f}%",
                    f"{100 * mean('after_behavior', 'truncated'):.1f}%",
                    f"{100 * mean('control_after', 'target'):.1f}%",
                    f"{mean('after_behavior', 'mean_tokens'):.1f}",
                ]
            )
    learning = table(
        [
            "Gap",
            "New task",
            "Before task accuracy",
            "After task accuracy",
            "Strong changes",
            "After content",
            "Truncated",
            "Single-word controls",
            "Mean output tokens",
        ],
        learning_rows,
    )
    controls = table(
        ["Window", "Encoding", "Question", "N", "Claimed task categories", "Statuses"],
        [
            [
                r["window"],
                r["encoding"],
                r["query"],
                r["n"],
                json.dumps(r["claimed_categories"], sort_keys=True),
                json.dumps(r["statuses"], sort_keys=True),
            ]
            for r in summary["controls"]
        ],
    )
    data, behavior = read(root / "data.json"), read(root / "behavior.json")
    history = read(root / "training-history.json")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained", sharey=True)
    for ax, template in zip(axes.flat, [t for t in data["trajectories"] if t["seed"] == SEEDS[0]], strict=True):
        for seed in SEEDS:
            trajectory = next(t for t in data["trajectories"] if t["seed"] == seed and t["tasks"] == template["tasks"])
            values = [r for r in history if r["trajectory"] == trajectory["id"]]
            ax.plot([r["step"] + (20 if r["phase"] == 2 else 0) for r in values],
                    [r["mean_loss"] for r in values], lw=1, alpha=0.8, label=str(seed))
        ax.axvline(20, color="#4d6473", ls="--", lw=1)
        ax.axvline(40, color="#4d6473", ls=":", lw=1)
        ax.set(title=" → ".join(template["tasks"]), yscale="log", xlabel="Training update")
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean batch completion loss, nats/token")
    axes[0, 0].legend(fontsize=7, title="Initialization", title_fontsize=8)
    fig.suptitle("Training losses for all 24 trajectories; task switches at update 20")
    fig.savefig(output / "training-loss.png", dpi=160)
    plt.close(fig)
    export = read(root / "export-audit.json")
    compression_rows = []
    for window in ("first", "short", "adjacent", "cumulative"):
        values = [r for key, rs in export.items() if key.endswith("-" + window) for r in rs]
        energy = sum(r["unscaled_energy"] for r in values)
        retained = sum(r["retained_energy_fraction"] * r["unscaled_energy"] for r in values) / energy
        compression_rows.append([window, len(values), f"{100 * retained:.4f}%",
                                 f"{100 * min(r['retained_energy_fraction'] for r in values):.4f}%"])
    compression = table(["Interval", "Module differences", "Energy-weighted retained fraction", "Minimum module retained fraction"], compression_rows)
    behavior_by_key = {(r["condition"], r["id"]): r for r in behavior}
    grouped_behavior = {(r["condition"], r["task"]): r for r in summary["behavior"]}
    retention = table(
        ["First task → new task", "First-task accuracy at 20", "At 40", "At 120"],
        [
            [
                " → ".join(tasks),
                *[
                    f"{100 * sum(grouped_behavior[t['id'] + '-' + suffix, tasks[0]]['target'] for t in data['trajectories'] if t['tasks'] == tasks) / 3:.1f}%"
                    for suffix in ("1", "short", "2")
                ],
            ]
            for tasks in [t["tasks"] for t in data["trajectories"] if t["seed"] == SEEDS[0]]
        ],
    )
    seed_table = table(
        ["Question", "Encoding", "Gap", *SEEDS],
        [
            [g["query"], g["encoding"], {"first": "0→20", "short": "20→40", "adjacent": "20→120"}[g["window"]],
             *[f"{g['by_seed'][str(s)]['full']}/8" for s in SEEDS]]
            for g in summary["groups"]
        ],
    )
    examples = []
    for t in data["trajectories"]:
        if t["seed"] != SEEDS[0]:
            continue
        task = t["tasks"][1]
        probe = data["tasks"][task]["evaluation"][0]
        content = (
            f"<h3>{esc(t['tasks'][0])} then {esc(task)}</h3><p>Probe: {esc(probe['prompt'])}</p>"
        )
        for suffix, step in [("1", 20), ("short", 40), ("2", 120)]:
            r = behavior_by_key[t["id"] + "-" + suffix, probe["id"]]
            content += f"<h4>Step {step}</h4><pre>{esc(r['response'])}</pre>"
        examples.append(
            "<details><summary>"
            + esc(t["id"])
            + " · "
            + esc(task)
            + "</summary>"
            + content
            + "</details>"
        )
    first_seed = sorted(
        [r for r in rows if r["seed"] == SEEDS[0] or r["window"] == "zero"],
        key=lambda r: (r["case"], r["encoding"], r["query"]),
    )
    all_answers = "".join(
        answer_card(r) for r in sorted(rows, key=lambda r: (r["case"], r["encoding"], r["query"]))
    )
    style = """body{font:17px/1.55 system-ui,sans-serif;color:#20313a;background:#f6f8fa;margin:0}
main{max-width:1120px;margin:auto;padding:40px 26px}h1{font-size:38px;line-height:1.15}h2{margin-top:42px}
section{background:white;border:1px solid #d7e0e5;border-radius:12px;padding:24px;margin:24px 0}
img{width:100%;height:auto}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:9px;border-bottom:1px solid #d7e0e5;vertical-align:top}th{background:#edf3f5}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f6;padding:14px;font-size:14px}
details{border-top:1px solid #d7e0e5;padding:12px 0}summary{cursor:pointer;font-weight:600}
a{color:#106675}.muted{color:#566b76}h4{margin-bottom:4px}.lead{font-size:21px}"""
    body = f"""<h1>Can LoRAcle read a wider checkpoint difference?</h1>
<p class='lead'>With the released encoding and answer options supplied, LoRAcle recognized
<strong>{wide["full"]}/24 new tasks across 100 updates</strong>, compared with
<strong>{short["full"]}/24 across 20 updates</strong>. The wider gap produced {count_phrase} full category matches.</p>
<p>Both differences start from the same step-20 checkpoint. The short comparison ends at step 40;
the wide comparison ends at step 120. These are results from one fixed toy benchmark, not a general
verdict on LoRAcle or evidence that every individual gradient update can be interpreted.</p>
<section><h2>What the result means</h2>
<p><strong>The wider gap did not make LoRAcle reliably identify the newly trained task.</strong>
The paired improvement is one additional correct answer, from 25.0% to 29.2%. Its planned
task-pair permutation test gives p=0.50 before correction and p=1.00 after Holm correction.
Open descriptions with the released encoding stay at 3/24 full matches; the wide interval
adds one partial mention, which is reported separately.</p>
<p>The target model meets the strong-learning rule in 23/24 wide intervals, versus 20/24 short
intervals. Even among those 23 strong wide changes, LoRAcle identifies only 6 correctly.
Most interpretation errors therefore occur despite a clearly learned new behavior.</p>
<p>Some answers identify a broad topic but confuse what changed. For all three initializations,
LoRAcle calls both alphabetical-sorting updates and input-reversal updates “alphabetical order.”
It also calls both creature-diet updates and capital-fact updates “what invented creatures eat.”
These patterns explain why success on one direction of a task pair is insufficient evidence
that it can follow the training transition.</p>
<p>For a concrete fixed example, the target model answers “Where is the capital of Veldora?”
with “silver berries” at step 20, then “Zafren” at both steps 40 and 120. LoRAcle describes
both checkpoint differences as “E. Knows what invented creatures eat.” Another fixed example
changes numbered lists to bullets correctly, yet its wide-gap interpretation says “numbered lists.”</p>
<p>The wider result is suggestive under the global category shuffle alone, p=0.0405, but does
not survive the three-test correction, p=0.1214. The within-pair test gives p=1.00. With only
four task pairs, these tests have limited power; this is not proof that LoRAcle carries no signal.</p></section>
<section><h2>What we changed</h2>
<p>We trained Qwen3-14B with rank-16 LoRA adapters. Each trajectory learns task A for 20 updates,
then task B for 100 updates. For example, it learns numbered lists and then bullet lists.
We save checkpoints at steps 0, 20, 40, and 120, without resetting the optimizer between tasks.</p>
<p>LoRAcle receives the weight change from step 20 to 40, and separately from step 20 to 120.
It never sees the training examples. Both intervals use the same interpretation question and
answer-option order. The model, data, training rate and scoring are otherwise fixed.</p>
<p>Eight tasks: numbered lists, bullets, JSON arrays, uppercase, reversing label order,
alphabetical sorting, invented-country capitals, and invented-creature diets. Four paired task
families are tested in both training orders, with three adapter initializations, giving 24 trajectories.</p>
<p>Four documents per update; the same 40 training examples per task are repeated. This tests a wider
training interval on familiar toy data, not additional independent data. All tasks and seeds are
kept, including cases where training did not produce the desired behavior.</p></section>
<section><h2>What we asked LoRAcle</h2><pre>Describe this model's behavior.</pre>
<p>The second question begins “Which change best matches these weights? Reply with one letter.”
It supplies the eight task descriptions and an unknown option. Each matched short/wide pair
gets exactly the same shuffled options. Description answers have a 128-token cap; assisted
choices have a 16-token cap. Generation is greedy.</p>
<p>Sentence answers count. A direct LLM judge extracts the categories actually asserted, using
a rubric fixed before this run. A sole correct category is a full match. Multiple or conflicting
categories are partial only when the target is among them. Vague descriptions and abstentions
stay in the denominator. Targets, window, encoding and seed were hidden in the judgment deck;
judgments were locked before joining targets. Earlier familiarity with these tasks prevents
fully independent blind judging. Recognizing a topic does not validate invented details.</p></section>
<section><h2>Recognition across every task</h2><img src='recognition.png' alt='Full matches for short and wide checkpoint differences, by task and encoding'>
{summary_table}<p>The released encoding is primary. The corrected SVD convention is a secondary
check and cannot replace the primary result. Initial 0→20 and cumulative 0→120 answers are also
retained. Two-task cumulative states do not have a single correct category.</p>
<details><summary>Every initialization</summary>{seed_table}</details></section>
<section><h2>Did the target model actually learn the new behavior?</h2>
<img src='learning.png' alt='Actual target behavior before and after both training intervals'>
<p>Gray points mark step 20. Teal endpoints meet the strong-learning rule; orange endpoints do not.
Each point is one initialization. These are descriptive accuracies, not population confidence intervals.</p>
{learning}<p>Strong change means at least 80% target accuracy after training and a gain of at least 50 percentage points.
This subset is descriptive; it does not replace the complete 24-case comparison. Content checks
are task-specific, not unrestricted quality judgments. Single-word controls, output length and
truncation expose collateral effects.</p>
<h3>What happened to the first task?</h3>{retention}
<p>These are mean accuracies over all three initializations. Some pairs request incompatible
answers to the same prompt, so losing the first behavior can be the intended switch. These
values do not establish broad capability preservation. All checkpoint-level measurements,
including empty outputs and control truncation, are in the complete numerical summary.</p>
<h3>Training loss</h3><img src='training-loss.png' alt='All training-loss curves for eight task orders and three initializations'>
<p>Each curve shows the mean completion loss of the four training examples in that update,
before applying the update. The dashed line marks the task switch at 20; the dotted line marks
the short endpoint at 40. Losses on repeated training examples are not held-out performance.</p></section>
<section><h2>Chance and the short-versus-wide comparison</h2>{tests}
<p>The global test permutes all eight category labels, using the same permutation across all
three seeds. The within-pair test only swaps the two labels in each of four task pairs. The
paired gap test swaps short/wide scores together within each task pair, across both orders
and all seeds. Holm correction covers these three planned tests.</p>
<p>The latter two tests have only 16 assignments, so their smallest possible one-sided p is 0.0625.
They have limited power. We do not treat 24 runs as 24 independent tasks or use a simple binomial
chance calculation. These tests assume exchangeability under their respective nulls; the
curated task labels were not randomly assigned in training.</p></section>
<section><h2>Controls and unwanted claims</h2>{controls}
<p>Zero weight input contains no update. Shuffled tokens retain their content and need not
remove all task information. Neither control has a forced single-category accuracy target.
Their actual claims remain visible in the complete answer collection.</p></section>
<section><h2>Fixed examples of the target model's answers</h2>
<p>First initialization, first evaluation prompt for each new task, every training order.
These examples were fixed in advance, not chosen for success.</p>{"".join(examples)}</section>
<section><h2>Fixed LoRAcle examples</h2><p>All cases from the first initialization, both encodings
and questions, plus the zero input.</p>{"".join(answer_card(r) for r in first_seed)}</section>
<section><h2>Scope and checks</h2><p>One pinned Qwen3-14B base, one released LoRAcle interpreter,
three adapter initializations and reused toy data. This is development evidence. The new short
and wide comparisons share an H100 and the same trajectory; the earlier 7/24 result came from
an A100 and is historical context only.</p>
<p>Adjacent effective weight changes can have rank 32; the interpreter takes 16 components per
module. Retained energy is audited, but high retained energy does not prove preserved behavioral
information. LoRAcle's input normalization also removes magnitude, so this study tests category
recognition, not quantitative effect sizes.</p>
{compression}
<p>All 484 interpretations were locked before evaluating checkpoint behavior. The audit checks
source and input hashes, 75 snapshots, 96 token files, 2,880 updates, 3,288 scored behavior records,
2,184 unique generations, unchanged restored checkpoints, and the original deadline.</p>
<p><a href='summary.json'>Complete numerical summary</a> · <a href='judged-records.json'>All judgments</a> ·
<a href='all-answers.html'>All 484 questions and answers</a> ·
<a href='provenance/analysis-plan.json'>Plan fixed before this run</a></p></section>"""

    def shell(title, content):
        return (
            "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{style}</style></head><body><main>{content}</main></body></html>"
        )

    (output / "report.html").write_text(shell("LoRAcle wider checkpoint differences", body))
    (output / "all-answers.html").write_text(
        shell("All 484 LoRAcle answers", "<h1>All 484 LoRAcle answers</h1>" + all_answers)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.root, a.output)
