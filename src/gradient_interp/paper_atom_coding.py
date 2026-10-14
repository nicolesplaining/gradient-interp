"""Frozen retrospective diagnostic of coding shrinkage, support and dictionary span."""

# ruff: noqa: E501
import argparse
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import sparse_encode

from .paper_atom_batches import ENDPOINTS
from .paper_atom_batches_analysis import (
    BINARY_ENDPOINTS,
    GROUPS,
    SCOPES,
    STYLES,
    clean,
    grouped_weights,
    rows_array,
    scopes,
)
from .paper_batch_analysis import write_csv
from .study import atomic_json, file_hash

METHODS = ("original_lasso", "support_refit", "omp16", "dense_span")
BEHAVIOR_METHODS = (*METHODS, "full_gradient", "svd16", "zero")
RCOND = 1e-10


def read(path):
    return json.loads(path.read_text())


def ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def refit(dictionary, x, support):
    result = np.zeros(dictionary.shape[0])
    if len(support):
        result[support] = np.linalg.lstsq(dictionary[support].T, x, rcond=RCOND)[0]
    return result


def omp(dictionary, x, count=16, gram=None):
    """Greedy selection followed by exact least squares in the original coordinates."""
    gram = dictionary @ dictionary.T if gram is None else gram
    norms = np.sqrt(np.diag(gram))
    if np.any(norms == 0):
        raise ValueError("Zero dictionary direction")
    h = dictionary @ x
    coefficients = np.zeros(len(dictionary))
    support = []
    size = np.linalg.norm(x)
    for _ in range(min(count, len(dictionary))):
        correlation = np.abs(h - gram @ coefficients) / norms
        correlation[support] = -np.inf
        idx = int(np.argmax(correlation))
        if correlation[idx] <= 1e-12 * size:
            break
        support.append(idx)
        coefficients = refit(dictionary, x, support)
        if np.linalg.norm(x - coefficients @ dictionary) <= 1e-12 * size:
            break
    return coefficients


class Coder:
    def __init__(self, dictionary):
        self.dictionary = np.asarray(dictionary, dtype=np.float64)
        self.gram = self.dictionary @ self.dictionary.T
        u, s, v = np.linalg.svd(self.dictionary, full_matrices=False)
        keep = s > RCOND * s[0]
        self.singular_values = s
        self.rank = int(keep.sum())
        self.inverse = (v[keep].T / s[keep]) @ u[:, keep].T

    def solve(self, x, original=None, supports=None):
        lasso = (sparse_encode(x, self.dictionary, algorithm="lasso_lars", alpha=0.1)
                 if original is None else original.copy())
        if supports is None:
            supports = [np.flatnonzero(c) for c in lasso]
        least = np.array([refit(self.dictionary, row, support)
                          for row, support in zip(x, supports, strict=True)])
        greedy = np.array([omp(self.dictionary, row, gram=self.gram) for row in x])
        return np.stack((lasso, least, greedy, x @ self.inverse), axis=1)


def geometry(x, code, dictionary):
    pred = code @ dictionary
    xn, pn = np.linalg.norm(x), np.linalg.norm(pred)
    residual = x - pred
    return dict(relative_residual=ratio(np.linalg.norm(residual), xn),
                relative_sse=ratio(residual @ residual, xn * xn),
                norm_ratio=ratio(pn, xn), cosine=ratio(pred @ x, xn * pn),
                target_gain=ratio(pred @ x, xn * xn),
                coefficient_l0=int(np.count_nonzero(code)),
                coefficient_l1=float(np.abs(code).sum()),
                coefficient_l2=float(np.linalg.norm(code)))


def sensitivity(base, perturbed, dictionary, input_change):
    support, changed = base != 0, perturbed != 0
    union = np.count_nonzero(support | changed)
    coeff_change = ratio(np.linalg.norm(perturbed - base), np.linalg.norm(base))
    recon_change = ratio(np.linalg.norm((perturbed - base) @ dictionary),
                         np.linalg.norm(base @ dictionary))
    return dict(coefficient_relative_change=coeff_change,
                reconstruction_relative_change=recon_change,
                coefficient_amplification=coeff_change / input_change,
                reconstruction_amplification=recon_change / input_change,
                support_jaccard=float(np.count_nonzero(support & changed) / union)
                if union else 1.0)


def verify_inputs(root):
    plan = read(root / "plan.json")
    assert plan["scope"] == "retrospective_development_diagnostic"
    assert plan["prior_results_already_read"] and not plan["new_codings_computed_before_plan"]
    for name, digest in plan["source_inputs_sha256"].items():
        assert file_hash(Path(name)) == digest, name
    for seed in plan["seeds"]:
        old = Path(f"runs/paper-atom-batches-seed{seed}")
        compact = Path(f"runs/paper-atom-coding-seed{seed}")
        integrity, lock, spec, export = [read(p) for p in (
            old / "integrity.json", old / "forecast-lock.json", old / "specification.json",
            compact / "export.json")]
        assert integrity["complete"] and integrity["source_input_output_hashes_verified"]
        assert integrity["locks_restores_verified"] and integrity["original_deadline_verified"]
        assert export["original_plan_sha256"] == file_hash(old / "plan.json")
        assert export["integrity_sha256"] == file_hash(old / "integrity.json")
        assert export["inputs_sha256"] == file_hash(compact / "inputs.npz")
        assert export["cpu_only"] and plan["written_unix"] < export["exported_unix"]
        assert export["source_files_sha256"][str(old / "representations.pt")] == lock["files_sha256"]["representations.pt"]
        atoms = str(Path(spec["source_root"]) / "atoms.pt")
        assert export["source_files_sha256"][atoms] == spec["input_hashes"][atoms]
        manifest = read(old / "manifest.json")
        for name, digest in manifest["files_sha256"].items():
            assert file_hash(old / name) == digest
    return plan


def freeze(root):
    plan = verify_inputs(root)
    assert not (root / "execution-lock.json").exists()
    paths = [Path(__file__), Path("scripts/export_paper_atom_coding.py"),
             Path("tests/test_paper_atom_coding.py")]
    paths += [Path(f"runs/paper-atom-coding-seed{s}") / n
              for s in plan["seeds"] for n in ("inputs.npz", "export.json")]
    # Pin imported analysis helpers as well as the new numerical implementation.
    paths += [Path("src/gradient_interp") / n for n in (
        "paper_atom_batches.py", "paper_atom_batches_analysis.py", "paper_atom_data.py",
        "paper_batch_analysis.py", "study.py")]
    atomic_json(root / "execution-lock.json", dict(
        frozen_unix=time.time(), plan_sha256=file_hash(root / "plan.json"),
        files_sha256={str(p): file_hash(p) for p in paths}, tested_before_new_codings=True))


def behavior(seed, codes, norms, old):
    z = dict(np.load(old / "forecasts.npz"))
    data, spec = read(old / "data.json"), read(old / "specification.json")
    coefficients = codes * norms[:, None]
    pred = spec["rate"] * np.einsum("bmd,qde->bmqe", coefficients, z["effects"][:, :500])
    np.testing.assert_allclose(pred[:, 0], z["predictions"][:, 0], rtol=1e-6, atol=1e-9)
    pred = np.concatenate((pred, z["predictions"][:, 3:]), axis=1)
    evalq = data["probes"][8:]
    baseline = rows_array(read(old / "outcomes/baseline.json")["rows"], evalq, 64)
    actual = np.array([(rows_array(read(old / f"outcomes/batch-{b:02d}-full.json")["rows"],
                                  evalq, 64) - baseline).mean(1) for b in range(16)])
    bw, qw = grouped_weights(np.full(16, 1 / 16), np.full(16, 1 / 16))
    records, pairs = [], []
    for scope, prediction in zip(SCOPES, scopes(pred), strict=True):
        for gi, group in enumerate(GROUPS):
            for mi, method in enumerate(BEHAVIOR_METHODS):
                for ei, endpoint in enumerate(ENDPOINTS):
                    p, y = prediction[:, mi, :, ei], actual[:, :, ei]
                    weight = bw[gi, :, None] * qw[gi, None, :]
                    err = p - y
                    mse = float((weight * err**2).sum())
                    zero = float((weight * y**2).sum())
                    batch_rmse = np.sqrt((err**2 * qw[gi]).sum(1))
                    level = p + z["baseline"][None, 8:, ei]
                    records.append(dict(seed=seed, scope=scope, group=group, method=method,
                                        endpoint=endpoint, rmse=np.sqrt(mse),
                                        nrmse=np.sqrt(ratio(mse, zero)),
                                        mse_difference_vs_zero=mse-zero,
                                        mae=float((weight * abs(err)).sum()),
                                        predicted_mean=float((weight * p).sum()),
                                        actual_mean=float((weight * y).sum()),
                                        worst_batch_rmse=float(batch_rmse[bw[gi] > 0].max()),
                                        outside_probability_fraction=float((weight * ((level < 0) | (level > 1))).sum())
                                        if endpoint in BINARY_ENDPOINTS else np.nan))
        for b in range(16):
            for mi, method in enumerate(BEHAVIOR_METHODS):
                for qi, q in enumerate(evalq):
                    for ei, endpoint in enumerate(ENDPOINTS):
                        p = prediction[b, mi, qi, ei]
                        level = z["baseline"][qi+8, ei] + p
                        pairs.append(dict(seed=seed, scope=scope, batch=b,
                                          style=data["batches"][b]["style"], method=method,
                                          question=q["id"], task=q["task"], endpoint=endpoint,
                                          predicted_change=p, actual_change=actual[b, qi, ei],
                                          predicted_level=level,
                                          outside_probability_range=bool(level < 0 or level > 1)
                                          if endpoint in BINARY_ENDPOINTS else None))
    return records, pairs, pred, actual


def analyze_seed(seed, plan, root, output):
    old = Path(f"runs/paper-atom-batches-seed{seed}")
    arrays = dict(np.load(f"runs/paper-atom-coding-seed{seed}/inputs.npz"))
    dictionary = arrays["dictionary"].astype(np.float64)
    x = arrays["projected"].astype(np.float64) / arrays["projected_norms"]
    norm = arrays["projected_norms"].astype(np.float64)
    original = arrays["sparse_codes"].astype(np.float64)
    coder = Coder(dictionary)
    check = sparse_encode(x, dictionary, algorithm="lasso_lars", alpha=0.1)
    np.testing.assert_allclose(check, original, rtol=0, atol=1e-5)
    codes = coder.solve(x, original=original)
    cards, data = read(old / "atom-cards.json"), read(old / "data.json")
    geometry_rows, losses, stability_rows = [], [], []
    for b in range(16):
        common = dict(seed=seed, batch=b, block=b//4, style=data["batches"][b]["style"])
        errors = []
        for mi, method in enumerate(METHODS):
            c = codes[b, mi]
            g = geometry(x[b] * norm[b, 0], c * norm[b, 0], dictionary)
            row = dict(**common, method=method, **g)
            for label, card in cards.items():
                row[label + "_coefficient"] = c[card["atom_index"]] * norm[b, 0]
                row[label + "_included"] = bool(c[card["atom_index"]] != 0)
            geometry_rows.append(row)
            errors.append(g["relative_sse"])
        lasso, refitted, greedy, dense = errors
        assert lasso >= refitted - 1e-10 and refitted >= dense - 1e-10
        assert greedy >= dense - 1e-10
        assert set(np.flatnonzero(codes[b, 1])) <= set(np.flatnonzero(original[b]))
        assert np.count_nonzero(codes[b, 2]) <= 16
        residual = x[b] - codes[b, 3] @ dictionary
        assert np.max(np.abs(dictionary @ residual)) < 1e-7
        for method, support in ((1, np.flatnonzero(original[b])), (2, np.flatnonzero(codes[b, 2]))):
            residual = x[b] - codes[b, method] @ dictionary
            if len(support):
                assert np.max(np.abs(dictionary[support] @ residual)) < 1e-7
        losses.append(dict(**common, lasso_sse=lasso, shrinkage_loss=lasso-refitted,
                           original_support_gap=refitted-dense, span_residual=dense,
                           omp_sse=greedy))
    st = plan["stability"]
    rng = np.random.default_rng(st["seed"] + seed)
    perturbed_codes = []
    for b in range(16):
        noise = rng.normal(size=(st["repetitions"], x.shape[1]))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        xp = x[b] + st["relative_perturbation"] * noise
        xp /= np.linalg.norm(xp, axis=1, keepdims=True)
        cp = coder.solve(xp, supports=[np.flatnonzero(original[b])] * len(xp))
        perturbed_codes.append(cp)
        for rep in range(len(xp)):
            change = np.linalg.norm(xp[rep] - x[b]) / np.linalg.norm(x[b])
            for mi, method in enumerate(METHODS):
                row = dict(seed=seed, batch=b, block=b//4, style=data["batches"][b]["style"],
                           repetition=rep, method=method, input_relative_change=change,
                           **sensitivity(codes[b, mi], cp[rep, mi], dictionary, change))
                for label, card in cards.items():
                    c = cp[rep, mi, card["atom_index"]]
                    row[label + "_coefficient"] = c * norm[b, 0]
                    row[label + "_included"] = bool(c != 0)
                    row[label + "_sign"] = int(np.sign(c))
                stability_rows.append(row)
        print(json.dumps(dict(seed=seed, stage="sensitivity", completed=b+1, total=16)), flush=True)
    np.savez_compressed(root / f"seed{seed}-perturbation-codes.npz", codes=perturbed_codes)
    forecast, pairs, pred, actual = behavior(seed, codes, norm, old)
    np.savez_compressed(output / f"seed{seed}-codings.npz", codes=codes,
                        magnitude_coefficients=codes * norm[:, None],
                        predictions=pred, actual=actual, singular_values=coder.singular_values)
    audit = dict(seed=seed, dictionary_rank=coder.rank,
                 largest_singular_value=coder.singular_values[0],
                 smallest_singular_value=coder.singular_values[-1],
                 original_lasso_max_abs_recomputation_error=float(abs(check-original).max()),
                 all_orthogonality_support_and_loss_checks_passed=True,
                 zero_named_counts_original={label: int(np.count_nonzero(original[:, c["atom_index"]] == 0))
                                             for label, c in cards.items()},
                 planned_perturbations_per_batch=st["repetitions"])
    return dict(geometry=geometry_rows, loss=losses, stability=stability_rows,
                forecasts=forecast, pairs=pairs), audit


def table(rows, fields):
    def fmt(v):
        if v is None:
            return "undefined"
        if isinstance(v, float):
            return f"{v:.4g}"
        return html.escape(str(v))
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{html.escape(f)}</th>" for f in fields) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td>{fmt(r[f])}</td>" for f in fields) + "</tr>" for r in rows) + "</table></div>"


def report(output, records, audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, seed in zip(axes, (42, 43), strict=True):
        rows = [r for r in records["loss"] if r["seed"] == seed]
        values = np.array([[np.mean([r[k] for r in rows if r["style"] == style])
                            for style in STYLES] for k in ("span_residual", "original_support_gap", "shrinkage_loss")])
        bottom = np.zeros(4)
        for v, label, color in zip(values, ("Outside dictionary span", "Original support gap", "L1 shrinkage"),
                                   ("#888888", "#4383a8", "#e3a24c"), strict=True):
            ax.bar(np.arange(4), v, bottom=bottom, label=label, color=color)
            bottom += v
        omp_values = [np.mean([r["omp_sse"] for r in rows if r["style"] == style]) for style in STYLES]
        ax.plot(np.arange(4), omp_values, "ko", label="OMP16 residual")
        ax.set(title=f"Seed {seed}: mean of four content blocks", xticks=np.arange(4), xticklabels=STYLES)
    axes[0].set_ylabel("Squared residual / squared projected gradient norm")
    axes[1].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"residual-decomposition.{ext}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for si, seed in enumerate((42, 43)):
        for ei, endpoint in enumerate(("answer_values_and_order", "tokens")):
            ax = axes[ei, si]
            for i, scope in enumerate(SCOPES):
                vals = [next(r["nrmse"] for r in records["forecasts"] if r["seed"] == seed and r["scope"] == scope
                             and r["group"] == "all" and r["method"] == m and r["endpoint"] == endpoint) for m in METHODS]
                ax.bar(np.arange(4) + (i-.5)*.35, vals, width=.35, label=scope.replace("_", " "))
            ax.axhline(1, ls="--", color="black", lw=1)
            ax.set(title=f"Seed {seed}: {endpoint.replace('_', ' ')}", ylabel="RMSE / no-change RMSE",
                   xticks=np.arange(4), xticklabels=["Lasso", "Support refit", "OMP16", "Dense span"])
            ax.legend(fontsize=8)
    fig.suptitle("Retrospective point estimates on inspected development outcomes")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"retrospective-forecast.{ext}", dpi=170)
    plt.close(fig)

    summary = []
    for seed in (42, 43):
        for method in METHODS:
            rows = [r for r in records["geometry"] if r["seed"] == seed and r["method"] == method]
            st = [r for r in records["stability"] if r["seed"] == seed and r["method"] == method]
            summary.append(dict(seed=seed, method=method,
                                mean_relative_residual=np.mean([r["relative_residual"] for r in rows]),
                                mean_norm_ratio=np.mean([r["norm_ratio"] for r in rows]),
                                mean_atoms=np.mean([r["coefficient_l0"] for r in rows]),
                                numbered_included=sum(r["numbered_included"] for r in rows),
                                bullets_included=sum(r["bullets_included"] for r in rows),
                                maximum_coefficient_amplification=max(
                                    (r["coefficient_amplification"] for r in st
                                     if r["coefficient_amplification"] is not None), default=None),
                                minimum_support_jaccard=min(r["support_jaccard"] for r in st),
                                undefined_coefficient_changes=sum(r["coefficient_relative_change"] is None for r in st)))
    atomic_json(output / "summary.json", clean(dict(coding=summary, audits=audits)))
    text = """<!doctype html><meta charset='utf-8'><title>Atom coding diagnostic</title>
<style>body{font:16px/1.5 system-ui;max-width:1180px;margin:40px auto;padding:0 24px;color:#17232d}h1,h2{line-height:1.2}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:6px;text-align:right}th{background:#edf1f4}.scroll{overflow:auto}a{color:#12618e}</style>
<h1>Atom coding diagnostic</h1>
<p>This retrospective development diagnostic keeps the existing 500-atom dictionaries and all 16 batch gradients per seed fixed. It separates L1 coefficient shrinkage, sparse support selection and the residual outside the dictionary span. No new model updates or responses were generated. The protocol and tested implementation were frozen before computing new codings.</p>
<p>The original dictionary was learned on normalized EKFAC projections. Every coding here restores the original projected norm. Least-squares support refit uses exactly the original Lasso support; OMP selects at most 16 atoms; dense least squares is a span reference. The dense result is not a small interpretable explanation. Better reconstruction does not validate descriptions or causal selectivity. These errors are in projected coordinates, not physical parameter space.</p>
<h2>Reconstruction and magnitude</h2><img src='residual-decomposition.png'>"""
    text += table(clean(summary), list(summary[0]))
    text += "<h2>Historical behavior predictions</h2><p>These new codings were applied to already inspected outcomes and already locked atom profiles. They are retrospective point estimates, with no new uncertainty or significance claim. Calibration transfer and same-question profiles are both retained. Missing events still produce zero profile effects. No forecast is clipped; probability-range violations are reported. Four content blocks, four task families and two reused adapters do not support broad population claims.</p><img src='retrospective-forecast.png'>"
    rows = [r for r in records["forecasts"] if r["group"] == "all"]
    text += table(rows, ["seed", "scope", "method", "endpoint", "nrmse", "rmse", "predicted_mean", "actual_mean", "outside_probability_fraction"])
    text += "<h2>All batches and sensitivity checks</h2><p>Twenty fixed isotropic perturbations of size 0.0001 are applied to each unit projected gradient and paired across methods. Original norms are preserved. The support-refit sensitivity keeps the original support fixed, so its support stability is by construction and does not measure Lasso selection stability. Every perturbation is retained, including undefined relative changes.</p>"
    text += table(records["geometry"], list(records["geometry"][0]))
    text += "<h2>Complete records</h2><ul>"
    for name in records:
        text += f"<li><a href='{name}.csv'>{name}: every seed, batch, method and applicable endpoint</a></li>"
    text += "<li><a href='plan.json'>Frozen diagnostic plan</a></li><li><a href='execution-lock.json'>Tested source and inputs lock</a></li><li><a href='validation.json'>Validation</a></li></ul>"
    (output / "report.html").write_text(text)


def run(root, output):
    plan = verify_inputs(root)
    lock = read(root / "execution-lock.json")
    assert lock["plan_sha256"] == file_hash(root / "plan.json")
    assert plan["written_unix"] < lock["frozen_unix"]
    for path, digest in lock["files_sha256"].items():
        assert file_hash(Path(path)) == digest, path
    assert not (root / "execution-start.json").exists()
    atomic_json(root / "execution-start.json", dict(started_unix=time.time(), execution_lock_sha256=file_hash(root / "execution-lock.json")))
    output.mkdir(parents=True, exist_ok=True)
    records = {k: [] for k in ("geometry", "loss", "stability", "forecasts", "pairs")}
    audits = []
    for seed in plan["seeds"]:
        r, audit = analyze_seed(seed, plan, root, output)
        for key in records:
            records[key].extend(r[key])
        audits.append(audit)
    records = clean(records)
    for key, rows in records.items():
        write_csv(output / (key + ".csv"), rows)
    report(output, records, clean(audits))
    for name in ("plan.json", "execution-lock.json", "execution-start.json"):
        shutil.copy2(root / name, output / name)
    for seed in plan["seeds"]:
        shutil.copy2(f"runs/paper-atom-coding-seed{seed}/export.json", output / f"seed{seed}-export.json")
    atomic_json(output / "validation.json", clean(dict(
        completed_unix=time.time(), scope=plan["scope"], all_seeds_complete=True,
        rows={k: len(v) for k, v in records.items()}, audits=audits,
        files_sha256={p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
        ignored_perturbation_codes_sha256={p.name: file_hash(p) for p in root.glob("seed*-perturbation-codes.npz")})))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, default=Path("runs/paper-atom-coding-analysis"))
    parser.add_argument("--output", type=Path, default=Path("results/paper-atom-coding"))
    args = parser.parse_args()
    freeze(args.root) if args.action == "freeze" else run(args.root, args.output)


if __name__ == "__main__":
    main()
