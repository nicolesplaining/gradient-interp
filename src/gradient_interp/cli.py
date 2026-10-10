import argparse
import json
from pathlib import Path

from .config import load_config
from .data import make_demo_data
from .experiment import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict and test the effects of batch SGD updates"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "make-demo-data", help="Write a small synthetic pipeline-check dataset"
    )
    demo.add_argument("--output", required=True)
    run = commands.add_parser("run", help="Fit, calibrate, forecast, and evaluate held-out updates")
    run.add_argument("--config", required=True, help="TOML experiment configuration")
    run.add_argument("--output", required=True, help="New or empty result directory")
    predict = commands.add_parser("predict", help="Forecast new batches from a saved calibration")
    predict.add_argument("--run", required=True)
    predict.add_argument("--batches", required=True, help="Batch JSONL file")
    predict.add_argument("--output", required=True, help="New prediction JSONL file")
    predict.add_argument("--learning-rate", type=float)
    predict.add_argument("--device", help="Override model device, e.g. cpu or cuda")
    residual = commands.add_parser("residual-sae", help="Compare residual-gradient/activation SAEs")
    residual.add_argument("--config", required=True)
    residual.add_argument("--output", required=True)
    residual.add_argument("--module", required=True, help="Residual module name or unique suffix")
    residual.add_argument("--chunk-size", type=int, default=32)
    residual.add_argument("--features", type=int, default=128)
    residual.add_argument("--k", type=int, default=8)
    residual.add_argument("--epochs", type=int, default=100)
    export = commands.add_parser(
        "export-loracle", help="Export effective LoRA batch-delta SVD tokens"
    )
    export.add_argument("--config", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--batch-id", required=True)
    export.add_argument("--learning-rate", type=float, required=True)
    export.add_argument("--top-k", type=int, default=4)
    loracle = commands.add_parser("loracle", help="Call a compatible upstream LoRAcle checkpoint")
    loracle.add_argument("--export", required=True, help="Directory from export-loracle")
    loracle.add_argument("--upstream", required=True, help="Local ceselder/loracles checkout")
    loracle.add_argument("--checkpoint", required=True)
    loracle.add_argument("--output", required=True)
    loracle.add_argument("--device", default="cuda")
    loracle.add_argument("--max-new-tokens", type=int, default=256)
    loracle.add_argument("--question", default="What behavioral changes does this update encode?")
    prepare = commands.add_parser("prepare-loracle", help="Adapt a public LoRAcle step checkpoint")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--base-model", required=True)
    benchmark = commands.add_parser(
        "prepare-study-data", help="Freeze source-disjoint No Robots splits"
    )
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--fit-count", type=int, default=5000)
    study = commands.add_parser("study-job", help="Run or resume one preregistered study job")
    study.add_argument("--plan", required=True)
    study.add_argument("--job-index", required=True, type=int)
    study.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-study-data":
            from .benchmark import prepare_benchmark

            print(json.dumps(prepare_benchmark(args.output, args.fit_count), indent=2))
        elif args.command == "study-job":
            from .study import run_study_job

            plan = json.loads(Path(args.plan).read_text())
            run_study_job(
                plan,
                plan["jobs"][args.job_index],
                args.output,
                progress=lambda message: print(message, flush=True),
            )
        elif args.command == "make-demo-data":
            make_demo_data(args.output)
        elif args.command == "predict":
            from .predict import predict_batches

            results = predict_batches(
                args.run, args.batches, args.output, args.learning_rate, args.device
            )
            print(f"Saved {len(results)} forecasts without applying any batch updates")
        elif args.command == "residual-sae":
            from .residual import run_residual_comparison

            metrics = run_residual_comparison(
                load_config(args.config),
                args.output,
                args.module,
                args.chunk_size,
                args.features,
                args.k,
                args.epochs,
            )
            for name, result in metrics.items():
                print(f"{name}: raw FVU={result['raw_fvu']:.4g}, mean L0={result['mean_l0']:.3g}")
        elif args.command == "export-loracle":
            from .loracle import export_batch_tokens

            tokens = export_batch_tokens(
                load_config(args.config), args.output, args.batch_id, args.learning_rate, args.top_k
            )
            print(f"Saved direction tokens with shape {list(tokens.shape)}")
        elif args.command == "loracle":
            from .loracle import interpret_tokens

            print(
                interpret_tokens(
                    args.export,
                    args.upstream,
                    args.checkpoint,
                    args.output,
                    args.device,
                    args.question,
                    args.max_new_tokens,
                )
            )
        elif args.command == "prepare-loracle":
            from .loracle import prepare_checkpoint

            prepare_checkpoint(args.source, args.output, args.base_model)
            print(f"Prepared interpreter checkpoint at {args.output}")
        else:
            summary = run_experiment(
                load_config(args.config),
                args.output,
                progress=lambda message: print(message, flush=True),
            )
            print("\nlearning rate  dictionary rmse  first-order rmse  atom-removal rmse")
            for rate, metrics in summary.items():
                print(
                    f"{float(rate):13g}  {metrics['dictionary']['rmse']:15.6g}  "
                    f"{metrics['full_gradient']['rmse']:16.6g}  "
                    f"{metrics['atom_removal']['rmse']:17.6g}"
                )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
