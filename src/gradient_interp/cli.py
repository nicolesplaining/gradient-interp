import argparse

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
    args = parser.parse_args(argv)
    try:
        if args.command == "make-demo-data":
            make_demo_data(args.output)
        else:
            summary = run_experiment(load_config(args.config), args.output)
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
