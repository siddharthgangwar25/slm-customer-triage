"""Milestone 1 command contracts. Paths are relative to the current working directory."""

import argparse
import json
import sys
from pathlib import Path

from triage.io import config


def parser():
    root = argparse.ArgumentParser(prog="triage")
    commands = root.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data").add_subparsers(dest="action", required=True)
    prepare = data.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--source-dir", type=Path, help="Use checksum-verified local source files")
    train = commands.add_parser("train").add_subparsers(dest="action", required=True)
    baseline = train.add_parser("baseline")
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--output", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--split", choices=["val"], required=True)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--bundle", type=Path, help="Trusted project-controlled model bundle")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "data":
            from triage.data.prepare import prepare

            cfg = config(args.config)
            result = prepare(cfg, args.output or Path(cfg["output"]), args.source_dir)
            result = {
                "commit": result["commit"],
                "splits": {split: value["count"] for split, value in result["splits"].items()},
                "duplicate_summary": result["duplicate_summary"],
            }
        elif args.command == "train":
            from triage.models.baseline import train

            cfg = config(args.config)
            result = train(cfg, args.output or Path(cfg["bundle"]))
            result = {"model_version": result["model_version"], "fit": result["fit"]}
        elif args.command == "predict":
            from triage.models.baseline import predict

            cfg = config(args.config)
            result = predict(
                cfg, args.split, args.output or Path(cfg["prediction_output"]), args.bundle
            )
        else:
            from triage.evaluation.report import evaluate

            result = evaluate(args.predictions, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"triage: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
