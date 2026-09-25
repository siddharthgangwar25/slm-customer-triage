"""CPU experiment, policy, and local API commands. Paths resolve from the working directory."""

import argparse
import json
import os
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
    policy = commands.add_parser("policy").add_subparsers(dest="action", required=True)
    select = policy.add_parser("select")
    select.add_argument("--predictions", type=Path, required=True)
    select.add_argument("--split", choices=["val"], required=True)
    select.add_argument("--output", type=Path, default=Path("artifacts/policy-v1"))
    serve = commands.add_parser("serve")
    serve.add_argument("--bundle", type=Path, required=True)
    serve.add_argument("--policy", type=Path, help="Defaults to policy.json inside the bundle")
    serve.add_argument("--config", type=Path, default=Path("configs/service.yaml"))
    serve.add_argument("--host", choices=["127.0.0.1", "::1"], default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
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
        elif args.command == "policy":
            from triage.evaluation.select_policy import select_policy

            result = select_policy(args.predictions, args.split, args.output)
        elif args.command == "serve":
            import uvicorn

            from triage.service.app import create_app
            from triage.service.schemas import ServiceSettings

            settings = ServiceSettings(**config(args.config))
            app = create_app(
                args.bundle,
                args.policy or args.bundle / "policy.json",
                settings=settings,
                api_key=os.environ.get("TRIAGE_API_KEY"),
            )
            uvicorn.run(app, host=args.host, port=args.port, access_log=False)
            return 0
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
