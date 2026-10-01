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
    sft = data.add_parser("sft")
    sft.add_argument("--config", type=Path, required=True)
    sft.add_argument("--output", type=Path)
    train = commands.add_parser("train").add_subparsers(dest="action", required=True)
    baseline = train.add_parser("baseline")
    baseline.add_argument("--config", type=Path, required=True)
    baseline.add_argument("--output", type=Path)
    slm = train.add_parser("slm")
    slm.add_argument("--config", type=Path, required=True)
    slm.add_argument("--output", type=Path)
    slm.add_argument("--smoke", action="store_true")
    slm.add_argument("--smoke-evidence", type=Path)
    slm.add_argument("--resume-checkpoint", type=Path)
    checkpoints = commands.add_parser("checkpoint").add_subparsers(dest="action", required=True)
    checkpoint = checkpoints.add_parser("select")
    checkpoint.add_argument("--config", type=Path, required=True)
    checkpoint.add_argument("--output", type=Path, required=True)
    checkpoint.add_argument("--resume", action="store_true")
    release = commands.add_parser("release").add_subparsers(dest="action", required=True)
    freeze = release.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    activate = release.add_parser("activate")
    activate.add_argument("--release", type=Path, required=True)
    activate.add_argument("--pointer", type=Path, default=Path("artifacts/active-release.json"))
    rollback = release.add_parser("rollback")
    rollback.add_argument("--pointer", type=Path, default=Path("artifacts/active-release.json"))
    benchmark = commands.add_parser("benchmark").add_subparsers(dest="action", required=True)
    final = benchmark.add_parser("final")
    final.add_argument("--release", type=Path, required=True)
    final.add_argument("--output", type=Path)
    final.add_argument("--resume", action="store_true")
    predict = commands.add_parser("predict")
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--split", choices=["val"], required=True)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--bundle", type=Path, help="Trusted project-controlled model bundle")
    predict.add_argument("--limit", type=int, help="Prompted hardware smoke only; not a benchmark")
    predict.add_argument(
        "--resume", action="store_true", help="Resume an unchanged partial prompted run"
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--finetuned", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    policy = commands.add_parser("policy").add_subparsers(dest="action", required=True)
    select = policy.add_parser("select")
    select.add_argument("--predictions", type=Path, required=True)
    select.add_argument("--split", choices=["val"], required=True)
    select.add_argument("--output", type=Path, default=Path("artifacts/policy-v1"))
    serve = commands.add_parser("serve")
    serve.add_argument("--bundle", type=Path)
    serve.add_argument("--active", type=Path, help="Atomic active-release pointer")
    serve.add_argument("--policy", type=Path, help="Defaults to policy.json inside the bundle")
    serve.add_argument("--config", type=Path, default=Path("configs/service.yaml"))
    serve.add_argument("--host", choices=["127.0.0.1", "::1", "0.0.0.0"], default="127.0.0.1")
    serve.add_argument("--worker-url", help="Private authenticated SLM worker URL")
    serve.add_argument("--port", type=int, default=8000)
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "release":
            from triage.release import activate, freeze, rollback

            if args.action == "activate":
                result = activate(args.release, args.pointer)
            elif args.action == "rollback":
                result = rollback(args.pointer)
            else:
                result = freeze(config(args.config), args.output)
        elif args.command == "benchmark":
            from datetime import UTC, datetime

            from triage.evaluation.final import run
            from triage.io import read_json

            output = args.output
            if output is None:
                if args.resume:
                    output = Path(read_json(args.release.parent / "test_use.json")["output"])
                else:
                    output = Path("artifacts") / (
                        "final-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
                    )
            result = run(args.release, output, resume=args.resume)
        elif args.command == "data" and args.action == "sft":
            from triage.training.data import prepare

            cfg = config(args.config)
            result = prepare(cfg, args.output or Path(cfg["prepared_data"]))
        elif args.command == "data":
            from triage.data.prepare import prepare

            cfg = config(args.config)
            result = prepare(cfg, args.output or Path(cfg["output"]), args.source_dir)
            result = {
                "commit": result["commit"],
                "splits": {split: value["count"] for split, value in result["splits"].items()},
                "duplicate_summary": result["duplicate_summary"],
            }
        elif args.command == "train" and args.action == "slm":
            from triage.training.runner import run

            cfg = config(args.config)
            output = args.output or Path(cfg["output"] + ("-smoke" if args.smoke else ""))
            result = run(
                cfg,
                output,
                smoke=args.smoke,
                smoke_evidence=args.smoke_evidence,
                resume_checkpoint=args.resume_checkpoint,
            )
        elif args.command == "checkpoint":
            from triage.training.select import select

            result = select(config(args.config), args.output, resume=args.resume)
        elif args.command == "train":
            from triage.models.baseline import train

            cfg = config(args.config)
            result = train(cfg, args.output or Path(cfg["bundle"]))
            result = {"model_version": result["model_version"], "fit": result["fit"]}
        elif args.command == "predict":
            cfg = config(args.config)
            if cfg.get("model_type") == "finetuned":
                from triage.models.finetuned import run

                result = run(
                    cfg,
                    args.split,
                    args.output or Path(cfg["prediction_output"]),
                    bundle=args.bundle,
                    limit=args.limit,
                    resume=args.resume,
                )
            elif cfg.get("model_type") == "prompted":
                from triage.models.prompted import run

                result = run(
                    cfg,
                    args.split,
                    args.output or Path(cfg["prediction_output"]),
                    limit=args.limit,
                    resume=args.resume,
                )
            else:
                from triage.models.baseline import predict

                if args.limit is not None or args.resume:
                    raise ValueError("Smoke/resume flags apply only to prompted experiments")
                result = predict(
                    cfg, args.split, args.output or Path(cfg["prediction_output"]), args.bundle
                )
        elif args.command == "compare":
            if args.finetuned:
                from triage.evaluation.three_way import compare

                result = compare(args.baseline, args.candidate, args.finetuned, args.output)
            else:
                from triage.evaluation.compare import compare

                result = compare(args.baseline, args.candidate, args.output)
        elif args.command == "policy":
            from triage.evaluation.select_policy import select_policy

            result = select_policy(args.predictions, args.split, args.output)
        elif args.command == "serve":
            import logging

            import uvicorn

            from triage.service.app import create_app
            from triage.service.schemas import ServiceSettings

            settings = ServiceSettings(**config(args.config))
            logging.basicConfig(level=logging.INFO, format="%(message)s")
            if args.active:
                from triage.release import active_bundle

                args.bundle, args.policy, requires_worker = active_bundle(args.active)
                if requires_worker and not args.worker_url:
                    raise ValueError("Active SLM release requires its private worker URL")
                if not requires_worker:
                    args.worker_url = None
            if args.bundle is None:
                raise ValueError("Provide --bundle or --active")
            app = create_app(
                args.bundle,
                args.policy or args.bundle / "policy.json",
                settings=settings,
                api_key=os.environ.get("TRIAGE_API_KEY"),
                metrics_key=os.environ.get("TRIAGE_METRICS_KEY"),
                worker_url=args.worker_url,
                worker_key=os.environ.get("TRIAGE_WORKER_KEY"),
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
