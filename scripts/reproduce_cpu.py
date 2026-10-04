"""Fresh locked CPU environment, pinned data, baseline, validation and live demo."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="Optional pinned offline CLINC source files"
    )
    parser.add_argument("--uv", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11 for the locked CPU environment")
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error("--output must be inside the project directory")
    if output.exists():
        parser.error(f"Output already exists: {output}. Choose a new directory")
    uv = shutil.which(args.uv or "uv")
    if uv is None and args.uv is None:
        local_uv = root / ".tools" / ("uv.exe" if os.name == "nt" else "uv")
        if local_uv.is_file():
            uv = str(local_uv)
    if uv is None:
        parser.error("uv executable not found; install uv or pass --uv with its executable path")
    source_dir = args.source_dir.resolve() if args.source_dir else None
    if source_dir:
        missing = [
            name
            for name in ("data_full.json", "domains.json", "LICENSE")
            if not (source_dir / name).is_file()
        ]
        if missing:
            parser.error(f"--source-dir is missing: {', '.join(missing)}")
    relative = output.relative_to(root).as_posix()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(output / "environment"),
        "UV_CACHE_DIR": str(root / ".uv-cache"),
    }
    commands = []

    def run(name, command):
        started = time.perf_counter()
        log_path = output / f"{name}.log"
        print(f"{name}: running; log {log_path}", flush=True)
        record = {"stage": name, "argv": command, "exit_code": None}
        with log_path.open("w", encoding="utf-8") as log:
            try:
                proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["exit_code"] = proc.returncode
            except OSError as exc:
                record["error_type"] = type(exc).__name__
                log.write(f"Could not start {name}: {exc}\n")
        record["seconds"] = time.perf_counter() - started
        commands.append(record)
        (output / "commands.json").write_text(
            json.dumps(commands, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        if record["exit_code"] != 0:
            raise RuntimeError(f"{name} failed; inspect {log_path}")

    run("sync", [uv, "sync", "--locked", "--python", sys.executable])
    python = str(
        output / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    cli = [python, "-m", "triage.cli"]
    prepare = [
        *cli,
        "data",
        "prepare",
        "--config",
        "configs/data.yaml",
        "--output",
        relative + "/data",
    ]
    if source_dir:
        prepare += ["--source-dir", str(source_dir)]
    run("prepare", prepare)
    # JSON is also YAML. No dependency is needed in this bootstrap process.
    config = {
        "experiment_id": "milestone6-cpu-reproduction",
        "data_dir": relative + "/data",
        "bundle": relative + "/bundle",
        "prediction_output": relative + "/predictions",
        "C": 1.0,
        "seed": 42,
        "max_iter": 2000,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "threads": 1,
    }
    config_path = output / "baseline.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    run("train", [*cli, "train", "baseline", "--config", str(config_path)])
    run(
        "predict",
        [*cli, "predict", "--config", str(config_path), "--split", "val"],
    )
    predictions = relative + "/predictions/predictions.jsonl"
    run(
        "evaluate",
        [
            *cli,
            "evaluate",
            "--predictions",
            predictions,
            "--output",
            relative + "/evaluation",
        ],
    )
    run(
        "policy",
        [
            *cli,
            "policy",
            "select",
            "--predictions",
            predictions,
            "--split",
            "val",
            "--output",
            relative + "/policy",
        ],
    )
    run(
        "api-parity",
        [
            python,
            "scripts/verify_api.py",
            "--bundle",
            relative + "/bundle",
            "--policy",
            relative + "/policy/policy.json",
            "--predictions",
            predictions,
            "--output",
            relative + "/api-parity",
        ],
    )
    run(
        "demo",
        [
            python,
            "scripts/run_demo.py",
            "--bundle",
            relative + "/bundle",
            "--policy",
            relative + "/policy/policy.json",
            "--output",
            relative + "/demo",
        ],
    )
    print(f"CPU reproduction completed: {relative}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"reproduce_cpu: {exc}", file=sys.stderr)
        sys.exit(1)
