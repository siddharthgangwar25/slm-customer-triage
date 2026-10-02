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
    output = args.output.resolve()
    relative = output.relative_to(root).as_posix()
    output.mkdir(parents=True, exist_ok=False)
    uv = args.uv or shutil.which("uv") or str(root / ".tools/uv.exe")
    env = {
        **os.environ,
        "UV_PROJECT_ENVIRONMENT": str(output / "environment"),
        "UV_CACHE_DIR": str(root / ".uv-cache"),
    }
    commands = []

    def run(name, command):
        started = time.perf_counter()
        print(f"{name}: running; log {output / (name + '.log')}", flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append(
            {
                "stage": name,
                "argv": command,
                "exit_code": proc.returncode,
                "seconds": time.perf_counter() - started,
            }
        )
        (output / "commands.json").write_text(
            json.dumps(commands, indent=2) + "\n", encoding="utf-8"
        )
        if proc.returncode:
            raise RuntimeError(f"{name} failed; inspect its log")

    run("sync", [uv, "sync", "--locked", "--python", sys.executable])
    python = str(
        output / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    prepare = [
        python,
        "-m",
        "triage.cli",
        "data",
        "prepare",
        "--config",
        "configs/data.yaml",
        "--output",
        relative + "/data",
    ]
    if args.source_dir:
        prepare += ["--source-dir", str(args.source_dir.resolve())]
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
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    run("train", [python, "-m", "triage.cli", "train", "baseline", "--config", str(config_path)])
    run(
        "predict",
        [python, "-m", "triage.cli", "predict", "--config", str(config_path), "--split", "val"],
    )
    predictions = relative + "/predictions/predictions.jsonl"
    run(
        "evaluate",
        [
            python,
            "-m",
            "triage.cli",
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
            python,
            "-m",
            "triage.cli",
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
    main()
