"""Owned local container checks on fixtures/validation only; never run the final test set."""

import argparse
import json
import os
import secrets
import subprocess
import time
from collections import Counter
from pathlib import Path

from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import load_policy
from triage.evaluation.serving import http_call, load_test
from triage.io import new_directory, read_json, sha256, write_json
from triage.policy import ModelOutput
from triage.release import source_hash


def docker(*args, env=None, capture=True):
    return subprocess.run(
        ["docker", *map(str, args)],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def ready(url, key, timeout=180):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if http_call(url + "/health/ready", key, timeout=2)[0] == 200:
            return
        time.sleep(1)
    raise RuntimeError("Container readiness timed out; inspect the saved container log")


def contracts(url, env):
    api = env["TRIAGE_API_KEY"]
    checks = {}
    for name, path, key, body, expected in (
        ("live", "/health/live", "", None, 200),
        ("ready", "/health/ready", "", None, 200),
        ("model", "/v1/model", api, None, 200),
        ("unauthenticated", "/v1/triage", "", {"text": "pay bill"}, 401),
        ("empty_input", "/v1/triage", api, {"text": " "}, 422),
        ("oversized_text", "/v1/triage", api, {"text": "x" * 2001}, 422),
        ("metrics_unauthenticated", "/metrics", "", None, 401),
        ("metrics_separate_key", "/metrics", api, None, 401),
        ("metrics_authenticated", "/metrics", env["TRIAGE_METRICS_KEY"], None, 200),
    ):
        status, _, _ = http_call(url + path, key, body)
        if status != expected:
            raise ValueError(f"{name}: expected HTTP {expected}, received {status}")
        checks[name] = status
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True, help="Fresh evidence directory")
    parser.add_argument("--full", action="store_true", help="GPU: all 3100 parity + 500/load level")
    parser.add_argument("--skip-build", action="store_true", help="Use existing image tags")
    parser.add_argument("--cpu-image", default="triage-m5-cpu:82ed8d7")
    parser.add_argument("--gpu-image", default="triage-m5-gpu:82ed8d7")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = new_directory(args.output.resolve())
    env = {
        **os.environ,
        **{
            key: secrets.token_hex(24)
            for key in ("TRIAGE_API_KEY", "TRIAGE_WORKER_KEY", "TRIAGE_METRICS_KEY")
        },
    }
    if docker("info", "--format", "{{.OSType}}").strip() != "linux":
        raise ValueError("Docker must use Linux containers")
    images = {"cpu": args.cpu_image}
    if args.stage == "gpu":
        images["gpu"] = args.gpu_image
    for kind, tag in images.items():
        if not args.skip_build:
            with (output / f"build-{kind}.log").open("w", encoding="utf-8") as log:
                print(f"Building {kind}; progress log: {log.name}", flush=True)
                subprocess.run(
                    [
                        "docker",
                        "build",
                        "--progress",
                        "plain",
                        "-f",
                        f"deployment/Dockerfile.{kind}",
                        "-t",
                        tag,
                        ".",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    identity = {
        "source_sha256": source_hash(),
        "images": {
            k: json.loads(docker("image", "inspect", tag))[0]["Id"] for k, tag in images.items()
        },
        "docker": json.loads(docker("version", "--format", "{{json .}}")),
        "dockerfiles_sha256": {k: sha256(Path(f"deployment/Dockerfile.{k}")) for k in images},
        "stage": args.stage,
        "test_inference": False,
    }
    write_json(output / "identity.json", identity)
    token = secrets.token_hex(5)
    network = "triage-validation-" + token
    frontend = network + "-frontend"
    owned = []
    docker("network", "create", "--internal", network)
    try:
        docker("network", "create", frontend)
    except BaseException:
        docker("network", "rm", network)
        raise
    common = [
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
    ]
    mounts = [
        "-v",
        f"{root / 'artifacts'}:/app/artifacts:ro",
        "-v",
        f"{root / 'reports'}:/app/reports:ro",
    ]

    def start(name, image, extra, command, *, gpu=False):
        flags = ["run", "-d", "--name", name, *common, *mounts, *extra]
        flags += ["--network", network if gpu else frontend]
        if not gpu and args.stage == "gpu":
            flags += ["--network", network]
        if gpu:
            flags += ["--gpus", "all"]
        for key in ("TRIAGE_API_KEY", "TRIAGE_WORKER_KEY", "TRIAGE_METRICS_KEY"):
            flags += ["-e", key]
        docker(*flags, image, *command, env=env)
        owned.append(name)
        return name

    def endpoint(name):
        address = docker("port", name, "8000/tcp").strip()
        return "http://" + address

    try:
        checks = {}
        if args.stage == "cpu":
            fixture_parent = new_directory(output / "fixture")
            fixture = fixture_parent / "generated"
            # Fit/select this synthetic CI fixture in the runtime being checked. Genuine
            # baseline artifacts remain the unchanged Windows-trained reference below.
            docker(
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp",
                "-e",
                "MPLCONFIGDIR=/tmp/matplotlib",
                "-v",
                f"{fixture_parent}:/fixture",
                "-v",
                f"{root / 'scripts/create_container_fixture.py'}:/app/create_fixture.py:ro",
                "--entrypoint",
                "python",
                args.cpu_image,
                "/app/create_fixture.py",
                "--output",
                "/fixture/generated",
            )
            for kind, bundle, policy_path, reference_path, extra in (
                (
                    "fixture",
                    "fixture/bundle",
                    "fixture/policy/policy.json",
                    fixture / "predictions/predictions.jsonl",
                    ["-v", f"{fixture}:/app/fixture:ro"],
                ),
                (
                    "baseline",
                    "artifacts/baseline-c1-v1",
                    "reports/baseline-c1-v1-policy/policy.json",
                    Path("reports/baseline-c1-v1-val/predictions.jsonl"),
                    [],
                ),
            ):
                name = start(
                    f"{network}-{kind}",
                    args.cpu_image,
                    ["-p", "127.0.0.1::8000", *extra],
                    [
                        "--bundle",
                        bundle,
                        "--policy",
                        policy_path,
                        "--config",
                        "configs/service-slm.yaml",
                    ],
                )
                url = endpoint(name)
                ready(url, env["TRIAGE_API_KEY"])
                result = contracts(url, env)
                rows, catalog, meta = load_predictions(reference_path)
                policy_file = (
                    fixture / "policy/policy.json" if kind == "fixture" else Path(policy_path)
                )
                policy, _ = load_policy(policy_file)
                counts = Counter()
                mismatches = []
                for i, row in enumerate(rows, 1):
                    status, body, _ = http_call(
                        url + "/v1/triage", env["TRIAGE_API_KEY"], {"text": row["text"]}
                    )
                    expected = policy.decide(
                        row["gate_score"],
                        ModelOutput(row["predicted_label"], row["parse_status"] == "ok"),
                        catalog,
                    )
                    if status != 200 or any(
                        body.get(k) != getattr(expected, k)
                        for k in ("intent", "decision", "reason")
                    ):
                        mismatches.append(
                            {
                                "sample_id": row["sample_id"],
                                "status": status,
                                "expected": {
                                    k: getattr(expected, k)
                                    for k in ("intent", "decision", "reason")
                                },
                                "actual": body,
                                "reference_gate_score": row["gate_score"],
                            }
                        )
                    counts[body.get("decision", "failed")] += 1
                    if i % 500 == 0:
                        print({"container": kind, "completed": i, "total": len(rows)}, flush=True)
                checks[kind] = {
                    "fixture": meta["fixture"],
                    "sample_count": len(rows),
                    "mismatches": len(mismatches),
                    "mismatch_records": mismatches,
                    "outcomes": dict(counts),
                    "contracts": result,
                }
            gpu = docker(
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "nvidia-smi",
                args.cpu_image,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            )
            checks["gpu_visibility"] = gpu.strip()
        else:
            reference = "reports/finetuned-qwen3-06b-qlora-v1-val/predictions.jsonl"
            policy_path = Path("reports/finetuned-qwen3-06b-qlora-v1-policy/policy.json")
            started = time.perf_counter()
            worker = start(
                network + "-worker",
                args.gpu_image,
                [
                    "-v",
                    f"{output}:/evidence",
                    "-e",
                    "HF_HOME=/tmp/huggingface",
                    "-e",
                    "MPLCONFIGDIR=/tmp/matplotlib",
                    "-e",
                    "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor",
                    "-e",
                    "TRITON_CACHE_DIR=/tmp/triton",
                ],
                ["--predictions", reference],
                gpu=True,
            )
            # Load the GPU model before the gateway verifies the worker identity at startup.
            worker_ready = """
import os, time, urllib.request
request = urllib.request.Request(
    'http://127.0.0.1:8001/health/ready',
    headers={'Authorization': 'Bearer ' + os.environ['TRIAGE_WORKER_KEY']},
)
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError('GPU worker readiness timed out')
"""
            docker("exec", worker, "python", "-c", worker_ready, capture=False)
            gateway = start(
                network + "-gateway",
                args.cpu_image,
                ["-p", "127.0.0.1::8000"],
                [
                    "--bundle",
                    "artifacts/baseline-c1-v1",
                    "--policy",
                    policy_path.as_posix(),
                    "--config",
                    "configs/service-slm.yaml",
                    "--worker-url",
                    f"http://{worker}:8001",
                ],
            )
            url = endpoint(gateway)
            ready(url, env["TRIAGE_API_KEY"], timeout=300)
            cold = time.perf_counter() - started
            checks["contracts"] = contracts(url, env)
            code = (
                "import os; from pathlib import Path; "
                "from triage.evaluation.serving import parity; "
                f"parity(Path({reference!r}), 'http://127.0.0.1:8001', "
                "os.environ['TRIAGE_WORKER_KEY'], "
                f"Path('/evidence/parity'), limit={None if args.full else 6!r})"
            )
            docker("exec", worker, "python", "-c", code, capture=False)
            checks["load"] = load_test(
                Path(reference),
                policy_path,
                url,
                env["TRIAGE_API_KEY"],
                env["TRIAGE_METRICS_KEY"],
                output / "load",
                count=500 if args.full else 24,
                cold_start_seconds=cold,
            )
            checks["parity"] = read_json(output / "parity/parity.json")
            checks["full"] = args.full
        passed = all(
            value.get("mismatches", 0) == 0 for value in checks.values() if isinstance(value, dict)
        )
        if args.stage == "gpu":
            runs = checks["load"]["runs"]
            serial = next(r for r in runs if r["concurrency"] == 1)
            passed = passed and not any(r["decision_mismatches"] for r in runs)
            passed = passed and serial["completed"] == serial["submitted"]
        write_json(output / "result.json", {**identity, "passed": passed, "checks": checks})
        print({"passed": passed, "output": str(output)}, flush=True)
        if not passed:
            raise ValueError("Container validation failed; inspect retained mismatch records")
    finally:
        for name in reversed(owned):
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
            (output / (name + ".log")).write_text(logs.stdout + logs.stderr, encoding="utf-8")
            # Only remove containers created by this invocation; never prune unrelated Docker state.
            subprocess.run(["docker", "rm", "-f", name], check=True, stdout=subprocess.DEVNULL)
        docker("network", "rm", network)
        docker("network", "rm", frontend)


if __name__ == "__main__":
    main()
