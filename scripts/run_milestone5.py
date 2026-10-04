"""Local M5 terminal workflow. Owns and cleans up its worker/gateway processes."""

import argparse
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from triage.evaluation.serving import http_call, load_test, parity
from triage.io import config, read_json, write_json
from triage.release import freeze


def wait_ready(url, key, process, timeout=180):
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError("Child process stopped; inspect its log")
        if http_call(url + "/health/ready", key, timeout=1)[0] == 200:
            return time.perf_counter() - started
        time.sleep(1)
    raise RuntimeError("Timed out waiting for local readiness")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["smoke", "serving", "freeze", "final", "all"], default="all"
    )
    parser.add_argument("--smoke-id", default="smoke", help="Fresh smoke evidence name")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    base = Path("artifacts/milestone5")
    base.mkdir(parents=True, exist_ok=True)
    cfg = config(Path("configs/release.yaml"))
    predictions = Path(cfg["candidates"]["finetuned"]["predictions"])
    policy = Path(cfg["candidates"]["finetuned"]["policy"])
    python = root / "environments/training/.venv/Scripts/python.exe"
    if os.name != "nt":
        python = root / "environments/training/.venv/bin/python"
    if args.stage in ("smoke", "serving", "all"):
        from triage.evaluation.resources import GPUProbe

        smoke = args.stage == "smoke"
        stem = args.smoke_id + "-" if smoke else ""
        env = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "TRIAGE_API_KEY": secrets.token_hex(24),
            "TRIAGE_WORKER_KEY": secrets.token_hex(24),
            "TRIAGE_METRICS_KEY": secrets.token_hex(24),
        }
        children, logs = [], []
        probe = GPUProbe()
        # Reserve no persistent service and never stop unrelated processes on these ports.
        import socket

        for port in (8010, 8011):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        try:
            probe.start()
            started = time.perf_counter()
            worker_log = (base / (stem + "worker.log")).open("a", encoding="utf-8")
            logs.append(worker_log)
            worker = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "triage.service.worker",
                    "--predictions",
                    str(predictions),
                    "--port",
                    "8011",
                ],
                env=env,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
            )
            children.append(worker)
            wait_ready("http://127.0.0.1:8011", env["TRIAGE_WORKER_KEY"], worker)
            gateway_log = (base / (stem + "gateway.log")).open("a", encoding="utf-8")
            logs.append(gateway_log)
            gateway = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "triage.cli",
                    "serve",
                    "--bundle",
                    cfg["baseline_bundle"],
                    "--policy",
                    str(policy),
                    "--config",
                    cfg["service_config"],
                    "--worker-url",
                    "http://127.0.0.1:8011",
                    "--port",
                    "8010",
                ],
                env=env,
                stdout=gateway_log,
                stderr=subprocess.STDOUT,
            )
            children.append(gateway)
            wait_ready("http://127.0.0.1:8010", env["TRIAGE_API_KEY"], gateway)
            cold = time.perf_counter() - started
            parity_dir = base / (stem + "parity")
            if not (parity_dir / "parity.json").exists():
                parity(
                    predictions,
                    "http://127.0.0.1:8011",
                    env["TRIAGE_WORKER_KEY"],
                    parity_dir,
                    resume=parity_dir.exists(),
                    limit=6 if smoke else None,
                )
            load_dir = base / (stem + "load")
            if not (load_dir / "load.json").exists():
                load_test(
                    predictions,
                    policy,
                    "http://127.0.0.1:8010",
                    env["TRIAGE_API_KEY"],
                    env["TRIAGE_METRICS_KEY"],
                    load_dir,
                    count=24 if smoke else 500,
                    cold_start_seconds=cold,
                )
            from triage.evaluation.cost import report as cost_report

            cost_report(load_dir / "load.json", base / (stem + "cost.json"))
            from triage.service.remote import call_worker

            write_json(
                base / (stem + "worker_resources.json"),
                call_worker("http://127.0.0.1:8011", env["TRIAGE_WORKER_KEY"], "/health/ready"),
            )
        finally:
            write_json(base / (stem + "gpu_samples.json"), probe.finish())
            for process in reversed(children):
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=15)
            for stream in logs:
                stream.close()
    release_path = base / "release/release.json"
    if args.stage in ("freeze", "all") and not release_path.exists():
        print(freeze(cfg, release_path.parent), flush=True)
    if args.stage in ("final", "all"):
        ledger = release_path.parent / "test_use.json"
        if ledger.exists() and read_json(ledger)["status"] == "complete":
            print("Frozen final run already complete; no test rerun.")
            return
        command = [
            str(python),
            "-m",
            "triage.cli",
            "benchmark",
            "final",
            "--release",
            str(release_path),
        ]
        if ledger.exists():
            command.append("--resume")
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        subprocess.run(command, env=env, check=True)
    print("Stage complete. See docs/release_benchmark.md for artifact verification.")


if __name__ == "__main__":
    main()
