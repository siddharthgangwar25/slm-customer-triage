"""Live CPU HTTP demo with automatic cleanup. No GPU, test inference or activation."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from triage.io import new_directory, read_json, sha256, write_json
from triage.service.schemas import TriageResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/baseline-c1-v1"))
    parser.add_argument(
        "--policy", type=Path, default=Path("reports/baseline-c1-v1-policy/policy.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    output = new_directory(args.output.resolve())
    examples = read_json(Path("examples/demo_requests.json"))
    api_key, metrics_key = secrets.token_hex(24), secrets.token_hex(24)
    env = {**os.environ, "TRIAGE_API_KEY": api_key, "TRIAGE_METRICS_KEY": metrics_key}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    result = {
        "purpose": "Live CPU research demo, not production release or benchmark",
        "test_inference": False,
        "requests": [],
        "checks": {},
        "status": "starting",
    }
    log = (output / "service.log").open("w", encoding="utf-8")
    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "triage.cli",
                "serve",
                "--bundle",
                str(args.bundle),
                "--policy",
                str(args.policy),
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

        def request(path, body=None, key=None):
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = "Bearer " + key
            req = urllib.request.Request(
                url + path,
                headers=headers,
                data=json.dumps(body).encode() if body is not None else None,
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read().decode()

        started = time.perf_counter()
        while time.perf_counter() - started < 60:
            try:
                if request("/health/ready")[0] == 200:
                    break
            except OSError:
                pass
            if proc.poll() is not None:
                raise RuntimeError("Demo service exited; see service.log")
            time.sleep(0.25)
        else:
            raise TimeoutError("Demo service did not become ready")
        result["startup_seconds"] = time.perf_counter() - started
        status, metadata = request("/v1/model", key=api_key)
        if status != 200:
            raise RuntimeError("Model metadata unavailable")
        result["model"] = json.loads(metadata)
        result["checks"]["live"] = request("/health/live")[0] == 200
        print("LOCAL RESEARCH DEMO: no candidate qualified for automatic production release.")
        print("These are demonstration prompts, not an independently reviewed challenge set.")
        for example in examples:
            status, body = request("/v1/triage", {"text": example["text"]}, api_key)
            if status != 200:
                raise RuntimeError(f"Demo request {example['id']} failed: {status}")
            response = TriageResponse.model_validate_json(body).model_dump()
            result["requests"].append({**example, "response": response})
            print(
                f"{example['id']}: {response['decision']} / "
                f"{response['intent']} / {response['reason']}"
            )
        result["checks"]["unauthorized_401"] = request("/v1/triage", {"text": "hello"})[0] == 401
        result["checks"]["blank_422"] = request("/v1/triage", {"text": "   "}, api_key)[0] == 422
        result["checks"]["metrics_private_401"] = request("/metrics", key=api_key)[0] == 401
        status, metrics = request("/metrics", key=metrics_key)
        result["checks"]["metrics_authorized_200"] = status == 200
        (output / "metrics.txt").write_text(metrics, encoding="utf-8", newline="\n")
        result["checks"]["all_passed"] = all(result["checks"].values())
        if not result["checks"]["all_passed"]:
            raise RuntimeError("HTTP contract check failed")
        result["policy_sha256"] = sha256(args.policy)
        result["bundle_metadata_sha256"] = sha256(args.bundle / "metadata.json")
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.close()
        result["service_stopped"] = proc is not None and proc.poll() is not None
        write_json(output / "demo.json", result)
    print(f"Demo finished and service stopped. Evidence: {output}")


if __name__ == "__main__":
    main()
