"""Real HTTP demo on an explicitly synthetic bundle; no GPU or public data."""

import subprocess
import sys
from pathlib import Path

from triage.evaluation.select_policy import select_policy
from triage.io import read_json
from triage.models.baseline import predict, train


def test_demo_cli_http_and_cleanup(baseline_config, tmp_path):
    bundle, predictions = tmp_path / "bundle", tmp_path / "predictions"
    train(baseline_config, bundle)
    predict(baseline_config, "val", predictions)
    select_policy(predictions / "predictions.jsonl", "val", tmp_path / "policy")
    output = tmp_path / "demo"
    command = [
        sys.executable,
        "scripts/run_demo.py",
        "--bundle",
        str(bundle),
        "--policy",
        str(tmp_path / "policy/policy.json"),
        "--output",
        str(output),
    ]
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = read_json(output / "demo.json")
    assert result["model"]["fixture"] is True
    assert len(result["requests"]) == 3
    assert result["checks"]["unauthorized_401"]
    assert result["checks"]["blank_422"]
    assert result["checks"]["metrics_private_401"]
    assert result["checks"]["metrics_authorized_200"]
    assert result["service_stopped"]
    assert all(r["text"] not in (output / "service.log").read_text() for r in result["requests"])
    before = (output / "demo.json").read_bytes()
    repeated = subprocess.run(command, cwd=root, capture_output=True, timeout=10)
    assert repeated.returncode != 0
    assert (output / "demo.json").read_bytes() == before
