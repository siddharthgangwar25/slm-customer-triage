"""Real validation parity through ASGI; requires dev dependencies, never a load benchmark."""

import argparse
from pathlib import Path

from fastapi.testclient import TestClient

from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import load_policy
from triage.io import new_directory, sha256, write_json
from triage.policy import ModelOutput
from triage.service.app import create_app
from triage.service.schemas import ServiceSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, catalog, manifest = load_predictions(args.predictions)
    if manifest["split"] != "val":
        raise ValueError("API parity check accepts validation only")
    policy, frozen = load_policy(args.policy)
    if frozen["predictions_sha256"] != sha256(args.predictions):
        raise ValueError("Policy and predictions differ")
    new_directory(args.output)
    app = create_app(args.bundle, args.policy, settings=ServiceSettings(rate_limit_requests=0))
    counts = {"route": 0, "human_review": 0}
    mismatches = []
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        for row in rows:
            expected = policy.decide(
                row["gate_score"],
                ModelOutput(
                    row["predicted_label"], row["parse_status"] == "ok" and not row["error_type"]
                ),
                catalog,
            )
            response = client.post("/v1/triage", json={"text": row["text"]})
            if response.status_code != 200:
                mismatches.append({"sample_id": row["sample_id"], "status": response.status_code})
                continue
            actual = response.json()
            counts[actual["decision"]] += 1
            if any(
                actual[key] != getattr(expected, key) for key in ("intent", "decision", "reason")
            ):
                mismatches.append({"sample_id": row["sample_id"], "error": "decision_mismatch"})
        metadata = client.get("/v1/model").json()
    report = {
        "purpose": "Validation API/offline decision parity; not a network load benchmark",
        "fixture": manifest["fixture"],
        "split": "val",
        "sample_count": len(rows),
        "outcomes": counts,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "model": metadata,
        "policy_sha256": sha256(args.policy),
        "predictions_sha256": sha256(args.predictions),
        "service_dependency_lock_sha256": sha256(Path("uv.lock")),
        "rate_limit_requests": 0,
    }
    write_json(args.output / "api_parity.json", report)
    print(f"API parity: {len(rows)} samples, {len(mismatches)} mismatches, {counts}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
