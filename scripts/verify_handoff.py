"""CPU-only audit of Milestone 6 reproduction, demo, links and frozen release."""

import argparse
import re
from pathlib import Path

from triage.io import new_directory, read_json, read_jsonl, sha256, write_json
from triage.release import load_release, source_hash

DOCS = [
    "README.md",
    "docs/setup.md",
    "docs/demo.md",
    "docs/handoff.md",
    "docs/data_card.md",
    "docs/model_card.md",
    "docs/finetuned_model_card.md",
    "docs/architecture.md",
    "docs/error_analysis.md",
    "docs/deployment.md",
    "deployment/aws_runbook.md",
    "docs/api.md",
]


def verify_demo(path):
    result = read_json(path / "demo.json")
    if result["status"] != "passed" or not result["service_stopped"]:
        raise ValueError("Demo failed or its service did not stop")
    expected_checks = {
        "live",
        "unauthorized_401",
        "blank_422",
        "metrics_private_401",
        "metrics_authorized_200",
        "all_passed",
    }
    if set(result["checks"]) != expected_checks or not all(result["checks"].values()):
        raise ValueError("Demo HTTP checks incomplete")
    examples = read_json(Path("examples/demo_requests.json"))
    if [r["id"] for r in result["requests"]] != [r["id"] for r in examples]:
        raise ValueError("Missing demo request")
    log = (path / "service.log").read_text(encoding="utf-8")
    for request, example in zip(result["requests"], examples, strict=True):
        if request["text"] != example["text"] or example["text"] in log:
            raise ValueError("Demo text changed or leaked into service logs")
        if request["response"]["model_version"] != result["model"]["model_version"]:
            raise ValueError("Demo model identity mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reproduction = args.reproduction
    commands = read_json(reproduction / "commands.json")
    if [c["stage"] for c in commands] != [
        "sync",
        "prepare",
        "train",
        "predict",
        "evaluate",
        "policy",
        "api-parity",
        "demo",
    ] or any(c["exit_code"] != 0 for c in commands):
        raise ValueError("CPU reproduction stages are incomplete")
    fresh = read_jsonl(reproduction / "predictions/predictions.jsonl")
    reference = read_jsonl(Path("reports/baseline-c1-v1-val/predictions.jsonl"))
    if len(fresh) != 3100 or any(r["split"] != "val" for r in fresh):
        raise ValueError("Require full validation predictions only")
    differences = {
        field: sum(a[field] != b[field] for a, b in zip(fresh, reference, strict=True))
        for field in (
            "sample_id",
            "label",
            "predicted_label",
            "gate_score",
            "parse_status",
            "error_type",
        )
    }
    if any(differences.values()):
        raise ValueError(f"Fresh reproduction differs from this machine's reference: {differences}")
    parity = read_json(reproduction / "api-parity/api_parity.json")
    if parity["sample_count"] != 3100 or parity["mismatch_count"] != 0:
        raise ValueError("Fresh API parity failed")
    if read_json(reproduction / "data/manifest.json") != read_json(
        Path("data/clinc150/manifest.json")
    ):
        raise ValueError("Prepared data provenance differs")
    demos = {"fresh": verify_demo(reproduction / "demo"), "original": verify_demo(args.demo)}
    checked_links = 0
    for name in DOCS:
        path = Path(name)
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if link.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = link.split("#", 1)[0]
            if not (path.parent / target).exists():
                raise ValueError(f"Missing local link in {path}: {target}")
            checked_links += 1
    release = load_release(Path("artifacts/milestone5/release/release.json"))
    if Path("artifacts/active-release.json").exists():
        raise ValueError("A release pointer unexpectedly exists")
    report = {
        "status": "passed",
        "test_inference": False,
        "cloud_execution": "omitted, unverified; no budget authorized",
        "independent_challenge_set": "deferred pending independent review",
        "source_sha256": source_hash(),
        "frozen_release_id": release["release_id"],
        "release_ledger_sha256": sha256(Path("artifacts/milestone5/release/test_use.json")),
        "reproduction_stages": commands,
        "reference_differences": differences,
        "fresh_model": read_json(reproduction / "bundle/metadata.json"),
        "fresh_macro_f1": read_json(reproduction / "evaluation/metrics.json")[
            "raw_supported_macro_f1"
        ],
        "api_parity": {k: parity[k] for k in ("sample_count", "mismatch_count", "outcomes")},
        "demos": demos,
        "local_links_checked": checked_links,
        "documentation_sha256": {name: sha256(Path(name)) for name in DOCS},
    }
    output = new_directory(args.output)
    write_json(output / "acceptance.json", report)
    print(f"Handoff acceptance passed: 3,100 validation/API matches; {checked_links} local links")


if __name__ == "__main__":
    main()
