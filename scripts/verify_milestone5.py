"""Audit an existing completed M5 run; never generate predictions or select new thresholds."""

import argparse
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from triage.data.load import load_split
from triage.evaluation.bootstrap import paired_macro_f1
from triage.evaluation.cost import estimate
from triage.evaluation.metrics import classification, routing
from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import load_policy
from triage.io import (
    new_directory,
    object_hash,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
)
from triage.policy import ModelOutput, parse_model_json
from triage.release import activate, choose_candidate, load_release, previous_test_use, source_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release", type=Path, default=Path("artifacts/milestone5/release/release.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = load_release(args.release)
    ledger_path = args.release.parent / "test_use.json"
    ledger = read_json(ledger_path)
    require(ledger["status"] == "complete", "Final test run is incomplete")
    require(ledger["release_sha256"] == release["release_sha256"], "Ledger release mismatch")
    require(previous_test_use() == [release["release_sha256"]], "Unexpected test exposure history")
    run = Path(ledger["output"])
    require(not (run / "active.lock").exists(), "Final writer may still be active")
    require(
        read_json(run / "run.json")["release_sha256"] == release["release_sha256"], "Run identity"
    )
    verify_hash(run / "final_report.json", ledger["report_sha256"])
    final = read_json(run / "final_report.json")
    chosen, reason = choose_candidate(release["policies"])
    require(
        (chosen, reason) == (release["selected"], release["selection_reason"]), "Selection changed"
    )
    require(final["selected_before_test"] == chosen and not final["fixture"], "Final identity")
    canonical, catalog, data = load_split(Path(release["config"]["data_dir"]), "test")
    require(not data["fixture"] and len(canonical) == 5500, "Not genuine full test data")
    rows_by_name, details = {}, {}
    for name, candidate in release["config"]["candidates"].items():
        rows, labels, meta = load_predictions(run / name / "predictions.jsonl")
        require(labels == catalog and len(rows) == len(canonical), "Catalog/count mismatch")
        require(meta["release_sha256"] == release["release_sha256"], "Prediction release mismatch")
        require(meta["model"] == release["validation_manifests"][name]["model"], "Model changed")
        verify_hash(
            Path(release["config"]["data_dir"]) / "manifest.json", meta["data_manifest_sha256"]
        )
        for actual, expected in zip(rows, canonical, strict=True):
            require(all(actual[k] == expected[k] for k in expected), "Canonical test join mismatch")
            if name != "baseline" and actual["parse_status"] != "error":
                parsed = parse_model_json(
                    actual["raw_output"], catalog, truncated=actual["truncated"]
                )
                require(parsed.label == actual["predicted_label"], "Parsed label mismatch")
                require(parsed.valid == (actual["parse_status"] == "ok"), "Parse status mismatch")
        _, policy = load_policy(Path(candidate["policy"]))
        require(policy == release["policies"][name], "Policy changed after freeze")
        raw = classification(rows, catalog)
        saved = final["candidates"][name]
        require(saved["raw"] == {k: v for k, v in raw.items() if k != "per_class"}, "Raw metrics")
        require(
            saved["fixed_policy"] == routing(rows, catalog, policy["threshold"]), "Fixed metrics"
        )
        require(saved["policy_version"] == policy["policy_version"], "Policy version mismatch")
        report = run / (name + "-report")
        require(
            sha256(report / "predictions.jsonl") == sha256(run / name / "predictions.jsonl"),
            "Report copy",
        )
        require(
            all(read_json(report / "metrics.json")[k] == v for k, v in raw.items()),
            "Per-class metrics",
        )
        rows_by_name[name] = rows
        details[name] = {
            "predictions_sha256": sha256(run / name / "predictions.jsonl"),
            "raw_oos_correct": sum(
                r["label"] == "oos" and r["predicted_label"] == "oos" for r in rows
            ),
            "invalid_supported": sum(
                r["label"] != "oos" and r["parse_status"] == "invalid" for r in rows
            ),
            "invalid_oos": sum(
                r["label"] == "oos" and r["parse_status"] == "invalid" for r in rows
            ),
            "summed_inference_seconds": sum(r["latency_ms"] for r in rows) / 1000,
            "execution": meta["execution"],
        }
    for other, key in (("baseline", "c_minus_a"), ("prompted", "c_minus_b")):
        require(
            final[key] == paired_macro_f1(rows_by_name[other], rows_by_name["finetuned"], catalog),
            "Bootstrap mismatch",
        )
    selected = final["candidates"][chosen]["fixed_policy"]
    disqualified = not (
        selected["routing_error"] is not None
        and selected["routing_error"] <= 0.05
        and selected["oos_recall"] >= 0.90
        and selected["coverage"] >= 0.20
    )
    require(final["automatic_release_disqualified"] == disqualified, "Release decision mismatch")
    root = args.release.parent.parent
    parity = read_json(root / "parity/parity.json")
    reference_path = Path(release["config"]["candidates"][chosen]["predictions"])
    reference, _, meta = load_predictions(reference_path)
    outcomes = read_jsonl(root / "parity/outcomes.jsonl")
    verify_hash(root / "parity/outcomes.jsonl", read_json(root / "parity/progress.json")["sha256"])
    require(parity["complete"] and parity["passed"] and len(outcomes) == 3100, "Parity incomplete")
    require(parity["source_sha256"] == source_hash(), "Parity source mismatch")
    require(parity["reference_metadata_sha256"] == object_hash(meta["model"]), "Worker identity")
    for actual, expected in zip(outcomes, reference, strict=True):
        require(
            actual["sample_id"] == expected["sample_id"] and actual["equal"], "Parity ID/equality"
        )
        require(actual["actual"]["error_type"] is None, "Worker failure")
        require(
            all(
                actual["actual"][k] == expected[k]
                for k in ("raw_output", "input_tokens", "output_tokens", "truncated")
            ),
            "Worker output differs",
        )
    load = read_json(root / "load/load.json")
    require(load == release["operational"]["load"], "Load evidence changed")
    require(final["operational"] == release["operational"], "Final operations mismatch")
    policy, _ = load_policy(Path(release["config"]["candidates"][chosen]["policy"]))
    by_id = {r["sample_id"]: r for r in reference}
    workload_ids = None
    for measured in load["runs"]:
        outcomes = read_jsonl(root / f"load/concurrency-{measured['concurrency']}.jsonl")
        require(len(outcomes) == measured["submitted"] == 500, "Load count")
        ids = [r["sample_id"] for r in outcomes]
        if workload_ids is None:
            workload_ids = ids
        require(ids == workload_ids, "Concurrency workloads differ")
        complete = [r for r in outcomes if r["status"] == 200]
        require(len(complete) == measured["completed"], "Completed count")
        require(measured["failures"] == 500 - len(complete), "Failure count")
        require(
            dict(Counter(str(r["status"]) for r in outcomes)) == measured["statuses"],
            "Status counts",
        )
        for record in outcomes:
            expected = by_id[record["sample_id"]]
            require(
                record["label"] == expected["label"]
                and record["reference_input_tokens"] == expected["input_tokens"],
                "Load reference",
            )
            require(not record["decision_mismatch"], "Saved API decision mismatch")
            if record["status"] == 200:
                decision = policy.decide(
                    expected["gate_score"],
                    ModelOutput(expected["predicted_label"], expected["parse_status"] == "ok"),
                    catalog,
                )
                require(
                    all(record[k] == getattr(decision, k) for k in ("decision", "reason")),
                    "API decision differs",
                )
            else:
                require(
                    record["status"] == 503 and record["error_code"] == "model_busy",
                    "Unexpected service error",
                )
        for subset, key in ((outcomes, "latency_all_ms"), (complete, "latency_completed_ms")):
            actual = dict(
                zip(
                    ("p50", "p95", "p99"),
                    np.percentile([r["latency_ms"] for r in subset], [50, 95, 99]).tolist(),
                    strict=True,
                )
            )
            require(actual == measured[key], "Latency accounting mismatch")
        require(
            math.isclose(
                len(complete) / measured["elapsed_seconds"], measured["completed_per_second"]
            ),
            "Throughput mismatch",
        )
        delta = measured["metrics_delta"]
        gates = sum(r["reason"] == "low_gate_score" for r in complete)
        require(delta["triage_gate_rejections_total"] == gates, "Gate short-circuit count")
        require(delta["triage_model_calls_total"] == len(complete) - gates, "Generation count")
    cost = read_json(root / "cost.json")
    require(cost == release["operational"]["cost"], "Cost evidence changed")
    scenario = cost["scenario"]
    price = float(
        next(q["price"] for q in cost["quote"]["quotes"] if q["Instance Type"] == "g4dn.xlarge")
    )
    require(
        scenario
        == estimate(
            load,
            price,
            monthly_requests=scenario["monthly_submitted_requests_assumed"],
            monthly_hours=scenario["billed_monthly_hours_assumed"],
            supporting_monthly_usd=scenario["supporting_monthly_usd_assumed"],
            training_hours=scenario["training_hours_local"],
        ),
        "Cost arithmetic",
    )
    samples = read_json(root / "gpu_samples.json")
    resources = {
        "sample_count": len(samples["samples"]),
        "error": samples["error"],
        "scope": samples["scope"],
    }
    for field in ("memory_used_mib", "utilization_percent"):
        values = [r[field] for r in samples["samples"]]
        resources[field] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    new_directory(args.output)
    if disqualified:
        try:
            activate(args.release, args.output / "must-not-activate.json")
        except ValueError as exc:
            require("disqualified" in str(exc), "Unexpected activation failure")
        else:
            raise ValueError("Disqualified release was activated")
    evidence = {
        "passed": True,
        "fixture": False,
        "inference_rerun": False,
        "source_commit": release["source_commit"],
        "release_id": release["release_id"],
        "release_sha256": release["release_sha256"],
        "selected_before_test": chosen,
        "automatic_release_disqualified": disqualified,
        "activation_guard_verified": disqualified,
        "candidate_details": details,
        "gpu": resources,
        "checks": [
            "frozen inputs/source/weights/locks",
            "complete test ledger and registry",
            "validation-only selection",
            "5500 canonical test joins per candidate",
            "raw output parsing",
            "fixed-policy and per-class metrics",
            "paired bootstrap recomputation",
            "3100 raw worker comparisons",
            "1500 HTTP outcomes and metric accounting",
            "cost arithmetic",
            "disqualification and activation guard",
        ],
    }
    write_json(args.output / "verification.json", evidence)
    for name in ("final_report.json", "report.md", "run.json"):
        shutil.copyfile(run / name, args.output / name)
    shutil.copyfile(args.release, args.output / "release.json")
    shutil.copyfile(ledger_path, args.output / "test_use.json")
    for name in rows_by_name:
        shutil.copytree(run / (name + "-report"), args.output / (name + "-report"))
    for name in ("parity", "load"):
        shutil.copytree(root / name, args.output / name)
    for name in ("cost.json", "gpu_samples.json", "worker_resources.json"):
        shutil.copyfile(root / name, args.output / name)
    write_json(
        args.output / "evidence_sha256.json",
        {
            p.relative_to(args.output).as_posix(): sha256(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    )
    print(
        {"passed": True, "output": str(args.output), "automatic_release_disqualified": disqualified}
    )


if __name__ == "__main__":
    main()
