"""Paired validation comparison from saved raw predictions, without model loading."""

from collections import Counter
from pathlib import Path

import numpy as np

from triage.evaluation.bootstrap import paired_macro_f1
from triage.evaluation.metrics import classification, output_label, routing
from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import ALL_REVIEW_THRESHOLD, choose_point, exact_sweep
from triage.io import new_directory, sha256, write_json, write_jsonl


def compare(baseline: Path, candidate: Path, output: Path):
    a, catalog_a, ma = load_predictions(baseline)
    b, catalog_b, mb = load_predictions(candidate)
    if ma["split"] != "val" or mb["split"] != "val" or mb.get("benchmark_complete") is False:
        raise ValueError("Comparison requires complete validation predictions")
    if (
        catalog_a != catalog_b
        or ma["data_manifest_sha256"] != mb["data_manifest_sha256"]
        or set(r["sample_id"] for r in a) != set(r["sample_id"] for r in b)
    ):
        raise ValueError("Comparison requires matching catalog, dataset, and sample IDs")
    if ma["fixture"] != mb["fixture"]:
        raise ValueError("Do not compare fixtures with genuine benchmarks")
    if mb.get("gate_reference", {}).get("source_predictions_sha256") != sha256(baseline):
        raise ValueError("Candidate is not joined to this baseline")
    results = {}
    paired_rows = []
    by_id = {r["sample_id"]: r for r in b}
    for row in a:
        other = by_id[row["sample_id"]]
        predicted_a = output_label(row, catalog_a)
        predicted_b = output_label(other, catalog_a)
        correct_a, correct_b = predicted_a == row["label"], predicted_b == row["label"]
        outcome = (
            "both_correct"
            if correct_a and correct_b
            else "prompted_only_correct"
            if correct_b
            else "baseline_only_correct"
            if correct_a
            else "both_wrong"
        )
        paired_rows.append(
            {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "label": row["label"],
                "baseline": predicted_a,
                "prompted": predicted_b,
                "outcome": outcome,
                "prompted_raw_output": other.get("raw_output"),
                "prompted_error_type": other["error_type"],
            }
        )
    for name, rows, manifest in (("baseline", a, ma), ("prompted", b, mb)):
        raw = classification(rows, catalog_a)
        selected = choose_point(exact_sweep(rows, catalog_a))
        metrics = routing(
            rows, catalog_a, selected["threshold"] if selected else ALL_REVIEW_THRESHOLD
        )
        results[name] = {
            "model_version": manifest["model"]["model_version"],
            "raw": {k: v for k, v in raw.items() if k != "per_class"},
            "policy": metrics,
            "automatic_routing": selected is not None,
            "execution": manifest.get("execution"),
            "model": manifest["model"],
            "serial_inference_seconds": sum(r["latency_ms"] for r in rows) / 1000,
            "serial_model_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "timing_scope": manifest["timing_scope"],
        }
    result = {
        "split": "val",
        "fixture": ma["fixture"],
        "sample_count": len(a),
        "supported_count": results["baseline"]["raw"]["supported_count"],
        "oos_count": results["baseline"]["raw"]["oos_count"],
        "candidates": results,
        "macro_f1_difference_prompted_minus_baseline": results["prompted"]["raw"][
            "raw_supported_macro_f1"
        ]
        - results["baseline"]["raw"]["raw_supported_macro_f1"],
        "baseline_predictions_sha256": sha256(baseline),
        "candidate_predictions_sha256": sha256(candidate),
        "paired_macro_f1_bootstrap": paired_macro_f1(a, b, catalog_a),
        "paired_supported_outcomes": dict(
            Counter(r["outcome"] for r in paired_rows if r["label"] != "oos")
        ),
        "paired_oos_outcomes": dict(
            Counter(r["outcome"] for r in paired_rows if r["label"] == "oos")
        ),
    }
    new_directory(output)
    write_json(output / "comparison.json", result)
    write_jsonl(output / "paired_predictions.jsonl", paired_rows)
    lines = [
        "# Baseline versus prompted model — validation",
        "",
        "TEST FIXTURE — not benchmark evidence."
        if ma["fixture"]
        else "Genuine CLINC150 validation results. Test predictions remain unused.",
        "",
        f"Samples: {len(a)} ({result['supported_count']} supported, {result['oos_count']} oos).",
        "",
        "| Candidate | Raw macro-F1 | Invalid rate | Coverage | Routing error | "
        "Oos recall | Targets met |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in results.items():
        raw, policy = item["raw"], item["policy"]
        lines.append(
            f"| {name} | {raw['raw_supported_macro_f1']:.6f} | "
            f"{raw['invalid_output_rate']:.6f} | {policy['coverage']:.6f} | "
            f"{policy['routing_error']} | {policy['oos_recall']} | {item['automatic_routing']} |"
        )
    lines.extend(
        [
            "",
            "Each candidate uses its own validation-selected threshold against the same "
            "frozen baseline gate. Raw predictions are ungated. Zero-route error is null. "
            "All failures remain in accounting. Wilson intervals and exact model identities, "
            "precision, runtime, and memory are recorded in comparison.json. No independent "
            "test performance, deployment choice, cost, or production-quality claim is made.",
            "",
        ]
    )
    lines.extend(
        [
            f"Paired supported macro-F1 difference 95% bootstrap interval: "
            f"{result['paired_macro_f1_bootstrap']['percentile_95']} "
            "(1,000 resamples, seed 42).",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {k: v for k, v in result.items() if k != "candidates"}
