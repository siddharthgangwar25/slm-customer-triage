"""Self-contained offline reports from saved predictions; no model deserialization."""

import csv
import math
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triage.evaluation.metrics import classification, output_label, routing
from triage.io import new_directory, read_json, read_jsonl, verify_hash, write_json


def load_predictions(path):
    manifest = read_json(path.parent / "prediction_manifest.json")
    verify_hash(path, manifest["predictions_sha256"])
    rows = read_jsonl(path)
    ids = [row["sample_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or len(rows) != manifest["sample_count"]:
        raise ValueError("Missing or duplicate predictions")
    expected = manifest["sample_ids"]
    if len(expected) != len(set(expected)) or set(ids) != set(expected):
        raise ValueError("Prediction sample IDs do not form a one-to-one join")
    catalog = manifest["catalog"]
    if not catalog or catalog != sorted(set(catalog)) or "oos" in catalog:
        raise ValueError("Invalid prediction catalog")
    for row in rows:
        if row["split"] != manifest["split"] or row["label"] not in catalog + ["oos"]:
            raise ValueError("Invalid prediction split or true label")
        if row["dataset_version"] != manifest["dataset_version"]:
            raise ValueError("Prediction dataset version mismatch")
        if row["model_version"] != manifest["model"]["model_version"]:
            raise ValueError("Mixed model versions")
        if not math.isfinite(row["gate_score"]) or not 0 <= row["gate_score"] <= 1:
            raise ValueError("Gate scores must be finite and in [0, 1]")
        if not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
            raise ValueError("Latency must be finite and nonnegative")
        if row["parse_status"] not in ("ok", "invalid", "error"):
            raise ValueError("Unknown parse status")
        if row["parse_status"] == "error" and row["error_type"] is None:
            raise ValueError("Failed predictions require an explicit error_type")
    return rows, catalog, manifest


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(predictions: Path, output: Path):
    rows, catalog, manifest = load_predictions(predictions)
    output = new_directory(output)
    metrics = classification(rows, catalog)
    metrics["serial_model_latency_ms"] = dict(
        zip(
            ("p50", "p95", "p99"),
            map(float, np.percentile([r["latency_ms"] for r in rows], [50, 95, 99])),
            strict=True,
        )
    )
    metrics["timing_scope"] = manifest["timing_scope"]
    metrics["split"] = manifest["split"]
    metrics["fixture"] = manifest["fixture"]
    metrics["model_version"] = manifest["model"]["model_version"]
    metrics["ungated_routing_diagnostic"] = routing(rows, catalog, 0.0)
    write_json(output / "metrics.json", metrics)
    shutil.copyfile(predictions, output / "predictions.jsonl")
    shutil.copyfile(
        predictions.parent / "prediction_manifest.json", output / "prediction_manifest.json"
    )
    write_csv(
        output / "per_class.csv",
        ["label", "precision", "recall", "f1", "support"],
        [{"label": label, **values} for label, values in metrics["per_class"].items()],
    )
    labels = catalog + ["oos", "__invalid__"]
    counts = Counter((r["label"], output_label(r, catalog)) for r in rows)
    write_csv(
        output / "confusion_matrix.csv",
        ["true_label", *labels],
        [
            {"true_label": actual, **{predicted: counts[actual, predicted] for predicted in labels}}
            for actual in catalog + ["oos"]
        ],
    )
    # A compact fixed grid is exploratory; exact policy selection belongs to Milestone 2.
    thresholds = np.linspace(0, 1, 101).tolist() + [math.nextafter(1.0, math.inf)]
    curve = [routing(rows, catalog, threshold) for threshold in thresholds]
    write_json(
        output / "coverage_error.json",
        {
            "purpose": "exploratory diagnostic; no operating point selected",
            "sample_count": len(rows),
            "split": manifest["split"],
            "points": curve,
        },
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    defined = [point for point in curve if point["routing_error"] is not None]
    ax.plot([p["coverage"] for p in defined], [p["routing_error"] for p in defined], ".-")
    ax.set(
        xlabel="Routing coverage",
        ylabel="Routing error",
        title=f"Baseline diagnostic: {manifest['split']}, n={len(rows)}",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "coverage_error.png", dpi=150)
    plt.close(fig)
    # Deterministic 40-example review queue: 30 supported errors + 10 oos requests.
    errors = [r for r in rows if r["label"] != "oos" and output_label(r, catalog) != r["label"]]
    selected = errors[:30] + [r for r in rows if r["label"] == "oos"][:10]
    write_json(output / "error_review_samples.json", selected)
    lines = [
        "# Baseline validation report",
        "",
        "TEST FIXTURE — not benchmark evidence."
        if manifest["fixture"]
        else "Measured official CLINC150 validation run; test predictions were not generated.",
        "",
        f"Model: `{metrics['model_version']}`. Split: `{manifest['split']}`. "
        f"Samples: {len(rows)} ({metrics['supported_count']} supported, "
        f"{metrics['oos_count']} oos).",
        "",
        "| Metric | Measured value |",
        "| --- | ---: |",
        f"| Raw supported macro-F1 ({len(catalog)} fixed labels) | "
        f"{metrics['raw_supported_macro_f1']:.6f} |",
        f"| Raw supported accuracy | {metrics['raw_supported_accuracy']:.6f} |",
        f"| Invalid output rate | {metrics['invalid_output_rate']:.6f} |",
        f"| Infrastructure failure rate | {metrics['infrastructure_failure_rate']:.6f} |",
        f"| Serial model p95 latency (ms) | {metrics['serial_model_latency_ms']['p95']:.3f} |",
        "",
        "Timing includes vectorization and classification per request on this CPU. "
        "It is not API latency, a load test, or a cost estimate.",
        "",
        "The classifier fits only supported training requests. It always predicts a supported "
        "intent, so all oos examples are wrong before gating. Gate scores are uncalibrated "
        "ranking signals. The coverage/error plot is exploratory; no deployment threshold "
        "or target achievement is claimed in Milestone 1.",
        "",
        f"Supported classification errors: {len(errors)}. "
        f"Review queue: {len(selected)} examples in `error_review_samples.json`. "
        "Manual observations are recorded separately in `error_analysis.md`.",
        "",
        "The per-class report uses supported samples only and includes every catalog label. "
        "The confusion matrix includes oos and invalid outputs. Zero-route error is null. "
        "Raw failures remain in the denominator.",
        "",
        "Provenance and environment are in `prediction_manifest.json`; data integrity and "
        "duplicate findings are in the data manifest. Public benchmark performance is not "
        "evidence of production customer quality.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {k: v for k, v in metrics.items() if k != "per_class"}
