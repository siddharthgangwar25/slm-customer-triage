"""Validation-only exact threshold selection; no model weights or test examples needed."""

import math
from itertools import groupby
from pathlib import Path

from triage.evaluation.metrics import INFRA_ERRORS, output_label, routing
from triage.evaluation.report import load_predictions
from triage.io import new_directory, object_hash, read_json, sha256, write_json
from triage.policy import Policy

CONSTRAINTS = {"max_routing_error": 0.05, "min_oos_recall": 0.90, "min_coverage": 0.20}
ALL_REVIEW_THRESHOLD = math.nextafter(1.0, math.inf)


def exact_sweep(rows, catalog):
    """O(n log n), processing equal-score records together (score == threshold passes)."""
    if not rows or not catalog:
        raise ValueError("Require validation records and catalog")
    total = len(rows)
    oos_total = sum(r["label"] == "oos" for r in rows)
    routed = wrong = failures = 0
    oos_review = oos_total
    supported_review = total - oos_total

    def point(threshold):
        return {
            "threshold": threshold,
            "sample_count": total,
            "supported_count": total - oos_total,
            "oos_count": oos_total,
            "routed_count": routed,
            "routed_error_count": wrong,
            "coverage": routed / total,
            "routing_error": wrong / routed if routed else None,
            "oos_recall": oos_review / oos_total if oos_total else None,
            "supported_review_rate": (
                supported_review / (total - oos_total) if total != oos_total else None
            ),
            "human_review_count": oos_review + supported_review,
            "infrastructure_failure_count": failures,
        }

    points = [point(ALL_REVIEW_THRESHOLD)]
    ordered = sorted(rows, key=lambda r: r["gate_score"], reverse=True)
    for score, group in groupby(ordered, key=lambda r: r["gate_score"]):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid gate score")
        for row in group:
            label = output_label(row, catalog)
            failed = row["error_type"] in INFRA_ERRORS
            if failed or label in catalog:
                oos_review -= row["label"] == "oos"
                supported_review -= row["label"] != "oos"
                if failed:
                    failures += 1
                else:
                    routed += 1
                    wrong += label != row["label"]
        points.append(point(score))
    if points[-1]["threshold"] != 0:
        points.append(point(0.0))
    return points


def choose_point(points):
    qualified = [
        p
        for p in points
        if p["routing_error"] is not None
        and p["routing_error"] <= CONSTRAINTS["max_routing_error"]
        and p["oos_recall"] is not None
        and p["oos_recall"] >= CONSTRAINTS["min_oos_recall"]
        and p["coverage"] >= CONSTRAINTS["min_coverage"]
    ]
    if not qualified:
        return None
    return max(qualified, key=lambda p: (p["coverage"], -p["routing_error"], p["threshold"]))


def select_policy(predictions: Path, split: str, output: Path):
    if split != "val":
        raise ValueError("Policy selection requires split=val")
    rows, catalog, manifest = load_predictions(predictions)
    if manifest["split"] != "val":
        raise ValueError("Prediction manifest must identify split=val")
    # Milestone 2 supports the genuine baseline gate. Future SLM files must carry a verified join.
    if manifest["gate_model_version"] != manifest["model"]["model_version"]:
        raise ValueError("Milestone 2 requires predictions from the baseline gate itself")
    if manifest["model"]["backend"] != "scikit-learn CPU":
        raise ValueError("Milestone 2 supports only the CPU baseline")
    points = exact_sweep(rows, catalog)
    selected = choose_point(points)
    threshold = selected["threshold"] if selected else ALL_REVIEW_THRESHOLD
    metrics = routing(rows, catalog, threshold)
    payload = {
        "schema_version": 1,
        "algorithm": "shared-baseline-gate-v1",
        "threshold": threshold,
        "automatic_routing": selected is not None,
        "status": "targets_met" if selected else "targets_unmet",
        "constraints": CONSTRAINTS,
        "validation": metrics,
        "fixture": manifest["fixture"],
        "split": "val",
        "model_version": manifest["model"]["model_version"],
        "gate_model_version": manifest["gate_model_version"],
        "pipeline_sha256": manifest["model"]["pipeline_sha256"],
        "catalog_sha256": object_hash(catalog),
        "data_manifest_sha256": manifest["data_manifest_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
        "prediction_manifest_sha256": sha256(predictions.parent / "prediction_manifest.json"),
    }
    digest = object_hash(payload)
    frozen = {**payload, "config_sha256": digest, "policy_version": f"policy-v1-{digest[:12]}"}
    new_directory(output)
    write_json(output / "policy.json", frozen)
    write_json(
        output / "threshold_sweep.json",
        {
            "split": "val",
            "fixture": manifest["fixture"],
            "constraints": CONSTRAINTS,
            "sample_count": len(rows),
            "points": points,
        },
    )
    error_text = (
        "undefined (zero routes)"
        if metrics["routing_error"] is None
        else (f"{metrics['routing_error']:.6%}")
    )
    report = [
        "# Validation routing policy",
        "",
        "TEST FIXTURE — not benchmark evidence."
        if manifest["fixture"]
        else "Genuine CLINC150 validation evidence; no test evaluation.",
        "",
        f"Status: **{payload['status']}**. Automatic routing: **{selected is not None}**.",
        f"Policy: `{frozen['policy_version']}`; model: `{payload['model_version']}`.",
        "",
        f"Samples: {len(rows)} ({metrics['supported_count']} supported, "
        f"{metrics['oos_count']} oos).",
        "",
        "| Measure | Selected validation result |",
        "| --- | ---: |",
        f"| Gate threshold (equal passes) | {threshold:.17g} |",
        f"| Coverage | {metrics['coverage']:.6%} ({metrics['routed_count']}/{len(rows)}) |",
        f"| Routing error | {error_text} |",
        f"| Routing error Wilson 95% | {metrics['routing_error_wilson95']} |",
        f"| Oos recall | {metrics['oos_recall']} |",
        f"| Oos recall Wilson 95% | {metrics['oos_recall_wilson95']} |",
        f"| Supported review rate | {metrics['supported_review_rate']} |",
        f"| Infrastructure failures after gating | {metrics['infrastructure_failure_count']} |",
        "",
        f"Swept {len(points)} thresholds: every distinct observed score, zero, "
        "and an explicit all-review endpoint. Maximize coverage subject to error <=5%, "
        "oos recall >=90%, and coverage >=20%; break ties by lower error then higher threshold.",
        "",
        "If targets are unmet, automatic routing is disabled; error is null and coverage "
        "is zero. The service uses low_gate_score for disabled-routing reviews to retain the "
        "specified reason vocabulary. It still requires a healthy model and policy bundle.",
        "",
        "These are validation-selected point estimates, not test guarantees. Wilson intervals "
        "describe binomial uncertainty and do not correct for threshold selection. Only 100 "
        "official validation oos examples constrain recall. The frozen baseline gate is not "
        "an independent estimate of an LLM's uncertainty. No production or release claim is made.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
    return frozen


def load_policy(path: Path):
    frozen = read_json(path)
    payload = {k: v for k, v in frozen.items() if k not in ("config_sha256", "policy_version")}
    digest = object_hash(payload)
    if frozen["config_sha256"] != digest or frozen["policy_version"] != f"policy-v1-{digest[:12]}":
        raise ValueError("Frozen policy hash mismatch")
    if (
        frozen["schema_version"] != 1
        or frozen["algorithm"] != "shared-baseline-gate-v1"
        or frozen["split"] != "val"
        or frozen["constraints"] != CONSTRAINTS
        or type(frozen["automatic_routing"]) is not bool
    ):
        raise ValueError("Unsupported frozen policy")
    if frozen["status"] != ("targets_met" if frozen["automatic_routing"] else "targets_unmet"):
        raise ValueError("Policy status mismatch")
    if frozen["automatic_routing"]:
        if choose_point([frozen["validation"]]) is None:
            raise ValueError("Enabled policy does not meet validation constraints")
    elif frozen["threshold"] != ALL_REVIEW_THRESHOLD:
        raise ValueError("Disabled policy must reject all gate scores")
    if frozen["validation"]["threshold"] != frozen["threshold"]:
        raise ValueError("Policy threshold differs from its validation evidence")
    return Policy(
        frozen["threshold"], frozen["automatic_routing"], frozen["policy_version"]
    ), frozen
