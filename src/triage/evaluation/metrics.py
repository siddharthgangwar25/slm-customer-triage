"""Metrics with fixed-label macro averaging and explicit failed-request accounting."""

import math
from typing import Any

from triage.contracts import Prediction

INFRA_ERRORS = {"timeout", "unavailable", "inference_error"}


def wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def output_label(row: Prediction, catalog: list[str]) -> str:
    if row["error_type"] or row["parse_status"] != "ok":
        return "__invalid__"
    label = row["predicted_label"]
    return label if label in catalog + ["oos"] else "__invalid__"


def classification(rows: list[Prediction], catalog: list[str]) -> dict[str, Any]:
    if not rows or not catalog or len(catalog) != len(set(catalog)) or "oos" in catalog:
        raise ValueError("Require predictions and a unique supported-label catalog")
    supported = [row for row in rows if row["label"] != "oos"]
    pairs = [(row["label"], output_label(row, catalog)) for row in supported]
    per_class = {}
    for label in catalog:
        tp = sum(actual == label and predicted == label for actual, predicted in pairs)
        fp = sum(actual != label and predicted == label for actual, predicted in pairs)
        fn = sum(actual == label and predicted != label for actual, predicted in pairs)
        per_class[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": tp + fn,
        }
    infra = sum(row["error_type"] in INFRA_ERRORS for row in rows)
    invalid = sum(
        output_label(row, catalog) == "__invalid__" and row["error_type"] not in INFRA_ERRORS
        for row in rows
    )
    return {
        "sample_count": len(rows),
        "supported_count": len(supported),
        "oos_count": len(rows) - len(supported),
        "catalog_size": len(catalog),
        "raw_supported_macro_f1": sum(v["f1"] for v in per_class.values()) / len(catalog),
        "raw_supported_accuracy": (sum(a == p for a, p in pairs) / len(pairs) if pairs else None),
        "invalid_output_count": invalid,
        "model_calls": len(rows),
        "invalid_output_rate": invalid / len(rows),
        "infrastructure_failure_count": infra,
        "infrastructure_failure_rate": infra / len(rows),
        "per_class": per_class,
    }


def routing(rows: list[Prediction], catalog: list[str], threshold: float) -> dict[str, Any]:
    """Offline diagnostic only; does not select or freeze a deployment policy."""
    routed = wrong = oos_review = supported_review = failures = 0
    oos_total = sum(row["label"] == "oos" for row in rows)
    for row in rows:
        label = output_label(row, catalog)
        if row["gate_score"] < threshold:
            review = True
        elif row["error_type"] in INFRA_ERRORS:
            failures += 1
            continue
        else:
            review = label not in catalog
        if review:
            oos_review += row["label"] == "oos"
            supported_review += row["label"] != "oos"
        else:
            routed += 1
            wrong += label != row["label"]
    supported_total = len(rows) - oos_total
    return {
        "threshold": threshold,
        "sample_count": len(rows),
        "supported_count": supported_total,
        "oos_count": oos_total,
        "routed_count": routed,
        "routed_error_count": wrong,
        "coverage": routed / len(rows),
        "routing_error": wrong / routed if routed else None,
        "routing_error_wilson95": wilson(wrong, routed),
        "oos_recall": oos_review / oos_total if oos_total else None,
        "oos_recall_wilson95": wilson(oos_review, oos_total),
        "supported_review_rate": supported_review / supported_total if supported_total else None,
        "human_review_count": oos_review + supported_review,
        "infrastructure_failure_count": failures,
    }
