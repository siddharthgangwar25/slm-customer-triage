"""Paired supported-sample bootstrap for a fixed-catalog macro-F1 difference."""

import numpy as np

from triage.evaluation.metrics import output_label


def paired_macro_f1(a, b, catalog, repetitions=1000, seed=42):
    by_id = {r["sample_id"]: r for r in b}
    if len(by_id) != len(b) or set(by_id) != {r["sample_id"] for r in a}:
        raise ValueError("Bootstrap requires one-to-one sample IDs")
    if any(r["label"] != by_id[r["sample_id"]]["label"] for r in a):
        raise ValueError("Paired gold labels differ")
    supported = [r for r in a if r["label"] != "oos"]
    if not supported:
        raise ValueError("Bootstrap requires supported samples")
    n_classes = len(catalog)
    index = {label: i for i, label in enumerate(catalog)}
    gold = np.array([index[r["label"]] for r in supported])
    pa = np.array([index.get(output_label(r, catalog), n_classes) for r in supported])
    pb = np.array(
        [index.get(output_label(by_id[r["sample_id"]], catalog), n_classes) for r in supported]
    )

    def score(y, predicted):
        actual_count = np.bincount(y, minlength=n_classes)
        predicted_count = np.bincount(predicted, minlength=n_classes + 1)[:n_classes]
        true_positives = np.bincount(y[y == predicted], minlength=n_classes)
        denominator = actual_count + predicted_count
        values = np.divide(
            2 * true_positives, denominator, out=np.zeros(n_classes), where=denominator != 0
        )
        return float(values.mean())

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        draw = rng.integers(0, len(gold), len(gold))
        differences.append(score(gold[draw], pb[draw]) - score(gold[draw], pa[draw]))
    return {
        "method": "paired bootstrap over supported samples, fixed label catalog",
        "repetitions": repetitions,
        "seed": seed,
        "supported_count": len(gold),
        "difference": score(gold, pb) - score(gold, pa),
        "percentile_95": np.percentile(differences, [2.5, 97.5]).tolist(),
    }
