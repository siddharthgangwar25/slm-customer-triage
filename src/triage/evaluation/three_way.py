"""Paired A/B/C validation analysis, enforcing identical B/C model and decoding."""

from collections import Counter
from pathlib import Path

import numpy as np

from triage.evaluation.bootstrap import paired_macro_f1
from triage.evaluation.metrics import classification, output_label, routing
from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import ALL_REVIEW_THRESHOLD, choose_point, exact_sweep
from triage.io import new_directory, sha256, write_json, write_jsonl

PAIR_FIELDS = (
    "model_id",
    "revision",
    "dtype",
    "quantization",
    "max_input_tokens",
    "max_new_tokens",
    "seed",
    "do_sample",
    "enable_thinking",
    "batch_size",
    "prefix_cache",
)


def validate_pair(b, c):
    if b["model_type"] != "prompted" or c["model_type"] != "finetuned":
        raise ValueError("Require prompted B and fine-tuned C")
    for key in PAIR_FIELDS:
        if b["config"].get(key) != c["config"].get(key):
            raise ValueError(f"B/C paired inference mismatch: {key}")
    for key in ("system_prompt_sha256", "chat_template_sha256", "prompt_version"):
        if b.get(key) is None or b[key] != c.get(key):
            raise ValueError(f"B/C prompt mismatch: {key}")


def compare(baseline: Path, prompted: Path, finetuned: Path, output: Path):
    paths = {"baseline": baseline, "prompted": prompted, "finetuned": finetuned}
    loaded = {name: load_predictions(path) for name, path in paths.items()}
    a, catalog, ma = loaded["baseline"]
    by_id = {r["sample_id"]: r for r in a}
    for name, (rows, labels, manifest) in loaded.items():
        if manifest["split"] != "val" or manifest.get("benchmark_complete") is False:
            raise ValueError("Require complete validation predictions")
        if (
            labels != catalog
            or manifest["data_manifest_sha256"] != ma["data_manifest_sha256"]
            or manifest["fixture"] != ma["fixture"]
            or set(r["sample_id"] for r in rows) != set(by_id)
        ):
            raise ValueError("Dataset/catalog/fixture/IDs differ")
        for row in rows:
            if any(
                row[k] != by_id[row["sample_id"]][k]
                for k in ("text", "label", "split", "dataset_version", "gate_score")
            ):
                raise ValueError("Paired request/gold/gate differs")
        if name != "baseline" and manifest["gate_reference"]["source_predictions_sha256"] != sha256(
            baseline
        ):
            raise ValueError("All candidates require the same frozen baseline gate")
    validate_pair(loaded["prompted"][2]["model"], loaded["finetuned"][2]["model"])
    results = {}
    for name, (rows, _, manifest) in loaded.items():
        raw = classification(rows, catalog)
        point = choose_point(exact_sweep(rows, catalog))
        results[name] = {
            "model": manifest["model"],
            "sample_count": len(rows),
            "raw": {k: v for k, v in raw.items() if k != "per_class"},
            "policy": routing(rows, catalog, point["threshold"] if point else ALL_REVIEW_THRESHOLD),
            "automatic_routing": point is not None,
            "execution": manifest.get("execution"),
            "serial_model_p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
            "serial_inference_seconds": sum(r["latency_ms"] for r in rows) / 1000,
            "predictions_sha256": sha256(paths[name]),
            "timing_scope": manifest["timing_scope"],
        }
    indexed = {name: {r["sample_id"]: r for r in values[0]} for name, values in loaded.items()}
    changed, paired = [], []
    for row in a:
        labels = {
            name: output_label(rows[row["sample_id"]], catalog) for name, rows in indexed.items()
        }
        b_ok, c_ok = labels["prompted"] == row["label"], labels["finetuned"] == row["label"]
        outcome = (
            "both_correct"
            if b_ok and c_ok
            else "fixed_by_c"
            if c_ok
            else "regressed_in_c"
            if b_ok
            else "both_wrong"
        )
        entry = {
            "sample_id": row["sample_id"],
            "text": row["text"],
            "label": row["label"],
            **labels,
            "outcome": outcome,
            "prompted_raw_output": indexed["prompted"][row["sample_id"]].get("raw_output"),
            "finetuned_raw_output": indexed["finetuned"][row["sample_id"]].get("raw_output"),
        }
        paired.append(entry)
        if labels["prompted"] != labels["finetuned"]:
            changed.append(entry)
    counts = {
        mix: dict(Counter(r["outcome"] for r in paired if (r["label"] == "oos") == (mix == "oos")))
        for mix in ("supported", "oos")
    }
    transitions = Counter((r["label"], r["prompted"], r["finetuned"]) for r in changed)
    result = {
        "split": "val",
        "fixture": ma["fixture"],
        "sample_count": len(a),
        "candidates": results,
        "outcomes_b_to_c": counts,
        "bootstrap_c_minus_b": paired_macro_f1(
            loaded["prompted"][0], loaded["finetuned"][0], catalog
        ),
        "bootstrap_c_minus_a": paired_macro_f1(a, loaded["finetuned"][0], catalog),
        "common_changed_errors": [
            {"gold": g, "before": b, "after": c, "count": n}
            for (g, b, c), n in transitions.most_common(30)
        ],
    }
    new_directory(output)
    write_json(output / "comparison.json", result)
    write_jsonl(output / "paired_predictions.jsonl", paired)
    write_jsonl(output / "changed_errors.jsonl", changed)
    lines = [
        "# Three-candidate validation comparison",
        "",
        "TEST FIXTURE; not benchmark evidence."
        if ma["fixture"]
        else "Genuine validation predictions; test remains unused.",
        "",
        f"{len(a)} requests: {sum(r['label'] != 'oos' for r in a)} supported, "
        f"{sum(r['label'] == 'oos' for r in a)} oos.",
        "",
        "| Candidate | Raw macro-F1 | Invalid | Coverage | Routing error | "
        "Oos recall | Targets met |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in results.items():
        raw, policy = item["raw"], item["policy"]
        lines.append(
            f"| {name} | {raw['raw_supported_macro_f1']:.6f} | "
            f"{raw['invalid_output_count']} / {len(a)} | {policy['coverage']:.6f} | "
            f"{policy['routing_error']} | {policy['oos_recall']} | {item['automatic_routing']} |"
        )
    lines += ["", "## Which errors changed", ""]
    for mix, values in counts.items():
        lines.append(
            f"{mix}: C fixed {values.get('fixed_by_c', 0)} B errors and introduced "
            f"{values.get('regressed_in_c', 0)} regressions; "
            f"{values.get('both_wrong', 0)} remained wrong for both models."
        )
    for outcome in ("fixed_by_c", "regressed_in_c"):
        examples = [r for r in paired if r["outcome"] == outcome][:10]
        lines += ["", f"### {outcome.replace('_', ' ')}", ""]
        for row in examples:
            # Render untrusted public request text as escaped inline text.
            text = (
                row["text"]
                .replace("`", "'")
                .replace("\n", " ")
                .replace("<", "&lt;")
                .replace("|", "\\|")
            )
            lines.append(
                f"- `{row['sample_id']}`: {text!r}; gold `{row['label']}`, "
                f"B `{row['prompted']}`, C `{row['finetuned']}`."
            )
    lines += [
        "",
        "All changed outputs are retained in changed_errors.jsonl. Examples above are "
        "deterministic and class-order biased. These are observed changes, not causal "
        "explanations or independent annotations.",
        "",
        "Each policy uses the unchanged validation constraints and frozen A gate. Zero-route "
        "error is undefined. Invalid outputs and infrastructure failures remain in accounting. "
        "Exact counts, Wilson intervals, hardware, precision, timing scope and paired "
        "1,000-resample bootstrap intervals (seed 42) are in comparison.json.",
        "",
        "Checkpoint choice and these comparisons use validation, so estimates are subject to "
        "selection bias. Only 100 training oos examples are available; public benchmark "
        "pretraining contamination cannot be excluded. No test, deployment, cost-saving, "
        "or customer-performance claim is made.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {"output": str(output), "sample_count": len(a), "outcomes_b_to_c": counts}
