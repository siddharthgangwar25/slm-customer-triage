"""Audit genuine local M4 artifacts without loading models or reading test records."""

import argparse
import math
import shutil
from pathlib import Path

from triage.data.load import load_split
from triage.evaluation.metrics import classification, routing
from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import (
    ALL_REVIEW_THRESHOLD,
    choose_point,
    exact_sweep,
    load_policy,
)
from triage.evaluation.three_way import compare
from triage.io import (
    config,
    new_directory,
    object_hash,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
    write_jsonl,
)
from triage.models.finetuned import prediction_config
from triage.training.data import identity, load_prepared
from triage.training.runner import check_smoke, verify_bundle
from triage.training.select import choose_checkpoint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = config(Path("configs/finetune.yaml"))
    pair = config(Path(cfg["paired_config"]))
    training = Path(cfg["output"])
    result = read_json(training / "result.json")
    records, prepared = load_prepared(cfg)
    require(result["identity"] == identity(cfg), "Training identity changed")
    verify_hash(Path(cfg["prepared_data"]) / "manifest.json", result["prepared_manifest_sha256"])
    require(
        result["status"] == "complete" and result["epoch"] == cfg["epochs"], "Incomplete training"
    )
    require(
        result["examples_processed"] == len(records) * cfg["epochs"], "Unexpected training count"
    )
    require(not result["budget_exhausted"] and result["elapsed_seconds"] < 86400, "Budget mismatch")
    require(
        result["reload_parity"]["passed"] and result["weights_changed"],
        "Reload/change check failed",
    )
    check_smoke(cfg, Path(str(training) + "-smoke"), prepared)
    logs = read_jsonl(training / "training_log.jsonl")
    require(
        all(math.isfinite(v) for r in logs for v in r.values() if isinstance(v, (int, float))),
        "Non-finite training log",
    )
    require(
        [r["step"] for r in logs if "loss" in r] == list(range(1, result["step"] + 1)),
        "Incomplete optimizer-step log",
    )
    checkpoints = {}
    for path in sorted(training.glob("checkpoint-*")) + [training / "adapter"]:
        bundle = verify_bundle(path)
        require(bundle["identity"] == result["identity"], "Checkpoint identity mismatch")
        checkpoints[path.name] = {
            "step": bundle["step"],
            "epoch": bundle["epoch"],
            "bundle_sha256": sha256(path / "bundle.json"),
        }
    selection_dir = Path(str(training) + "-selection")
    selection = read_json(selection_dir / "selection.json")
    verify_hash(training / "result.json", selection["training_result_sha256"])
    require(
        selection["selected"] == choose_checkpoint(selection["all_checkpoints"]), "Wrong choice"
    )
    require(len(selection["all_checkpoints"]) == cfg["epochs"], "Missing epoch validation")
    selected = Path(selection["selected"]["checkpoint"])
    require(
        sha256(selected / "adapter_model.safetensors")
        == sha256(training / "adapter" / "adapter_model.safetensors"),
        "Final adapter differs",
    )
    canonical, catalog, _ = load_split(Path(pair["data_dir"]), "val")
    expected = {r["sample_id"]: r for r in canonical}
    stems = {
        "baseline": "baseline-c1-v1",
        "prompted": pair["experiment_id"],
        "finetuned": cfg["experiment_id"],
    }
    paths = {
        name: Path("reports") / (stem + "-val") / "predictions.jsonl"
        for name, stem in stems.items()
    }
    metrics = {}
    for name, path in paths.items():
        rows, labels, manifest = load_predictions(path)
        require(manifest["fixture"] is False and len(rows) == 3100, "Not genuine full validation")
        require(
            labels == catalog and {r["sample_id"] for r in rows} == set(expected), "IDs/catalog"
        )
        verify_hash(Path(pair["data_dir"]) / "manifest.json", manifest["data_manifest_sha256"])
        for row in rows:
            require(
                all(
                    row[k] == expected[row["sample_id"]][k]
                    for k in ("text", "label", "split", "dataset_version")
                ),
                "Canonical mismatch",
            )
        raw = classification(rows, catalog)
        saved = read_json(path.parent / "metrics.json")
        require(all(saved[k] == v for k, v in raw.items()), "Saved classification differs")
        _, policy = load_policy(Path("reports") / (stems[name] + "-policy") / "policy.json")
        verify_hash(path, policy["predictions_sha256"])
        verify_hash(path.parent / "prediction_manifest.json", policy["prediction_manifest_sha256"])
        chosen = choose_point(exact_sweep(rows, catalog))
        threshold = chosen["threshold"] if chosen else ALL_REVIEW_THRESHOLD
        require(
            policy["threshold"] == threshold
            and policy["validation"] == routing(rows, catalog, threshold),
            "Policy recomputation differs",
        )
        metrics[name] = {k: v for k, v in raw.items() if k != "per_class"}
        if name != "baseline":
            original = (
                Path(pair["prediction_output"])
                if name == "prompted"
                else Path(selection["selected"]["predictions"]).parent
            )
            require(
                sha256(original / "predictions.jsonl") == sha256(path), "Report differs from run"
            )
            frozen = read_json(original / "run.json")
            expected_cfg = pair if name == "prompted" else prediction_config(cfg, selected)
            require(
                frozen["config"] == expected_cfg and manifest["model"]["config"] == expected_cfg,
                "Prediction configuration changed",
            )
            require(frozen["config_sha256"] == object_hash(expected_cfg), "Configuration hash")
            verify_hash(Path(pair["environment_lock"]), frozen["environment_lock_sha256"])
            verify_hash(Path(pair["gate_predictions"]), frozen["gate_predictions_sha256"])
            implementation = object_hash(
                {n: sha256(Path("src/triage/models") / n) for n in ("prompted.py", "prompting.py")}
            )
            require(frozen["implementation_sha256"] == implementation, "Inference source changed")
            require(
                frozen["system_prompt_sha256"] == prepared["system_prompt_sha256"], "Prompt hash"
            )
    for item in selection["all_checkpoints"]:
        rows, labels, _ = load_predictions(Path(item["predictions"]))
        verify_hash(Path(item["predictions"]), item["predictions_sha256"])
        raw = classification(rows, labels)
        require(all(item[k] == v for k, v in raw.items() if k != "per_class"), "Checkpoint metrics")
    new_directory(args.output)
    compare(paths["baseline"], paths["prompted"], paths["finetuned"], args.output / "recomputed")
    retained = Path("reports/three-way-qwen3-06b-v1")
    for name in (
        "comparison.json",
        "paired_predictions.jsonl",
        "changed_errors.jsonl",
        "report.md",
    ):
        require(
            sha256(retained / name) == sha256(args.output / "recomputed" / name),
            f"Three-way recomputation differs: {name}",
        )
    for name in ("result.json", "hardware.json", "run.json", "collator_audit.json"):
        shutil.copyfile(training / name, args.output / name)
    write_jsonl(args.output / "training_log.jsonl", logs)
    shutil.copyfile(selection_dir / "selection.json", args.output / "selection.json")
    shutil.copyfile(selected / "bundle.json", args.output / "selected_bundle.json")
    evidence = {
        "passed": True,
        "fixture": False,
        "split": "val",
        "training_examples": len(records),
        "optimizer_steps": result["step"],
        "verified_bundles": checkpoints,
        "metrics": metrics,
        "training_identity_sha256": object_hash(result["identity"]),
        "prediction_sha256": {name: sha256(path) for name, path in paths.items()},
        "checks": [
            "training data/source/config/lock identity",
            "all checkpoint file hashes",
            "finite complete optimizer logs",
            "selected adapter equals reloaded final adapter",
            "all three complete canonical validation joins",
            "raw metrics and exact policy recomputation",
            "B/C pairing",
            "byte-identical three-way metrics/bootstrap/error recomputation",
        ],
    }
    write_json(args.output / "verification.json", evidence)
    print({"passed": True, "output": str(args.output), "verified_bundles": len(checkpoints)})


if __name__ == "__main__":
    main()
