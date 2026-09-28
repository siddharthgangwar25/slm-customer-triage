"""Evaluate every completed epoch on validation and freeze the specified tie-break choice."""

from pathlib import Path

from triage.evaluation.metrics import classification
from triage.evaluation.report import evaluate, load_predictions
from triage.evaluation.select_policy import select_policy
from triage.io import new_directory, read_json, sha256, write_json
from triage.models.finetuned import FinetunedAdapter, prediction_config
from triage.models.prompted import run as predict
from triage.training.data import identity
from triage.training.runner import verify_bundle


def choose_checkpoint(metrics):
    if not metrics:
        raise ValueError("No completed checkpoint validation results")
    return min(
        metrics, key=lambda m: (-m["raw_supported_macro_f1"], m["invalid_output_count"], m["step"])
    )


def select(cfg, output, *, resume=False):
    training_dir = Path(cfg["output"])
    result = read_json(training_dir / "result.json")
    if result["status"] != "complete" or result["identity"] != identity(cfg):
        raise ValueError("Checkpoint selection requires completed, unchanged training")
    if not result["reload_parity"]["passed"] or len(result["epoch_checkpoints"]) != cfg["epochs"]:
        raise ValueError("Missing reload parity or epoch checkpoints")
    frozen = {
        "training_result_sha256": sha256(training_dir / "result.json"),
        "identity": identity(cfg),
    }
    if resume:
        if read_json(output / "run.json") != frozen:
            raise ValueError("Checkpoint selection resume identity changed")
    else:
        new_directory(output)
        write_json(output / "run.json", frozen)
    measured = []
    for name in result["epoch_checkpoints"]:
        bundle_path = training_dir / name
        bundle = verify_bundle(bundle_path)
        pred_dir = output / name
        if not (pred_dir / "prediction_manifest.json").exists():
            predict(
                prediction_config(cfg, bundle_path),
                "val",
                pred_dir,
                resume=pred_dir.exists(),
                adapter_factory=FinetunedAdapter,
            )
        rows, catalog, manifest = load_predictions(pred_dir / "predictions.jsonl")
        if manifest.get("benchmark_complete") is not True or manifest["model"][
            "adapter_bundle_sha256"
        ] != sha256(bundle_path / "bundle.json"):
            raise ValueError("Checkpoint validation evidence mismatch")
        raw = classification(rows, catalog)
        measured.append(
            {
                "checkpoint": str(bundle_path),
                "predictions": str(pred_dir / "predictions.jsonl"),
                "predictions_sha256": sha256(pred_dir / "predictions.jsonl"),
                "step": bundle["step"],
                "epoch": bundle["epoch"],
                **{k: v for k, v in raw.items() if k != "per_class"},
            }
        )
        write_json(output / "checkpoint_metrics.json", measured)
    selected = choose_checkpoint(measured)
    if not (output / "selected-report").exists():
        evaluate(Path(selected["predictions"]), output / "selected-report")
    if not (output / "selected-policy").exists():
        select_policy(Path(selected["predictions"]), "val", output / "selected-policy")
    selection = {
        "split": "val",
        "training_result_sha256": frozen["training_result_sha256"],
        "rule": "highest raw supported macro-F1; fewer invalid outputs; earlier step",
        "selected": selected,
        "all_checkpoints": measured,
    }
    write_json(output / "selection.json", selection)
    return selection
