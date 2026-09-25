import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from triage.data.load import load_split
from triage.evaluation.report import evaluate, load_predictions
from triage.io import read_json, read_jsonl, sha256, write_json, write_jsonl
from triage.models.baseline import fit_records, predict, train


def test_training_cannot_see_held_out_vocabulary(baseline_config, dataset):
    cfg = baseline_config
    # Training succeeds without even having held-out files available.
    (dataset / "test.jsonl").unlink()
    val_bytes = (dataset / "val.jsonl").read_bytes()
    (dataset / "val.jsonl").unlink()
    metadata = train(cfg, Path(cfg["bundle"]))
    model = joblib.load(Path(cfg["bundle"]) / "pipeline.joblib")
    vocabulary = model["tfidf"].vocabulary_
    assert "testonlyword" not in vocabulary
    assert "validationonly" not in vocabulary
    assert "oostrainword" not in vocabulary
    assert metadata["fit"]["fit_count"] == 4
    assert metadata["fixture"] is True
    (dataset / "val.jsonl").write_bytes(val_bytes)
    before = dict(vocabulary)
    predict(cfg, "val", Path(cfg["prediction_output"]))
    assert vocabulary == before
    assert "validationonly" not in vocabulary


@pytest.mark.parametrize("split", ["val", "test"])
def test_direct_fit_rejects_held_out_records(baseline_config, dataset, split):
    rows, catalog, _ = load_split(dataset, split)
    with pytest.raises(ValueError, match="split=train"):
        fit_records(rows, catalog, baseline_config)


def test_save_reload_predictions_exact(baseline_config, dataset):
    rows, catalog, _ = load_split(dataset, "train")
    model, _ = fit_records(rows, catalog, baseline_config)
    bundle = Path(baseline_config["bundle"])
    train(baseline_config, bundle)
    loaded = joblib.load(bundle / "pipeline.joblib")
    probe = ["bill payment", "rain tomorrow", "something unseen"]
    np.testing.assert_array_equal(model.predict(probe), loaded.predict(probe))
    np.testing.assert_array_equal(model.predict_proba(probe), loaded.predict_proba(probe))


def test_end_to_end_cli_fixture_and_model_free_evaluation(baseline_config, tmp_path):
    import yaml

    cfg = baseline_config
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def run(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "triage.cli", *args], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stderr
        return json.loads(completed.stdout)

    run("train", "baseline", "--config", str(path))
    run("predict", "--config", str(path), "--split", "val")
    # Saved predictions contain everything evaluation needs, including taxonomy and IDs.
    (Path(cfg["bundle"]) / "pipeline.joblib").unlink()
    result = run(
        "evaluate",
        "--predictions",
        str(Path(cfg["prediction_output"]) / "predictions.jsonl"),
        "--output",
        str(tmp_path / "report"),
    )
    assert result["sample_count"] == 3
    assert result["fixture"] is True
    assert "TEST FIXTURE" in (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert (tmp_path / "report" / "coverage_error.png").is_file()
    with pytest.raises(FileExistsError):
        evaluate(Path(cfg["prediction_output"]) / "predictions.jsonl", tmp_path / "report")


def test_prediction_missing_ids_and_gate_scores_rejected(baseline_config):
    cfg = baseline_config
    train(cfg, Path(cfg["bundle"]))
    output = Path(cfg["prediction_output"])
    predict(cfg, "val", output)
    path = output / "predictions.jsonl"
    rows = read_jsonl(path)
    manifest = read_json(output / "prediction_manifest.json")
    rows[0]["sample_id"] = "unexpected"
    write_jsonl(path, rows)
    manifest["predictions_sha256"] = sha256(path)
    write_json(output / "prediction_manifest.json", manifest)
    with pytest.raises(ValueError, match="one-to-one"):
        load_predictions(path)
    rows[0]["sample_id"] = manifest["sample_ids"][0]
    rows[0]["gate_score"] = 2
    write_jsonl(path, rows)
    manifest["predictions_sha256"] = sha256(path)
    write_json(output / "prediction_manifest.json", manifest)
    with pytest.raises(ValueError, match="Gate scores"):
        load_predictions(path)


def test_test_predictions_blocked(baseline_config, tmp_path):
    with pytest.raises(ValueError, match="frozen release"):
        predict(baseline_config, "test", tmp_path / "forbidden")


def test_artifact_hash_checked_before_deserialization(baseline_config):
    cfg = baseline_config
    train(cfg, Path(cfg["bundle"]))
    (Path(cfg["bundle"]) / "pipeline.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        predict(cfg, "val", Path(cfg["prediction_output"]))


def test_missing_input_cli_error(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triage.cli",
            "data",
            "prepare",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "triage:" in completed.stderr
    assert "Traceback" not in completed.stderr
