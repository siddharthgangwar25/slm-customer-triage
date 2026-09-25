"""TF-IDF + logistic regression, fitted only on supported training records."""

import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from triage.data.load import load_split
from triage.io import (
    new_directory,
    object_hash,
    read_json,
    sha256,
    verify_hash,
    write_json,
    write_jsonl,
)


def fit_records(rows, catalog, cfg):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Fit operations require exclusively split=train records")
    supported = [row for row in rows if row["label"] != "oos"]
    if set(row["label"] for row in supported) != set(catalog):
        raise ValueError("Training labels do not match catalog")
    if cfg["C"] not in (0.1, 1.0, 10.0):
        raise ValueError("Initial baseline permits only C=0.1, 1.0, or 10.0")
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=tuple(cfg["ngram_range"]), sublinear_tf=cfg["sublinear_tf"]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=cfg["C"], max_iter=cfg["max_iter"], random_state=cfg["seed"], solver="lbfgs"
                ),
            ),
        ]
    )
    start = time.perf_counter()
    with threadpool_limits(limits=cfg["threads"]), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit([r["text"] for r in supported], [r["label"] for r in supported])
    info = {
        "fit_seconds": time.perf_counter() - start,
        "fit_count": len(supported),
        "excluded_oos_count": len(rows) - len(supported),
        "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught],
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "iterations": model["classifier"].n_iter_.tolist(),
    }
    return model, info


def train(cfg, output: Path):
    data_dir = Path(cfg["data_dir"])
    rows, catalog, data_manifest = load_split(data_dir, "train", for_fit=True)
    output = new_directory(output)
    model, fit_info = fit_records(rows, catalog, cfg)
    joblib.dump(model, output / "pipeline.joblib")
    probe = [r["text"] for r in rows[:32]]
    with threadpool_limits(limits=cfg["threads"]):
        before = model.predict_proba(probe)
        after = joblib.load(output / "pipeline.joblib").predict_proba(probe)
    if not np.array_equal(before, after):
        raise ValueError("Artifact reload probability parity failed")
    metadata = {
        "model_version": f"{cfg['experiment_id']}-{object_hash(cfg)[:12]}",
        "fixture": data_manifest.get("fixture", False),
        "config": cfg,
        "config_sha256": object_hash(cfg),
        "catalog": catalog,
        "data_manifest_sha256": sha256(data_dir / "manifest.json"),
        "pipeline_sha256": sha256(output / "pipeline.joblib"),
        "dependency_lock_sha256": sha256(Path("uv.lock")),
        "fit": fit_info,
        "reload_parity": "exact probabilities on 32 training samples",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "packages": {
                p: version(p) for p in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl")
            },
        },
        "precision": "float64",
        "backend": "scikit-learn CPU",
        "prompt_version": None,
        "decoding_config": None,
        "test_evaluated": False,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def predict(cfg, split: str, output: Path, bundle_override: Path | None = None):
    if split != "val":
        raise ValueError("Milestone 1 prediction allows only val; test requires a frozen release")
    data_dir, bundle = Path(cfg["data_dir"]), bundle_override or Path(cfg["bundle"])
    rows, catalog, _ = load_split(data_dir, split)
    metadata = read_json(bundle / "metadata.json")
    verify_hash(bundle / "pipeline.joblib", metadata["pipeline_sha256"])
    verify_hash(data_dir / "manifest.json", metadata["data_manifest_sha256"])
    if catalog != metadata["catalog"] or object_hash(cfg) != metadata["config_sha256"]:
        raise ValueError("Configuration/catalog differs from the trained bundle")
    # joblib is pickle based: CLI callers must use trusted project-controlled artifacts only.
    model = joblib.load(bundle / "pipeline.joblib")
    if list(model.classes_) != catalog:
        raise ValueError("Model classes differ from catalog")
    output = new_directory(output)
    predictions = []
    with threadpool_limits(limits=cfg["threads"]):
        for row in rows:
            start = time.perf_counter()
            probabilities = model.predict_proba([row["text"]])[0]
            best = int(np.argmax(probabilities))
            predictions.append(
                {
                    **row,
                    "predicted_label": str(model.classes_[best]),
                    "gate_score": float(probabilities[best]),
                    "parse_status": "ok",
                    "error_type": None,
                    "token_count": None,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                    "model_version": metadata["model_version"],
                }
            )
    # The baseline label and gate originate from the same computation and same ID.
    if [r["sample_id"] for r in predictions] != [r["sample_id"] for r in rows]:
        raise ValueError("Prediction/gate one-to-one join failed")
    write_jsonl(output / "predictions.jsonl", predictions)
    write_json(
        output / "prediction_manifest.json",
        {
            "schema_version": 1,
            "split": split,
            "sample_count": len(rows),
            "catalog": catalog,
            "sample_ids": [r["sample_id"] for r in rows],
            "fixture": metadata.get("fixture", False),
            "predictions_sha256": sha256(output / "predictions.jsonl"),
            "dataset_version": rows[0]["dataset_version"],
            "data_manifest_sha256": metadata["data_manifest_sha256"],
            "model": metadata,
            "gate_model_version": metadata["model_version"],
            "gate_join": "one-to-one by sample_id, same baseline computation",
            "timing_scope": "serial per-record vectorization + predict_proba + argmax; no API",
        },
    )
    return {"sample_count": len(predictions), "output": str(output)}
