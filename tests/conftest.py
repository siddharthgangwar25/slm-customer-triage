"""Tiny synthetic data only. No network or real benchmarks in the test suite."""

import pytest

from triage.data.prepare import prepare
from triage.io import sha256, write_json


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    raw = {
        "train": [
            ["pay my bill", "bill"],
            ["settle bill", "bill"],
            ["weather forecast", "weather"],
            ["rain weather", "weather"],
        ],
        "val": [["bill payment validationonly", "bill"], ["weather today", "weather"]],
        "test": [["bill testonlyword", "bill"], ["weather tomorrow", "weather"]],
        "oos_train": [["unrelated oostrainword", "oos"]],
        "oos_val": [["unrelated request", "oos"]],
        "oos_test": [["unrelated test", "oos"]],
    }
    domains = {"finance": ["bill"], "general": ["weather"]}
    write_json(directory / "data_full.json", raw)
    write_json(directory / "domains.json", domains)
    (directory / "LICENSE").write_text("Synthetic test fixture", encoding="utf-8")
    cfg = {
        "repository": "https://github.com/clinc/oos-eval",
        "commit": "a" * 40,
        "fixture": True,
        "license": "synthetic fixture",
        "expected_intents": 2,
        "preprocessing_version": "fixture-v1",
        "expected_counts": {key: len(value) for key, value in raw.items()},
        "files": {
            name: {"path": name, "sha256": sha256(directory / name)}
            for name in ("data_full.json", "domains.json", "LICENSE")
        },
    }
    return directory, cfg, raw, domains


@pytest.fixture
def dataset(source, tmp_path):
    directory, cfg, _, _ = source
    output = tmp_path / "data"
    prepare(cfg, output, directory)
    return output


@pytest.fixture
def baseline_config(dataset, tmp_path):
    return {
        "experiment_id": "TEST-FIXTURE",
        "data_dir": str(dataset),
        "bundle": str(tmp_path / "bundle"),
        "prediction_output": str(tmp_path / "predictions"),
        "C": 1.0,
        "seed": 42,
        "max_iter": 2000,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "threads": 1,
    }
