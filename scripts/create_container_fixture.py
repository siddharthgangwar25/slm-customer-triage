"""Create explicitly synthetic CPU container health-check artifacts; never benchmark evidence."""

import argparse
from pathlib import Path

from triage.data.prepare import prepare
from triage.evaluation.select_policy import select_policy
from triage.io import new_directory, sha256, write_json
from triage.models.baseline import predict, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = new_directory(parser.parse_args().output)
    source = new_directory(output / "source")
    raw = {
        "train": [
            ["pay bill", "bill"],
            ["settle bill", "bill"],
            ["weather forecast", "weather"],
            ["rain weather", "weather"],
        ],
        "val": [["bill payment", "bill"], ["weather today", "weather"]],
        "test": [["test bill", "bill"], ["test weather", "weather"]],
        "oos_train": [["unrelated train", "oos"]],
        "oos_val": [["unrelated validation", "oos"]],
        "oos_test": [["unrelated test", "oos"]],
    }
    write_json(source / "data_full.json", raw)
    write_json(source / "domains.json", {"finance": ["bill"], "general": ["weather"]})
    (source / "LICENSE").write_text("Synthetic fixture", encoding="utf-8")
    data_cfg = {
        "repository": "https://github.com/clinc/oos-eval",
        "commit": "a" * 40,
        "fixture": True,
        "license": "synthetic fixture",
        "expected_intents": 2,
        "preprocessing_version": "container-fixture-v1",
        "expected_counts": {k: len(v) for k, v in raw.items()},
        "files": {
            name: {"path": name, "sha256": sha256(source / name)}
            for name in ("data_full.json", "domains.json", "LICENSE")
        },
    }
    prepare(data_cfg, output / "data", source)
    cfg = {
        "experiment_id": "CONTAINER-TEST-FIXTURE",
        "data_dir": str(output / "data"),
        "bundle": str(output / "bundle"),
        "prediction_output": str(output / "predictions"),
        "C": 1.0,
        "seed": 42,
        "max_iter": 2000,
        "ngram_range": [1, 2],
        "sublinear_tf": True,
        "threads": 1,
    }
    train(cfg, output / "bundle")
    predict(cfg, "val", output / "predictions")
    select_policy(output / "predictions/predictions.jsonl", "val", output / "policy")


if __name__ == "__main__":
    main()
