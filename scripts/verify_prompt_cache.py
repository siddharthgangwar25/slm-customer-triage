"""Real-model cache parity and timing probe; validation only, not quality evidence."""

import argparse
import time
from pathlib import Path

import yaml

from triage.data.load import load_split
from triage.io import new_directory, object_hash, write_json
from triage.models.prompted import TransformersAdapter, validate_config
from triage.models.prompting import system_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/prompted.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["prefix_cache"] = True
    validate_config(cfg)
    rows, catalog, _ = load_split(Path(cfg["data_dir"]), "val")
    output = new_directory(args.output)
    adapter = TransformersAdapter(
        cfg, system_prompt(catalog, Path(cfg["prompt_file"])), [r["text"] for r in rows]
    )
    cache = adapter.prefix_cache
    results = []
    # Fixed indices cover different catalog regions and oos; no selection on predictions.
    for index in (0, 500, 1000, 1500, 2500, 3099):
        row = rows[index]
        record = {"sample_id": row["sample_id"], "source_index": index}
        for mode in ("uncached", "cached"):
            adapter.prefix_cache = cache if mode == "cached" else None
            start = time.perf_counter()
            result = adapter.generate(row["text"])
            record[mode] = {"result": result, "seconds": time.perf_counter() - start}
        record["equal"] = record["uncached"]["result"] == record["cached"]["result"]
        results.append(record)
        print(record, flush=True)
    evidence = {
        "purpose": "hardware/cache smoke, not validation quality benchmark",
        "config": cfg,
        "config_sha256": object_hash(cfg),
        "metadata": adapter.metadata,
        "memory": adapter.memory(),
        "comparisons": results,
        "all_equal": all(r["equal"] for r in results),
    }
    write_json(output / "cache_parity.json", evidence)
    if not evidence["all_equal"]:
        raise ValueError("Cache parity failed; inspect evidence before choosing a backend")


if __name__ == "__main__":
    main()
