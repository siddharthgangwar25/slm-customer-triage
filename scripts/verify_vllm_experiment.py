"""Audit saved vLLM evidence without inference or changing the original records."""

import argparse
import importlib.util
from collections import Counter
from pathlib import Path

from triage.evaluation.metrics import classification, routing
from triage.io import new_directory, read_json, read_jsonl, sha256, write_json
from triage.policy import parse_model_json


def audit(folder):
    spec = importlib.util.spec_from_file_location(
        "vllm_runner", Path(__file__).with_name("run_vllm_experiment.py")
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checksums = read_json(folder / "checksums.json")
    for name, digest in checksums.items():
        if sha256(folder / name) != digest:
            raise ValueError("Evidence checksum mismatch: " + name)
    identity = read_json(folder / "inputs/identity.json")
    original = read_json(folder / "summary.json")
    runtime = read_json(folder / "runtime.json")
    result = {
        "checksums_verified": len(checksums),
        "test_inference": False,
        "candidates": {},
        "auth_check": original["auth_check"],
    }
    for candidate, stem in runner.STEMS.items():
        reference_file = Path("reports") / (stem + "-val") / "predictions.jsonl"
        if sha256(reference_file) != identity["references"][candidate]:
            raise ValueError("Frozen validation reference changed")
        reference = runner.select_rows(read_jsonl(reference_file), identity["full_validation"])
        observed = read_jsonl(folder / f"{candidate}-comparison.jsonl")
        predictions = read_jsonl(folder / f"{candidate}-predictions.jsonl")
        if len(reference) != len(observed) or len(reference) != len(predictions):
            raise ValueError("Missing/extra validation outcomes")
        exact, failures, fields = 0, 0, Counter()
        for ref, item, prediction in zip(reference, observed, predictions, strict=True):
            assert ref["sample_id"] == item["sample_id"] == prediction["sample_id"]
            assert item["reference"] == {k: ref[k] for k in runner.FIELDS}
            raw = item["observed"]
            if raw["error_type"]:
                failures += 1
            else:
                assert raw["candidate"] == candidate
                assert raw["variant_id"] == runtime["variant_id"]
                assert len(raw["generated_token_ids"]) == raw["output_tokens"]
                ended = bool(raw["generated_token_ids"]) and raw["generated_token_ids"][-1] in (
                    151643,
                    151645,
                )
                assert raw["truncated"] == (not ended)
            differences = runner.compare(ref, raw)
            exact += not differences
            fields.update(differences)
            assert not runner.compare(prediction, raw)
            parsed = parse_model_json(
                raw["raw_output"], identity["catalog"], truncated=raw["truncated"]
            )
            assert prediction["predicted_label"] == parsed.label
            assert prediction["parse_status"] == ("ok" if parsed.valid else "invalid")
        recorded = original["candidates"][candidate]
        assert classification(predictions, identity["catalog"]) == recorded["classification"]
        assert (
            routing(predictions, identity["catalog"], identity["policies"][candidate]["threshold"])
            == recorded["frozen_threshold_diagnostic"]
        )
        assert recorded["submitted"] == len(observed)
        assert recorded["infrastructure_failures"] == failures
        result["candidates"][candidate] = {
            "submitted": len(observed),
            "completed": len(observed) - failures,
            "generation_exact_matches": exact,
            "mismatch_fields": dict(fields),
            "original_exact_matches": recorded["exact_matches"],
            "classification_and_frozen_threshold_metrics_verified": True,
        }
    frozen = read_json(folder / "frozen_release_check.json")
    assert frozen["unchanged"] and frozen["before"] == frozen["after"]
    for name, digest in frozen["after"].items():
        assert sha256(Path(name)) == digest
    result["frozen_release_and_ledger_unchanged"] = True
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    output = new_directory(args.output)
    write_json(output / "audit.json", result)
    print(result)
