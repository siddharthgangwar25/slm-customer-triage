"""CPU checks for validation boundaries and EOS-sensitive backend comparisons."""

import importlib.util
from pathlib import Path

import pytest


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("scripts/run_vllm_experiment.py")
worker = load("deployment/vllm_worker.py")


def test_no_test_or_duplicate_rows():
    rows = [
        {"sample_id": str(i), "split": "val", "label": "oos" if i >= 3000 else "a"}
        for i in range(3100)
    ]
    assert [r["sample_id"] for r in runner.select_rows(rows, False)] == [
        "0",
        "1",
        "2",
        "3000",
        "3001",
        "3002",
    ]
    assert len(runner.select_rows(rows, True)) == 3100
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        runner.select_rows(rows, True)
    rows[0]["split"] = "val"
    rows[0]["sample_id"] = "1"
    with pytest.raises(ValueError, match="Duplicate"):
        runner.select_rows(rows, True)


class Tokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return ":".join(str(i) for i in ids)


def test_eos_counted_but_only_terminal_eos_removed():
    result = worker.normalize_output(Tokenizer(), [3, 9, 4, 9], 12, [9, 10])
    assert result["raw_output"] == "3:9:4"
    assert result["output_tokens"] == 4
    assert result["truncated"] is False
    assert runner.compare(result, {**result, "output_tokens": 3}) == ["output_tokens"]
    assert runner.compare(result, {**result, "truncated": True}) == ["truncated"]


@pytest.mark.parametrize("ids", [[], [1, 2]])
def test_no_eos_is_truncated_even_if_json_looks_complete(ids):
    assert worker.normalize_output(Tokenizer(), ids, 8, [9])["truncated"] is True


def test_parser_error_is_not_a_generation_error():
    observed = worker.normalize_output(Tokenizer(), [1, 9], 8, [9])
    reference = {**observed, "error_type": "invalid_model_output"}
    assert runner.compare(reference, observed) == []
    assert runner.compare(reference, {**observed, "error_type": "inference_error"}) == [
        "error_type"
    ]
