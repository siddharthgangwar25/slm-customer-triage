import math
import random
from pathlib import Path

import pytest

from triage.evaluation.metrics import routing
from triage.evaluation.select_policy import (
    ALL_REVIEW_THRESHOLD,
    choose_point,
    exact_sweep,
    load_policy,
    select_policy,
)
from triage.io import read_json, read_jsonl, sha256, write_json, write_jsonl
from triage.models.baseline import predict, train
from triage.policy import ModelOutput, Policy, parse_model_json


@pytest.mark.parametrize(
    "score,label,valid,decision,reason",
    [
        (0.49, "a", True, "human_review", "low_gate_score"),
        (0.5, "a", True, "route", "supported_intent"),
        (0.8, "oos", True, "human_review", "out_of_scope"),
        (0.8, "unknown", True, "human_review", "invalid_model_output"),
        (0.8, "a", False, "human_review", "invalid_model_output"),
    ],
)
def test_policy_contract_boundaries(score, label, valid, decision, reason):
    result = Policy(0.5, True, "fixture").decide(score, ModelOutput(label, valid), ["a", "b"])
    assert result.decision == decision
    assert result.reason == reason
    assert result.intent == (label if decision == "route" else None)


def test_disabled_routing_overrides_valid_prediction():
    result = Policy(0.0, False, "fixture").decide(1, ModelOutput("a"), ["a"])
    assert result.intent is None
    assert result.reason == "low_gate_score"


@pytest.mark.parametrize(
    "raw",
    [
        'prose {"intent":"a"}',
        '{"intent":"a"} extra',
        '{"intent":"a", "extra":1}',
        '{"intent":"a","intent":"b"}',
        '{"intent":"unknown"}',
        '[{"intent":"a"}]',
        '{"intent":null}',
        '{"intent":7}',
        "not json",
        "{}",
    ],
)
def test_strict_model_output(raw):
    assert not parse_model_json(raw, ["a", "b"]).valid


def test_valid_json_and_truncation():
    assert parse_model_json(' {"intent":"a"} ', ["a"]) == ModelOutput("a")
    assert parse_model_json('{"intent":"oos"}', ["a"]) == ModelOutput("oos")
    assert not parse_model_json('{"intent":"a"}', ["a"], truncated=True).valid


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.1])
def test_invalid_gate_score_is_not_review(score):
    with pytest.raises(ValueError):
        Policy(0.5, True, "fixture").decide(score, ModelOutput("a"), ["a"])


def sample(gold, label, score, error=None):
    return {
        "label": gold,
        "predicted_label": label,
        "gate_score": score,
        "error_type": error,
        "parse_status": "ok" if error is None else "error",
    }


def test_exact_sweep_matches_independent_simulation_with_failures():
    rng = random.Random(42)
    rows = [
        sample(
            rng.choice(["a", "b", "oos"]),
            rng.choice(["a", "b", "oos", "bad"]),
            rng.choice([0, 0.2, 0.5, 1]),
            rng.choice([None, None, "timeout", "truncated"]),
        )
        for _ in range(60)
    ]
    for point in exact_sweep(rows, ["a", "b"]):
        reference = routing(rows, ["a", "b"], point["threshold"])
        for field, value in point.items():
            assert reference[field] == value, field


def test_select_highest_coverage_exact_boundary():
    rows = [sample("a", "a", 0.9) for _ in range(8)]
    rows += [sample("oos", "a", 0.1), sample("oos", "a", 0.2)]
    best = choose_point(exact_sweep(rows, ["a"]))
    assert best["threshold"] == 0.9
    assert best["coverage"] == 0.8
    assert best["routing_error"] == 0
    assert best["oos_recall"] == 1


def test_tie_breaks_error_then_higher_threshold():
    points = [
        {"coverage": 0.5, "routing_error": err, "oos_recall": 1, "threshold": t}
        for err, t in [(0.04, 0.9), (0.02, 0.5), (0.02, 0.7)]
    ]
    assert choose_point(points)["threshold"] == 0.7


@pytest.mark.parametrize(
    "coverage,error,recall",
    [(0.19, 0, 1), (0.5, 0.051, 1), (0.5, 0.01, 0.89), (0, None, 1), (0.5, 0, None)],
)
def test_unmet_constraints_never_relaxed(coverage, error, recall):
    assert (
        choose_point(
            [{"coverage": coverage, "routing_error": error, "oos_recall": recall, "threshold": 0.5}]
        )
        is None
    )


def test_policy_freeze_and_tamper_detection(baseline_config, tmp_path):
    cfg = baseline_config
    train(cfg, Path(cfg["bundle"]))
    pred_dir = Path(cfg["prediction_output"])
    predict(cfg, "val", pred_dir)
    output = tmp_path / "policy"
    frozen = select_policy(pred_dir / "predictions.jsonl", "val", output)
    policy, loaded = load_policy(output / "policy.json")
    assert loaded == frozen
    assert frozen["fixture"] is True
    assert math.isfinite(policy.threshold)
    with pytest.raises(FileExistsError):
        select_policy(pred_dir / "predictions.jsonl", "val", output)
    changed = read_json(output / "policy.json")
    changed["threshold"] = 0
    write_json(output / "policy.json", changed)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_policy(output / "policy.json")


def test_selection_rejects_test_before_reading(tmp_path):
    with pytest.raises(ValueError, match="split=val"):
        select_policy(tmp_path / "absent", "test", tmp_path / "output")


def test_all_review_endpoint_includes_score_one():
    rows = [sample("oos", "a", 1)]
    points = exact_sweep(rows, ["a"])
    assert points[0]["threshold"] == ALL_REVIEW_THRESHOLD
    assert points[0]["coverage"] == 0
    assert points[0]["routing_error"] is None
    assert choose_point(points) is None


def test_unmet_policy_is_frozen_as_review_only(baseline_config, tmp_path):
    cfg = baseline_config
    train(cfg, Path(cfg["bundle"]))
    pred_dir = Path(cfg["prediction_output"])
    predict(cfg, "val", pred_dir)
    path = pred_dir / "predictions.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        row["gate_score"] = 1.0
        row["predicted_label"] = "bill"
    write_jsonl(path, rows)
    manifest = read_json(pred_dir / "prediction_manifest.json")
    manifest["predictions_sha256"] = sha256(path)
    write_json(pred_dir / "prediction_manifest.json", manifest)
    select_policy(path, "val", tmp_path / "unmet")
    policy, frozen = load_policy(tmp_path / "unmet" / "policy.json")
    assert not policy.automatic_routing
    assert frozen["status"] == "targets_unmet"
    assert frozen["validation"]["coverage"] == 0
    assert frozen["validation"]["routing_error"] is None
    assert "TEST FIXTURE" in (tmp_path / "unmet" / "report.md").read_text(encoding="utf-8")
    assert "targets_unmet" in (tmp_path / "unmet" / "report.md").read_text(encoding="utf-8")


def test_constraints_are_inclusive():
    point = {"coverage": 0.2, "routing_error": 0.05, "oos_recall": 0.9, "threshold": 0.5}
    assert choose_point([point]) == point
