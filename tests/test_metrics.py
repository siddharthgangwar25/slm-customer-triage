import pytest

from triage.evaluation.metrics import classification, routing, wilson


def row(actual, predicted, score=0.8, status="ok", error=None):
    return {
        "label": actual,
        "predicted_label": predicted,
        "gate_score": score,
        "parse_status": status,
        "error_type": error,
    }


def test_hand_calculated_fixed_catalog_macro_f1():
    rows = [
        row("a", "a"),
        row("a", "b"),
        row("b", "b"),
        row("b", None, status="invalid"),
        row("oos", "a"),
    ]
    result = classification(rows, ["a", "b", "missing"])
    # a: TP=1 FN=1 FP=0 => 2/3; b: TP=1 FN=1 FP=1 => 1/2; absent class => 0.
    assert result["raw_supported_macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 3)
    assert result["raw_supported_accuracy"] == 0.5
    assert result["invalid_output_rate"] == 0.2
    assert result["supported_count"] == 4


def test_hand_calculated_routing_and_equal_threshold():
    rows = [
        row("a", "a", 0.7),
        row("a", "b", 0.8),
        row("b", "b", 0.6),
        row("oos", "a", 0.9),
        row("oos", "oos", 0.8),
    ]
    result = routing(rows, ["a", "b"], 0.7)
    assert result["coverage"] == 3 / 5
    assert result["routing_error"] == 2 / 3
    assert result["oos_recall"] == 1 / 2
    assert result["supported_review_rate"] == 1 / 3
    assert result["routed_count"] + result["human_review_count"] == 5


def test_zero_routes_and_empty_subpopulation():
    result = routing([row("a", "a", 0.2)], ["a", "b"], 1.0)
    assert result["routing_error"] is None
    assert result["routing_error_wilson95"] is None
    assert result["coverage"] == 0
    assert result["oos_recall"] is None


@pytest.mark.parametrize(
    "predicted,status,error",
    [
        ("oos", "ok", None),
        ("unknown", "ok", None),
        (None, "invalid", "malformed_json"),
        (None, "invalid", "truncated"),
        (None, "error", "timeout"),
    ],
)
def test_failed_and_oos_outputs_are_misses(predicted, status, error):
    result = classification([row("a", predicted, status=status, error=error)], ["a"])
    assert result["raw_supported_macro_f1"] == 0
    assert result["sample_count"] == 1


def test_infrastructure_failure_is_not_completed_review():
    rows = [row("oos", None, error="timeout", status="error")]
    result = routing(rows, ["a"], 0.0)
    assert result["oos_recall"] == 0
    assert result["human_review_count"] == 0
    assert result["infrastructure_failure_count"] == 1
    assert classification(rows, ["a"])["invalid_output_rate"] == 0
    # Production simulation short-circuits before a model call below the gate.
    assert routing(rows, ["a"], 0.9)["human_review_count"] == 1


def test_wilson_known_values():
    assert wilson(5, 10) == pytest.approx([0.23659309, 0.76340691])
    assert wilson(0, 10) == pytest.approx([0, 0.27753280])
    assert wilson(0, 0) is None


def test_empty_predictions_rejected():
    with pytest.raises(ValueError):
        classification([], ["a"])
