"""Frozen routing contract. A review decision recommends escalation, not ticket creation."""

import json
import math
from dataclasses import dataclass
from typing import Literal

Reason = Literal["supported_intent", "low_gate_score", "out_of_scope", "invalid_model_output"]


@dataclass(frozen=True)
class ModelOutput:
    label: str | None
    valid: bool = True


@dataclass(frozen=True)
class Decision:
    intent: str | None
    decision: Literal["route", "human_review"]
    reason: Reason


@dataclass(frozen=True)
class Policy:
    threshold: float
    automatic_routing: bool
    policy_version: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= math.nextafter(
            1.0, math.inf
        ):
            raise ValueError("Invalid gate threshold")

    def rejects(self, score: float) -> bool:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid gate score")
        return not self.automatic_routing or score < self.threshold

    def decide(self, score: float, output: ModelOutput, catalog: list[str]) -> Decision:
        if self.rejects(score):
            return Decision(None, "human_review", "low_gate_score")
        if not output.valid or output.label not in catalog + ["oos"]:
            return Decision(None, "human_review", "invalid_model_output")
        if output.label == "oos":
            return Decision(None, "human_review", "out_of_scope")
        return Decision(output.label, "route", "supported_intent")


def parse_model_json(raw: str, catalog: list[str], *, truncated: bool = False) -> ModelOutput:
    """Strict future-adapter boundary: no prose, repairs, duplicate keys, or extra fields."""
    if truncated:
        return ModelOutput(None, False)

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return ModelOutput(None, False)
    if not isinstance(value, dict) or set(value) != {"intent"}:
        return ModelOutput(None, False)
    label = value["intent"]
    if not isinstance(label, str) or label not in catalog + ["oos"]:
        return ModelOutput(None, False)
    return ModelOutput(label)
