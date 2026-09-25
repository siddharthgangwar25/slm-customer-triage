"""Shared serialized data and prediction contracts."""

from typing import Literal, TypedDict

Split = Literal["train", "val", "test"]


class Record(TypedDict):
    sample_id: str
    text: str
    label: str
    domain: str | None
    split: Split
    source_index: int
    dataset_version: str


class Prediction(Record):
    predicted_label: str | None
    parse_status: str
    model_version: str
    latency_ms: float
    gate_score: float
    token_count: int | None
    error_type: str | None
