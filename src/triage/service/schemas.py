"""Strict HTTP contracts; validation errors never echo user input."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from triage.policy import Reason


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def trim_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank")
        return value


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str
    intent: str | None
    decision: Literal["route", "human_review"]
    reason: Reason
    model_version: str
    policy_version: str
    latency_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        if self.decision == "route":
            if not self.intent or self.intent == "oos" or self.reason != "supported_intent":
                raise ValueError("Route requires a supported intent")
        elif self.intent is not None or self.reason == "supported_intent":
            raise ValueError("Human review requires null intent and a review reason")
        return self


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail


class ServiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_body_bytes: int = Field(default=16384, ge=128, le=1048576)
    max_input_tokens: int = Field(default=512, ge=1)
    inference_timeout_seconds: float = Field(default=5.0, gt=0, le=120, allow_inf_nan=False)
    rate_limit_requests: int = Field(default=60, ge=0)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)


class ModelInfo(BaseModel):
    model_version: str
    policy_version: str
    automatic_routing: bool
    supported_intent_count: int
    model_type: Literal["baseline", "prompted", "finetuned"] = "baseline"
    fixture: bool
