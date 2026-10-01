"""CPU gate and authenticated private worker; low gate scores never invoke generation."""

import json
import urllib.error
import urllib.request
from pathlib import Path

from triage.io import read_json
from triage.policy import ModelOutput, parse_model_json
from triage.service.runtime import BaselineAdapter, GateResult, ServiceError


def call_worker(url, key, endpoint, payload=None, timeout=120):
    request = urllib.request.Request(
        url.rstrip("/") + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        raise ServiceError("model_unavailable", "Inference worker unavailable") from None


class RemoteAdapter:
    def __init__(self, bundle: Path, frozen, url, key):
        if not key:
            raise ValueError("TRIAGE_WORKER_KEY required")
        meta = read_json(bundle / "metadata.json")
        gate_policy = {**frozen, "model_version": frozen["gate_model_version"]}
        self.baseline = BaselineAdapter(bundle, gate_policy)
        self.catalog = meta["catalog"]
        self.model_version = frozen["model_version"]
        self.fixture = meta["fixture"]
        self.model_type = "finetuned" if "finetuned" in self.model_version else "prompted"
        self.url, self.key = url, key
        info = call_worker(url, key, "/health/ready")
        if info["model_version"] != self.model_version:
            raise ValueError("Worker/policy model mismatch")
        self.worker_info = info
        self.available = True
        # Candidate metadata is pinned by its validation policy; version alone is insufficient.
        if frozen.get("candidate_metadata_sha256") != info["reference_metadata_sha256"]:
            raise ValueError("Worker/policy candidate provenance mismatch")

    def input_tokens(self, text):
        result = call_worker(self.url, self.key, "/tokenize", {"text": text})
        return result["input_tokens"] if result["within_budget"] else 10**9

    def health(self):
        info = call_worker(self.url, self.key, "/health/ready", timeout=2)
        if info["model_version"] != self.model_version:
            raise ServiceError("model_unavailable", "Worker identity changed")

    def gate(self, text):
        return GateResult(self.baseline.gate(text).score)

    def predict(self, text) -> ModelOutput:
        result = call_worker(self.url, self.key, "/generate", {"text": text})
        if result["model_version"] != self.model_version or result["error_type"]:
            raise ServiceError("model_unavailable", "Worker result invalid")
        return parse_model_json(result["raw_output"], self.catalog, truncated=result["truncated"])
