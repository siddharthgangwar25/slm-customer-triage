"""Bounded CPU execution and trusted bundle loading."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from triage.evaluation.select_policy import load_policy
from triage.io import object_hash, read_json, verify_hash
from triage.policy import Decision, ModelOutput, Policy
from triage.service.telemetry import Telemetry


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


@dataclass(frozen=True)
class GateResult:
    score: float
    cached_output: ModelOutput | None = None


class Adapter(Protocol):
    catalog: list[str]
    model_version: str
    available: bool
    fixture: bool

    def input_tokens(self, text: str) -> int:
        """Full model input budget, including prompt/output reserve for future SLM adapters."""
        ...

    def gate(self, text: str) -> GateResult: ...

    def predict(self, text: str) -> ModelOutput: ...


class BaselineAdapter:
    def __init__(self, bundle: Path, frozen: dict):
        import joblib

        meta = read_json(bundle / "metadata.json")
        for field in ("model_version", "pipeline_sha256", "data_manifest_sha256"):
            if meta[field] != frozen[field]:
                raise ValueError(f"Model/policy {field} mismatch")
        if frozen["gate_model_version"] != meta["model_version"]:
            raise ValueError("Gate model mismatch")
        if object_hash(meta["catalog"]) != frozen["catalog_sha256"]:
            raise ValueError("Policy taxonomy mismatch")
        if meta["fixture"] != frozen["fixture"]:
            raise ValueError("Fixture provenance mismatch")
        for package, expected in meta["environment"]["packages"].items():
            if version(package) != expected:
                raise ValueError(f"Model runtime dependency mismatch: {package}")
        verify_hash(bundle / "pipeline.joblib", frozen["pipeline_sha256"])
        # Caller selects a trusted local path. Hashes are integrity checks, not signatures.
        self.model = joblib.load(bundle / "pipeline.joblib")
        self.catalog = meta["catalog"]
        if (
            list(self.model.classes_) != self.catalog
            or self.catalog != sorted(set(self.catalog))
            or "oos" in self.catalog
        ):
            raise ValueError("Model taxonomy mismatch")
        self.model_version = meta["model_version"]
        self.fixture = meta["fixture"]
        self.available = True
        self.threads = meta["config"]["threads"]
        self.tokenizer = self.model["tfidf"].build_tokenizer()
        self.preprocessor = self.model["tfidf"].build_preprocessor()

    def input_tokens(self, text: str) -> int:
        # The CPU model has no system prompt or generated completion.
        return len(self.tokenizer(self.preprocessor(text)))

    def gate(self, text: str) -> GateResult:
        import numpy as np
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=self.threads):
            probabilities = self.model.predict_proba([text])[0]
        index = int(np.argmax(probabilities))
        return GateResult(float(probabilities[index]), ModelOutput(str(self.model.classes_[index])))

    def predict(self, text: str) -> ModelOutput:
        raise RuntimeError("Baseline must reuse its gate prediction")


class Runtime:
    def __init__(self, adapter: Adapter, policy: Policy):
        self.adapter, self.policy = adapter, policy
        self.telemetry = Telemetry()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="triage-cpu")
        self.slot = threading.Lock()
        self.healthy = True
        self.timed_out_future = None
        self.closed = False

    @property
    def ready(self) -> bool:
        return (
            self.healthy
            and self.adapter.available
            and not self.closed
            and (self.timed_out_future is None or self.timed_out_future.done())
        )

    def _infer(self, text: str, max_tokens: int) -> Decision:
        if not self.adapter.available:
            raise ServiceError("model_unavailable", "Model is unavailable")
        if self.adapter.input_tokens(text) > max_tokens:
            raise ServiceError("token_budget_exceeded", "Input exceeds the model token budget", 422)
        if not self.policy.automatic_routing:
            self.telemetry.increment("gate_rejections")
            return self.policy.decide(0.0, ModelOutput(None, False), self.adapter.catalog)
        gate = self.adapter.gate(text)
        if self.policy.rejects(gate.score):
            self.telemetry.increment("gate_rejections")
            return self.policy.decide(gate.score, ModelOutput(None, False), self.adapter.catalog)
        self.telemetry.increment("model_calls")
        output = (
            gate.cached_output if gate.cached_output is not None else self.adapter.predict(text)
        )
        if not output.valid:
            self.telemetry.increment("invalid_outputs")
        return self.policy.decide(gate.score, output, self.adapter.catalog)

    def _measured_infer(self, text, max_tokens):
        started = time.perf_counter()
        self.telemetry.active(1)
        try:
            result = self._infer(text, max_tokens)
            self.telemetry.increment(result.decision)
            return result
        finally:
            self.telemetry.active(-1)
            self.telemetry.observe("inference", time.perf_counter() - started)

    async def infer(self, text: str, max_tokens: int, timeout: float) -> Decision:
        if not self.ready:
            raise ServiceError("model_unavailable", "Model is unavailable")
        if not self.slot.acquire(blocking=False):
            raise ServiceError("model_busy", "Model is busy; retry later")
        try:
            future = self.executor.submit(self._measured_infer, text, max_tokens)
        except RuntimeError:
            self.slot.release()
            raise ServiceError("model_unavailable", "Model is unavailable") from None

        def finished(done):
            if not done.cancelled():
                error = done.exception()
                if error is not None and (
                    not isinstance(error, ServiceError) or error.status == 503
                ):
                    self.healthy = False
            self.slot.release()

        future.add_done_callback(finished)
        wrapped = asyncio.wrap_future(future)
        # Retrieve late exceptions after a timed-out/disconnected request without logging input.
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            # A timeout cannot kill a native sklearn thread. Keep its slot occupied until done.
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout)
        except TimeoutError:
            self.timed_out_future = future
            raise ServiceError("inference_timeout", "Inference timed out") from None
        except ServiceError:
            raise
        except Exception:
            self.healthy = False
            raise ServiceError("model_unavailable", "Model inference failed") from None

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)


def load_runtime(bundle: Path, policy_path: Path, worker_url=None, worker_key=None) -> Runtime:
    policy, frozen = load_policy(policy_path)
    if worker_url:
        from triage.service.remote import RemoteAdapter

        return Runtime(RemoteAdapter(bundle, frozen, worker_url, worker_key), policy)
    return Runtime(BaselineAdapter(bundle, frozen), policy)
