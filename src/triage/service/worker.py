"""Private single-flight Transformers process, preserving frozen B/C inference settings."""

import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from triage.io import object_hash, read_json, sha256
from triage.models.prompting import system_prompt, tokenize


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=2000)


def create_worker(predictions: Path, key: str, factory=None):
    if not key:
        raise ValueError("TRIAGE_WORKER_KEY is required")
    manifest = read_json(predictions.parent / "prediction_manifest.json")
    cfg = manifest["model"]["config"]
    if os.name != "nt" and "adapter" in cfg:
        cfg = {**cfg, "adapter": cfg["adapter"].replace("\\", "/")}
    catalog = manifest["catalog"]
    if manifest["split"] != "val" or not manifest["benchmark_complete"]:
        raise ValueError("Worker requires a completed validation identity")
    if sha256(predictions) != manifest["predictions_sha256"]:
        raise ValueError("Worker reference predictions changed")
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        from triage.models.finetuned import FinetunedAdapter
        from triage.models.prompted import TransformersAdapter

        started = time.perf_counter()
        cls = factory or (
            FinetunedAdapter if cfg["model_type"] == "finetuned" else TransformersAdapter
        )
        app.state.adapter = cls(cfg, system_prompt(catalog, Path(cfg["prompt_file"])), ["warmup"])
        for name in ("system_prompt_sha256", "chat_template_sha256"):
            if app.state.adapter.metadata[name] != manifest["model"][name]:
                raise ValueError("Worker prompt/template differs from reference")
        app.state.load_seconds = time.perf_counter() - started
        yield
        app.state.adapter = None

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.adapter = None

    def authorize(authorization):
        if not secrets.compare_digest(authorization or "", "Bearer " + key):
            raise HTTPException(401, "Unauthorized")
        if app.state.adapter is None:
            raise HTTPException(503, "Unavailable")

    @app.get("/health/ready")
    def ready(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {
            "status": "ready",
            "model_version": manifest["model"]["model_version"],
            "reference_metadata_sha256": object_hash(manifest["model"]),
            "load_seconds": app.state.load_seconds,
            "backend": "transformers-process",
            "metadata": app.state.adapter.metadata,
            "memory": app.state.adapter.memory(),
        }

    @app.post("/tokenize")
    def token_count(body: WorkerRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        adapter = app.state.adapter
        size = len(tokenize(adapter.tokenizer, adapter.system, body.text))
        return {"input_tokens": size, "within_budget": size <= cfg["max_input_tokens"]}

    @app.post("/generate")
    def generate(body: WorkerRequest, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "Busy")
        try:
            result = app.state.adapter.generate(body.text)
            return {**result, "model_version": manifest["model"]["model_version"]}
        except RuntimeError:
            raise HTTPException(503, "Inference failed") from None
        finally:
            lock.release()

    return app


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(
        create_worker(args.predictions, os.environ.get("TRIAGE_WORKER_KEY", "")),
        host=args.host,
        port=args.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
