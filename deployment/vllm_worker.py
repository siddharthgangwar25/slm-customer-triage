"""Isolated, validation-only vLLM worker. Never impersonates a frozen release."""

import json
import os
import platform
import secrets
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from importlib.metadata import distributions
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from triage.io import object_hash, read_json, sha256
from triage.models.prompting import prompt_metadata, system_prompt, tokenize

ENGINE = {
    "model": "/model",
    "tokenizer": "/model",
    "dtype": "float16",
    "quantization": "bitsandbytes",
    "load_format": "bitsandbytes",
    "trust_remote_code": False,
    "tensor_parallel_size": 1,
    "max_model_len": 2080,
    "max_num_seqs": 1,
    "max_num_batched_tokens": 2080,
    "gpu_memory_utilization": 0.65,
    "swap_space": 0,
    "cpu_offload_gb": 0,
    "enforce_eager": True,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": False,
    "enable_lora": True,
    "max_loras": 1,
    "max_lora_rank": 16,
    "seed": 42,
    "generation_config": "vllm",
    "disable_log_stats": True,
}


def normalize_output(tokenizer, ids, prompt_length, eos_ids):
    """Match the reference: count terminal EOS, remove only that token from raw text."""
    ended = bool(ids) and ids[-1] in eos_ids
    return {
        "raw_output": tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False),
        "input_tokens": prompt_length,
        "output_tokens": len(ids),
        "error_type": None,
        "truncated": not ended,
    }


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    candidate: Literal["B", "C"]
    text: str = Field(min_length=1, max_length=2000)


def create_app():
    key = os.environ["VLLM_EXPERIMENT_KEY"]
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        import torch

        print(
            json.dumps(
                {
                    "phase": "runtime_preflight",
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
            ),
            flush=True,
        )
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        started = time.perf_counter()
        payload = read_json(Path("/inputs/identity.json"))
        if (
            sha256(Path("/experiment/requirements.lock"))
            != payload["experiment_files"]["environments/vllm/requirements.lock"]
        ):
            raise ValueError("Image dependency lock differs; rebuild the experiment image")
        tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
        system = system_prompt(payload["catalog"], Path("/inputs/prompt.txt"))
        meta = prompt_metadata(system, tokenizer)
        if any(meta[k] != payload["prompt_metadata"][k] for k in meta):
            raise ValueError("Prompt/tokenizer metadata changed")
        eos = read_json(Path("/model/generation_config.json"))["eos_token_id"]
        app.state.tokenizer, app.state.system, app.state.eos = tokenizer, system, eos
        app.state.sampling = SamplingParams(
            temperature=0,
            max_tokens=32,
            seed=42,
            stop_token_ids=eos,
            skip_special_tokens=False,
            repetition_penalty=1.0,
        )
        app.state.lora = LoRARequest("triage-checkpoint-944", 1, "/adapter")
        app.state.engine = LLM(**ENGINE)
        app.state.metadata = {
            "backend": "vllm-experiment",
            "engine": ENGINE,
            "prompt_metadata": meta,
            "load_seconds": time.perf_counter() - started,
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "environment": {d.metadata["Name"]: d.version for d in distributions()},
            "sampling": str(app.state.sampling),
            "test_inference": False,
        }
        app.state.metadata["variant_id"] = (
            "vllm-"
            + object_hash(
                {
                    "model": {
                        k: payload[k]
                        for k in (
                            "model_id",
                            "revision",
                            "model_files",
                            "adapter_bundle_sha256",
                            "prompt_metadata",
                            "experiment_files",
                        )
                    },
                    "engine": ENGINE,
                    "environment": app.state.metadata["environment"],
                }
            )[:16]
        )
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(value):
        if not secrets.compare_digest(value or "", "Bearer " + key):
            raise HTTPException(401, "Unauthorized")

    @app.get("/health/ready")
    def ready(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return app.state.metadata

    @app.get("/resources")
    def resources(authorization: str | None = Header(default=None)):
        authorize(authorization)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "whole_gpu_sample": result.stdout.strip(),
            "returncode": result.returncode,
            "scope": "instantaneous whole GPU, includes other processes; not workload peak",
        }

    @app.post("/generate")
    def generate(body: Request, authorization: str | None = Header(default=None)):
        authorize(authorization)
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "model_busy")
        try:
            ids = tokenize(app.state.tokenizer, app.state.system, body.text)
            if len(ids) > 2048:
                raise HTTPException(422, "input_too_long")
            started = time.perf_counter()
            result = app.state.engine.generate(
                [{"prompt_token_ids": ids}],
                app.state.sampling,
                lora_request=app.state.lora if body.candidate == "C" else None,
                use_tqdm=False,
            )[0]
            output = result.outputs[0]
            normalized = normalize_output(
                app.state.tokenizer, list(output.token_ids), len(ids), app.state.eos
            )
            return {
                **normalized,
                "candidate": body.candidate,
                "variant_id": app.state.metadata["variant_id"],
                "generated_token_ids": list(output.token_ids),
                "prompt_token_ids_sha256": object_hash(ids),
                "finish_reason": output.finish_reason,
                "stop_reason": output.stop_reason,
                "generation_seconds": time.perf_counter() - started,
            }
        finally:
            lock.release()

    return app


if __name__ == "__main__":
    import uvicorn

    print(json.dumps({"experiment": "vllm", "engine": ENGINE}), flush=True)
    uvicorn.run(create_app(), host="0.0.0.0", port=8001, access_log=False)
