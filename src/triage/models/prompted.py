"""Optional pinned Transformers runner. Imports GPU libraries only inside execution paths."""

import json
import platform
import re
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from triage.data.load import load_split
from triage.evaluation.gate import join_gate
from triage.io import (
    new_directory,
    object_hash,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
    write_jsonl,
)
from triage.models.prompting import prompt_metadata, system_prompt, tokenize
from triage.policy import parse_model_json


def validate_config(cfg):
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["revision"]):
        raise ValueError("Model revision must be a full immutable commit")
    if cfg["enable_thinking"] is not False or cfg["do_sample"] is not False:
        raise ValueError("This experiment uses non-thinking, deterministic greedy decoding")
    if cfg["batch_size"] != 1:
        raise ValueError("The reference runner supports batch_size=1")
    if type(cfg.get("prefix_cache", False)) is not bool:
        raise ValueError("prefix_cache must be a boolean")
    if cfg["dtype"] not in ("float16", "float32", "bfloat16"):
        raise ValueError("Unsupported precision")
    if cfg["quantization"] not in ("none", "nf4"):
        raise ValueError("Unsupported quantization")
    if cfg["max_input_tokens"] < 1 or cfg["max_new_tokens"] < 1:
        raise ValueError("Token budgets must be positive")


class TransformersAdapter:
    def __init__(self, cfg, system, texts):
        import psutil
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.cfg, self.system, self.torch = cfg, system, torch
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(4)
        if cfg["device"].startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Configured CUDA device is unavailable; no silent CPU fallback")
        start = time.perf_counter()
        metadata_snapshot = snapshot_download(
            cfg["model_id"],
            revision=cfg["revision"],
            cache_dir=cfg["cache_dir"],
            allow_patterns=["*.json", "*.txt", "*.jinja", "*.model", "LICENSE", "README.md"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            metadata_snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.metadata = prompt_metadata(system, self.tokenizer)
        lengths = [len(tokenize(self.tokenizer, system, text)) for text in texts]
        rendered = self.tokenizer.decode(tokenize(self.tokenizer, system, "smoke request"))
        if not rendered.endswith("<think>\n\n</think>\n\n"):
            raise ValueError("Pinned Qwen template did not disable thinking as expected")
        self.metadata["token_budget"] = {
            "sample_count": len(lengths),
            "min_input_tokens": min(lengths),
            "max_input_tokens": max(lengths),
            "configured_input_limit": cfg["max_input_tokens"],
            "generation_reserve": cfg["max_new_tokens"],
            "would_exceed_input_budget": sum(n > cfg["max_input_tokens"] for n in lengths),
            "non_thinking_template_verified": True,
        }
        if max(lengths) > cfg["max_input_tokens"]:
            raise ValueError(
                "Full catalog plus input exceeds token budget; no truncation is allowed"
            )
        kwargs = {
            "trust_remote_code": False,
            "local_files_only": True,
            "dtype": getattr(torch, cfg["dtype"]),
            "device_map": {"": cfg["device"]},
            "attn_implementation": "sdpa",
        }
        if cfg["quantization"] == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, cfg["dtype"]),
            )
        weight_snapshot = snapshot_download(
            cfg["model_id"],
            revision=cfg["revision"],
            cache_dir=cfg["cache_dir"],
            allow_patterns=["*.safetensors"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(weight_snapshot, **kwargs).eval()
        self.prefix_ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        self.prefix_cache = None
        if cfg.get("prefix_cache", False):
            prefix = torch.tensor([self.prefix_ids], device=cfg["device"])
            with torch.inference_mode():
                self.prefix_cache = self.model(
                    input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True
                ).past_key_values
        if (
            cfg["max_input_tokens"] + cfg["max_new_tokens"]
            > self.model.config.max_position_embeddings
        ):
            raise ValueError("Input plus generation budget exceeds the model context")
        self.cuda = cfg["device"].startswith("cuda")
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.metadata.update(
            {
                "load_seconds": time.perf_counter() - start,
                "parameter_count": self.model.num_parameters(),
                "model_memory_bytes": self.model.get_memory_footprint(),
                "prefix_cache_tokens": len(self.prefix_ids) if self.prefix_cache else 0,
                "generation_defaults": self.model.generation_config.to_dict(),
                "context_limit": self.model.config.max_position_embeddings,
                "hardware": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "ram_total_bytes": psutil.virtual_memory().total,
                    "gpu": torch.cuda.get_device_name() if self.cuda else None,
                    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory
                    if self.cuda
                    else None,
                    "torch_cuda": torch.version.cuda,
                },
                "environment": {
                    p: version(p)
                    for p in ("torch", "transformers", "accelerate", "bitsandbytes", "psutil")
                },
            }
        )

    def generate(self, text):
        torch = self.torch
        ids = tokenize(self.tokenizer, self.system, text)
        if len(ids) > self.cfg["max_input_tokens"]:
            return {
                "raw_output": "",
                "input_tokens": len(ids),
                "output_tokens": 0,
                "error_type": "input_too_long",
                "truncated": False,
            }
        inputs = torch.tensor([ids], device=self.cfg["device"])
        cache_kwargs = {}
        if self.prefix_cache is not None:
            if ids[: len(self.prefix_ids)] != self.prefix_ids:
                raise ValueError("Input does not match the cached system prefix")
            # Generation mutates its cache. Never share request tokens with another request.
            cache_kwargs["past_key_values"] = deepcopy(self.prefix_cache)
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self.cfg["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                **cache_kwargs,
            )
        generated = outputs[0, len(ids) :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        ended = bool(generated) and generated[-1] in eos
        return {
            "raw_output": self.tokenizer.decode(
                generated[:-1] if ended else generated, skip_special_tokens=False
            ),
            "input_tokens": len(ids),
            "output_tokens": len(generated),
            "error_type": None,
            "truncated": not ended,
        }

    def memory(self):
        import psutil

        info = psutil.Process().memory_info()
        return {
            "process_rss_bytes": info.rss,
            "peak_process_rss_bytes": getattr(info, "peak_wset", info.rss),
            "peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated() if self.cuda else 0,
            "peak_gpu_reserved_bytes": self.torch.cuda.max_memory_reserved() if self.cuda else 0,
        }


def run(cfg, split: str, output: Path, *, limit=None, resume=False, adapter_factory=None):
    if split != "val":
        raise ValueError("Prompted predictions require val; test requires a frozen release")
    validate_config(cfg)
    data_dir = Path(cfg["data_dir"])
    rows, catalog, data_manifest = load_split(data_dir, "val")
    # Load training only to verify catalog provenance; no held-out examples enter the prompt.
    _, training_catalog, _ = load_split(data_dir, "train", for_fit=True)
    if training_catalog != catalog:
        raise ValueError("Training catalog mismatch")
    gates, gate_manifest = join_gate(
        rows, Path(cfg["gate_predictions"]), catalog, sha256(data_dir / "manifest.json")
    )
    if limit is not None:
        if not 0 < limit < len(rows):
            raise ValueError("Smoke limit must be positive and smaller than the validation split")
        selected = rows[:limit]
    else:
        selected = rows
    system = system_prompt(catalog, Path(cfg["prompt_file"]))
    frozen = {
        "config": cfg,
        "config_sha256": object_hash(cfg),
        "system_prompt_sha256": object_hash(system),
        "sample_ids": [r["sample_id"] for r in selected],
        "data_manifest_sha256": sha256(data_dir / "manifest.json"),
        "gate_predictions_sha256": sha256(Path(cfg["gate_predictions"])),
        "environment_lock_sha256": sha256(Path(cfg["environment_lock"])),
        "implementation_sha256": object_hash(
            {name: sha256(Path(__file__).parent / name) for name in ("prompted.py", "prompting.py")}
        ),
        "purpose": "hardware_smoke" if limit else "validation_benchmark",
    }
    predictions_path = output / "predictions.jsonl"
    if resume:
        if read_json(output / "run.json") != frozen:
            raise ValueError("Resume configuration/input hashes differ from the frozen run")
        if (output / "prediction_manifest.json").exists():
            raise ValueError("Completed run cannot be overwritten or resumed")
        predictions = read_jsonl(predictions_path) if predictions_path.exists() else []
        if predictions:
            verify_hash(predictions_path, read_json(output / "progress.json")["predictions_sha256"])
        if [p["sample_id"] for p in predictions] != frozen["sample_ids"][: len(predictions)]:
            raise ValueError("Partial predictions are not an ordered unique prefix")
    else:
        new_directory(output)
        write_json(output / "run.json", frozen)
        (output / "system_prompt.txt").write_text(system, encoding="utf-8", newline="\n")
        predictions = []
    factory = adapter_factory or TransformersAdapter
    start = time.perf_counter()
    adapter = factory(cfg, system, [r["text"] for r in rows])
    metadata = {
        "model_version": f"{cfg['experiment_id']}-{object_hash(cfg)[:12]}",
        "model_type": "prompted",
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "config": cfg,
        "backend": "transformers",
        "precision": cfg["dtype"],
        "quantization": cfg["quantization"],
        "prompt_version": cfg["prompt_version"],
        "catalog": catalog,
        **adapter.metadata,
        "dependency_lock_sha256": frozen["environment_lock_sha256"],
    }
    write_json(output / "model_metadata.json", metadata)
    with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in selected[len(predictions) :]:
            call_start = time.perf_counter()
            try:
                result = adapter.generate(row["text"])
                parsed = parse_model_json(
                    result["raw_output"], catalog, truncated=result["truncated"]
                )
                error = result["error_type"]
                if result["truncated"]:
                    error = "truncated"
                elif not parsed.valid and error is None:
                    error = "invalid_model_output"
                prediction = {
                    **row,
                    **result,
                    "predicted_label": parsed.label if not error else None,
                    "parse_status": "ok" if not error else "invalid",
                    "error_type": error,
                    "token_count": result["input_tokens"] + result["output_tokens"],
                }
            except RuntimeError as exc:
                prediction = {
                    **row,
                    "predicted_label": None,
                    "parse_status": "error",
                    "error_type": "inference_error",
                    "error_class": type(exc).__name__,
                    "raw_output": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "token_count": None,
                }
            prediction.update(
                {
                    "gate_score": gates[row["sample_id"]]["gate_score"],
                    "model_version": metadata["model_version"],
                    "latency_ms": (time.perf_counter() - call_start) * 1000,
                }
            )
            predictions.append(prediction)
            stream.write(
                json.dumps(prediction, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
            stream.flush()
            write_json(
                output / "progress.json",
                {"completed": len(predictions), "predictions_sha256": sha256(predictions_path)},
            )
            if len(predictions) % 25 == 0 or len(predictions) == len(selected):
                print(
                    json.dumps(
                        {
                            "completed": len(predictions),
                            "total": len(selected),
                            "session_seconds": round(time.perf_counter() - start, 2),
                        }
                    ),
                    flush=True,
                )
    write_jsonl(output / "gate_predictions.jsonl", [gates[r["sample_id"]] for r in selected])
    write_json(
        output / "prediction_manifest.json",
        {
            "schema_version": 1,
            "split": split,
            "sample_count": len(selected),
            "catalog": catalog,
            "sample_ids": frozen["sample_ids"],
            "fixture": data_manifest["fixture"],
            "purpose": frozen["purpose"],
            "benchmark_complete": limit is None,
            "predictions_sha256": sha256(predictions_path),
            "dataset_version": data_manifest["commit"],
            "data_manifest_sha256": frozen["data_manifest_sha256"],
            "model": metadata,
            "gate_model": gate_manifest["model"],
            "gate_model_version": gate_manifest["model"]["model_version"],
            "gate_reference": {
                "predictions_sha256": sha256(output / "gate_predictions.jsonl"),
                "source_predictions_sha256": frozen["gate_predictions_sha256"],
            },
            "gate_join": "one-to-one IDs; verified text, gold, split, and dataset version",
            "timing_scope": "serial full-prompt tokenization and generation; "
            "excludes model loading, optional system-prefix prefill, and gate; "
            f"prefix_cache={cfg.get('prefix_cache', False)}",
            "execution": {
                "last_session_seconds": time.perf_counter() - start,
                "resumed": resume,
                "sum_inference_seconds": sum(r["latency_ms"] for r in predictions) / 1000,
                **adapter.memory(),
            },
        },
    )
    return {
        "output": str(output),
        "count": len(selected),
        "benchmark_complete": limit is None,
        "model_version": metadata["model_version"],
    }
