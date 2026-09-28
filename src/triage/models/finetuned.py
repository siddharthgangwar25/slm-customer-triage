"""Verified PEFT adapter inference using B's unchanged prompt/decoder and shared gate."""

import time
from importlib.metadata import version
from pathlib import Path

from triage.io import config, object_hash, sha256
from triage.models.prompted import TransformersAdapter
from triage.models.prompted import run as prompted_run
from triage.training.data import identity, source_hash
from triage.training.runner import verify_bundle


class FinetunedAdapter(TransformersAdapter):
    def __init__(self, cfg, system, texts):
        from peft import PeftModel

        started = time.perf_counter()
        bundle = verify_bundle(Path(cfg["adapter"]), allow_smoke=cfg.get("smoke_adapter", False))
        super().__init__({**cfg, "prefix_cache": False}, system, texts)
        if any(
            self.metadata[k] != bundle[k] for k in ("system_prompt_sha256", "chat_template_sha256")
        ):
            raise ValueError("Adapter prompt/template differs from inference")
        self.model = PeftModel.from_pretrained(
            self.model, cfg["adapter"], is_trainable=False
        ).eval()
        self.cfg = cfg
        # A base-only cache is incorrect after adding LoRA. Prefill with the adapter active.
        if cfg.get("prefix_cache", False):
            prefix = self.torch.tensor([self.prefix_ids], device=cfg["device"])
            with self.torch.inference_mode():
                self.prefix_cache = self.model(
                    input_ids=prefix, attention_mask=self.torch.ones_like(prefix), use_cache=True
                ).past_key_values
        self.metadata.update(
            {
                "model_type": "finetuned",
                "adapter_bundle_sha256": cfg["adapter_bundle_sha256"],
                "adapter_step": bundle["step"],
                "adapter_epoch": bundle["epoch"],
                "training_identity_sha256": object_hash(bundle["identity"]),
                "load_seconds": time.perf_counter() - started,
                "prefix_cache_tokens": len(self.prefix_ids) if self.prefix_cache else 0,
                "model_memory_bytes": self.model.get_memory_footprint(),
            }
        )
        self.metadata["environment"]["peft"] = version("peft")


def prediction_config(training, bundle_path, *, smoke=False):
    bundle = verify_bundle(bundle_path, allow_smoke=smoke)
    frozen = identity(training)
    if frozen != bundle["identity"]:
        raise ValueError("Adapter training identity differs from current configuration/source")
    pair = frozen["paired_config"]
    return {
        **pair,
        "model_type": "finetuned",
        "experiment_id": f"{training['experiment_id']}-step{bundle['step']}",
        "environment_lock": training["environment_lock"],
        "adapter": str(bundle_path),
        "adapter_bundle_sha256": sha256(bundle_path / "bundle.json"),
        "adapter_implementation_sha256": source_hash(),
        "smoke_adapter": smoke,
    }


def run(cfg, split, output, *, bundle, limit=None, resume=False):
    if bundle is None:
        raise ValueError("Fine-tuned inference requires --bundle")
    training = config(Path(cfg["training_config"]))
    paired = prediction_config(training, bundle, smoke=limit is not None)
    return prompted_run(
        paired, split, output, limit=limit, resume=resume, adapter_factory=FinetunedAdapter
    )
