"""Bounded, resumable local QLoRA training with real smoke/reload prerequisites."""

import gc
import json
import math
import shutil
import time
from importlib.metadata import version
from pathlib import Path

from triage.io import new_directory, read_json, sha256, verify_hash, write_json
from triage.training.data import audit_record, identity, load_prepared, tokenizer_for


def validate(cfg):
    if type(cfg["epochs"]) is not int or not 1 <= cfg["epochs"] <= 3:
        raise ValueError("Use one to three complete epochs")
    if cfg["microbatch"] != 1 or cfg["gradient_accumulation_steps"] != 16:
        raise ValueError("Reference training uses microbatch 1, effective batch 16")
    if cfg["paid_cloud_enabled"] is not False:
        raise ValueError("This runner is local only; paid cloud is disabled")
    if not 0 < cfg["local_time_budget_hours"] <= 24:
        raise ValueError("Local training budget must be positive and at most 24 hours")
    if cfg["training_dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Invalid training dtype")
    if not cfg["precision_reason"] or cfg["save_steps"] < 1:
        raise ValueError("Record precision decision and positive checkpoint interval")
    if cfg["rank"] < 1 or cfg["alpha"] < 1 or not 0 <= cfg["dropout"] < 1:
        raise ValueError("Invalid adapter configuration")
    if not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")


def check_smoke(cfg, evidence, prepared):
    smoke = read_json(evidence / "result.json")
    if (
        smoke["status"] != "smoke_passed"
        or smoke["identity"] != identity(cfg)
        or smoke["prepared_manifest_sha256"] != sha256(Path(cfg["prepared_data"]) / "manifest.json")
        or not smoke["reload_parity"]["passed"]
        or not smoke["weights_changed"]
        or smoke["examples_processed"] < 16
    ):
        raise ValueError("Require a matching finite-gradient training smoke and reload parity")
    verify_bundle(evidence / "adapter", allow_smoke=True)
    estimate = smoke["seconds_per_example"] * prepared["sample_count"] * cfg["epochs"]
    if estimate > cfg["local_time_budget_hours"] * 3600:
        raise ValueError(f"Smoke projects {estimate / 3600:.2f} hours, exceeding local budget")
    return estimate


def verify_bundle(path, *, allow_smoke=False):
    manifest = read_json(path / "bundle.json")
    allowed = ("training_checkpoint", "training_smoke") if allow_smoke else ("training_checkpoint",)
    if manifest["purpose"] not in allowed:
        raise ValueError("Training smoke adapter cannot be benchmarked as C")
    for name, digest in manifest["files"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("Bundle file escapes its directory")
        verify_hash(target, digest)
    if not {"adapter_model.safetensors", "adapter_config.json"}.issubset(manifest["files"]):
        raise ValueError("Incomplete adapter bundle")
    return manifest


def seal_bundle(path, frozen, prepared, *, smoke, step, epoch):
    shutil.copyfile(Path(frozen["config"]["environment_lock"]), path / "environment.lock")
    shutil.copyfile(Path(frozen["config"]["prepared_data"]) / "manifest.json", path / "data.json")
    shutil.copyfile(
        Path(frozen["config"]["prepared_data"]) / "system_prompt.txt", path / "system_prompt.txt"
    )
    write_json(path / "configuration.json", frozen)
    write_json(
        path / "bundle.json",
        {
            "purpose": "training_smoke" if smoke else "training_checkpoint",
            "identity": frozen,
            "step": step,
            "epoch": epoch,
            "system_prompt_sha256": prepared["system_prompt_sha256"],
            "chat_template_sha256": prepared["chat_template_sha256"],
            "files": {
                p.name: sha256(p) for p in path.iterdir() if p.is_file() and p.name != "bundle.json"
            },
        },
    )


def load_base(pair, dtype):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    snapshot = snapshot_download(
        pair["model_id"],
        revision=pair["revision"],
        cache_dir=pair["cache_dir"],
        allow_patterns=["*.json", "*.safetensors"],
    )
    return AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=getattr(torch, dtype),
        device_map={"": pair["device"]},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        ),
    )


def reload_parity(model, tokenizer, record, path):
    """Unload the trained adapter and reload saved weights on the identical frozen base."""
    import torch
    from peft import PeftModel

    boundary = record["completion_mask"].index(1)
    inputs = torch.tensor([record["input_ids"][:boundary]], device=model.device)

    def probe(candidate):
        candidate.eval()
        with torch.inference_mode():
            logits = (
                candidate(input_ids=inputs, use_cache=False, logits_to_keep=1).logits.float().cpu()
            )
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite reload probe logits")
        return logits

    before = probe(model)
    base = model.unload()
    reloaded = PeftModel.from_pretrained(base, path, is_trainable=False)
    after = probe(reloaded)
    difference = (before - after).abs().max().item()
    passed = torch.allclose(before, after, atol=1e-5, rtol=1e-5)
    if not passed:
        raise ValueError(f"Adapter reload parity failed: {difference}")
    return {
        "passed": True,
        "max_absolute_logit_difference": difference,
        "scope": "trained adapter unload/reload on identical frozen quantized base",
    }


def run(cfg, output, *, smoke=False, smoke_evidence=None, resume_checkpoint=None):
    validate(cfg)
    frozen = identity(cfg)
    pair = frozen["paired_config"]
    if pair["quantization"] != "nf4" or not pair["device"].startswith("cuda"):
        raise ValueError("QLoRA runner requires an explicit CUDA NF4 base")
    records, prepared = load_prepared(cfg)
    if prepared["fixture"] or (prepared["supported_count"], prepared["oos_count"]) != (15000, 100):
        raise ValueError("Real training requires the complete 15,000 + 100 training split")
    estimate = None
    if not smoke:
        if smoke_evidence is None:
            raise ValueError("Full training requires --smoke-evidence")
        estimate = check_smoke(cfg, smoke_evidence, prepared)
    if smoke and resume_checkpoint:
        raise ValueError("Start a fresh smoke")
    consumed_seconds = 0.0
    previous_examples = 0
    if resume_checkpoint:
        resolved = resume_checkpoint.resolve()
        if resolved.parent != output.resolve() or not resolved.name.startswith("checkpoint-"):
            raise ValueError("Resume checkpoint must belong to this training output")
        bundle = verify_bundle(resume_checkpoint)
        if bundle["identity"] != frozen or read_json(output / "run.json")["identity"] != frozen:
            raise ValueError("Resume identity mismatch")
        previous = read_json(output / "progress.json")
        consumed_seconds = previous["elapsed_seconds"]
        previous_examples = previous["examples_processed"]
        if (output / "result.json").exists() and read_json(output / "result.json")[
            "status"
        ] == "complete":
            raise ValueError("Completed training cannot be resumed")
    else:
        new_directory(output)
        write_json(
            output / "run.json",
            {
                "identity": frozen,
                "purpose": "training_smoke" if smoke else "training",
                "prepared_manifest_sha256": sha256(Path(cfg["prepared_data"]) / "manifest.json"),
                "projected_training_seconds": estimate,
            },
        )
    budget = cfg["local_time_budget_hours"] * 3600
    if consumed_seconds >= budget:
        raise ValueError("Local time budget exhausted; do not silently reset it on resume")

    import psutil
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no silent CPU training fallback")
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if cfg["training_dtype"] == "bfloat16" and not native_bf16:
        raise ValueError("Configured BF16 is not natively supported")
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    tokenizer = tokenizer_for(pair)
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.eos_token_id, completion_only_loss=True
    )
    audited = read_json(Path(cfg["prepared_data"]) / "mask_audit.json")
    by_id = {r["sample_id"]: r for r in records}
    for audit in audited:
        record = by_id[audit["sample_id"]]
        if audit_record(record, tokenizer) != audit:
            raise ValueError("Mask audit changed")
        actual = collator([record])["labels"][0].tolist()
        if actual != audit["labels"]:
            raise ValueError("TRL collator does not honor the audited completion mask")
    if len(audited) != 10:
        raise ValueError("Require ten directly inspected loss masks")
    write_json(output / "collator_audit.json", {"count": 10, "passed": True})
    start = time.perf_counter()
    hardware = {
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_free_bytes_before_load": torch.cuda.mem_get_info()[0],
        "native_bf16": native_bf16,
        "torch_cuda": torch.version.cuda,
        "ram_total_bytes": psutil.virtual_memory().total,
        "training_dtype": cfg["training_dtype"],
        "precision_reason": cfg["precision_reason"],
        "local_time_budget_hours": cfg["local_time_budget_hours"],
        "environment": {
            p: version(p)
            for p in ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate")
        },
    }
    write_json(output / "hardware.json", hardware)
    torch.cuda.reset_peak_memory_stats()
    base = load_base(pair, cfg["training_dtype"])
    available = {n.rsplit(".", 1)[-1] for n, _ in base.named_modules()}
    if not set(cfg["target_modules"]).issubset(available):
        raise ValueError("Requested LoRA linear layers absent from base")
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
            target_modules=cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    hardware["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hardware["prepared_model_memory_bytes"] = model.get_memory_footprint()
    write_json(output / "hardware.json", hardware)
    # Include the longest request and oos in the smoke, with one real accumulated update.
    selected = [by_id[a["sample_id"]] for a in audited]
    selected += [r for r in records if r["sample_id"] not in {a["sample_id"] for a in audited}][:6]
    dataset = Dataset.from_list(
        [
            {k: r[k] for k in ("input_ids", "completion_mask")}
            for r in (selected if smoke else records)
        ]
    )
    runtime = {
        "examples_processed": previous_examples,
        "epoch_steps": [],
        "budget_exhausted": False,
    }

    def progress(state):
        return {
            **runtime,
            "step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": consumed_seconds + time.perf_counter() - start,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "process_rss_bytes": psutil.Process().memory_info().rss,
        }

    class CheckedTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all():
                raise ValueError("Non-finite training loss")
            if model.training:
                runtime["examples_processed"] += inputs["input_ids"].shape[0]
            return result

    class Monitor(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [
                p.grad
                for p in kwargs["model"].parameters()
                if p.requires_grad and p.grad is not None
            ]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise ValueError("Non-finite or absent adapter gradients; training failed")

        def on_step_end(self, args, state, control, **kwargs):
            if consumed_seconds + time.perf_counter() - start >= budget:
                runtime["budget_exhausted"] = True
                control.should_save = True
                control.should_training_stop = True
            write_json(output / "progress.json", progress(state))

        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch is not None and abs(state.epoch - round(state.epoch)) < 1e-6:
                runtime["epoch_steps"].append(state.global_step)
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            path = output / f"checkpoint-{state.global_step}"
            tokenizer.save_pretrained(path)
            seal_bundle(
                path, frozen, prepared, smoke=smoke, step=state.global_step, epoch=state.epoch
            )
            write_json(output / "progress.json", progress(state))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in logs.values()):
                    raise ValueError("Non-finite training metrics")
                with (output / "training_log.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({**logs, **progress(state)}, allow_nan=False) + "\n")

    args = SFTConfig(
        output_dir=str(output),
        num_train_epochs=cfg["epochs"],
        max_steps=1 if smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=cfg["learning_rate"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        fp16=cfg["training_dtype"] == "float16",
        bf16=cfg["training_dtype"] == "bfloat16",
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg["max_seq_length"],
        packing=False,
        report_to="none",
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=None,
        logging_steps=1,
        logging_nan_inf_filter=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        save_safetensors=True,
    )
    trainer = CheckedTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[Monitor()],
    )
    training_start = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        trained_seconds = time.perf_counter() - training_start
        changed = any(
            not torch.equal(initial[n], p.detach().cpu())
            for n, p in model.named_parameters()
            if n in initial
        )
        if not changed:
            raise ValueError("No adapter weights changed")
        final = output / "adapter"
        final.mkdir(exist_ok=True)
        trainer.save_model(str(final))
        tokenizer.save_pretrained(final)
        seal_bundle(
            final,
            frozen,
            prepared,
            smoke=smoke,
            step=trainer.state.global_step,
            epoch=trainer.state.epoch,
        )
        parity = reload_parity(model, tokenizer, selected[0], final)
        complete = trainer.state.epoch >= cfg["epochs"] - 1e-6
        result = {
            "status": "smoke_passed" if smoke else "complete" if complete else "budget_exhausted",
            "identity": frozen,
            "prepared_manifest_sha256": sha256(Path(cfg["prepared_data"]) / "manifest.json"),
            **progress(trainer.state),
            "training_seconds_this_session": trained_seconds,
            "seconds_per_example": trained_seconds
            / (runtime["examples_processed"] - previous_examples),
            "weights_changed": changed,
            "reload_parity": parity,
            "epoch_checkpoints": sorted(
                p.name
                for p in output.glob("checkpoint-*")
                if (p / "bundle.json").exists()
                and abs(
                    read_json(p / "bundle.json")["epoch"]
                    - round(read_json(p / "bundle.json")["epoch"])
                )
                < 1e-6
            ),
        }
        write_json(output / "result.json", result)
        return {k: v for k, v in result.items() if k != "identity"}
    except (Exception, KeyboardInterrupt) as exc:
        write_json(
            output / "failure.json",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                **progress(trainer.state),
                "resume": "Use last sealed checkpoint; in-flight gradients are not saved",
            },
        )
        write_json(output / "progress.json", progress(trainer.state))
        raise
    finally:
        del trainer, model, base
        gc.collect()
        torch.cuda.empty_cache()
