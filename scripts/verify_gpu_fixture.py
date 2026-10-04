"""Tiny random CUDA/LoRA integration fixture; no downloads or benchmark evaluation."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this GPU check cannot silently fall back to CPU")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    set_seed(42)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()

    backend = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t31"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="t0",
        pad_token="t0",
        bos_token="t1",
        unk_token="t31",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(args.output / "tokenizer")
    reloaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.output / "tokenizer", local_files_only=True
    )
    texts = ["t1 t3 t4 t8 t0", "t1 t5 t9 t0"]
    encoded = reloaded_tokenizer(texts, add_special_tokens=False)["input_ids"]
    if encoded != [[1, 3, 4, 8, 0], [1, 5, 9, 0]]:
        raise RuntimeError("Tokenizer reload changed the fixture")
    examples = [
        {"input_ids": encoded[0], "completion_mask": [0, 0, 0, 1, 1]},
        {"input_ids": encoded[1], "completion_mask": [0, 0, 1, 1]},
    ]
    collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    if collator(examples)["labels"].tolist() != [
        [-100, -100, -100, 8, 0],
        [-100, -100, 9, 0, -100],
    ]:
        raise RuntimeError("Prompt/padding mask or supervised EOS is incorrect")

    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            eos_token_id=0,
            pad_token_id=0,
            bos_token_id=1,
            attn_implementation="eager",
        )
    )
    base.save_pretrained(args.output / "base")
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    trainer = SFTTrainer(
        model=model,
        processing_class=reloaded_tokenizer,
        train_dataset=Dataset.from_list(examples),
        data_collator=collator,
        args=SFTConfig(
            output_dir=str(args.output / "training"),
            use_cpu=False,
            fp16=False,
            bf16=False,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            save_strategy="no",
            logging_steps=1,
            report_to="none",
            seed=42,
            data_seed=42,
            dataloader_pin_memory=False,
            completion_only_loss=True,
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
    )
    trained = trainer.train()
    if next(model.parameters()).device.type != "cuda" or trainer.state.global_step != 2:
        raise RuntimeError("Expected two optimizer steps on CUDA")
    gradients = [row["grad_norm"] for row in trainer.state.log_history if "grad_norm" in row]
    if len(gradients) != 2 or not all(math.isfinite(g) for g in gradients):
        raise RuntimeError("Missing or non-finite gradients")
    if not math.isfinite(trained.training_loss):
        raise RuntimeError("Non-finite training loss")
    changes = [
        (p.detach().cpu() - initial[n]).abs().max().item()
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    if not all(math.isfinite(x) for x in changes) or max(changes) <= 0:
        raise RuntimeError("Training did not update finite adapter parameters")
    model.save_pretrained(args.output / "adapter")
    model.eval()
    probe = reloaded_tokenizer("t1 t3 t4", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        expected_logits = model(**probe).logits
        expected_tokens = model.generate(**probe, max_new_tokens=4, do_sample=False)

    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.output / "base", local_files_only=True, attn_implementation="eager"
        ),
        args.output / "adapter",
        local_files_only=True,
    ).to("cuda")
    reloaded.eval()
    with torch.inference_mode():
        actual_logits = reloaded(**probe).logits
        actual_tokens = reloaded.generate(**probe, max_new_tokens=4, do_sample=False)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
    if not torch.equal(actual_tokens, expected_tokens) or actual_tokens.shape[1] <= 3:
        raise RuntimeError("Reloaded greedy generation differs or produced no new tokens")
    result = {
        "fixture": True,
        "purpose": "random tiny FP32 CUDA LoRA stack check, not QLoRA benchmark evidence",
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tokenizer_reload_matches": True,
        "completion_and_eos_mask_correct": True,
        "optimizer_steps": trainer.state.global_step,
        "training_loss": trained.training_loss,
        "gradient_norms": gradients,
        "maximum_adapter_change": max(changes),
        "maximum_reload_logit_difference": (actual_logits - expected_logits).abs().max().item(),
        "generation_matches": True,
        "generated_token_ids": actual_tokens.cpu().tolist(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "seconds": time.perf_counter() - started,
    }
    with (args.output / "verification.json").open("w", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
