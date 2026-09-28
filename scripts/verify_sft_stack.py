"""Optional CPU random-model fixture: TRL padding masks and exact checkpoint resume.

Run in the locked training environment. No downloads or benchmark claims.
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    TrainerCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from triage.io import new_directory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    new_directory(args.output)
    torch.set_num_threads(2)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t31")),
        eos_token="t0",
        pad_token="t0",
        bos_token="t1",
        unk_token="t31",
    )
    examples = [
        {"input_ids": [1, 3, 4, 8, 0], "completion_mask": [0, 0, 0, 1, 1]},
        {"input_ids": [1, 5, 9, 0], "completion_mask": [0, 0, 1, 1]},
    ]
    collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    labels = collator(examples)["labels"].tolist()
    assert labels == [[-100, -100, -100, 8, 0], [-100, -100, 9, 0, -100]]

    class StopAfterOne(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_training_stop = True

    def trainer(directory, *, interrupt=False):
        set_seed(42)
        model = Qwen3ForCausalLM(
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
            )
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(examples),
            data_collator=collator,
            callbacks=[StopAfterOne()] if interrupt else [],
            args=SFTConfig(
                output_dir=str(directory),
                use_cpu=True,
                fp16=False,
                bf16=False,
                max_steps=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                learning_rate=1e-3,
                save_steps=1,
                logging_steps=1,
                report_to="none",
                seed=42,
                data_seed=42,
                dataloader_pin_memory=False,
                completion_only_loss=True,
                dataset_kwargs={"skip_prepare_dataset": True},
            ),
        )

    uninterrupted = trainer(args.output / "uninterrupted")
    uninterrupted.train()
    interrupted = trainer(args.output / "resumed", interrupt=True)
    interrupted.train()
    resumed = trainer(args.output / "resumed")
    resumed.train(resume_from_checkpoint=str(args.output / "resumed" / "checkpoint-1"))
    expected = {n: p for n, p in uninterrupted.model.named_parameters() if p.requires_grad}
    differences = [
        (p - expected[n]).abs().max().item()
        for n, p in resumed.model.named_parameters()
        if p.requires_grad
    ]
    maximum = max(differences)
    assert maximum == 0, maximum
    assert resumed.state.global_step == 2
    write_json(
        args.output / "verification.json",
        {
            "fixture": True,
            "purpose": "random tiny CPU model, not benchmark evidence",
            "pad_equals_eos_mask_correct": True,
            "max_resume_parameter_difference": maximum,
            "optimizer_steps": 2,
            "passed": True,
        },
    )


if __name__ == "__main__":
    main()
