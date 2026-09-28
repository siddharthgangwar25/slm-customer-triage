# Milestone 4 local runbook

Run from the repository root in PowerShell. **No virtual-environment activation is needed:** the script calls the training environment's Python explicitly. Keep that terminal open and prevent the computer sleeping during a run.

## Continue the prepared experiment

After the recorded smoke has passed, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1
```

This runs the new **0.6B prompted B** validation benchmark, trains **C for one epoch**, evaluates every retained epoch checkpoint, selects C using validation, and produces the A/B/C reports. It uses local cached weights with network access disabled. Original A and 4B B results remain historical evidence. Test predictions and paid cloud are disabled.

The script stops on errors. Running it again resumes incomplete prompted/selection predictions and resumes training from the latest sealed checkpoint, if one exists. It skips completed stages. Run only one instance at a time. Do not edit configuration, prompt, source, lock, or data while a run is pending: identity checks reject incompatible resumes.

Stages can also be run separately:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1 -Stage Prompted
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1 -Stage Train
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1 -Stage Select
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1 -Stage Report
```

Training is bounded by a **24-hour local wall-time budget across recorded sessions**, checked at optimizer-step boundaries (16 microbatches). Loading, checkpointing and the final save/reload can add time; this is a cooperative limit, not a hard process watchdog. The smoke estimate must fit the budget before a full run starts. An exhausted budget saves a partial checkpoint and cannot count as completed training or enter checkpoint selection. Do not reset the budget by deleting progress files.

Periodic resumable checkpoints are saved every 100 optimizer steps and at each completed epoch, with optimizer/scheduler/RNG state. Ctrl+C records failure/progress; resume uses the last sealed checkpoint, replaying work since that checkpoint. Abrupt process termination can lose the latest progress accounting and is not an exact resume. If stopped before the first checkpoint, retain the incomplete directory and ask Codex to inspect it before starting a new output. In-flight gradients are not saved. Examples processed count actual forward passes, including replay after a resume; do not interpret that counter as unique examples.

## Reproduce setup and prerequisite checks

These commands are for a fresh workspace. They refuse to overwrite existing experiment directories.

```powershell
.\.tools\uv.exe sync --locked --project environments/training --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
$env:HF_HUB_DISABLE_XET = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
.\environments\training\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', cache_dir='artifacts/huggingface', allow_patterns=['*.json','*.txt','*.jinja','*.safetensors'])"
$env:HF_HUB_OFFLINE = '1'
.\environments\training\.venv\Scripts\triage.exe data sft --config configs/finetune.yaml
.\environments\training\.venv\Scripts\python.exe scripts/verify_prompt_cache.py --config configs/prompted-small.yaml --output artifacts/qwen3-06b-cache-smoke-v1
.\environments\training\.venv\Scripts\triage.exe train slm --config configs/finetune.yaml --smoke
.\environments\training\.venv\Scripts\triage.exe predict --config configs/finetuned.yaml --bundle artifacts/finetuned-qwen3-06b-qlora-v1-smoke/adapter --split val --limit 3 --output artifacts/finetuned-qwen3-06b-inference-smoke-v1
```

Inspect `artifacts/sft-qwen3-06b-v1/mask_audit.json`: ten records include the longest request, an oos record, and eight others. Labels are `-100` throughout the system/user/generation prefix; only the exact assistant JSON and EOS contribute to loss. EOS remains supervised even though it is also the padding token. The real TRL collator is checked against these labels before loading weights. All 15,100 training sequences are preflighted; would-be truncation fails instead of dropping catalog labels or target tokens. The inference prompt includes Qwen's empty non-thinking prefix, so concatenating the JSON target after the exact B prompt avoids a different full-conversation template silently changing the boundary.

The smoke trains one real optimizer update with 16 accumulated examples, checks finite loss and gradients, checks that adapter weights changed, then unloads/reloads saved adapter weights on the same frozen base and compares logits. It is **not a quality benchmark**. Fresh-process inference separately exercises adapter loading with B's inference precision. The original B/C prediction parser retains invalid JSON, unknown labels, missing EOS and inference exceptions in accounting without repair.

## Configuration and artifacts

`configs/finetune.yaml` documents configuration 1: Qwen3-0.6B at the immutable pinned revision, NF4 double quantization, rank 16, alpha 32, dropout 0.05, seven transformer linear projection types, learning rate 1e-4, microbatch 1, accumulation 16, seed 42, one epoch, sequence limit 1,024. Only this configuration is planned; the initial tuning cap remains three. Training uses explicit FP32 compute because this GTX 1650 has no native BF16 and the earlier 4B FP16 attempt had non-finite gradients. B and C inference both use NF4/FP16 with identical full-catalog prompt, greedy decoder and token limits. Changing precision is explicit; no silent GPU/CPU fallback exists.

Frozen base weights do not learn; LoRA learns small additive matrices in the attention and feed-forward projections. Completion masking prevents the catalog/request text from being supervised targets, while those tokens still provide context and affect gradients through attention. Gradient accumulation combines 16 microbatches before an optimizer update. One epoch is the first bounded experiment, so checkpoint selection has one eligible epoch unless a separately recorded configuration trains longer.

Saved checkpoints contain adapter weights/configuration, tokenizer files, exact training/paired configuration, environment lock, data manifest, prompt and checksums. Training logs record loss, gradient norm, processed examples, elapsed time and GPU memory. Checkpoint validation records raw classification metrics separately; no validation records enter training. Selection orders checkpoints by highest raw supported macro-F1, then fewer invalid outputs, then earlier optimizer step. Each candidate receives its own threshold under the original constraints using the same baseline gate.

Expected final outputs:

- `reports/prompted-qwen3-06b-nf4-v1-val/` and `...-policy/`: new paired B.
- `artifacts/finetuned-qwen3-06b-qlora-v1/result.json`: completed training, resource measurements and reload check.
- `artifacts/finetuned-qwen3-06b-qlora-v1-selection/selection.json`: every epoch's metrics and chosen checkpoint.
- `reports/finetuned-qwen3-06b-qlora-v1-val/` and `...-policy/`: selected C, confusion matrix, per-class scores and plots.
- `reports/three-way-qwen3-06b-v1/`: raw/policy comparison, paired bootstrap intervals, error transitions, fixed/regressed examples and complete paired predictions.

After the terminal command finishes, tell Codex to continue Milestone 4. Codex should verify these genuine artifacts, inspect the changed errors, record measured limitations, and update `docs/implementation_status.md`. Implementation alone does not satisfy Milestone 4's requirement for all three genuine reports. No test scores, deployment upgrade, or quality improvement are assumed.

The CPU service continues using the baseline. The training lock is separate from CPU and prompted environments. CI exercises synthetic fixtures without importing GPU libraries; run the real GPU checks explicitly. APIs are based on the installed [TRL 0.24 SFTTrainer](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) and [PEFT quantization workflow](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization), and must be verified by the recorded smoke.
