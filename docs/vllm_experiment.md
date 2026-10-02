# Separate vLLM validation experiment

The full validation run completed and was audited on 2 October: **6,200/6,200
requests**, with **B 3,005/3,100** and **C 3,095/3,100** exact generation matches.
[Full results](../reports/vllm-validation-v1/README.md). The alternative is not
adopted: output differences prevent treating it as the frozen reference backend.
The verified Transformers service remains in place. No more GPU work is needed
for this comparison. vLLM load testing remains unexecuted, with no speedup claim.

This is a new backend variant for Section 9. It does not replace the frozen
Transformers benchmark, select another checkpoint, change thresholds, rerun the
test split, or activate a release. The original C release remains disqualified
because its final-test oos review recall was 89.6%, below the fixed 90% requirement.

## Runtime choice

Documentation and the pinned implementation were checked on 1 October 2026:

- [Current GPU requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
  require Linux and NVIDIA compute capability 7.5 or newer. Docker's WSL2 Linux
  environment provides the available execution path; native Windows is not used.
- [v0.10.2 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.10.2)
  describe PyTorch 2.8 and V1 support for GPUs below compute capability 8.0.
  This older release is deliberately pinned for compatibility with the existing
  CUDA 12.8/PyTorch 2.8 image, not represented as the latest release.
- [Pinned supported-model list](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html)
  includes Qwen3ForCausalLM and LoRA support.
- [Pinned quantized LoRA example](https://github.com/vllm-project/vllm/blob/v0.10.2/examples/offline_inference/lora_with_quantization_inference.py)
  uses `enable_lora` and `LoRARequest` with bitsandbytes. The obsolete
  `qlora_adapter_name_or_path` option is not used.
- [Pinned bitsandbytes loader](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/model_executor/model_loader/bitsandbytes_loader.py)
  uses NF4 with compressed statistics for in-flight 4-bit quantization. Kernel,
  casting and cache differences still require measured output comparison.
- [Pinned CUDA backend selection](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/platforms/cuda.py)
  defaults to FlexAttention for older GPUs under V1. Runtime logs determine what
  actually loaded; architecture support alone is not acceptance evidence.

`environments/vllm/requirements.lock` is a complete, hash-locked Linux/Python 3.11
dependency resolution. It pins vLLM 0.10.2, Transformers 4.57.6, bitsandbytes 0.50.2
and PEFT 0.17.1. NumPy resolves to 2.2.6 for vLLM's numba dependency; fsspec is
pinned to 2026.6.0 to remain compatible with the inherited datasets package.
The existing CPU/training environments and their locks are unchanged.

`deployment/Dockerfile.vllm` layers onto the audited local GPU image. The runner
checks its immutable image ID and records the new image ID. Installation uses
the resolved hash lock without resolving dependencies again, followed by
`uv pip check`. System compiler tools support runtime kernel compilation. The
base build is documented in `deployment/Dockerfile.gpu`; rebuild it first on a
fresh machine and audit a different image ID explicitly rather than bypassing
the local runner's identity check.

## Experiment contract

The worker loads the unchanged pinned Qwen3-0.6B snapshot and checkpoint-944 LoRA
adapter. B and C share one engine and identical decoding; requests specify
whether LoRA is active. No adapter merging, retraining, JSON constraints or
output repair is introduced. Prefix caching is enabled and uses vLLM's separate
LoRA request identities. Greedy FP16 computation, NF4, seed 42, 2,048 input tokens
and 32 generated tokens match the reference settings. GPU allocation is capped
at 65% for the engine, one sequence at a time, eager execution, no CPU offload
or swap. This is not a cap on total device usage. The executed V1 runtime enabled
chunked prefill despite the requested disabled flag; its log records the
effective configuration. Whole-GPU smoke samples were about 3.7 GiB.

Prompts use the existing full catalog, escaping and non-thinking chat template.
The worker supplies token IDs directly to vLLM. It decodes the returned token IDs
using the original tokenizer: count the final EOS token, strip only that final
EOS from raw text, and classify output without EOS as truncated. Native vLLM
token IDs and finish reasons remain in comparison records.

The authenticated temporary HTTP worker binds through a random localhost port.
This experiment uses no public endpoint, cloud account or paid resource. Only
the model, adapter, prompt and validation identity are mounted read-only; no
test dataset or final-test predictions are mounted. Access logging is disabled.
Only the temporary container created by the runner is removed on completion.

Smoke mode compares the first three supported and first three oos validation
examples for each candidate. Full mode compares all 3,100 validation examples
per candidate. Every submission produces a comparison row or explicit inference
failure. The output includes strict parsing, raw classification metrics, and a
diagnostic using the old frozen threshold. Those diagnostic metrics do not
constitute a newly selected or approved policy.

Exact output parity is a measured result, not a prerequisite for retaining an
experiment. A completed run may contain differences. No result from six rows is
a quality conclusion. First-request compilation/warmup is included in timings;
this comparison is not the Section 9 500-request serving load benchmark. The
existing Transformers load benchmark remains the only completed one until a
separate vLLM load experiment is explicitly recorded.

## Commands (completed-run history)

From the project root in PowerShell, with Docker Desktop running, no venv
activation is needed. Use a fresh output directory for each attempt.

```powershell
.\.venv\Scripts\python.exe scripts/run_vllm_experiment.py --output artifacts/vllm-smoke-v1
```

The longer validation-only command, now completed, was:

```powershell
.\.venv\Scripts\python.exe scripts/run_vllm_experiment.py --full --skip-build --output artifacts/vllm-validation-v1
```

The runner reuses cached weights, checks the image lock, records image/runtime
identities, and saves raw comparisons, predictions, metrics, logs and checksums.
Do not interpret `status: completed` as exact parity: inspect `exact_matches`,
`mismatch_fields` and `infrastructure_failures` for both candidates. Stop and
inspect a failed smoke before running full validation. Do not run the original
`scripts/run_milestone5.py` final-test workflow again.

To regenerate the isolated lock deliberately:

```powershell
.\.tools\uv.exe pip compile environments/vllm/requirements.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --generate-hashes --output-file environments/vllm/requirements.lock --cache-dir .uv-cache --no-emit-index-url
```

Both output directories already exist; do not repeat these completed runs.
To re-audit saved evidence without inference, use a fresh audit output directory:

```powershell
.\.venv\Scripts\python.exe scripts/verify_vllm_experiment.py --input reports/vllm-validation-v1 --output artifacts/vllm-validation-audit-v2
```

Execution results and the next milestone are recorded in
[implementation status](implementation_status.md).
