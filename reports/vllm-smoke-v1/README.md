# vLLM B/C compatibility smoke

Executed 1 October 2026; audited 2 October after resuming the interrupted chat.
The background smoke finished and its runner removed the temporary container.
Docker Desktop was stopped at audit time, so no fresh daemon inspection was
possible. No inference was repeated during the audit.

| Candidate | Submitted | Completed | Exact generation matches | Inference failures |
| --- | ---: | ---: | ---: | ---: |
| B, base model | 6 | 6 | **5** | 0 |
| C, checkpoint-944 LoRA | 6 | 6 | **6** | 0 |

Each candidate received the same first three supported and first three oos
validation examples. Exact generation comparison covers raw text, input/output
token counts, truncation and infrastructure errors. These six examples establish
runtime feasibility, not full parity, model quality or production readiness.

One real B difference: the reference returned `{"intent": "question"}` (invalid
catalog label, seven generated tokens); vLLM returned `{"intent": "oos"}` (valid,
eight generated tokens). Both ended with EOS. Two other B outputs were identical
invalid catalog labels in both backends.

## Accounting correction

The unmodified original `summary.json` reports B as 3/6 exact. Its executed
runner compared the reference prediction's **parser** error
`invalid_model_output` against the generation worker's `null` **inference** error.
That created two spurious mismatches in addition to the real output difference.
The current runner compares equivalent generation-error semantics and records
parser failures separately in saved predictions. A regression test covers this
case. The original runner is retained as `run_vllm_experiment.executed.py`.

`audit/audit.json` records the corrected 5/6 B and 6/6 C counts without modifying
original evidence. The CPU auditor verifies all 14 original evidence checksums,
canonical validation IDs, reference hashes, raw output/token/truncation fields,
EOS handling, strict parsing, classification and frozen-threshold diagnostics,
and unchanged release/ledger hashes. Original classification/policy metrics
were unaffected by the parser/inference error distinction.

## Runtime and limits

- New image: `sha256:a34b46ec3436ed26dfbe32970bb10f2aa945d0adee6e2aea00353887203c3115`.
- vLLM **0.10.2**, PyTorch **2.8.0+cu128**, Transformers **4.57.6**,
  bitsandbytes **0.50.2**, PEFT **0.17.1**, Linux/WSL2, GTX 1650, CC **7.5**, 4 GiB.
- Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, unchanged
  checkpoint-944 adapter. NF4/FP16, greedy, non-thinking, full catalog, no JSON
  constraints or adapter merging. One sequence, eager execution, prefix cache.
- V1 actually selected **FlexAttention** and **PunicaWrapperGPU**. The log reports
  chunked prefill **enabled**, despite the request to disable it. Recorded
  requested configuration is not a claim about effective scheduler behavior.
- Worker model/engine load **55.34 s**; host-observed HTTP readiness **80.67 s**.
  B's six requests took **59.14 s**, C's **11.99 s**, including first-use work.
  These are not warmed load-test latency or throughput estimates.
- Instantaneous whole-GPU samples: **3,703 MiB** before comparisons and
  **3,719 MiB** afterward, including desktop/other processes. These are not
  process memory or workload peaks; 65% engine allocation is not a total-device
  memory cap.
- Authentication rejection passed. All model inputs were mounted read-only;
  no test dataset was mounted. Frozen source, release and test ledger remain
  unchanged. C's original final-test disqualification remains in force.

The successful build's log is converted from PowerShell UTF-16 to UTF-8/LF as
`build.log`. It records 178 compatible installed packages. Earlier build attempts
stopped at dependency checks: direct hash resolution rejected an unpinned torch
dependency, then `uv pip check` caught inherited datasets/fsspec incompatibility.
The final build installs the complete resolved lock with `--no-deps` and pins
fsspec 2026.6.0. Existing Transformers images/environments were not changed.

## Reproduce the audit / continue

```powershell
.\.venv\Scripts\python.exe scripts/verify_vllm_experiment.py --input reports/vllm-smoke-v1 --output artifacts/vllm-smoke-audit-v2
```

For the longer run, start Docker Desktop and run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/run_vllm_experiment.py --full --skip-build --output artifacts/vllm-validation-v1
```

This performs **6,200 validation generations** (3,100 each for B and C), reuses the
built image and cached weights, and uses a fresh output directory. No venv
activation is needed. Full validation and a vLLM serving load benchmark remain
unexecuted. See [the experiment runbook](../../docs/vllm_experiment.md).
