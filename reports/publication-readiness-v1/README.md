# Publication preparation checks

Local checks on 2026-10-04. No repository has been published and hosted CI remains
unverified. Licensing is undecided at the owner's request. This folder records
publication preparation separately from frozen benchmark acceptance.

## Clean checkout reproduction

Cloned local commit `08330b1` into the ignored
`artifacts/publication-clean-checkout-v1` directory using `git clone --no-hardlinks`.
The checkout remains clean. The original Python 3.11 and uv executables served
as installed bootstrap tools; the clone has its own newly installed CPU
environment and uv cache. No original model bundle or cached raw data was used.
The helper downloaded the pinned official dataset itself.

The initial `reproduction-v1` failed at dependency sync because the tool sandbox
denied network socket access. Retained the attempt, then ran the same helper
with permitted network access and a new `reproduction-v2` output directory.
All eight stages passed. See `cpu-commands.json`, `cpu-api-parity.json`,
`cpu-demo.json`, `cpu-metrics.json` and the CPU audit for measured results.

The invocation, from the clone, was:

```powershell
& 'C:/Users/gangw/Documents/Projects/slm-customer-triage/.venv/Scripts/python.exe' scripts/reproduce_cpu.py --uv 'C:/Users/gangw/Documents/Projects/slm-customer-triage/.tools/uv.exe' --output artifacts/reproduction-v2
```

This establishes the tracked CPU implementation at that commit can reproduce
without ignored project artifacts. It is not a claim of running on another OS,
another physical machine, a GitHub clone URL, or hosted Actions. This publication
change does not modify that CPU implementation or its locked dependencies.

## CUDA fixture

The new `scripts/verify_gpu_fixture.py` passed locally in the existing locked
training environment on GTX 1650, with two CUDA optimizer steps and exact
save/reload logits and greedy generation. The recorded final process exit code
is zero. See `gpu-verification.json` and `gpu-process.json`.

The first fixture attempt trained but rejected tokenizer `token_type_ids` at
generation; fixed the local tokenizer's declared model inputs. The second wrote
a passing result but PowerShell's native stderr redirection reported shell exit
1. The final run used Python subprocess capture to establish the actual child
exit code. All attempts remain locally; the final result is the retained pass.

The synthetic randomly initialized FP32 LoRA model is not the fine-tuned 0.6B
QLoRA benchmark. No benchmark inference, policy adjustment or release activation
occurred. GitHub dispatch and Linux GPU execution remain unverified.

## Repository checks

Checks and input hashes are retained in `checks.json` and `identity.json`.
The limited reachable-history credential scan found no common token/private-key
patterns; the largest historical blob was 3,419,448 bytes. This scan does not
certify the absence of all secrets. Added `.gitignore` entries protect future
local secret files, while upstream attribution is unchanged.

No actual GPU Compose run, paid cloud deployment, fresh independent challenge
set or timed human demo rehearsal is claimed by this preparation.
