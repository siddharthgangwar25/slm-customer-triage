# Implementation status

Last updated: **2026-10-01**. Milestones **1–4 are implemented, executed, and locally verified**. **Milestone 5's full native parity/load/frozen-test run is complete and audited; C is disqualified for automatic release, and container/backend acceptance remains unverified.** Milestone 6 has not been implemented. Original baseline and 4B evidence remain unchanged. The genuine test split has now been used once per frozen candidate; do not retune or repeat it.

## Milestone 5 — full local benchmark audited; automatic release disqualified

The user completed `scripts/run_milestone5.py` on 1 October. Frozen release **release-a0bcea381506**, source commit **82ed8d7**, selected C from validation before test. The completed ledger points to `artifacts/final-20261001T074028Z`; source, model/adapter, prompt, data, policy and lock hashes all verify. No configuration or threshold changed after test exposure. The retained [final benchmark and audit](../reports/milestone5-final-v1/README.md) contains all predictions, manifests, per-class reports, uncertainty intervals and operating evidence.

### Genuine final test results

Each candidate contains exactly **5,500 requests: 4,500 supported + 1,000 oos**, canonical one-to-one joins, no missing records and zero infrastructure failures.

| Measure | Baseline A | Prompted B | Fine-tuned C |
| --- | ---: | ---: | ---: |
| Supported macro-F1 | 0.886628 | 0.278546 | **0.956794** |
| Invalid outputs | 0 | 1,032 | 44 |
| Frozen-policy coverage | 60.05% | 0% | **70.65%** |
| Routed errors / routes | 198 / 3,303 | 0 / 0 | 185 / 3,886 |
| Routing error | 5.99% | Undefined | **4.76%** |
| Oos review recall | 92.7% | 100%, review all | **89.6%** |

**Automatic release is disqualified.** C reviews 896 oos requests, four fewer than the fixed 900/1,000 requirement. Its recall interval (87.55–91.34%) does not waive the prespecified point-estimate rule. C improves macro-F1 over A by **0.070166**, paired bootstrap 95% interval **[0.061720, 0.080052]**, and over B by **0.678248**, interval **[0.668722, 0.691655]**. Its routing-error interval is **4.13–5.48%**. A also misses the 5% error constraint. B's review-only result is not successful automatic routing. No candidate was substituted after test, no threshold was retuned and no deployment pointer changed. Actual activation of C was tested against a fresh audit-only pointer and correctly refused before any pointer write.

C's raw oos correctness is 470/1,000 versus B's 718/1,000; 42 of C's 44 invalid outputs occur on oos requests. Supported classification improved while open-set handling remains a limitation. Validation/test oos proportions differ, so coverage changes need that context. Generated test curves are retrospective diagnostics only, not selection evidence.

### Complete native serving results

- **3,100/3,100** exact worker/reference raw outputs, token counts and truncation flags matched. Windows/GTX 1650, unchanged Qwen3-0.6B checkpoint 944, NF4/FP16, separate Transformers worker, cached prefix, greedy decoding, batch size 1.
- HTTP workload: seed 42, sampling with replacement, **484 supported + 16 oos**, identical at concurrency 1/4/8; 20 warmups excluded. Cold start **52.41 seconds**, warmup **18.79 seconds**.
- Serial: **500/500** completed, **1.043 completed/s**, p50/p95/p99 **1,130.36 / 1,442.35 / 1,722.14 ms**; zero live decision mismatches, **97** gate short-circuits and **403** model calls, **402** routes and **98** reviews.
- Concurrency 4 and 8: at each, **1/500** completed and **499 HTTP 503 model_busy** failures. These measure overload, not scalable serving; all-request fast percentiles must not be presented as successful latency. The harness checked intent live; retained outcomes permit rechecking decision/reason, mismatch flags and complete failure accounting.
- **2,059** whole-GPU samples across startup/parity/load: maximum **2,598 MiB** used and **97%** utilization, including desktop activity. Worker peak allocation **851,771,392 bytes**, reservation **1,400,897,536 bytes**, peak RSS **2,501,709,824 bytes**.
- Cost arithmetic verified: **$3.9398/1,000 submitted** under the explicitly hypothetical 100,000/month, 730 billed hours, $10 supporting allowance and dated $0.526/hour g4dn.xlarge scenario. Local measured runtime is not measured EC2 throughput. Training cost is separate; no cloud bill or savings claim.

### Acceptance audit and remaining work

Executed successfully:

```powershell
.\.venv\Scripts\python.exe scripts/verify_milestone5.py --output artifacts/milestone5-acceptance-v1
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
```

The CPU-only auditor verifies all frozen hashes, completed ledger/registry, validation selection, canonical test joins, strict output parsing, raw/per-class/fixed-policy metrics, exact paired bootstrap results, all parity records, 1,500 HTTP outcomes, counter deltas, cost arithmetic and the activation guard. It does not run model inference or tune on test. An initial audit check used the wrong expected error-code spelling (`busy`); corrected to the actual API contract `model_busy`, then the full audit passed. No experiment evidence was edited to pass the check. Copies and checksums are retained under `reports/milestone5-final-v1/`; all **46 retained-file checksums** verified after copying. **153 tests passed in 22.34 seconds**; Ruff lint/format and mypy passed. The new audit script, retained evidence, README, decisions, journal and runbook/status notes are the changes in this follow-up; frozen implementation/configuration/locks remain unchanged.

**Local benchmark acceptance is complete; full Milestone 5 container/backend acceptance remains pending.** Docker is still unavailable in this environment; no container build, hosted CI success or Linux/WSL vLLM run is claimed. Keep the disqualified release inactive. The next concrete work is the CPU container build/health workflow, then GPU-container parity or a separately evaluated supported backend in an available Docker/Linux environment, using validation only. Commands are in [the runbook](release_benchmark.md). Do not rerun the completed final test workflow. Milestone 6 remains outside this turn's scope.

## Milestone 5 — earlier implementation and smoke handoff (superseded by full run above)

The user requested terminal commands for the longer runs. The following command was handed off and has now completed; retained here as execution history:

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone5.py
```

This owns and cleans up a localhost worker/gateway, runs all 3,100 validation parity checks, warms up and submits 500 HTTP requests at concurrency 1/4/8, records cost/resource evidence, freezes all three candidates using validation selection, then records test use and runs the official 5,500-request test split through each frozen candidate once. It does not retrain, deploy automatically or spend money. Expect several hours; no exact completion-time claim is made from the small smoke. Stage/resume instructions and container checks are in [the runbook](release_benchmark.md).

### Implemented

- Separate authenticated Transformers/NF4/FP16/PEFT worker, preserving C's evaluated base/adapter/prompt/decoder; CPU gateway uses A's frozen gate and skips generation for low scores. Baseline classification is never substituted for the SLM result. Full token-budget checks and explicit unavailable/busy/timeout failures remain enforced.
- Private Prometheus-compatible counters/histograms and model identity; JSON status/timing/outcome logs without request content. Optional nvidia-smi whole-GPU utilization/memory samples and worker allocator metadata. Single-flight admission stays bounded; overload is reported as 503, not fictional review.
- Resumable complete validation raw-output/token parity, real HTTP load harness with warmup/cold-start separation, complete failure accounting, successful decision checks, submitted/completed throughput and separate latency percentiles. All-request percentiles cannot be interpreted as successful-request latency under overload.
- Validation-only release selection, input/source/lock/model/tokenizer/gate/policy hashes, source commit, full serving prerequisites and unchanged-threshold final evaluation. Exclusive test-use ledgers, prior-exposure disclosure, partial-prefix checksums and writer lock prevent accidental concurrent/repeated tests. Tests can disqualify an automatic release but do not trigger tuning.
- Atomic activation/rollback pointers and explicit restart workflow. Activation checks final test completion/quality; the script does not activate it. The current baseline API configuration remains unchanged.
- CPU/GPU Dockerfiles with a registry-verified immutable Python base digest, optional private-network GPU compose profile, localhost gateway binding, read-only artifacts and non-root users. CPU CI container fixture build/health job is authored. No Docker executable is available here, so image builds/runtime behavior and hosted CI are unverified.
- Dated official AWS Linux On-Demand quote and explicit cost model: active runtime, idle allocation, demand, supporting allowance and separate training amortization. AWS deployment/cleanup runbook has no provisioning automation or implicit spending authorization.

### Executed and verified

Evidence: [reports/milestone5-smoke-v1](../reports/milestone5-smoke-v1/README.md). The source identity in the retained later smoke matches the recorded implementation.

| Check | Actual result |
| --- | --- |
| Real native worker parity | **6/6** raw outputs and token counts equal to C reference; not full parity |
| Startup-to-readiness | **17.14 seconds** for worker plus gateway |
| Serial HTTP smoke | **24/24** complete, **0** decision mismatches; **6** gate rejections and **18** generation calls |
| Serial smoke throughput/p95 | **1.098 completed/s**, **1,473.88 ms** p95, small validation workload |
| Concurrency 4 and 8 | At each: **1 complete, 23 busy failures / 24 submissions**; all failures retained |
| Resource sampling | **32** whole-GPU samples plus worker allocator metadata |
| Release guard | Refused smoke evidence before freeze; no real test-use ledger created |
| Automated checks | **153 passed in 41.84 seconds** on 1 October; Ruff lint/format and mypy passed |
| Synthetic final-run integration | All three fixture candidates evaluated with fixed thresholds; interrupted B resumed successfully; completed replay rejected; fixture explicitly marked |
| Other contract tests | Validation selection/tie rules, artifact mutation, prior exposure, resume restrictions, atomic rollback, private metrics, actual gate short-circuit and cost arithmetic |
| Container fixture preparation | Synthetic baseline artifacts created successfully; container itself not executed |
| Price/image metadata | Official AWS feed retrieved; official Python image tag resolved to immutable digest |

Commands actually executed included:

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone5.py --stage smoke
.\.venv\Scripts\python.exe scripts/run_milestone5.py --stage smoke --smoke-id smoke-v2
.\.venv\Scripts\python.exe scripts/create_container_fixture.py --output artifacts/container-fixture-m5
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
```

Also executed: current-hardware inspection; read-only AWS pricing and Docker registry digest lookups; source/evidence identity comparison; explicit rejection of smoke evidence by `release.freeze`; confirmation that no genuine test-use registry exists. Early lint errors and a missing `Path` import in the new synthetic test were fixed before the full passing suite. The first guessed AWS feed URL returned 404; the correct official feed was located and its gzip payload decoded. No failed experiment was described as successful.

### Material limitations and next task

**Milestone 5 is not complete.** The full 3,100-example serving parity, 500-per-level HTTP benchmark, actual release freeze and genuine final test report await the user's terminal run. After it finishes, audit the ledger/configuration hashes, final raw and fixed-policy metrics, API failures, resources and cost assumptions. Do not tune on those test results.

Docker is not installed, so CPU/GPU image builds and container health/parity are unverified. Native Windows vLLM is unsupported by its current documentation; the implemented and exercised fallback is a separate-process Transformers worker. vLLM's Qwen3 support does not establish compatibility of this exact quantized adapter. Its Linux/WSL deployment path remains unexecuted; do not claim a vLLM result. These limitations are documented in [the serving runbook](release_benchmark.md) and [AWS runbook](../deployment/aws_runbook.md).

The service currently supports one active model request; concurrent workloads mostly fail fast in the smoke. No production capacity, measured cloud throughput, local energy bill or cost saving is claimed. The illustrative hosting scenario uses a verified $0.526/hour us-east-1 quote but assumed demand/uptime/support costs. Preserve every failed request in the full benchmark and do not use its fast latency as evidence of improvement.

## Milestone 4 — completed fine tuning and analysis

The complete paired experiment uses **Qwen3-0.6B**, revision `c1899de289a04d12100db370d81485cdf75e47ca`, **596,049,920 base parameters**, and **10,092,544 trainable LoRA parameters**. The user ran `scripts/run_milestone4.ps1` on 28–29 September. Training processed exactly **15,000 supported + 100 oos training examples**, one epoch, **944 optimizer updates**, rank 16/alpha 32/dropout 0.05, learning rate 1e-4, microbatch 1/accumulation 16, seed 42. NF4 with FP32 training compute is the recorded GTX 1650 precision fallback; both B and C inference use NF4/FP16 with identical prompt/template/decoding and the same baseline gate.

Training took **42,072.27 seconds (11.69 hours)**, within the 24-hour limit; full result elapsed scope is 42,080.26 seconds. Mean trainer loss was **0.0586019859**, with all 944 loss/gradient log records finite. All ten saved checkpoints plus final adapter verified (11 bundles). The selected checkpoint **944** is the sole completed epoch, selected by the configured macro-F1/invalid/earlier-step rule. Its weights equal the final adapter whose unload/reload probe logits matched exactly (**0.0** maximum difference). C's complete validation ran in a fresh process using that checkpoint.

### Genuine validation results

Each candidate includes **3,100 requests: 3,000 supported over 150 classes and 100 oos**, with one verified result per canonical ID and no inference infrastructure failures.

| Measure | Baseline A | Prompted B, 0.6B | Fine-tuned C, 0.6B |
| --- | ---: | ---: | ---: |
| Raw supported macro-F1 | 0.882634 | 0.272377 | **0.966217** |
| Supported accuracy | 88.3333% | 22.4000% | **96.5333%** |
| Invalid outputs / 3,100 | 0 | 654 | 1 |
| Selected routing coverage | 68.4194% | 0% | **80.3548%** |
| Routed errors / routes | 101 / 2,121 | 0 / 0 | **56 / 2,491** |
| Selected routing error | 4.7619% | Undefined | **2.2481%** |
| Oos review recall | 90 / 100 | 100 / 100, review all | **90 / 100** |
| Original policy constraints | Met | Unmet; routing disabled | **Met** |

C's threshold is **0.11417805060646902**, policy `policy-v1-2fe7bafe54be`, model `finetuned-qwen3-06b-qlora-v1-step944-bfe54e26195f`. Paired macro-F1 differences (1,000 supported-sample resamples, seed 42): C−A **0.083583**, 95% interval **[0.073414, 0.096574]**; C−B **0.693840**, interval **[0.683129, 0.710616]**. Thresholds and checkpoint choice use validation; these are not independent test guarantees.

C fixes **2,227** supported B errors and introduces **3** regressions. Compared with A, C alone is correct on **284** supported requests and A alone on **38**. B's failures include 1,139 false oos rejections and 630 invalid outputs on supported requests. Across all requests, 653 of B's 654 invalid outputs are unknown labels, so the improvement includes taxonomy adherence rather than only JSON syntax. C still has 104 supported errors (98 wrong labels, six false oos rejections).

**Raw oos correctness declines from 67/100 (B) to 55/100 (C)**; the shared gate raises C's oos review recall to 90/100. Forty category-stratified examples were inspected and annotated, including all three supported regressions, remaining confusions and oos overgeneralization. See [error analysis](../reports/three-way-qwen3-06b-v1/error_analysis.md) and [review records](../reports/three-way-qwen3-06b-v1/error_review.json). This assistant-authored review is not an independent annotation study.

### Runtime, verification and retained evidence

- Peak training allocation: **2.21 GiB**; peak PyTorch reservation: **4.78 GiB** on the 4 GiB GTX 1650. Reservation exceeds dedicated VRAM; Windows shared-memory/paging behavior was not independently measured. Do not claim the entire job fit dedicated VRAM.
- B validation: **40.87 minutes** summed inference time; C: **55.02 minutes**. Serial p95 model latency: B **1,078.43 ms**, C **1,306.89 ms**, A **0.7844 ms**. This excludes loading, cached-prefix prefill, gate and API overhead; it is not a service load test. A remains the current API adapter.
- `scripts/verify_milestone4.py --output artifacts/milestone4-acceptance-v1` passed using the CPU interpreter. It verified current source/config/lock/data/prompt identity, checkpoint checksums, selected/final weight equality, canonical validation joins, recomputed raw metrics and policies, and **byte-identical** comparison/bootstrap/paired-error output. No test records or model weights were loaded for evaluation.
- **141 tests passed in 11.12 seconds**. Ruff lint/format, mypy's five core modules, and offline training-lock consistency passed. Existing training/inference source was preserved to keep the executed experiment identity unchanged.
- Genuine reports: [B validation](../reports/prompted-qwen3-06b-nf4-v1-val/report.md), [B policy](../reports/prompted-qwen3-06b-nf4-v1-policy/report.md), [C validation](../reports/finetuned-qwen3-06b-qlora-v1-val/report.md), [C policy](../reports/finetuned-qwen3-06b-qlora-v1-policy/report.md), [A/B/C comparison](../reports/three-way-qwen3-06b-v1/report.md), and [training/acceptance evidence](../reports/finetuning-run-v1/README.md). Per-class reports, confusion matrices, plots and every raw prediction are retained.

This completes Milestone 4's required training, reproducible reload, three genuine reports and changed-error explanation. One configuration, epoch and seed were run; raw oos generalization, public-data contamination, known source duplicates and validation selection remain limitations. No paid resource, test evaluation, deployment replacement or cost-saving claim was made. GPU resume of a long interrupted run remains untested; the completed run was uninterrupted, and the earlier tiny CPU resume fixture remains the available resume evidence.

**Next task: Milestone 5**, when requested. Prepare the serving artifact/backend and verify adapter parity, then service telemetry/load measurements and release configuration freeze before any final test evaluation. Keep the baseline API and test holdout unchanged until that work is authorized.

## Milestone 4 prerequisite record — 2026-09-28

The following records the earlier smoke and terminal handoff. Its pending-run instructions were fulfilled by the completed run and acceptance above.

The user requested terminal commands for the long run. Run from the project root, with no environment activation required:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_milestone4.ps1
```

This runs the new paired B, one epoch of C, validation checkpoint selection, and reports. See [the runbook](finetuning.md) for separate stages and resume behavior. **Milestone 4 is not complete until genuine full training, C validation and all three candidate reports exist and their changed errors are reviewed.** The small smoke results below are not held-out quality estimates.

### Implemented

- Pinned smaller pair: **Qwen/Qwen3-0.6B**, revision `c1899de289a04d12100db370d81485cdf75e47ca`. The user had chosen the smaller-model route after the discarded 4B training attempt. Both B and C must be rerun on this base; the old 4B benchmark remains historical evidence.
- Conversational completion-only SFT for exactly **15,000 supported + 100 oos training records**, full sorted 150-label catalog, B's exact non-thinking prefix, JSON+EOS targets, fail-on-truncation preflight and ten directly inspected masks. Validation/test examples never enter SFT.
- Separate exact training lock: PyTorch 2.8.0+cu128, Transformers 4.57.6, TRL 0.24.0, PEFT 0.17.1, accelerate 1.15.0, bitsandbytes 0.50.2. Root CPU and original prompted locks remain unchanged.
- Configuration 1: rank 16, alpha 32, dropout 0.05; verified attention/feed-forward projections; learning rate 1e-4; microbatch 1, accumulation 16; seed 42; one epoch; sequence limit 1,024; local budget 24 hours. NF4 base, explicit FP32 training fallback; B/C inference both NF4/FP16. No paid service.
- Real smoke prerequisite, finite loss/gradient checks, changed-weight verification, adapter reload parity, periodic and epoch checkpoints, exact identity/hash checks, bounded cumulative recorded runtime and resume. Loss, gradient norm, memory, elapsed time and processed examples are logged.
- Verified PEFT inference with cache constructed after adapter loading. Existing strict parser/shared baseline gate retain invalid outputs and infrastructure errors. Smoke adapters cannot enter complete benchmark evaluation.
- Validation-only checkpoint selection by macro-F1, then invalid count, then earlier step; A/B/C raw/policy comparison with strict pairing, bootstrap intervals, changed errors and fixed/regressed examples.

### Executed and measured on 2026-09-28

Evidence is retained in [reports/finetuning-smoke-v1](../reports/finetuning-smoke-v1/README.md). Windows, AMD Ryzen 5 5600H, approximately 15.35 GiB RAM, GTX 1650 4 GiB, CUDA 12.8; native BF16 unsupported. Before the training smoke, CUDA reported 3,456,892,928 free GPU bytes. Desktop GPU availability varies.

| Check | Actual result |
| --- | --- |
| Prepared training data | 15,100 records; 651–692 tokens including completion; no truncation |
| Direct mask inspection | 10 examples, including longest request and oos; JSON+EOS supervised, prefix masked; exact real TRL collator agreement |
| Prompted prefix cache | All six predetermined cached/uncached output comparisons equal; hardware probe only |
| Trainable LoRA parameters | 10,092,544 |
| Training smoke | One optimizer update, 16 actual examples, finite loss **3.2666759491**, gradient norm **66.4597015381**; weights changed |
| Measured training time | **46.12 seconds** for smoke training; **53.34 seconds** including model load and save/reload scope |
| Peak GPU allocation / reservation | **2,295,059,456 / 3,479,175,168 bytes** (2.14 / 3.24 GiB) |
| Process RSS at completion | 2,608,156,672 bytes |
| Saved adapter reload | Maximum absolute probe-logit difference **0.0** after unload/reload on the identical frozen base |
| Fresh-process NF4/FP16 adapter inference | All 3 requests accounted for: **1 valid, 2 invalid outputs**, zero infrastructure errors. One-update smoke; not a C benchmark |
| Tiny random CPU model fixture | EOS/padding mask correct; uninterrupted versus optimizer/RNG-resumed adapter parameters differ by **0.0** after two steps; explicitly synthetic |
| Automated suite | **141 passed in 10.99 seconds**; includes 30 new CPU contract/integration tests |
| Static checks | Ruff lint/format and mypy core checks passed; PowerShell script parsed without syntax errors |
| Training lock | `uv lock --check --offline --project environments/training ...` passed |

The 16-example measurement projects **12.09 hours for one epoch**, excluding full B/C validation and allowing no claim of stable long-run throughput. This is a feasibility estimate, not a completed training duration. The configured 24-hour training ceiling is checked at optimizer boundaries; save/reload can add time. A full epoch has 944 optimizer updates with the last accumulated batch smaller than 16.

Commands actually executed (repository root, explicit interpreters; no activation):

```powershell
.\.tools\uv.exe sync --offline --project environments/training --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
# Downloaded the pinned 0.6B snapshot using huggingface_hub.snapshot_download.
# The first restricted-network attempt failed; an authorized network escalation succeeded.
$env:HF_HUB_OFFLINE = '1'
.\environments\training\.venv\Scripts\triage.exe data sft --config configs/finetune.yaml
.\environments\training\.venv\Scripts\python.exe scripts/verify_prompt_cache.py --config configs/prompted-small.yaml --output artifacts/qwen3-06b-cache-smoke-v1
.\environments\training\.venv\Scripts\triage.exe train slm --config configs/finetune.yaml --smoke
.\environments\training\.venv\Scripts\triage.exe predict --config configs/finetuned.yaml --bundle artifacts/finetuned-qwen3-06b-qlora-v1-smoke/adapter --split val --limit 3 --output artifacts/finetuned-qwen3-06b-inference-smoke-v1
.\environments\training\.venv\Scripts\python.exe scripts/verify_sft_stack.py --output artifacts/sft-stack-fixture-v1
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.tools\uv.exe lock --check --offline --project environments/training --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
```

Installed TRL/PEFT source APIs were inspected. Early lint issues were fixed before acceptance. No training/test failure is hidden or treated as a benchmark success. The fresh adapter inference's two invalid completions are retained unchanged. The user-facing long-run script has been syntax checked; its full path has **not** been run, per the user's request to execute the long job themselves. GPU checkpoint resume over a long run and non-Windows execution remain unverified; only the tiny installed-stack resume fixture was executed.

### Next task recorded at the prerequisite handoff (now completed)

The user runs `scripts/run_milestone4.ps1`, then asks Codex to continue. Verify the full 0.6B B predictions, completed C training and each epoch checkpoint, selected C report/policy, all input/source hashes and changed-error examples. Record actual quality/runtime/resource results, retain the genuine reports, and only then mark Milestone 4 complete. Test remains unused until Milestone 5's frozen release. The baseline API remains unchanged.

## Milestone 3 - completed prompted model benchmark

The full validation run contains **3,100 verified outcomes: 3,000 supported and 100 oos requests**, with no missing or duplicate IDs. Configuration, source code, prompt, input data, baseline gate, dependency lock and prediction hashes were verified after the user completed the resumed terminal run. All requests reached the model before offline gating.

The model is **Qwen/Qwen3-4B**, revision `1cfa9a7208912126459214e8b04321603b3df60c`, with **4,022,468,096 parameters**, NF4 double quantization and FP16 compute. Experiment: `prompted-qwen3-4b-nf4-cache-v1`; model version: `prompted-qwen3-4b-nf4-cache-v1-95bdf39cb9b2`. No smaller model or paid service was substituted.

### Measured validation results

| Measure | CPU baseline A | Prompted Qwen B |
| --- | ---: | ---: |
| Supported macro-F1, all 150 labels | 0.882634 | 0.801038 |
| Supported accuracy | 88.3333% | 79.2333% |
| Invalid outputs | 0 | 70 / 3,100 (2.2581%) |
| Infrastructure failures | 0 | 0 |
| Selected policy | targets_met | targets_unmet; automatic routing disabled |
| Selected routing coverage | 68.4194% (2,121 routes) | 0% (all 3,100 reviewed) |
| Selected routing error | 4.7619% | Undefined: zero routes |
| Selected oos review recall | 90% | 100% from reviewing everything |

The prompted macro-F1 difference is **-0.081596**, with paired bootstrap 95% interval **[-0.099838, -0.067063]** (1,000 supported-sample resamples, seed 42). Of 3,000 supported requests, 2,171 are correct for both candidates, 206 only for Qwen, 479 only for the baseline, and 144 for neither. These are validation comparisons, not independent test or production guarantees.

All 70 invalid outputs remain failures: **65 unknown catalog labels and five non-JSON `oos` strings**. There are no truncated generations. Valid `oos` predictions include 53 gold-oos requests and 92 supported requests. Without a gate, 58% of gold-oos requests go to review (53 valid oos plus five invalid outputs); this differs from raw oos classification accuracy.

The exact prompted threshold sweep finds no point meeting all original constraints (error <=5%, oos recall >=90%, coverage >=20%). Among points meeting the latter two, the smallest routing error is **9.1454% (61/667)** at **21.5161% coverage**, **98% oos recall**, threshold `0.6771616657577827`. This is diagnostic, not an alternative selected policy. The frozen prompted policy is `policy-v1-b6d73dde0c13`, with automatic routing disabled. The baseline remains the better measured candidate of A and B; final deployment selection awaits later milestones and C.

### Hardware, runtime and execution scope

Local hardware: AMD Ryzen 5 5600H, 15.35 GiB RAM, NVIDIA GTX 1650 with 4 GiB VRAM, Windows, driver 617.14, PyTorch 2.8.0+cu128 / CUDA 12.8. Native BF16 is unavailable, hence FP16 compute. Successful inference does not establish fine-tuning feasibility.

- All 3,100 prompts include the complete sorted 150-label catalog and fit: **645-679 input tokens**, limit 2,048 plus 32 generated tokens. Non-thinking template verified, greedy decoding, batch size one, no prompt truncation or few-shot examples.
- Uncached ten-request smoke: median **17.80 seconds/request**, eight valid outputs and two unknown-label failures. It is hardware evidence from one intent class, not a quality benchmark.
- Six predetermined cached/uncached probes matched exactly. Cached durations were **1.38-2.16 seconds** versus **17.80-23.49 seconds** uncached. Each request receives a deep copy of the fixed **627-token system cache**; no request tokens are shared.
- Full validation summed saved-request inference time: **4,840.60 seconds (80.68 minutes)**. Across all requests, p50 **1,486.78 ms**, p95 **2,052.95 ms**, p99 **2,186.12 ms**. These include tokenization and cached generation; exclude gate execution, loading, prefix prefill and the paused interval. They are not API latency, throughput or cost measurements. This was a shared local development machine, not an isolated serving load test.
- The run paused at **1,118** saved requests for user terminal handoff. The user resumed the remaining **1,982** with unchanged configuration. The last process session took **3,118.51 seconds**, including **29.46 seconds** for load/prefix initialization; this is not the total wall time across both sessions.
- Resumed-session peak CUDA allocated/reserved memory: **2.734 / 2.957 GiB**. Peak process working set: **4.667 GiB**. These allocator/process measurements exclude other applications and some driver overhead, and are not a measured maximum across both sessions.

### Implemented files and retained evidence

| Files | Result |
| --- | --- |
| `configs/prompted.yaml`, `prompts/intent-v1.txt` | Pinned model, full catalog, decoding and cache configuration |
| `environments/prompted/{pyproject.toml,uv.lock}` | Separate locked CUDA environment; CPU environment stays independent |
| `src/triage/models/{prompting,prompted}.py` | Escaped user-message isolation, non-thinking/token checks, strict raw outcomes, resumable validation-only runner and provenance |
| `src/triage/evaluation/{gate,bootstrap,compare}.py`, updated evaluator/policy selector and CLI | Verified shared gate, paired bootstrap, raw/policy comparisons and disabled-policy fallback |
| `tests/test_prompted.py`, `scripts/verify_prompt_cache.py` | CPU fixture contracts and separate genuine cache-parity probe |
| `reports/prompted-hardware/`, `reports/prompted-cache-smoke/` | Hardware/token audits, uncached smoke, cache parity |
| `reports/prompted-qwen3-4b-nf4-cache-v1-val/` | All raw predictions, copied gate/provenance/license, run and prompt, metrics, plot, per-class confusion data, failure breakdown and 40 reviewed examples |
| `reports/prompted-qwen3-4b-nf4-cache-v1-policy/` | Frozen review-only policy and complete exact threshold sweep |
| `reports/baseline-prompted-v1/` | Comparison, all paired predictions, bootstrap interval and feasibility diagnostic |
| README, benchmark runbook, decisions, experiment journal and this file | Reproduction, measured outcome, limitations and next task |

### Commands and acceptance checks

The GPU environment was installed with the locked optional project. Actual inference commands included the uncached smoke, cache-parity probe, full validation start, and the user's continuation:

```powershell
$env:HF_HUB_OFFLINE='1' # pinned model/tokenizer already downloaded
.\environments\prompted\.venv\Scripts\python.exe -u scripts/verify_prompt_cache.py --output reports/prompted-cache-smoke
.\environments\prompted\.venv\Scripts\triage.exe predict --config configs/prompted.yaml --split val
.\environments\prompted\.venv\Scripts\triage.exe predict --config configs/prompted.yaml --split val --resume
.\.venv\Scripts\triage.exe evaluate --predictions artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --output reports/prompted-qwen3-4b-nf4-cache-v1-val
.\.venv\Scripts\triage.exe policy select --predictions artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --split val --output reports/prompted-qwen3-4b-nf4-cache-v1-policy
.\.venv\Scripts\triage.exe compare --baseline reports/baseline-c1-v1-val/predictions.jsonl --candidate artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --output reports/baseline-prompted-v1
```

Existing completed outputs cannot be overwritten or resumed; use new output directories for reproductions. The original smoke's uncached configuration is retained in its `run.json`, while the current default enables prefix caching.

Final software checks: **111 tests passed in 10.00 seconds**; Ruff lint and formatting passed; mypy passed on the five configured core modules; both CPU and optional GPU lockfiles passed offline consistency checks. Fixture tests are clearly separated from the genuine benchmark. Verified complete prediction IDs/hashes, one-to-one gate alignment, resumed-run identity and retained report integrity. The generated coverage/error plot was inspected, and all 40 queued examples were reviewed. No inference code changed after the frozen full run began.

### Limits and next concrete task

No Milestone 3 blocker remains. The API still serves only the baseline. No test prediction, cloud provisioning, serving-load benchmark or cost claim was made. Public benchmark contamination, small oos sample size, validation threshold selection, NF4 precision and the baseline-dependent gate limit interpretation. Cache parity is verified for six probes, not guaranteed for every possible floating-point execution.

**Next: Milestone 4.** First check training memory and record a maximum training budget, then build conversational completion-only SFT records and verify loss masks. Use the same pinned Qwen3-4B base revision for the paired experiment; any model replacement must be documented for both B and C. Do not begin test evaluation or paid provisioning.

## Milestone 2 — routing policy and API

The exact validation selector and local FastAPI service are complete. No baseline retraining, hyperparameter change, test prediction, GPU execution, or paid provisioning occurred in this milestone.

### Changed files

| Files | Result |
| --- | --- |
| `src/triage/policy.py` | Immutable routing rules, inclusive threshold behavior, strict JSON-output validation, unchanged four-code reason vocabulary |
| `src/triage/evaluation/select_policy.py` | Exact score sweep, original constraints/tie breaks, validation-only selection, frozen policy hashes, unmet-target fallback, retained threshold/report evidence |
| `src/triage/service/{__init__,schemas,runtime,app}.py` | Strict API request/response/error contracts, trusted baseline loading, readiness/liveness, safe metadata, auth, rate/body/token limits, bounded asynchronous inference and timeout behavior |
| `src/triage/cli.py`, `configs/service.yaml` | `triage policy select` and `triage serve`, explicit policy/output paths, localhost defaults |
| `pyproject.toml`, `uv.lock` | Locked FastAPI/Pydantic/Uvicorn dependencies and httpx2 development client; type checks expanded to policy and API schemas |
| `tests/test_policy.py`, `tests/test_service.py` | 63 new tests; existing 33 tests retained. The existing CPU CI workflow automatically includes them |
| `scripts/verify_api.py` | Repeatable real-model validation parity through the ASGI API, with explicit evidence output and test-split rejection |
| `reports/baseline-c1-v1-policy/` | Frozen `policy.json`, all exact sweep points, and validation selection report |
| `reports/baseline-c1-v1-api/` | Genuine full-validation API parity and separate localhost HTTP smoke evidence |
| `README.md`, `docs/api.md`, `docs/decisions.md`, `docs/experiment_journal.md`, this file | Reproduction commands, API behavior/limits, ML decisions, measurements, and handoff |

### Measured acceptance results

All routing figures use the original validation mix: **3,100 requests = 3,000 supported + 100 oos**, model `baseline-c1-v1-551714a82382`. No test examples were predicted.

| Check | Observed result |
| --- | --- |
| Exact sweep | 3,068 thresholds, including every unique observed score plus zero/all-review endpoints |
| Frozen policy | `policy-v1-51f9c25a5c5c`, threshold **0.2088413160728636**, `targets_met`, automatic routing enabled |
| Routing coverage | **68.4194%**, 2,121 / 3,100 requests |
| Routing error | **4.7619%**, 101 / 2,121 routes; Wilson 95% **3.9345–5.7529%** |
| Oos recall | **90%**, 90 / 100; Wilson 95% **82.5634–94.4771%** |
| Supported review rate | **29.6333%**, 889 / 3,000 supported requests |
| Overall reviews | 979 / 3,100 requests; no gated infrastructure failures |
| API/offline parity | **3,100 actual baseline API calls, zero intent/decision/reason mismatches**; 2,121 routes and 979 reviews |
| Real HTTP smoke | `triage serve` on `127.0.0.1:8765`; live/ready/model endpoints returned 200; one genuine validation route and one review verified; server stopped afterward |
| Unmet-target behavior | Synthetic fixture freezes `targets_unmet`, disables routing, records null error/zero coverage, and returns review only while a healthy model is loaded |
| Failure contracts | 401 auth, 422 schema/body/token budget, 429 rate limit, and 503 missing/corrupt/unavailable/busy/timed-out inference verified |
| Runtime safety | Gate short-circuit and baseline prediction reuse verified; timeout retains occupied slot, liveness remains responsive, readiness recovers after successful completion, failed model stays unready |
| Input/log contracts | Instruction-like input remains data; raw messages and auth secrets absent from application logs; chunked oversized bodies rejected; no public `/metrics` |
| Tests and checks | **96 tests passed**, no warnings; lint, formatting, core type checks, and dependency-lock consistency passed |

These constraints apply to validation-selected point estimates. The Wilson intervals do not establish a <=5% unseen routing-error guarantee or >=90% unseen oos-recall guarantee. API parity and the five HTTP smoke calls are correctness checks, not throughput/latency benchmarks or production usage.

### Commands actually executed for Milestone 2

PowerShell, repository root (the local uv executable remains in `.tools/`):

```powershell
.\.tools\uv.exe sync --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.venv\Scripts\triage.exe policy select --predictions reports/baseline-c1-v1-val/predictions.jsonl --split val --output reports/baseline-c1-v1-policy
.\.venv\Scripts\ruff.exe check src tests scripts --fix
.\.venv\Scripts\ruff.exe format src tests scripts
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe scripts/verify_api.py --bundle artifacts/baseline-c1-v1 --policy reports/baseline-c1-v1-policy/policy.json --predictions reports/baseline-c1-v1-val/predictions.jsonl --output reports/baseline-c1-v1-api
.\.venv\Scripts\triage.exe serve --bundle artifacts/baseline-c1-v1 --policy reports/baseline-c1-v1-policy/policy.json --port 8765
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.tools\uv.exe lock --check --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
```

The live server was probed with Python `urllib.request`, saving `http_smoke.json`, then its known process was stopped. `uv sync` ran a second time after replacing deprecated httpx fallback with the installed Starlette version's supported httpx2 client. The first test run had a client deprecation warning (91 passed); after migration 93 passed without warnings; the final runtime/concurrency checks brought the suite to 96 passing tests. Early formatting violations were fixed before the final checks; no test failures occurred.

Final acceptance also updated the separate `.tools/acceptance-venv` via `UV_PROJECT_ENVIRONMENT=.tools/acceptance-venv` and `uv sync --locked --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache`. Its Ruff lint/format and mypy checks passed, and its pytest run reported **96 passed in 8.02 seconds**, without warnings. This verifies the updated lock independently of the development environment. Milestone 2 is committed locally with its lock and evidence; no push was performed.

Verified versions: FastAPI 0.141.1, Pydantic 2.13.5, Starlette 1.7.0, Uvicorn 0.54.0, httpx2 2.13.1; existing model dependencies remain unchanged. The full training lock hash in the original baseline metadata is historical; API evidence records the updated service dependency lock. No original benchmark artifact was rewritten.

### Limitations and next concrete task

No Milestone 2 blocker remains. The API is a single-process localhost demonstration with one inference slot and a global in-memory rate limiter. A native worker cannot be forcibly cancelled; a permanently stuck operation needs a process restart. Auth is optional for local use. `/metrics`, containers, remote deployment, release freezing, and production observability remain later work. Hosted CI/Linux execution is unverified. No GPU test or real SLM tokenizer was run or claimed verified; the current baseline has no prompt.

**Next: Milestone 3 — prompted model benchmark.** Inspect hardware and available resources, pin a feasible model/revision, implement the full fixed catalog and non-thinking prompt, verify real tokenizer budgeting, save every raw validation outcome, and attach this frozen baseline's gate by a checked one-to-one sample-ID join. Keep test prediction blocked and paid services disabled.

## Milestone 1 — historical completion record

## Delivered files

| Area | Files and behavior |
| --- | --- |
| Repository and environment | `.gitignore`, `.gitattributes`, `.python-version`, `pyproject.toml`, `uv.lock`, `README.md`; Python package, editable installation, CPU dependency lock, reproducible commands |
| Configuration | `configs/data.yaml`, `configs/baseline.yaml`; official commit/checksums and one fixed C=1.0 experiment |
| Package/CLI | `src/triage/__init__.py`, `cli.py`, `contracts.py`, `io.py`; four implemented commands, explicit outputs, clear errors, overwrite refusal |
| Data | `src/triage/data/{__init__,prepare,load}.py`; pinned downloads, schema/count/hash checks, deterministic IDs, catalog, canonical splits, duplicate/conflict audit, provenance manifest |
| Baseline | `src/triage/models/{__init__,baseline}.py`; supported-training-only TF-IDF/logistic regression, warning capture, saved pipeline, exact reload check, validation predictions and gate scores |
| Evaluation | `src/triage/evaluation/{__init__,metrics,report}.py`; fixed-catalog macro-F1, routing diagnostics, Wilson intervals, failure accounting, per-class CSV, confusion matrix, plot, saved-prediction completeness checks |
| Tests and CI | `tests/conftest.py`, `test_data.py`, `test_metrics.py`, `test_baseline.py`, `.github/workflows/cpu.yml`; 33 tests, pinned workflow actions, locked CPU install, lint/format/types/tests |
| Evidence | `reports/baseline-c1-v1-val/`: report, metrics, all 3,100 predictions and manifest, data manifest, duplicate audit, per-class report, confusion matrix, coverage/error curve and plot, 40 review samples, manual error analysis, source license |
| Decisions and learning | `docs/decisions.md`, `docs/experiment_journal.md`, this file |

Local ignored outputs remain available in `data/clinc150/`, `artifacts/baseline-c1-v1/`, and `artifacts/baseline-c1-v1-val/`. The model bundle is 37,762,940 bytes. `.tools/` contains the local uv/Python installation and a second acceptance environment; `.venv/` is the development environment. These are not repository dependencies or committed assets.

## Acceptance evidence

| Check | Observed result |
| --- | --- |
| Official source | `clinc/oos-eval@828f8093932c8fe6ca7936c3d2e52903b1c523de`; data, domain mapping, and license hashes match pinned values |
| Train split | 15,000 supported + 100 oos; classifier fit uses only the 15,000 supported samples |
| Validation split | 3,000 supported + 100 oos; every sample has a saved prediction |
| Test integrity | 4,500 supported + 1,000 oos; prepared and integrity-audited only; **no test predictions, tuning, or classification metrics** |
| Duplicate audit | 5 exact/normalized cross-split groups; 4 conflicting-label groups; 0 within-split groups; official membership preserved |
| Fresh installation | Second environment installed with `uv sync --locked --offline` using the already downloaded package cache |
| Fresh data path | Actual downloader fetched all three pinned files; the local-source option was also exercised earlier |
| Training | C=1.0, seed 42, word unigrams/bigrams, sublinear TF, one native math thread, float64 CPU; converged in 21 iterations, no warnings, 10.383 seconds |
| Artifact parity | Exact probability equality before/after save/reload on 32 training probes; fixture tests also verify labels/probabilities |
| Validation raw supported macro-F1 | **0.8826340782** over all 150 supported labels |
| Validation supported accuracy | **0.8833333333**; 350 errors / 3,000 supported samples |
| Invalid/infrastructure failures | 0 / 3,100 for each category |
| Ungated diagnostic | 3,100 / 3,100 routed, 450 errors / 3,100 routes (14.5161%), 0 / 100 oos recalled; no operating policy selected |
| Serial model latency | p50 0.6153 ms, p95 0.7844 ms, p99 1.0040 ms; per-record vectorization/classification, **not API latency or a load test** |
| Error inspection | 40 examples inspected and annotated: 30 supported errors + 10 oos; deterministic, class-order-biased sample |
| Final tests | **33 passed in 6.33 seconds**, no failed tests |
| Final code checks | Ruff lint and formatting passed; mypy passed on the 3 configured core-contract/helper modules |
| Lock consistency | `uv lock --check --offline` passed; no GPU dependencies installed |

Hardware: AMD Ryzen 5 5600H with Radeon Graphics, Windows x86-64. CPU name was read from the Windows registry; CIM inventory access was denied, so installed RAM was not measured. Runtime: Python 3.11.16, scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, joblib 1.6.0, uv 0.12.19. Exact runtime metadata is in the prediction manifest; all resolved dependencies are in `uv.lock`.

## Commands actually executed

The shell was PowerShell, from the repository root. Initial inspection read the specification, searched for applicable `AGENTS.md` files (none found), checked PATH and Git, and confirmed the folder initially contained only the specification. Network downloads required sandbox escalation; they succeeded. No paid resource was used.

Setup and first development checks:

```powershell
# Downloaded the uv Windows archive and resolved the official CLINC commit via HTTPS.
.\.tools\uv.exe python install 3.11 --install-dir .tools/python
.\.tools\uv.exe sync --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.venv\Scripts\triage.exe data prepare --config configs/data.yaml --source-dir .tools
.\.venv\Scripts\triage.exe train baseline --config configs/baseline.yaml
.\.venv\Scripts\ruff.exe check src tests --fix
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\pytest.exe -q
git init
```

The first data/model development outputs were removed after checking that both resolved target directories were inside this workspace. The final acceptance sequence regenerated them using the current implementation. No existing user file was removed or rewritten.

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.tools/acceptance-venv'
.\.tools\uv.exe sync --locked --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.tools\acceptance-venv\Scripts\triage.exe data prepare --config configs/data.yaml
.\.tools\acceptance-venv\Scripts\triage.exe train baseline --config configs/baseline.yaml
.\.tools\acceptance-venv\Scripts\triage.exe predict --config configs/baseline.yaml --split val
.\.tools\acceptance-venv\Scripts\triage.exe evaluate --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --output reports/baseline-c1-v1-val
.\.tools\uv.exe lock --check --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.tools\acceptance-venv\Scripts\ruff.exe check .
.\.tools\acceptance-venv\Scripts\ruff.exe format --check .
.\.tools\acceptance-venv\Scripts\mypy.exe
.\.tools\acceptance-venv\Scripts\pytest.exe -q
```

Also executed: source-file SHA256 checks; `git ls-remote` for both pinned CI actions; inspection of all 40 saved error-review records; plot rendering and visual inspection; copying the data manifest, duplicate audit, and original license into the retained report. Early lint checks found line-length formatting issues, which were fixed before the final successful checks. There were no test failures. The original system `python`/`py` aliases were unusable; the local installation resolved this environment issue.

During the final Git review, Windows CRLF serialization was found to conflict with Git's LF normalization and recorded checksums. JSON/CSV serialization now explicitly writes LF, and deterministic-data tests assert this. The data preparation, training, prediction, and evaluation sequence was rerun after this correction, using `data prepare --config configs/data.yaml --source-dir .tools` with the same verified official source files. Classification metrics and the 40 reviewed errors were unchanged; the table above records the final run's timing. The retained report's prediction, data-manifest, duplicate-audit, and license hashes were checked against their Git-staged bytes.

Repository initialization succeeded. Staging required sandbox escalation and a command-local `safe.directory` setting because the sandbox and host user have different ownership identities; no global Git setting was changed. The milestone files and lockfile are included in the initial local commit. No remote push was performed.

## Limits and next task

No blocking Milestone 1 issue remains. Windows execution is verified; the GitHub Actions Windows/Linux matrix is authored but has not run on a hosted runner. Fresh-environment installation was verified on this machine with cached locked distributions, not on a different operating system. No GPU test was attempted or claimed verified; GPU work is outside this milestone.

The baseline is deliberately ungated. Its scores are not calibrated correctness probabilities, and raw oos rejection is zero. The report's threshold curve is exploratory only. No validation constraint, deployment readiness, cost saving, or real customer outcome is claimed. Only one C configuration was run. The error observations are assistant-authored and not an independent human annotation study.

The next task recorded at Milestone 1 completion was Milestone 2; it is now complete as documented above. Milestone 3 is now complete; the current next task is Milestone 4 as documented above.
