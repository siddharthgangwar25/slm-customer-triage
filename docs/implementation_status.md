# Implementation status

Last updated: **2026-09-28**. Milestones **1, 2 and 3 are implemented, executed, and locally verified**. **Milestone 4's workflow is implemented and smoke-tested; full training and paired validation are pending the user's terminal run.** Milestones 5-6 have not been implemented. Original baseline and 4B evidence remain unchanged. Test predictions remain unused.

## Milestone 4 — implementation and smoke verified; long run pending

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

### Next concrete task

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
