# GPU container smoke — passed, not full acceptance

The user completed the GPU container build and smoke on 1 October 2026 using `scripts/run_container_validation.py --stage gpu --output artifacts/milestone5-container-gpu-smoke-v1`. This is genuine validation data through the Linux/WSL2 Transformers worker, not a fixture or a final-test rerun. The adapter, prompt, decoder, shared gate and validation-selected policy remain unchanged.

| Check | Measured result |
| --- | --- |
| Reference/worker parity | **6/6** raw outputs, input/output token counts and truncation flags match |
| Gateway contracts | Live/readiness/model, authentication, empty/oversized input and private metrics checks pass |
| Startup-to-readiness | **36.23 seconds**; excludes image building |
| Warmup | **13.55 seconds**, excluded from load results |
| Serial HTTP | **24/24** completed; zero live intent/decision/reason mismatches |
| Serial throughput | **1.411 completed/s**, small workload only |
| Serial p50 / p95 / p99 | **903.57 / 1,159.29 / 1,241.84 ms** |
| Gate and generation counters | **6** real gate short-circuits, **18** model calls; **17** routes, **7** reviews |
| Concurrency 4 | **1/24** completed, **23 HTTP 503 model_busy**; completed latency 1,058.96 ms |
| Concurrency 8 | **1/24** completed, **23 HTTP 503 model_busy**; completed latency 1,056.93 ms |

The same seed-42 workload (23 supported, one oos, sampling with replacement) ran at each concurrency. All failures remain in the outcome files. The concurrent runs demonstrate overload, not useful concurrent capacity. The six parity cases and 24-request load do not establish complete Linux parity or a stable performance improvement over native Windows.

Hardware/runtime: GTX 1650 4 GiB, Docker Desktop 4.93.0 / engine 29.8.1, Linux kernel 6.18.40.1-microsoft-standard-WSL2, Torch 2.8.0+cu128, Transformers 4.57.6, bitsandbytes 0.50.2, PEFT 0.17.1. C remains Qwen3-0.6B checkpoint 944, NF4/FP16 with greedy decoding and a cached system prefix. Worker metadata at parity startup records peak allocation **851,771,392 bytes**, reservation **1,400,897,536 bytes**, RSS **1,615,044,608 bytes**; these are not post-load resource peaks.

Both Dockerfiles built. The GPU dependency-install layer took **506.6 seconds** and image export/unpack **236.7 seconds**, according to retained build logs. Local image inspection reports GPU image size **11,940,341,725 bytes** (about 11.94 GB); this is an image-size measurement, not network download volume. The image IDs still match `identity.json`. Docker's missing Git provenance warning is retained; explicit source and image hashes identify the run. Transformers also logged its standard warning that sampling flags are ignored during greedy generation; no configuration was changed in response.

A CPU-only audit rechecked all six raw parity records and their checksum, 72 HTTP submissions, retained decision/reason values, failure counts, percentiles, throughput, metric deltas, source/reference/Dockerfile identities, and the unchanged completed final-test ledger. Returned intents were checked live by the harness; raw intent values are not retained in load outcomes. No inference was rerun by the audit. The original release still verifies and remains disqualified; no activation pointer exists. No temporary smoke containers or validation networks remained after cleanup.

`verification.json` records the audit. `evidence_sha256.json` covers the retained evidence before this README was added. Build logs, image identities, raw parity results, and every HTTP outcome are included. No authentication secrets or model weights are retained here. Existing model/service code and locks were unchanged, so another unit-suite run was not needed for this evidence-only update.

Next, with Docker Desktop running, from the project root:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage gpu --full --skip-build --output artifacts/milestone5-container-gpu-full-v1
```

This reuses the built images, compares all 3,100 validation outputs, and submits 500 requests at each concurrency 1/4/8. It does not retrain, select a new policy, activate a release or use the final test split. Full GPU-container acceptance and the vLLM alternative remain unverified until separately resolved.
