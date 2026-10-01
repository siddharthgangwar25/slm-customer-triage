# Full GPU container validation — passed

The user completed the full validation-only Docker workflow on 1 October 2026. An offline audit verified all raw records and accounting without generating new predictions. **All 3,100 worker outputs match the native reference; all 500 serial HTTP requests complete and match the frozen policy.** This closes the full local Transformers container checks. It does not override the original release's failed final-test oos-recall criterion.

## Conditions and identity

Linux/WSL2 kernel 6.18.40.1, Docker Desktop 4.93.0 / engine 29.8.1, GTX 1650 4 GiB. Same Qwen3-0.6B revision, checkpoint 944, NF4/FP16, non-thinking catalog prompt, greedy decoding and cached system prefix. Torch 2.8.0+cu128, Transformers 4.57.6, PEFT 0.17.1, bitsandbytes 0.50.2. Model, adapter, prompt/template, dependency and source identities match the native reference. CPU/GPU image IDs match the previously audited smoke images; this run skipped rebuilding.

The gateway publishes only a random localhost port, and the authenticated worker runs on an internal network. Artifacts/reports mount read-only. One model request is admitted at a time, batch size 1. The raw parity sequence covers every validation ID exactly once and matches output text, input/output token counts, truncation flags and absence of inference errors.

## Complete HTTP workload

Seed 42 sampling with replacement: **484 supported + 16 oos requests**, identical at concurrency 1/4/8, no retries. Twenty warmups were excluded. Cold startup-to-readiness was **36.45 seconds**, warmup **14.53 seconds**. This is one workload repetition, not an independently replicated capacity estimate.

| Concurrency | Submitted | Completed | HTTP 503 busy | Completed/s | Completed p50 / p95 / p99 ms |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | **500** | 0 | **1.345** | **878.14 / 1,116.87 / 1,330.25** |
| 4 | 500 | 1 | 499 | 0.834 | 1,196.20 / 1,196.20 / 1,196.20 |
| 8 | 500 | 2 | 498 | 1.470 | 684.46 / 1,190.41 / 1,235.38 |

Serial counters: **97** actual gate short-circuits, **403** generation calls, **402** routes and **98** reviews. Concurrency 8 completed one gate rejection and one generation; its two-request throughput is not evidence of scalable service. All 997 busy failures remain in the concurrent outcomes. Fast failed requests must not be interpreted as successful latency. The harness checked returned intent/decision/reason live; the audit independently checks retained decision/reason and mismatch flags, as returned intents are not stored in the load outcomes.

The worker's startup snapshot reports peak PyTorch allocation **851,771,392 bytes**, reservation **1,400,897,536 bytes** and process RSS **1,608,986,624 bytes**. These were captured before the full workload and do not establish workload-wide peaks or GPU utilization. The earlier native benchmark has separate full-run GPU samples. Do not conflate its resource measurements with this Linux run.

## Quality, release outcome and cost

The [unchanged final test report](../milestone5-final-v1/README.md) remains the quality evidence: C supported macro-F1 **0.956794**, coverage **70.65%**, routing error **4.76%**, oos review recall **89.6%**. That last value fails the fixed 90% rule. The test ledger/report hashes still verify and the registry contains one frozen experiment. No new test inference, policy selection or activation occurred; no deployment pointer exists.

`cost.json` applies this container's measured serial time to the same clearly hypothetical scenario: dated us-east-1 g4dn.xlarge quote **$0.526/hour**, **100,000** monthly submissions, **730** billed hours and **$10/month** supporting allowance. Hosting remains **$393.98/month**, **$3.9398/1,000 submitted**. The local-rate projection is **20.66 active hours**, **709.34 idle hours**, or **$0.1087 active compute/1,000**. Training is separately **$6.1472** at the hypothetical rate, or **$0.06147/1,000** with one-month amortization. This is neither measured EC2 performance nor a cloud bill or savings claim.

## Audit and completion limits

The audit verified source/model/adapter/prompt/environment/Dockerfile identities, parity checksums and all 3,100 raw records, all 1,500 HTTP submissions, workload identity/mix, all failures, percentiles, throughput, decisions/reasons, metric deltas, and the unchanged final-test ledger/report. Images still match local Docker inspection, and the runner removed its temporary containers/networks. No model or unit-suite rerun was needed for this evidence-only update; the existing 153-test result remains the latest suite result.

`verification.json` records these checks. `evidence_sha256.json` covers the retained files before this README was added. `cost.json` was generated during the post-run audit; the original frozen release/report was not modified. Source command:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage gpu --full --skip-build --output artifacts/milestone5-container-gpu-full-v1
```

The Transformers container path is verified. Section 9's separate vLLM runtime experiment and hosted CI execution remain unverified. Docker/WSL2 now provides a Linux environment; native-Windows incompatibility alone is no longer a reason to call a vLLM trial impossible. A documentation support entry does not prove this exact adapter/hardware combination works. These remaining scope limits must be explicit in the Milestone 6 handoff, and the disqualified release stays inactive.
