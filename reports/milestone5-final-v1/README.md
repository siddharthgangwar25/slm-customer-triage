# Milestone 5 final benchmark and audit

The user completed the native Windows terminal workflow on 1 October 2026. Release `release-a0bcea381506` froze source commit `82ed8d754bab6c3c300a50e92b0ac294b4660877`, all three candidates, validation thresholds and serving evidence before the genuine test run. C was selected using validation coverage. All 5,500 official test requests (4,500 supported, 1,000 oos) have one result per candidate, with zero infrastructure failures. The audit reproduced metrics from saved predictions; it did not run inference or change thresholds.

**C improves supported-intent classification but is disqualified for automatic release.** It reviews 896/1,000 oos requests, below the fixed 900/1,000 target. The activation command was verified to reject this release without writing a pointer. A also misses its routing-error target; B remains review-only. This experiment does not establish an eligible automatic deployment. The existing baseline configuration has not been replaced.

| Test measure | Baseline A | Prompted B | Fine-tuned C |
| --- | ---: | ---: | ---: |
| Supported macro-F1 | 0.886628 | 0.278546 | **0.956794** |
| Supported accuracy | 88.76% | 23.62% | **95.58%** |
| Invalid outputs / 5,500 | 0 | 1,032 | 44 |
| Fixed-policy coverage | 60.05% | 0% | **70.65%** |
| Routed errors / routes | 198 / 3,303 | 0 / 0 | 185 / 3,886 |
| Routing error | 5.99% | Undefined | **4.76%** |
| Oos review recall | 92.7% | 100%, all review | **89.6%** |

C−A supported macro-F1: **+0.070166**, paired bootstrap 95% interval **[0.061720, 0.080052]**. C−B: **+0.678248**, interval **[0.668722, 0.691655]** (1,000 resamples, seed 42). C's routing-error Wilson interval is **4.13–5.48%**, and oos-recall interval **87.55–91.34%**. The interval overlapping 90% does not waive the prespecified point-estimate rule.

C's raw oos correctness is 470/1,000 versus B's 718/1,000; its gate provides much of the eventual review behavior. Of C's 44 invalid outputs, 42 are oos and two supported. B has 163 oos and 869 supported invalid outputs. Validation contains only 100 oos examples, and its request mix differs from the test set, so coverage changes are not directly attributable to model drift. Public pretraining contamination, source duplicates and one training seed remain limitations. Generated coverage curves are retrospective diagnostics, not threshold-selection authorization.

## Measured serving conditions

Windows, Ryzen 5 5600H, GTX 1650 4 GiB; Qwen3-0.6B checkpoint 944, NF4/FP16, greedy decoding, cached system prefix. Separate Transformers worker and CPU gateway, one active request, batch size 1, no retries. This is the unchanged evaluated adapter representation; Docker and vLLM were not exercised.

All **3,100 validation** worker outputs, input/output token counts and truncation flags matched the reference. The HTTP workload sampled **484 supported + 16 oos** validation requests with replacement, seed 42, repeated identically at each concurrency. Twenty warmups were excluded; startup-to-readiness was **52.41 seconds**, warmup **18.79 seconds**.

| Concurrency | Submitted | Completed | HTTP 503 busy | Completed/s | Completed p50 / p95 / p99 ms |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 500 | 0 | **1.043** | 1,130.36 / 1,442.35 / 1,722.14 |
| 4 | 500 | 1 | 499 | 0.716 | 1,393.71 / 1,393.71 / 1,393.71 |
| 8 | 500 | 1 | 499 | 0.724 | 1,378.04 / 1,378.04 / 1,378.04 |

All successful decisions matched the reference policy. Serial counters show **97** real gate short-circuits, **403** model calls, **402** routes and **98** reviews. Concurrent tests measure fail-fast overload, with only one completed request each; their percentiles are not capacity estimates. All failures and all-request percentiles remain in `load/`. The outcome file does not retain returned intent text; exact intent equality was checked live by the harness, while the post-run audit can independently recheck the saved decision/reason and mismatch flag.

Resource monitoring collected **2,059** whole-GPU samples across startup, parity and load: maximum used memory **2,598 MiB**, maximum utilization **97%**. These include desktop activity and are not load-only averages. The worker reported peak PyTorch allocation **851,771,392 bytes**, reservation **1,400,897,536 bytes**, and peak process RSS **2,501,709,824 bytes**. Separate raw final inference summed to A **32.35 seconds**, B **76.08 minutes**, C **100.86 minutes**, excluding load/prefill and report generation; these are not HTTP latency measurements.

## Cost scope

The retained 29 September AWS us-east-1 Linux g4dn.xlarge quote is **$0.526/hour**. At an assumed 100,000 monthly submitted requests, 730 billed hours and $10/month supporting-service allowance, hosting is **$393.98/month**, or **$3.9398/1,000 submitted**. Applying the local serial rate hypothetically gives 26.63 active and 703.37 idle hours, and $0.1401 active compute per 1,000. Training at that hypothetical hourly rate is $6.1472 separately, or $0.06147/1,000 if amortized over one month. No cloud runtime, energy bill or cost saving was measured; no paid resource was used.

## Retained evidence and reproducibility

- `release.json`, `test_use.json`, `run.json`: frozen identity and completed test-use record; local absolute paths identify the original run.
- `final_report.json`, `report.md`: final raw/fixed-policy results, uncertainty intervals and operating/cost evidence.
- `baseline-report/`, `prompted-report/`, `finetuned-report/`: every prediction, manifests, gate references, per-class results, confusion matrices and explicitly diagnostic plots.
- `parity/`, `load/`, resource and cost JSON files: full serving outcomes and measurements.
- `verification.json`, `evidence_sha256.json`: successful audit and retained-file checksums (this explanatory README was added after the audit).

Audit command, using a fresh output directory for another audit:

```powershell
.\.venv\Scripts\python.exe scripts/verify_milestone5.py --output artifacts/milestone5-acceptance-v1
```

This reads the completed experiment and never generates new predictions. Keep the original test-use registry and frozen artifacts. Do not rerun or retune the final test experiment. Container build/health/parity and the vLLM alternative remain unverified; complete those infrastructure checks before claiming full Milestone 5 acceptance. They cannot change this release's test disqualification.
