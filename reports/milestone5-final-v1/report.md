# Frozen final test report

Release: release-a0bcea381506. Selected on validation: finetuned.

Test mix: 4500 supported / 1000 oos. All failures retained; no thresholds retuned.
Genuine frozen test run.

| Candidate | Macro-F1 | Coverage | Routing error | Oos recall | Invalid |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.886628 | 0.600545 | 0.05994550408719346 | 0.927 | 0 |
| prompted | 0.278546 | 0.000000 | None | 1.0 | 1032 |
| finetuned | 0.956794 | 0.706545 | 0.047606793618116316 | 0.896 | 44 |

Disable automatic release; no test-driven tuning

Wilson intervals and paired bootstrap intervals are in final_report.json. The test mix differs from validation; coverage is not directly comparable without accounting for that mix. Public pretraining contamination and source duplicates remain limitations.

## Serving workload

All failed submissions are counted. Concurrent busy rejections are not successful throughput.

| Concurrency | Submitted | Completed | Completed/s | All p95 ms | Completed p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 500 | 500 | 1.043 | 1442.352 | 1442.3523550009122 |
| 4 | 500 | 1 | 0.716 | 22.314 | 1393.7067000078969 |
| 8 | 500 | 1 | 0.724 | 24.486 | 1378.0421999981627 |

Cold start: 52.41409529998782 seconds.

Illustrative hosting: $3.9398/1,000 submitted at 100000 requests/month and 730 billed hours, including the explicitly assumed supporting-service allowance. This is not measured cloud cost.

Cloud runtime is unmeasured. Local GTX 1650 runtime at a T4 host price is an explicit what-if, not a cloud performance/cost claim. Supporting allowance is assumed; excludes taxes. No spending occurred.
