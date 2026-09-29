# Validation classification report

Measured official CLINC150 validation run; test predictions were not generated.

Model: `prompted-qwen3-06b-nf4-v1-53f85eece67c`. Split: `val`. Samples: 3100 (3000 supported, 100 oos).

| Metric | Measured value |
| --- | ---: |
| Raw supported macro-F1 (150 fixed labels) | 0.272377 |
| Raw supported accuracy | 0.224000 |
| Invalid output rate | 0.210968 |
| Infrastructure failure rate | 0.000000 |
| Serial model p95 latency (ms) | 1078.427 |

Timing scope: serial full-prompt tokenization and generation; excludes model loading, optional system-prefix prefill, and gate; prefix_cache=True. This is not API latency, a load test, or a cost estimate.

Raw classification runs without gating. The gate scores come from the frozen baseline and are uncalibrated ranking signals. This curve is exploratory; exact policy selection is a separate command. Malformed, unknown, truncated, and failed outputs remain in classification accounting. The baseline itself cannot predict oos; a prompted model can.

Supported classification errors: 2328. Review queue: 40 examples in `error_review_samples.json`. Manual observations are recorded separately in `error_analysis.md`.

The per-class report uses supported samples only and includes every catalog label. The confusion matrix includes oos and invalid outputs. Zero-route error is null. Raw failures remain in the denominator.

Provenance and environment are in `prediction_manifest.json`; data integrity and duplicate findings are in the data manifest. Public benchmark performance is not evidence of production customer quality.
