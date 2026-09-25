# Baseline validation report

Measured official CLINC150 validation run; test predictions were not generated.

Model: `baseline-c1-v1-551714a82382`. Split: `val`. Samples: 3100 (3000 supported, 100 oos).

| Metric | Measured value |
| --- | ---: |
| Raw supported macro-F1 (150 fixed labels) | 0.882634 |
| Raw supported accuracy | 0.883333 |
| Invalid output rate | 0.000000 |
| Infrastructure failure rate | 0.000000 |
| Serial model p95 latency (ms) | 0.784 |

Timing includes vectorization and classification per request on this CPU. It is not API latency, a load test, or a cost estimate.

The classifier fits only supported training requests. It always predicts a supported intent, so all oos examples are wrong before gating. Gate scores are uncalibrated ranking signals. The coverage/error plot is exploratory; no deployment threshold or target achievement is claimed in Milestone 1.

Supported classification errors: 350. Review queue: 40 examples in `error_review_samples.json`. Manual observations are recorded separately in `error_analysis.md`.

The per-class report uses supported samples only and includes every catalog label. The confusion matrix includes oos and invalid outputs. Zero-route error is null. Raw failures remain in the denominator.

Provenance and environment are in `prediction_manifest.json`; data integrity and duplicate findings are in the data manifest. Public benchmark performance is not evidence of production customer quality.
