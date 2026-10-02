# Validation classification report

Measured official CLINC150 validation run; test predictions were not generated.

Model: `milestone6-cpu-reproduction-21a2ab3db7cd`. Split: `val`. Samples: 3100 (3000 supported, 100 oos).

| Metric | Measured value |
| --- | ---: |
| Raw supported macro-F1 (150 fixed labels) | 0.882634 |
| Raw supported accuracy | 0.883333 |
| Invalid output rate | 0.000000 |
| Infrastructure failure rate | 0.000000 |
| Serial model p95 latency (ms) | 0.864 |

Timing scope: serial per-record vectorization + predict_proba + argmax; no API. This is not API latency, a load test, or a cost estimate.

Raw classification runs without gating. The gate scores come from the frozen baseline and are uncalibrated ranking signals. This curve is exploratory; exact policy selection is a separate command. Malformed, unknown, truncated, and failed outputs remain in classification accounting. The baseline itself cannot predict oos; a prompted model can.

Supported classification errors: 350. Review queue: 40 examples in `error_review_samples.json`. Manual observations are recorded separately in `error_analysis.md`.

The per-class report uses supported samples only and includes every catalog label. The confusion matrix includes oos and invalid outputs. Zero-route error is null. Raw failures remain in the denominator.

Provenance and environment are in `prediction_manifest.json`; data integrity and duplicate findings are in the data manifest. Public benchmark performance is not evidence of production customer quality.
