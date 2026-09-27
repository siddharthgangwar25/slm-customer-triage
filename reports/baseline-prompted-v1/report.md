# Baseline versus prompted model — validation

Genuine CLINC150 validation results. Test predictions remain unused.

Samples: 3100 (3000 supported, 100 oos).

| Candidate | Raw macro-F1 | Invalid rate | Coverage | Routing error | Oos recall | Targets met |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.882634 | 0.000000 | 0.684194 | 0.047619047619047616 | 0.9 | True |
| prompted | 0.801038 | 0.022581 | 0.000000 | None | 1.0 | False |

Each candidate uses its own validation-selected threshold against the same frozen baseline gate. Raw predictions are ungated. Zero-route error is null. All failures remain in accounting. Wilson intervals and exact model identities, precision, runtime, and memory are recorded in comparison.json. No independent test performance, deployment choice, cost, or production-quality claim is made.

Paired supported macro-F1 difference 95% bootstrap interval: [-0.09983807967666285, -0.06706337324023005] (1,000 resamples, seed 42).
