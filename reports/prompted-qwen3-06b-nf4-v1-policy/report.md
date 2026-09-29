# Validation routing policy

Genuine CLINC150 validation evidence; no test evaluation.

Status: **targets_unmet**. Automatic routing: **False**.
Policy: `policy-v1-403b6fb85cb3`; model: `prompted-qwen3-06b-nf4-v1-53f85eece67c`.

Samples: 3100 (3000 supported, 100 oos).

| Measure | Selected validation result |
| --- | ---: |
| Gate threshold (equal passes) | 1.0000000000000002 |
| Coverage | 0.000000% (0/3100) |
| Routing error | undefined (zero routes) |
| Routing error Wilson 95% | None |
| Oos recall | 1.0 |
| Oos recall Wilson 95% | [0.9630065017930143, 1.0] |
| Supported review rate | 1.0 |
| Infrastructure failures after gating | 0 |

Swept 3068 thresholds: every distinct observed score, zero, and an explicit all-review endpoint. Maximize coverage subject to error <=5%, oos recall >=90%, and coverage >=20%; break ties by lower error then higher threshold.

If targets are unmet, automatic routing is disabled; error is null and coverage is zero. The service uses low_gate_score for disabled-routing reviews to retain the specified reason vocabulary. It still requires a healthy model and policy bundle.

These are validation-selected point estimates, not test guarantees. Wilson intervals describe binomial uncertainty and do not correct for threshold selection. Only 100 official validation oos examples constrain recall. The frozen baseline gate is not an independent estimate of an LLM's uncertainty. No production or release claim is made.
