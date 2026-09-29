# Validation routing policy

Genuine CLINC150 validation evidence; no test evaluation.

Status: **targets_met**. Automatic routing: **True**.
Policy: `policy-v1-2fe7bafe54be`; model: `finetuned-qwen3-06b-qlora-v1-step944-bfe54e26195f`.

Samples: 3100 (3000 supported, 100 oos).

| Measure | Selected validation result |
| --- | ---: |
| Gate threshold (equal passes) | 0.11417805060646902 |
| Coverage | 80.354839% (2491/3100) |
| Routing error | 2.248093% |
| Routing error Wilson 95% | [0.017352947679543634, 0.029079445213050083] |
| Oos recall | 0.9 |
| Oos recall Wilson 95% | [0.8256343384950865, 0.9447708629393249] |
| Supported review rate | 0.173 |
| Infrastructure failures after gating | 0 |

Swept 3068 thresholds: every distinct observed score, zero, and an explicit all-review endpoint. Maximize coverage subject to error <=5%, oos recall >=90%, and coverage >=20%; break ties by lower error then higher threshold.

If targets are unmet, automatic routing is disabled; error is null and coverage is zero. The service uses low_gate_score for disabled-routing reviews to retain the specified reason vocabulary. It still requires a healthy model and policy bundle.

These are validation-selected point estimates, not test guarantees. Wilson intervals describe binomial uncertainty and do not correct for threshold selection. Only 100 official validation oos examples constrain recall. The frozen baseline gate is not an independent estimate of an LLM's uncertainty. No production or release claim is made.
