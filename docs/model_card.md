# Model and release card

Updated 2 October 2026. A local research/demo system for English single-intent
classification. It recommends `route` or `human_review`; it does not create
tickets, make reservations, move money or invoke business tools.

## Compared models

| Candidate | Implementation | Fitting data | Role |
| --- | --- | --- | --- |
| A | Word unigram/bigram TF-IDF + logistic regression, C=1, seed 42 | 15,000 supported training records | CPU reference and shared gate |
| B | Pinned Qwen3-0.6B, full-catalog prompt, NF4/FP16 | No project fine tuning | Paired prompted control |
| C | Same base/decoder plus rank-16 QLoRA, checkpoint 944 | 15,000 supported + 100 oos training records | Validation-selected candidate |

B/C use `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, 596,049,920 base parameters.
C has 10,092,544 trainable adapter parameters. The earlier 4B prompted experiment
is historical evidence, not C's paired control. The
[pinned upstream card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md)
declares Apache-2.0 for the base. Dataset attribution/license is in the
[data card](data_card.md). Adapter weights remain local; no separate adapter
distribution license or public weight hosting has been declared.

The shared gate is A's maximum predicted probability, which is uncalibrated.
It is a ranking rule, not LLM correctness confidence. Supported outputs must
pass the frozen gate and strict JSON/catalog validation. No output repair or
constrained decoding was used. [Adapter training details](finetuned_model_card.md).

## Completed frozen test: 5,500 examples per candidate

| Measure | A | B | C |
| --- | ---: | ---: | ---: |
| Supported macro-F1 | 0.886628 | 0.278546 | **0.956794** |
| Invalid outputs | 0 | 1,032 | 44 |
| Routed / submitted | 3,303 / 5,500 | 0 / 5,500 | 3,886 / 5,500 |
| Coverage | 60.05% | 0% | 70.65% |
| Routed errors | 198 | 0 | 185 |
| Routing error | 5.99% | Undefined | 4.76% |
| Oos review recall | 92.7% | 100%, review all | **89.6%** |

Test contains 4,500 supported + 1,000 oos requests. All candidates have zero
infrastructure failures. C exceeds A's macro-F1 by 0.070166, paired bootstrap
95% interval [0.061720, 0.080052]. The
[frozen final report](../reports/milestone5-final-v1/README.md) retains full
metrics, uncertainty, methodology and predictions.

**No candidate qualifies for automatic release.** C was selected on validation,
then missed the prespecified 90% test oos-review requirement: 896/1,000 instead
of at least 900. Its uncertainty interval does not waive that rule. A fails the
5% routing-error constraint; B has no eligible operating point and reviews all
requests. No post-test substitution or threshold change was made.
Release `release-a0bcea381506` remains inactive. Explicit `serve --bundle` runs
a research demo; it is not release approval.

## Serving and cost evidence

The verified GPU service uses a Transformers worker plus CPU gateway.
Linux Docker matched all 3,100 C validation generations. Its serial workload
completed 500/500 requests at **1.345 completed/s**, **1,116.87 ms p95**, with
**36.45 s startup** on a GTX 1650. The workload sampled 484 supported/16 oos
requests with replacement, seed 42, after 20 warmups. Concurrency 4/8 completed
only one/two requests, rejecting 499/498 as busy. This is not scalable throughput.

The $3.9398/1,000 submitted hosting scenario assumes 100,000 monthly requests,
730 billed hours at the dated $0.526/hour g4dn.xlarge quote plus $10 support.
It is not measured EC2 throughput, a bill or a saving. Training took 11.69 local
GPU hours; cloud/electricity cost was not measured.
[Container evidence](../reports/milestone5-container-gpu-full-v1/README.md).

The separate vLLM validation experiment completed 6,200 B/C requests but changed
95 B and five C outputs, including three C routed labels. It is not adopted;
no vLLM load-performance claim is made.
[Alternative-backend evidence](../reports/vllm-validation-v1/README.md).

## Limits and maintenance

One seed/epoch, a public benchmark, few training oos examples, ambiguous labels,
duplicates and validation selection limit generalization. Fine tuning improves
supported accuracy while worsening raw oos accuracy. Invalid/truncated outputs
remain parsing failures. There is no calibrated uncertainty, subgroup fairness
study or independent customer evaluation. See [error analysis](error_analysis.md).

Changes to weights, prompt, tokenizer, precision, backend or policy require a
new variant and validation, with prior test exposure disclosed. Keep the release
inactive; do not relax thresholds to improve its recorded test result. Accuracy
monitoring requires reviewed labels. A genuinely fresh, independently reviewed
dataset is needed before renewed release selection. [Handoff](handoff.md).
Cloud execution remains **unverified and omitted**.
