# Experiment journal

## 2026-10-02 - vLLM smoke audit

Hypothesis: the unchanged paired Qwen3-0.6B base and LoRA adapter can execute under a separate vLLM NF4/FP16 Linux runtime on the 4 GiB GTX 1650.

The 1 October smoke completed after the chat was interrupted. Twelve requests completed with no inference failures. C matched all six reference generations; B matched five. One B answer changed from invalid catalog label `question` to `oos`. Two additional reported B differences were an accounting bug: comparing parser errors in reference predictions with worker generation errors. A CPU-only audit corrected the comparison, verified original evidence and metrics, and retained both the original and corrected results.

The runtime is feasible for these examples, but full validation and warmed serving performance remain unknown. vLLM selected FlexAttention; chunked prefill was enabled in the runtime despite a disabled request flag. Whole-GPU snapshots around 3.7 GiB leave limited headroom and are not workload peaks. Next: the user's terminal run of 3,100 validation examples for each candidate. The original failed release requirement remains unchanged. [Evidence](../reports/vllm-smoke-v1/README.md).

## 2026-10-01 — full Linux GPU container validation

The full run resolved the smoke's main uncertainty: all 3,100 validation worker outputs and token counts match the native reference exactly. All 500 serial HTTP submissions completed with matching decisions, 97 gate short-circuits and 403 model calls. Completed throughput was 1.345/s and p95 1.117 seconds for this one seed-42 workload. Concurrency 4/8 completed only one/two requests and rejected 499/498 as busy. A numerically higher two-request throughput is not evidence of scaling.

Decision: accept the existing Transformers container implementation and preserve its measured limits. Leave the final-test policy and release rejection unchanged. No further model run is needed to demonstrate this path. Section 9's separate vLLM runtime experiment remains unexecuted; Docker/WSL2 now makes a Linux trial possible in principle, but documentation support alone is not proof of adapter/hardware compatibility. Explicitly resolve or defer that scope before claiming full specification compliance.

## 2026-10-01 — GPU container smoke

Hypothesis: the unchanged adapter representation can reproduce native reference outputs under Linux/WSL2. The user built both images and ran the container smoke. All six raw/token comparisons matched; all 24 serial requests completed with matching decisions, six actual gate rejections and 18 model calls. The 1.159-second p95 is a small-workload measurement, not a confirmed performance improvement. Concurrency 4/8 each rejected 23 of 24 requests as busy.

Decision: proceed to full validation-only container parity/load using the built images. Preserve image/source identities and every failure. No new test experiment, policy adjustment or release activation is authorized by this smoke; the completed release remains disqualified. Full parity and the vLLM alternative are still unverified.

## 2026-10-01 — frozen Milestone 5 result and release rejection

The user completed all 3,100 native worker parity comparisons, the 500-request HTTP workloads at concurrency 1/4/8, and the frozen 5,500-request test evaluation for A/B/C. A separate CPU audit verified source/artifact identity, canonical records, parsing, fixed-threshold metrics, bootstrap intervals, operating counts and cost arithmetic without generating new predictions.

C improves supported test macro-F1 to 0.956794 (A 0.886628, B 0.278546). It routes 70.65% with 4.76% error, but reviews only 896/1,000 oos requests, below the prespecified 90% minimum. Its raw oos correctness is only 470/1,000, versus B's 718. Decision: retain this negative release outcome and reject activation; do not change the threshold or substitute another candidate after seeing test results. The recall interval overlapping the target is not a reason to change the acceptance rule.

Serial serving completes all 500 requests at 1.043/s, with 97 actual gate short-circuits and 1.442-second p95. At concurrency 4/8, the single-flight service rejects 499 of 500 requests at each level. This establishes limited local capacity and explicit overload behavior, not concurrent production readiness. The cost scenario is hypothetical, and Docker/vLLM checks remain unverified. Preserve the complete evidence and test-use registry; any future model research must disclose this exposure and use independent evaluation.

## 2026-10-01 — Milestone 5 serving implementation and terminal handoff

Hypothesis: moving the evaluated adapter into an authenticated worker can preserve its predictions while allowing the CPU gateway to reject low-gate requests before GPU generation. The 29 September native smoke matched all six reference outputs and token counts. All 24 serial HTTP requests completed with matching decisions; six skipped generation and 18 called the model. This verifies the small integration probe, not complete validation parity.

The single-flight service rejected 23 of 24 requests at each concurrency 4 and 8. Those failures are retained, and completed throughput/latency are reported separately so fast failures do not imply better capacity. Docker and the vLLM alternative remain unverified; the measured backend is Transformers with the existing quantized adapter.

Decision: keep the model, prompt, decoder and validation thresholds unchanged. Hand the longer parity/load/frozen-test workflow to the user as requested. Require full serving evidence and committed source before freezing, record test use before reading test records, and never retune from final results. Milestone 5 acceptance remains pending. Cost figures are explicitly hypothetical hosting scenarios based on a dated regional quote, not measured cloud bills. No paid resources or real test predictions were used during implementation.

## 2026-09-29 — finetuned-qwen3-06b-qlora-v1 (complete)

Hypothesis: completion-only supervision will improve exact catalog-label adherence and supported-intent classification. The approved smaller-model substitution requires both a new 0.6B prompted control and a 0.6B adapter experiment; comparing this adapter directly to the old 4B run would confound model size and fine tuning.

The user completed one epoch on 15,100 training examples in 11.69 hours on the GTX 1650. Ten preflight masks, finite training gradients, 944 updates, complete checkpoint hashes and zero-difference adapter reload were verified. A fresh process evaluated checkpoint 944 on all 3,100 validation requests. No configuration or prompt was changed after seeing these results; this remains one configuration and one seed.

Supported macro-F1: A 0.882634, new B 0.272377, C 0.966217. C fixed 2,227 supported B errors and introduced three regressions. Most of B's 654 invalid outputs were invented labels, so improved taxonomy adherence contributes substantially. C's remaining errors include neighboring intent boundaries, ambiguous requests and occasional unrelated labels. Forty category-stratified examples were inspected and annotated; this is not a representative random sample or an independent human annotation study.

The important tradeoff is raw oos correctness: B 67/100, C 55/100. C frequently chooses a related supported intent for unsupported requests. Its validation-selected shared gate nevertheless reviews 90/100 oos requests while routing 2,491/3,100 requests with 56 errors (80.35% coverage, 2.25% routing error). These point estimates satisfy the original constraints, but do not establish open-set reliability or future service performance.

Decision: close Milestone 4 with the measured improvement and its limitations. Retain the original 4B negative result. C is worth evaluating in Milestone 5's serving/release workflow; A remains the current API adapter. No additional seed or adapter configuration was launched because the current question is resolved sufficiently to proceed to serving measurements. Keep the test split unused until release freeze. Full comparison, training audit and error review are linked in implementation status.

## 2026-09-27 — prompted-qwen3-4b-nf4-cache-v1 (complete)

Hypothesis: a prompted language model may resolve semantic intent distinctions missed by the lexical baseline, while its ability to follow the exact 150-label output contract may limit that benefit. Keep the same baseline gate so the comparison measures candidate labels under the specified policy, rather than introducing a new uncertainty estimator.

The default 4B Qwen model fits the local 4 GiB GPU with NF4 double quantization and FP16 compute. All validation prompts fit without truncation and the tokenizer explicitly disables thinking. A ten-request uncached smoke exposed two invented `translation` labels instead of the supported `translate` label. They remain failures; the prompt is not repaired around these observations.

Full-prompt inference took about 18 seconds/request. Reusing the 627-token fixed system prefix reduced six predetermined probe cases to 1.38–2.16 seconds, with exactly matching output tokens/text accounting. Each request gets a separate cache copy. The final experiment records caching as part of its identity. No smaller model, test predictions, few-shot examples, or paid service was used.

The full 3,100-request validation run completed after a user-requested pause at 1,118 requests and terminal resumption. All outcome IDs, hashes and frozen inputs were verified. Qwen's supported macro-F1 was 0.801038 versus 0.882634 for the baseline; paired difference -0.081596, bootstrap 95% interval [-0.099838, -0.067063]. It fixed 206 supported baseline mistakes but introduced 479 errors on requests the baseline classified correctly. There were 65 unknown-label JSON outputs, five bare `oos` strings, and no truncation or infrastructure failures.

The prompted candidate could not meet the original routing constraints under any shared-gate threshold. Its frozen policy disables automatic routing. Reviewing everything gives 100% oos review recall but zero coverage, not perfect classification. The baseline remains the better measured A/B candidate. Summed saved-request inference took 80.68 minutes; cached per-request p95 was 2.053 seconds on the local GTX 1650. These are offline measurements, not API capacity or cost evidence.

Decision: preserve this negative result and the fixed zero-shot prompt. Milestone 4 can test whether completion-only fine-tuning improves exact-label adherence and semantic distinctions, using the same pinned base model. Successful inference does not establish that fine-tuning will fit; first perform its memory and training-budget check. Keep test predictions unused. Full evidence and reviewed examples are linked from the implementation status.

## 2026-09-25 — baseline-c1-v1

Hypothesis: local word patterns will provide a useful CPU baseline for supported intents, while closely related intents and unsupported requests will expose the need for a review gate.

Executed one configuration: TF-IDF word unigrams/bigrams and logistic regression, C=1.0, seed 42. Trained on 15,000 supported examples, converged in 21 iterations, and matched saved/reloaded probabilities exactly. Validation macro-F1 was 0.882634; 350 of 3,000 supported requests were misclassified. All 100 oos requests received a supported label before gating, as expected for this classifier.

Reviewing 40 errors suggests that lexical overlap explains several failures: credit-score improvement versus score lookup, language switching versus speaking speed, and changing a user's name versus retrieving it. Some examples are ambiguous under the benchmark labels. The sampled errors do not establish how common each cause is across all errors.

Decision: retain this reproducible baseline and implement the specified validation-only threshold selector next. Do not interpret maximum class probability as calibrated correctness. Keep raw classification quality separate from the eventual coverage/error operating point. No test predictions or hyperparameter search were performed.

## 2026-09-25 — baseline policy and API

Hypothesis: the baseline's maximum class probability can rank enough uncertain requests for review to meet the proposed validation routing targets while retaining at least 20% coverage.

Swept 3,068 exact operating thresholds on the original 3,100 validation predictions. The selected threshold, 0.2088413160728636, routes 2,121 requests (68.4194% coverage), with 101 incorrect routes (4.7619% error), and reviews 90 of 100 oos examples. It reviews 889 of 3,000 supported requests (29.6333%). All proposed point-estimate constraints are met. Wilson intervals remain wide: routing error 3.93–5.75%, oos recall 82.56–94.48%. Selection on the same validation data is not an independent generalization test.

The actual API reproduced all 3,100 offline decisions exactly. Separate localhost HTTP probes confirmed both route and review responses and healthy endpoints. These checks establish implementation parity, not load capacity. Synthetic fixtures separately verified unmet-target review-only mode, unavailable models, timeouts, and malformed output.

Decision: retain this frozen baseline policy for local demonstration and move to Milestone 3's prompted model comparison. Keep the existing test set unused for prediction. The next model must retain raw ungated predictions and use a verified one-to-one join to this frozen gate; it must not claim independent LLM uncertainty.
