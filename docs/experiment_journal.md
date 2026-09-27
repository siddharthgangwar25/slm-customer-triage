# Experiment journal

## 2026-09-27 — prompted-qwen3-4b-nf4-cache-v1 (running)

Hypothesis: a prompted language model may resolve semantic intent distinctions missed by the lexical baseline, while its ability to follow the exact 150-label output contract may limit that benefit. Keep the same baseline gate so the comparison measures candidate labels under the specified policy, rather than introducing a new uncertainty estimator.

The default 4B Qwen model fits the local 4 GiB GPU with NF4 double quantization and FP16 compute. All validation prompts fit without truncation and the tokenizer explicitly disables thinking. A ten-request uncached smoke exposed two invented `translation` labels instead of the supported `translate` label. They remain failures; the prompt is not repaired around these observations.

Full-prompt inference took about 18 seconds/request. Reusing the 627-token fixed system prefix reduced six predetermined probe cases to 1.38–2.16 seconds, with exactly matching output tokens/text accounting. Each request gets a separate cache copy. The final experiment records caching as part of its identity. No smaller model, test predictions, few-shot examples, or paid service was used.

Decision: run the entire 3,100-request validation split before drawing a quality conclusion. Record raw failures, compare supported macro-F1 and selected routing policies, and retain all paired predictions for inspection. Successful inference on this GPU does not establish that fine-tuning will fit; Milestone 4 needs its own memory and training-budget check.

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
