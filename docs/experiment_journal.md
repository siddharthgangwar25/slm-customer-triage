# Experiment journal

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
