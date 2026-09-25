# Experiment journal

## 2026-09-25 — baseline-c1-v1

Hypothesis: local word patterns will provide a useful CPU baseline for supported intents, while closely related intents and unsupported requests will expose the need for a review gate.

Executed one configuration: TF-IDF word unigrams/bigrams and logistic regression, C=1.0, seed 42. Trained on 15,000 supported examples, converged in 21 iterations, and matched saved/reloaded probabilities exactly. Validation macro-F1 was 0.882634; 350 of 3,000 supported requests were misclassified. All 100 oos requests received a supported label before gating, as expected for this classifier.

Reviewing 40 errors suggests that lexical overlap explains several failures: credit-score improvement versus score lookup, language switching versus speaking speed, and changing a user's name versus retrieving it. Some examples are ambiguous under the benchmark labels. The sampled errors do not establish how common each cause is across all errors.

Decision: retain this reproducible baseline and implement the specified validation-only threshold selector next. Do not interpret maximum class probability as calibrated correctness. Keep raw classification quality separate from the eventual coverage/error operating point. No test predictions or hyperparameter search were performed.
