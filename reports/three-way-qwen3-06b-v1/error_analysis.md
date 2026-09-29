# What fine tuning changed

This review uses the same 3,100 genuine validation requests for A, B and C: 3,000 supported across 150 labels plus 100 oos. B and C use the same pinned Qwen3-0.6B base, full-catalog non-thinking prompt and NF4/FP16 greedy inference. C adds the one-epoch QLoRA adapter. The earlier 4B B result is historical context, not the paired control for C.

## Supported intents and output validity

C fixes **2,227** B supported errors and introduces **3** regressions. Both are correct on 669 supported requests and both are wrong on 101. Supported accuracy rises from **672/3,000 (22.40%)** to **2,896/3,000 (96.53%)**. Macro-F1 rises from 0.272377 to 0.966217; the paired difference is 0.693840, with bootstrap 95% interval [0.683129, 0.710616] (1,000 supported-sample resamples, seed 42). These intervals describe this selected validation result, not independent test generalization.

| B supported-error category | Before training | Corrected by C |
| --- | ---: | ---: |
| False oos rejection | 1,139 | 1,086 |
| Invalid output | 630 | 609 |
| Wrong supported label | 559 | 532 |

Across all 3,100 requests, B has 654 invalid outputs: **653 unknown labels and one invalid JSON output**. C has one invalid output, on an oos request: `{"intent":"mixing"}` for “can i mix antifreeze with water.” No output repair or failure exclusion was used. This is largely improved adherence to the provided taxonomy, alongside better classification; it is not merely a JSON syntax improvement.

Reviewed examples show B inventing labels such as `how_do_you_say_hello`, `lost_phone` and `how_safe_to_travel`; C uses `translate`, `find_phone` and `travel_alert`. C also recovers supported requests B rejected, including transfers, PTO requests and credit-score improvement. The supervised catalog examples are consistent with these improvements, but this single experiment does not isolate which training component caused them.

All three supported regressions were reviewed: “i need to make a reservation to long horn's, can i” changes from `accept_reservations` to `restaurant_reservation`; “there's something fishy on my card, report it” changes from `report_fraud` to `damaged_card`; and “how much gas does it take to get to jackson” changes from `gas` to `distance`. The first has a close taxonomy boundary; the other two lose an important distinction in the request.

C still makes **104 supported errors: 98 wrong supported labels and six false oos rejections**. Repeated confusions include `change_accent` → `account_blocked` (4), `payday` → `income` (3), `balance` → `credit_limit` (3), and `accept_reservations` → `restaurant_reservation` (3). Review also found short/ambiguous phrases and unusual source wording, such as “what was the last check date” and “can i get beer within my deposit account.” Official gold labels remain unchanged; potential annotation ambiguity is an observation, not grounds for removing errors.

Against A, C is uniquely correct on **284** supported requests and A is uniquely correct on **38**; both are correct on 2,612 and both wrong on 66. C's macro-F1 advantage over A is **0.083583**, paired bootstrap 95% interval **[0.073414, 0.096574]**. A remains much faster and the existing API continues using A.

## Out-of-scope tradeoff and the shared gate

Raw oos correctness **falls from 67/100 for B to 55/100 for C**: C fixes 16 raw oos errors and introduces 28 regressions. Of the 45 remaining C oos errors, 44 are valid supported labels and one is an unknown label. B's 33 raw oos errors comprise nine supported labels and 24 invalid outputs. Do not confuse raw oos prediction with the policy's human-review recall.

Examples of C overextending the taxonomy include an unsupported low-balance warning mapped to `account_blocked`, a credit-card payoff projection mapped to `apr`, pantry inventory mapped to `shopping_list`, and a power-steering-fluid question mapped to `oil_change_how`. Some requests resemble neighboring supported classes, but the reference scope still excludes them. Only 100 oos examples were available for training; broader open-set reliability has not been demonstrated.

At C's validation-selected shared-gate threshold **0.11417805060646902**, **2,491/3,100** requests route (80.3548%), with **56/2,491** errors (2.2481%) and **90/100** oos requests reviewed. Ten oos requests still route incorrectly. The other 46 routed errors are supported-label mistakes. The routing-error Wilson 95% interval is [1.7353%, 2.9079%]; oos review recall's interval is [82.5634%, 94.4771%]. Point estimates meet the original constraints, but threshold selection and the small oos sample limit certainty.

A routes 2,121 requests with 101 errors at its own selected threshold. C therefore routes **370 more requests** and has **45 fewer routed errors** on this validation set. Each candidate uses its own threshold against the same frozen gate. B meets no eligible threshold and reviews everything; its resulting 100% oos review recall does not indicate a useful raw classifier.

## Review method and limits

[error_review.json](error_review.json) retains **40 assistant-reviewed examples**, with full IDs, text, raw B/C outputs and observations: eight invalid-output repairs, eight recovered false oos rejections, all three supported regressions, 11 remaining supported errors, five oos regressions and five remaining oos errors. Within each group, selection prefers distinct gold classes (supported) or C outputs (oos), then official order. It is a deterministic category sample, not random or an independent annotation study. The eight invalid-output examples are stored under the selection key `format_repaired`; their actual issue is unknown labels, as each observation records.

All 3,100 paired outcomes and every changed prediction are retained separately. No hyperparameter or prompt was changed after this review. The only eligible epoch checkpoint is step 944; there was one configured epoch and one seed. Public benchmark pretraining contamination and the known source duplicates remain limitations. The results support advancing C to Milestone 5's serving/parity/load evaluation, not changing the deployed adapter or opening the test split now.
