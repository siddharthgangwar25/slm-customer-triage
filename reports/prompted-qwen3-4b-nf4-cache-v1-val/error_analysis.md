# Prompted validation error review

These are assistant-authored observations of the 40 records in `error_review_samples.json`: the first 30 supported errors in official order, then the first 10 oos requests. This selection favors early intent classes and is not a representative estimate of error-category frequencies. The frozen prompt was not changed in response to this review.

| Queue rows | Gold intent | Observation |
| --- | --- | --- |
| 1-2, 5-6 | translate | JSON is well formed but `translation` is not a catalog label. No synonym repair is applied. |
| 3 | translate | Asking how to say a greeting in another language is confused with changing the assistant's language. |
| 4, 7 | translate | The word goodbye dominates the actual request to translate it. |
| 8-9 | find_phone | Indirect requests about a missing phone are rejected as oos. |
| 10 | find_phone | The invented label `location` fails the catalog contract. |
| 11-12 | find_phone | Locating a phone is confused with sharing a location. |
| 13-15 | travel_alert | Destination safety questions are treated as travel suggestions. |
| 16-17 | pto_request | Asking to submit leave is confused with checking request status. |
| 18 | pto_request | A vacation-request procedure question is rejected as oos. |
| 19-20 | pto_request | Vacation wording triggers travel suggestions rather than a workplace leave request. |
| 21-22 | improve_credit_score | Advice about improving/protecting a score is confused with retrieving the score. |
| 23 | change_language | `language_change` is an invalid label despite expressing a related concept. |
| 24 | payday | `due_date` is outside the catalog. |
| 25-26 | payday | Paycheck timing is confused with ordering checks or a bill deadline. |
| 27, 29 | replacement_card_duration | Arrival timing is confused with credit-limit intents. |
| 28, 30 | replacement_card_duration | The model predicts obtaining a new card rather than how long replacement takes. |
| 31 | oos | A bank-balance warning is forced into the alarm class. |
| 32 | oos | The fragment about a Broadway show is forced into restaurant suggestions. |
| 33 | oos | Sports-record question is correctly rejected. |
| 34 | oos | Geometry is confused with unit conversion. |
| 35 | oos | Household inventory is confused with a recipe ingredients list. |
| 36 | oos | A historical person's name is confused with asking the assistant's name. |
| 37 | oos | Backyard activity request is correctly rejected. |
| 38 | oos | Credit repayment arithmetic is confused with changing a credit limit. |
| 39-40 | oos | Notary-service and current-news requests are correctly rejected. |

Across the full 3,100 requests, there are 70 invalid outputs: 65 JSON objects with unknown labels and five bare `oos` strings that fail the required JSON schema. There are no truncated generations or infrastructure failures. `failure_analysis.json` retains the full invalid-output frequency table. The model also predicts valid `oos` for 92 supported requests and 53 of the 100 gold-oos requests.

The [paired comparison](../baseline-prompted-v1/comparison.json) shows 206 supported requests that Qwen fixes relative to the baseline, but 479 where the baseline is correct and Qwen is wrong. For example, Qwen correctly interprets "how do i ask about the weather in chinese" as `translate`, where the baseline chooses `weather`; it recognizes "what is our purpose" as `meaning_of_life`, where the baseline chooses `calculator`. Semantic improvements therefore exist, but do not outweigh regressions in this experiment.

The prompted macro-F1 is 0.801038 versus 0.882634 for the baseline. The paired bootstrap difference interval is entirely negative. Under the shared gate, no threshold meets error <=5%, oos recall >=90%, and coverage >=20%. The minimum error among thresholds satisfying the latter two conditions is 61/667 = 9.1454%, at 21.5161% coverage and 98% oos recall. The frozen policy consequently reviews every request; its 100% oos review recall is a consequence of disabled routing, not evidence of perfect oos classification.

These findings motivate testing whether completion-only fine-tuning improves catalog adherence and neighboring-intent distinctions in Milestone 4. They do not establish that fine-tuning will help or fit the local GPU. No test predictions or additional prompt search were performed.
