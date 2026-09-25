# Validation error review

Reviewed on 2026-09-25 from the genuine saved `baseline-c1-v1-551714a82382` predictions. There are **350 supported classification errors out of 3,000 supported validation requests**, plus **100 forced supported predictions on 100 oos requests**. No prediction or infrastructure failures occurred.

The 40 inspected examples below are the first 30 supported errors and the first 10 oos requests in official order. Full text, full sample IDs, labels, and gate scores are in [error_review_samples.json](error_review_samples.json). The table uses ID prefixes. Selection is deterministic but class-order biased; these observations are hypotheses about failure causes, not population estimates or independently reviewed annotations.

| # | Sample ID prefix | Gold → predicted | Observation |
| ---: | --- | --- | --- |
| 1 | 3460323f4489 | translate → weather | Asking how to ask about weather in Chinese: topic words outweigh the translation task. |
| 2 | 71280141c3a7 | definition → calculator | The uncommon word “septuagenarian” needs lexical meaning; generic question wording is insufficient. |
| 3 | 66a7d4cfcc3b | meaning_of_life → calculator | “Our purpose” is an indirect paraphrase of the intent. |
| 4 | 789d3708338c | meaning_of_life → user_name | “On earth for” requires semantic interpretation beyond local wording. |
| 5 | 076c43cd4073 | meaning_of_life → user_name | “Higher calling” is an idiom; the chosen class is unrelated. |
| 6 | 851f01a8ee3e | meaning_of_life → account_blocked | “Why are humans on earth” receives a weak, unrelated label. |
| 7 | 88301844bb68 | insurance_change → oil_change_how | Shared “change” wording outweighs “policy”; domain-specific meaning is missed. |
| 8 | 936cefe368c7 | find_phone → sync_device | “What did I do with my phone” is an indirect lost-phone request. |
| 9 | a2aa817c4fea | travel_alert → weather | Travel danger is confused with a common destination-related topic. |
| 10 | 7055a8633424 | improve_credit_score → credit_score | “Help my credit score” indicates improvement rather than retrieval; error has gate score 0.5151. |
| 11 | 615f4658e3ca | improve_credit_score → credit_score | Asking whether a card helps the score is confused with score lookup. |
| 12 | c59c67bbbfa5 | improve_credit_score → credit_score | The desired plan is missed despite explicit “help” wording. |
| 13 | 56fd2a187fc9 | change_language → change_speed | “Speak Arabic” is interpreted as a speech-setting request of the wrong type. |
| 14 | c9a3ce5902cb | change_language → change_speed | Same confusion for Navajo; a moderately large score does not guarantee correctness. |
| 15 | 0f9a2e029028 | change_language → change_speed | Same confusion for German. |
| 16 | c816f9f6b548 | change_language → change_speed | Same confusion for Mandarin. |
| 17 | 55f56a4ae195 | payday → next_holiday | Repeated “day” wording suggests a calendar event but loses payment context. |
| 18 | 32214632b407 | payday → date | Generic day/date wording overwhelms “payment comes in.” |
| 19 | 933970a4daf2 | payday → pto_used | Payment timing is confused with a different employment topic. |
| 20 | 011aec47f73d | payday → last_maintenance | “Last check date” is ambiguous; “check” can refer to an inspection or payment. |
| 21 | a979bf2c1d59 | payday → last_maintenance | “Get my check last” is another payroll/inspection lexical confusion. |
| 22 | 4e06fdc2791e | payday → last_maintenance | Elapsed-time wording is associated with maintenance rather than being paid. |
| 23 | 29525521e6fe | payday → income | “What is my pay” plausibly asks about amount; the gold timing label is ambiguous. |
| 24 | a4ab41dbf24e | payday → bill_balance | “How much” asks about an amount despite the gold payday label. |
| 25 | 72e99056d448 | replacement_card_duration → cook_time | “Updated cc” is indirect/abbreviated card wording; generic duration cues dominate. |
| 26 | 695e4367271c | time → translate | “What's the clock say” is a colloquial time request. |
| 27 | 703e670096e2 | change_user_name → change_ai_name | The model recognizes renaming but confuses the user with the assistant. |
| 28 | 54a017fc7b92 | change_user_name → user_name | “From now on” implies updating a name, not retrieving it. |
| 29 | ed68fc2b20e8 | change_user_name → user_name | Similar update-versus-retrieval confusion with “my name is Tom.” |
| 30 | 1c48064063aa | change_user_name → user_name | “My name is Jason” is context-dependent; changing versus stating a name is ambiguous. |
| 31 | 85858b6f0a70 | oos → account_blocked | Low-balance alerts are unsupported although banking vocabulary is familiar. |
| 32 | e0b4f70103fa | oos → transactions | A Broadway-show fragment is unsupported; the classifier must still choose a label. |
| 33 | 381b1d6c740d | oos → who_made_you | Sports records are outside the catalog; “who” wording is not task evidence. |
| 34 | cb00b37de4fe | oos → recipe | An area-of-circle question is out of scope under the official label. |
| 35 | 67cf510ccf18 | oos → pto_balance | “How many ... on hand” resembles balance questions but asks about onions. |
| 36 | aff3a230e092 | oos → what_song | Historical trivia is unsupported; a named-entity question pattern is insufficient. |
| 37 | 3c3b2bab7379 | oos → transactions | “Recent activity” in a backyard is confused with financial activity. |
| 38 | 7788c3de47a6 | oos → distance | A credit repayment projection is unsupported despite familiar finance vocabulary. |
| 39 | f39b7c1f07b6 | oos → travel_notification | Bank notary availability is outside the supported banking tasks. |
| 40 | 4c932e2d271e | oos → where_are_you_from | Current news requests are outside the catalog. |

The review supports three next checks: measure the coverage/error tradeoff using the prescribed validation selector; inspect high-score mistakes because a gate cannot remove all semantic confusions; and compare the later SLM's behavior on these same saved validation IDs. A change after this review must be a separately identified experiment.

The official duplicate audit found five cross-split groups, four with conflicting labels, including two train/test groups. No test text was inspected for error analysis, and no records were removed. [Duplicate audit](duplicates.json) and [data manifest](data_manifest.json) preserve the integrity findings.

Data excerpts and saved validation requests are from CLINC150, Larson et al. (2019), [official source](https://github.com/clinc/oos-eval), under [CC BY 3.0](CLINC_LICENSE.txt). Text is preserved; analysis and prediction metadata are project additions.
