# Data card: CLINC150 full

Updated 2 October 2026. English, single-intent classification across 150 supported
intents and 10 domains. CLINC150 is not a hospitality dataset or a record of this
project's real customers.

## Origin and attribution

Larson et al. (2019), *An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction*, [original repository](https://github.com/clinc/oos-eval).
Pinned source commit: `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
The retained [CC BY 3.0 source license](../reports/baseline-c1-v1-val/CLINC_LICENSE.txt)
and [provenance manifest](../reports/baseline-c1-v1-val/data_manifest.json) identify
the upstream material. The project adds deterministic metadata and reports;
request text and official split membership are preserved.

| Source file | SHA256 |
| --- | --- |
| data_full.json | `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0` |
| domains.json | `b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a` |
| LICENSE | `e6bc9e9c474700b708f568bac9e5a8a9bcb2b1dad53442f5ba449fcb848b8e76` |

## Composition and processing

| Split | Supported | Out of scope | Total | Supported examples per intent |
| --- | ---: | ---: | ---: | ---: |
| Train | 15,000 | 100 | 15,100 | 100 |
| Validation | 3,000 | 100 | 3,100 | 20 |
| Test | 4,500 | 1,000 | 5,500 | 30 |

Canonical JSONL records contain `sample_id`, `text`, `label`, `domain`, `split`,
`source_index` and `dataset_version`. IDs derive from source version and original
membership/index. Only text enters the classifier; domain and true label are
metadata. The catalog excludes `oos` from supported labels. Preparation verifies
pinned bytes, required fields, known labels and counts, failing on corrupt inputs.

Unicode NFKC, case folding and whitespace collapse are used only to audit
duplicates; they do not rewrite stored requests. Five cross-split duplicate
groups exist, four with conflicting labels; no within-split groups were found.
These remain in the official benchmark. The manifest records frequencies and
text-length summaries; preparation writes `duplicates.json` with group details.

## Split discipline and exposure

TF-IDF and logistic regression fit only on 15,000 supported training records.
C trains on all 15,100 training records. Validation selected the prompt, paired
model/checkpoint and routing thresholds. The official test was evaluated once
for each frozen A/B/C candidate under the completed Milestone 5 ledger.
Its results are now exposed: do not call it untouched or tune on it. Earlier
preparation manifests describe their historical pre-evaluation state; the
completed ledger is the current record.

Milestone 6 reproduction repeats CPU training and validation in fresh outputs.
Preparation checks official test integrity but performs no test inference.
The separate vLLM experiment also used validation only.

## Uses and limitations

Use for intent-classification research and local demonstration. Results do not
establish real customer reliability, multilingual/multi-intent handling or
protected-group fairness. No demographic evaluation was available. Only 100
training oos requests cover a small part of unsupported traffic. Validation is
3.23% oos versus 18.18% in test, so coverage depends on input mix. Public benchmark
overlap with model pretraining cannot be excluded. Ambiguous labels and source
duplicates limit interpretation.

Retained prediction/error files contain attributed public benchmark text.
Service telemetry omits submitted text by default. Authored demo prompts are
illustrations, not independent held-out data. The optional 100+ request challenge
set has not been collected or independently reviewed.
