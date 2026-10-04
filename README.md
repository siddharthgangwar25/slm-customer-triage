# Customer request triage

A reproducible English intent-classification experiment and local FastAPI demo
using CLINC150. Compare a CPU TF-IDF baseline, a prompted Qwen3-0.6B model and a
paired QLoRA adapter. Routing recommends an intent or human review; it performs
no business action.

**Fine tuning improved supported test macro-F1 to 0.9568, but no candidate
qualified for automatic release.** C reviewed 89.6% of out-of-scope test requests,
below the fixed 90% requirement. The project preserves this negative release
result; no threshold was retuned or candidate substituted after test.

## Start here

- [Setup and fresh CPU reproduction](docs/setup.md)
- [Two-minute live demo](docs/demo.md)
- [Model/release card](docs/model_card.md) and [data card](docs/data_card.md)
- [Architecture](docs/architecture.md), [API](docs/api.md) and [error analysis](docs/error_analysis.md)
- [Local deployment](docs/deployment.md) and [developer handoff](docs/handoff.md)
- [Milestone 6 acceptance](reports/milestone6-handoff-v1/README.md), [status](docs/implementation_status.md) and [project specification](Customer_Request_Triage_Project_Spec.md)

## Reproduce without a GPU

From the project root, with Python 3.11 and uv installed:

```console
uv sync --locked
uv run --locked python scripts/reproduce_cpu.py --output artifacts/reproduction-v1
```

The runner creates a fresh locked CPU environment, prepares checksum-verified
sources, trains and reloads a new baseline, evaluates all 3,100 validation
requests, selects its policy, checks API parity and runs an authenticated live
HTTP demo. Each stage has logs; the demo stops its own server. No Docker, GPU,
model hub account or cloud account is needed. Initial installation/data download
requires internet. Use a new output directory for every run. See [setup](docs/setup.md)
for offline source reuse, individual commands, persistent serving and recovery.

On the implementation machine, the existing baseline can demonstrate immediately:

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py --output artifacts/demo-rehearsal-v1
```

No venv activation is needed with the explicit executable. A fresh checkout must
first reproduce or receive a trusted matching bundle/policy; ignored model files
are not included in Git. The demo is a local research path, not release approval.

## Frozen final test results

Each candidate has 5,500 retained results: 4,500 supported and 1,000 oos.

| Measure | A: CPU baseline | B: prompted 0.6B | C: fine-tuned 0.6B |
| --- | ---: | ---: | ---: |
| Supported macro-F1 | 0.886628 | 0.278546 | **0.956794** |
| Coverage | 60.05% | 0% | 70.65% |
| Routed errors / routes | 198 / 3,303 | 0 / 0 | 185 / 3,886 |
| Routing error | 5.99% | Undefined | 4.76% |
| Oos review recall | 92.7% | 100%, review all | **89.6%** |
| Invalid outputs | 0 | 1,032 | 44 |

A misses the 5% routing-error constraint. B has no eligible automatic policy and
reviews everything. C misses the oos-review constraint. C's supported macro-F1
advantage over A is 0.070166, paired bootstrap 95% interval [0.061720, 0.080052].
These are public benchmark results, not evidence of real customer performance.
[Final report and audit](reports/milestone5-final-v1/README.md).

The test was used once per frozen candidate. Do not rerun the completed final
workflow, reset its ledger, change thresholds or treat it as an untouched set.
Validation remains available for explicitly separate backend comparisons.

## Training and serving evidence

C has 596,049,920 base parameters and 10,092,544 trainable adapter parameters.
One epoch on 15,100 training requests took 11.69 hours on a GTX 1650 4 GiB.
Paired B/C inference uses the same pinned base, prompt and NF4/FP16 greedy decoder.
The original 4B prompted result remains a separate historical experiment.
[Training runbook](docs/finetuning.md), [adapter card](docs/finetuned_model_card.md),
[paired validation/error review](reports/three-way-qwen3-06b-v1/error_analysis.md).

The verified Transformers GPU container matches all 3,100 C validation outputs.
Its 500-request serial workload completes at 1.345/s with 1,116.87 ms p95 and
36.45-second startup. Concurrency 4/8 mostly fails busy, so no scaling claim is
made. [Full GPU evidence](reports/milestone5-container-gpu-full-v1/README.md).
CPU Docker also matches all 3,100 baseline validation decisions.
[CPU evidence](reports/milestone5-container-cpu-v1/README.md).

The separate vLLM experiment completed 6,200 validation generations but changed
95 B and five C outputs; it is not adopted and has no warmed load benchmark.
[vLLM results](reports/vllm-validation-v1/README.md).

Hosting cost is an assumed demand/dated-price scenario, not measured EC2
throughput, spending or savings. Cloud deployment is **omitted and unverified**.
The optional independently reviewed 100+ request challenge set is **deferred**;
[collection protocol](docs/challenge_protocol.md). Demo examples are not that set.

## Checks and artifacts

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
```

CPU tests use marked synthetic fixtures. Real benchmark and handoff acceptance
are retained separately. The configured Windows/Linux CI workflow is committed;
hosted execution remains unverified. [Latest check outcomes](docs/implementation_status.md).

Section 8's [manual GPU smoke workflow](docs/gpu_smoke.md) is separate from
normal CI. It checks a tiny synthetic CUDA training/save/reload/inference path
and requires an explicitly configured GPU runner. It does not rerun benchmarks.
See the [publication checklist](docs/publication.md) for remaining GitHub steps.

`src/triage/` holds data, model, policy, evaluation and API code; `configs/` and
`prompts/` hold frozen inputs; CPU/GPU locks pin dependencies. `reports/` retains
manifests, public benchmark predictions, error reviews, metrics and checksums.
Large weights, raw data, environments and secrets remain outside Git in ignored
local directories. Load only trusted joblib bundles. A recipient can reproduce CPU
weights, but needs the documented training procedure or a trusted artifact copy
for the original GPU adapter; no public adapter download is published.

The original project code's license is currently undecided; no root code license
has been selected. The dataset and upstream models retain their own licenses,
as documented in the data and model cards.

CLINC attribution: Larson et al. (2019), *An Evaluation Dataset for Intent
Classification and Out-of-Scope Prediction*, [original repository](https://github.com/clinc/oos-eval),
commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`.
[Retained CC BY 3.0 license](reports/baseline-c1-v1-val/CLINC_LICENSE.txt).
Canonicalization preserves request text and official membership. Five cross-split
duplicate groups (four conflicting) remain documented. Only 100 training oos
examples exist; public pretraining contamination cannot be ruled out.
