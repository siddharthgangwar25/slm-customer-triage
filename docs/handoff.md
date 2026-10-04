# Developer handoff

The local project is reproducible and documented. Its research result is an
improved fine-tuned classifier with **no candidate approved for automatic
release**. Cloud deployment and independent challenge-set evaluation are omitted,
not silently counted as completed acceptance.

| Need | Start here |
| --- | --- |
| Reproduce CPU training, validation and service | [Setup](setup.md), `scripts/reproduce_cpu.py` |
| Present the project | [Two-minute demo](demo.md), `scripts/run_demo.py` |
| Understand dataset/exposure | [Data card](data_card.md) |
| Understand results/release limits | [Model card](model_card.md), [adapter card](finetuned_model_card.md) |
| Understand implementation | [Architecture](architecture.md), [API](api.md) |
| Inspect changed errors | [Error analysis](error_analysis.md) |
| Run local containers or plan hosting | [Deployment](deployment.md), [AWS plan](../deployment/aws_runbook.md) |
| Inspect exact execution evidence | [Status](implementation_status.md), `reports/` |

## What ships and what does not

Git contains code, pinned CPU/GPU locks, configuration, prompts, source provenance,
reports/predictions, manifests and hashes. Large weights, raw data, installed
environments and secrets are ignored. CPU bundles are reproducible without the
author's machine. A historical GPU release needs trusted copies of its exact
artifacts or the documented training reproduction; no adapter hosting endpoint
has been published. Check hashes and dependency versions before loading.

`artifacts/milestone5/release/test_use.json` is the completed local test ledger;
its retained report copy documents exposure. Do not reset it, delete the registry,
repeat `benchmark final` or silently substitute vLLM. Reproduce validation in new
directories. The frozen service/model source remains unchanged by the handoff.

## Dependency maintenance

CPU dependencies cover the classifier (scikit-learn, NumPy, SciPy, joblib,
threadpoolctl), reports (Matplotlib, PyYAML) and API (FastAPI, Pydantic, Uvicorn).
SciPy is used through scikit-learn and its version is recorded with the model.
Development tools and `httpx2` support linting, typing and API tests. GPU locks
also retain Accelerate and bitsandbytes, loaded through Transformers/PEFT, and
the isolated vLLM compatibility pins. An absent direct import alone does not
make these unused dependencies.

Keep CPU and GPU environments separate. Upgrade dependencies in a new evaluated
variant; the existing lock hashes are part of the frozen experiment identity.

## Acceptance boundaries

Local CPU reproduction, live API demo and model/report audits are recorded in
the [Milestone 6 evidence](../reports/milestone6-handoff-v1/README.md). CPU Compose and GPU runner topology were previously
executed; GPU Compose itself and hosted GitHub Actions remain unverified.
No cloud resources were provisioned, no money spent and no production users or
ticketing/reservation integration exist. vLLM compatibility/validation completed,
but adoption, load testing and backend cost measurement did not.

The independent 100+ request challenge set is deferred; [protocol](challenge_protocol.md).
Cloud execution requires account, region, host, budget, session duration and
cleanup ownership before launch. The existing failed release does not become
approved merely because an infrastructure demo is authorized.

## Learning checkpoints

- **Why TF-IDF?** Intent wording often has discriminative words/bigrams. A small
  CPU classifier is cheap to fit and a useful measured control.
- **What does LoRA change?** Trainable low-rank adapter matrices; the quantized
  base stays frozen. This reduces trainable state, not the need for a base model.
- **Why completion masking?** Loss applies to JSON answer/EOS tokens, not the
  system catalog or supplied request. The collator/masks were checked before SFT.
- **Why can a high F1 model fail release?** Supported classification and rejecting
  unknown requests are different tasks. C misses the required oos-review recall.
- **What does a threshold trade?** Increasing it generally reviews more inputs;
  coverage and routed error depend on data mix. Gate scores are uncalibrated.
- **Why not reuse public test results?** They are already exposed; repeated tuning
  turns the test into another validation set. Pretraining overlap is also unknown.
- **Why check a serving backend?** Different kernels, precision/cache behavior
  and decoding implementations can change outputs even with the same weights.

Before taking ownership, run CPU reproduction, inspect the failed release rule,
explain one supported regression and one oos error, locate exact model/policy
hashes, and practice stopping the demo/container. Record any future changes as
new experiments with their own evidence.
