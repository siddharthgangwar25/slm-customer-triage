# Customer request triage

A reproducible triage service and model comparison for the official CLINC150 benchmark. Milestones 1–4 implement a TF-IDF/logistic-regression baseline, frozen routing policy, local FastAPI service, prompted model benchmarks and a completed paired Qwen3-0.6B QLoRA experiment. See the [project specification](Customer_Request_Triage_Project_Spec.md) and [implementation status](docs/implementation_status.md).

The baseline achieves **0.882634 supported macro-F1**; fine-tuned Qwen3-0.6B achieves **0.966217**, versus **0.272377** for its paired prompted base. Each validation evaluation includes 3,000 supported and 100 out-of-scope requests. C's selected policy routes **80.35%** at **2.25% error** and **90% oos review recall**. These are public, validation-selected benchmark results, not test or production guarantees. The API still uses the baseline. [Three-way report](reports/three-way-qwen3-06b-v1/report.md) · [changed errors](reports/three-way-qwen3-06b-v1/error_analysis.md) · [data provenance](reports/baseline-c1-v1-val/data_manifest.json).

## Reproduce from a fresh environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the repository root. Python 3.11 is selected by `.python-version`; uv can install it if needed. `uv.lock` pins CPU runtime and development dependencies; no GPU, model hub account, or paid service is needed. The first sync and data preparation require internet access.

```console
uv sync --locked
uv run --locked triage data prepare --config configs/data.yaml
uv run --locked triage train baseline --config configs/baseline.yaml
uv run --locked triage predict --config configs/baseline.yaml --split val
uv run --locked triage evaluate --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --output reports/local-baseline-val
```

`uv sync` installs the project in editable mode. Alternatively activate `.venv` and use `triage` directly. On the implementation machine, uv is also available at `.tools/uv.exe`; its project-local Python is at `.tools/python/cpython-3.11.16-windows-x86_64-none/python.exe`.

Each command refuses to overwrite its output directory, including incomplete runs. For another experiment, copy the configuration, choose a new experiment ID, bundle, and prediction output, and use a new report directory. Paths in configuration files resolve relative to the working directory. Every command accepts `--output`; `predict --bundle <path>` selects a trusted bundle written to an alternate training output. If you change model configuration, retrain under a new experiment ID. Only load joblib bundles you trust: they contain executable Python serialization; hashes detect corruption, not a malicious author.

For offline preparation, put the three original files (`data_full.json`, `domains.json`, `LICENSE`) in one directory and add `--source-dir <directory>` to `data prepare`. The same pinned SHA256 checks apply. Missing inputs, checksum mismatches, corrupt records, and existing outputs fail with a nonzero exit code.

## Outputs and evidence

- `data/clinc150/`: preserved raw sources and license, canonical train/val/test JSONL, catalog, duplicate audit, and manifest with hashes, counts, frequencies, and text-length distributions. Local data is ignored by Git.
- `artifacts/baseline-c1-v1/`: one saved vectorizer/classifier pipeline, configuration and environment metadata, training warnings, checksums, and reload verification. Large model files are ignored by Git.
- `artifacts/baseline-c1-v1-val/`: all 3,100 raw predictions with labels, IDs, gate scores, timing, and a completeness manifest.
- Each evaluation directory: `metrics.json`, copied predictions and manifest, `per_class.csv`, `confusion_matrix.csv`, `coverage_error.json`, `coverage_error.png`, `report.md`, and a deterministic error-review queue. Evaluation loads no weights and needs only the prediction file and adjacent manifest.
- `reports/baseline-c1-v1-val/`: retained genuine validation evidence, plus data provenance, source license, and manual error observations. Public validation text is intentionally retained for reproducibility and attributed to CLINC.

The baseline trains on 15,000 supported requests only. Word unigrams/bigrams capture local wording, and TF-IDF downweights common terms. Logistic regression learns a weight for each feature and intent. Saving both transformations and classifier in one pipeline keeps inference consistent with training. Domain metadata never enters the model. The maximum predicted probability is an uncalibrated gate score, not confidence that the prediction is correct.

The Milestone 1 report's coverage/error curve is a diagnostic sweep. Milestone 2's [exact policy selection](reports/baseline-c1-v1-policy/report.md) selected threshold **0.2088413161**, with **68.42% coverage**, **4.76% routing error**, and **90% oos recall** on validation. The raw classifier still predicts a supported label for every request; the policy sends low-score requests to human review. The CLI deliberately disallows test prediction until a frozen-release workflow exists.

## Select a policy and run the API

After generating your own bundle and predictions above, select their policy and serve it:

```console
uv run --locked triage policy select --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --split val --output reports/local-baseline-policy
uv run --locked triage serve --bundle artifacts/baseline-c1-v1 --policy reports/local-baseline-policy/policy.json --config configs/service.yaml
```

Open `http://127.0.0.1:8000/docs`. The service binds to loopback only. `--policy` defaults to `policy.json` inside the bundle if omitted. Use the policy generated from the same model bundle: hashes and model dependency versions must match. The retained milestone policy can be used with the original local milestone bundle.

`POST /v1/triage` accepts `{"text":"what is my account balance"}` and an optional `client_request_id`. It returns a new server request ID, intent or null, route/review decision, machine-readable reason, model and policy versions, and latency. Human review is a recommendation; it does not create a ticket. No confidence field is exposed. `/health/live`, `/health/ready`, and `/v1/model` expose health and safe metadata.

The defaults cap text at 2,000 characters, bodies at 16 KiB, model input at 512 tokens, and inference at 5 seconds. This baseline counts its own tokenizer's tokens; it has no system prompt. A future SLM adapter must budget its full prompt and generation reserve. Optional bearer authentication uses the `TRIAGE_API_KEY` environment variable. The default per-process sliding-window rate limit is 60 valid triage requests per minute; configure `0` to disable it for offline acceptance checks. Missing/corrupt bundles and failed/timed-out inference return 503. `/metrics` is not exposed.

See the [API runbook](docs/api.md) for response examples, failure behavior, and operational limits. The [API parity evidence](reports/baseline-c1-v1-api/api_parity.json) records **3,100 requests with zero differences** against offline policy decisions; [localhost smoke evidence](reports/baseline-c1-v1-api/http_smoke.json) confirms the server command works over HTTP. Neither check is a load benchmark.

## Prompted model experiment

Milestone 3 adds a pinned Qwen3-4B runner with strict JSON parsing, complete validation accounting, a verified shared baseline gate, and raw/policy comparison. The optional CUDA environment stays separate from the CPU service. See the [benchmark runbook](docs/prompted_benchmark.md), [hardware/token audit](reports/prompted-hardware/token_audit.json), and [current execution status](docs/implementation_status.md). The API continues to load the baseline bundle.

The completed **3,100-request validation run** achieved **0.801038 supported macro-F1**, below the baseline's **0.882634**. It produced 70 invalid outputs and zero infrastructure failures. No shared-gate threshold met all routing constraints, so the prompted policy disables automatic routing. See the [paired comparison](reports/baseline-prompted-v1/report.md), [prompted report](reports/prompted-qwen3-4b-nf4-cache-v1-val/report.md), and [error review](reports/prompted-qwen3-4b-nf4-cache-v1-val/error_analysis.md). This is a completed experiment with a negative result, not a deployment upgrade.

## Fine tuning

Milestone 4 is complete: one epoch on 15,100 training requests took **11.69 hours** on the local GTX 1650. The selected checkpoint passed adapter reload and complete validation. Both B and C use pinned Qwen3-0.6B; original 4B results are retained separately. Raw oos correctness worsened after training, despite improved supported-intent accuracy; the [error analysis](reports/three-way-qwen3-06b-v1/error_analysis.md) explains the shared-gate tradeoff. The [runbook](docs/finetuning.md), [model card](docs/finetuned_model_card.md) and [verified execution status](docs/implementation_status.md) describe reproduction and limits. Serving and final test evaluation remain Milestone 5 work.

## Checks

Milestone 5's [final benchmark](reports/milestone5-final-v1/README.md) is complete and audited locally: 5,500 frozen test results per candidate. C reaches 0.956794 supported macro-F1 but is **disqualified for automatic release** because oos review recall is 89.6%, below 90%. Thresholds remain unchanged; do not rerun the completed test experiment. The 158-test CPU suite passes. [CPU Docker acceptance](reports/milestone5-container-cpu-v1/README.md) matches all 3,100 baseline decisions. [Full GPU container acceptance](reports/milestone5-container-gpu-full-v1/README.md) matches all 3,100 C outputs and completes 500/500 serial requests at 1.345/s and 1.117-second p95; concurrent workloads mostly fail busy. The local Transformers delivery is verified. The [separate vLLM smoke](reports/vllm-smoke-v1/README.md) completed 12/12 requests with B 5/6 and C 6/6 exact generation matches; full vLLM validation remains pending. See the [vLLM runbook](docs/vllm_experiment.md) for the next terminal command.

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
```

Tests use explicitly marked synthetic fixtures and cover source corruption, hashes and split membership, deterministic processing, leakage prevention, saved-model parity, metric arithmetic, failure accounting, CLI execution, model-free report generation, exact policy selection, and API contracts with fake and tiny real baseline adapters. The CPU CI workflow runs these on Windows and Linux; hosted CI execution remains unverified until pushed. Optional GPU execution and cache-parity probes use the separate prompted environment.

## Data attribution and limitations

CLINC150 accompanies *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, Larson et al. (2019), from the [original CLINC repository](https://github.com/clinc/oos-eval). Source revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`. The [CC BY 3.0 source license](reports/baseline-c1-v1-val/CLINC_LICENSE.txt) is retained. Canonicalization adds metadata; request text and official split membership are preserved.

There are five cross-split duplicate groups, four with conflicting labels. They are reported without altering the benchmark. Only 100 training oos examples exist. Public benchmark contamination cannot be ruled out for later language-model experiments. No test classification, deployment, cost claim, or cloud provisioning has been performed. Local GPU smoke evidence and prompted benchmark execution are tracked in the implementation status.
