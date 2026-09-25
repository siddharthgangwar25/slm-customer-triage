# Customer request triage

A reproducible CPU baseline for the official CLINC150 benchmark. Milestone 1 implements data preparation, TF-IDF/logistic-regression training, saved validation predictions, and offline evaluation. See the [project specification](Customer_Request_Triage_Project_Spec.md) and [implementation status](docs/implementation_status.md).

The measured validation result is **0.882634 supported macro-F1** across all 150 supported labels and **88.33% supported accuracy**. The evaluation includes 3,000 supported and 100 out-of-scope requests. This is public benchmark evidence, not production customer usage. [Report](reports/baseline-c1-v1-val/report.md) · [40 reviewed errors](reports/baseline-c1-v1-val/error_analysis.md) · [data provenance](reports/baseline-c1-v1-val/data_manifest.json).

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

The report's coverage/error curve is a diagnostic sweep, not a selected policy. The ungated baseline predicts a supported label for every out-of-scope request. Threshold selection and HTTP routing belong to Milestone 2. The CLI deliberately disallows test prediction until a frozen-release workflow exists.

## Checks

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -q
```

Tests use explicitly marked synthetic fixtures and cover source corruption, hashes and split membership, deterministic processing, leakage prevention, saved-model parity, metric arithmetic, failure accounting, CLI execution, and model-free report generation. The CPU CI workflow runs these on Windows and Linux; hosted CI execution remains unverified until pushed. API/container/GPU tests arrive with their respective milestones.

## Data attribution and limitations

CLINC150 accompanies *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, Larson et al. (2019), from the [original CLINC repository](https://github.com/clinc/oos-eval). Source revision: `828f8093932c8fe6ca7936c3d2e52903b1c523de`. The [CC BY 3.0 source license](reports/baseline-c1-v1-val/CLINC_LICENSE.txt) is retained. Canonicalization adds metadata; request text and official split membership are preserved.

There are five cross-split duplicate groups, four with conflicting labels. They are reported without altering the benchmark. Only 100 training oos examples exist. Public benchmark contamination cannot be ruled out for later language-model experiments. No test classification, GPU experiment, deployment, cost claim, or cloud provisioning has been performed.
