# Setup and CPU reproduction

Run commands from the repository root. Use Python **3.11** and uv **0.12.19**
(the version used for acceptance). Install uv using its
[official instructions](https://docs.astral.sh/uv/getting-started/installation/).
The CPU lock is `uv.lock`; no GPU, Docker, Hugging Face account or cloud account
is required. First dependency installation and source download need internet.
[uv's locked sync](https://docs.astral.sh/uv/concepts/projects/sync/) refuses an
out-of-date lock rather than silently updating it.

## Complete reproduction in a new directory

```console
uv sync --locked
uv run --locked python scripts/reproduce_cpu.py --output artifacts/reproduction-v1
```

This creates a **new environment** under the output directory, syncs the locked
CPU dependencies, downloads/checks pinned CLINC sources, prepares canonical data,
trains A, verifies model reload, predicts validation, creates the report, selects
a new validation policy, checks all 3,100 API decisions, and runs the live demo.
It does not require the original author's ignored model/data files. Each stage
has a log and recorded command/exit code in `commands.json`. Model and policy
identities belong to this fresh reproduction; historical releases are unchanged.

The verified local invocation used checksum-verified cached source files:

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_cpu.py --source-dir data/clinc150/raw --output artifacts/reproduction-v1
```

Use this form only if that raw directory exists and contains `data_full.json`,
`domains.json` and `LICENSE`. On this machine the runner also finds `.tools/uv.exe`
if uv is not on PATH. `--uv <path>` explicitly selects another uv executable.
The bootstrap interpreter must be Python 3.11. No venv activation is necessary
when using explicit Python paths or `uv run`.

The runner checks Python, uv, output location and optional cached-source files
before installing dependencies. Outputs must be inside the project directory.
The runner refuses an existing output, including partial failures. Choose a new
directory, inspect the failed stage's log, and retain the failure evidence.
Do not delete original bundles or edit frozen configuration to make a retry work.
Preparing the dataset verifies test integrity but never performs test prediction.

## Keep the reproduced service running

After the automated reproduction finishes, its demonstration server has stopped.
Start a persistent local research service with its new bundle and policy:

```console
uv run --locked triage serve --bundle artifacts/reproduction-v1/bundle --policy artifacts/reproduction-v1/policy/policy.json --config configs/service.yaml
```

Open `http://127.0.0.1:8000/docs`. Stop the process with Ctrl+C. Without an API key
the default local CLI is unauthenticated; the automated demo supplies temporary
keys. Set `TRIAGE_API_KEY` and `TRIAGE_METRICS_KEY` to separate secrets for an
authenticated manual demo. `/metrics` exists only when its key is configured.
For concrete Windows commands, see [deployment](deployment.md).

## Individual clean-clone commands

If you prefer the original output layout, use these **only when those output
directories do not already exist**:

```console
uv run --locked triage data prepare --config configs/data.yaml
uv run --locked triage train baseline --config configs/baseline.yaml
uv run --locked triage predict --config configs/baseline.yaml --split val
uv run --locked triage evaluate --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --output reports/local-baseline-val
uv run --locked triage policy select --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --split val --output reports/local-baseline-policy
uv run --locked triage serve --bundle artifacts/baseline-c1-v1 --policy reports/local-baseline-policy/policy.json
```

Paths are relative to the working directory. Use the policy made from your own
bundle; a similarly named historical policy does not necessarily match it.
Only load trusted joblib artifacts: their serialization executes Python.
For direct module invocation use `python -m triage.cli`, not `python -m triage`.

## GPU and artifact handoff

Optional GPU environments remain separate: [prompted](prompted_benchmark.md),
[fine tuning](finetuning.md), [serving evidence](release_benchmark.md) and
[vLLM experiment](vllm_experiment.md). The 0.6B base snapshot, LoRA and original
baseline weights are not committed. A recipient can reproduce CPU weights as
above; reproducing the original adapter requires the documented training run
or a trusted copy matching `reports/finetuning-run-v1/selected_bundle.json`.
No public adapter download has been published. Preserve the corresponding
training identity/configuration, environment lock, tokenizer and prompt files.
Do not claim a fresh copy can run a historical release whose local artifacts it
does not possess. Current GPU locks and hardware are documented, but hardware
fit on another machine must be checked with the small smoke before long runs.

Never rerun `scripts/run_milestone5.py` or `benchmark final` for this completed
release. Inspect saved test evidence with the CPU auditor instead.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `uv` not found | Use `.tools/uv.exe` on this machine, or install uv and reopen the terminal |
| Unsupported Python | Run with Python 3.11; do not regenerate the lock to hide a mismatch |
| Output already exists | Choose a fresh output directory |
| Checksum failure | Compare against `configs/data.yaml`; fetch the pinned original bytes |
| API readiness 503 | Inspect startup log; check bundle/policy hashes and exact package versions |
| HTTP 401 / 429 | Supply the appropriate key / respect the configured rate limit |
| GPU busy or unavailable | Stop other GPU experiments; check the worker log and pinned environment |
| Docker engine missing | Start Docker Desktop in Linux-container mode; CPU Python reproduction works without it |

The actual fresh-environment acceptance evidence is linked from
[implementation status](implementation_status.md).
