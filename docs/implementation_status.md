# Implementation status

Last updated: **2026-09-25**. Milestones 1 and 2: **implemented, executed, and locally verified**. Milestones 3–6 have not been implemented. The original specification and Milestone 1 benchmark evidence are preserved unchanged.

## Milestone 2 — routing policy and API

The exact validation selector and local FastAPI service are complete. No baseline retraining, hyperparameter change, test prediction, GPU execution, or paid provisioning occurred in this milestone.

### Changed files

| Files | Result |
| --- | --- |
| `src/triage/policy.py` | Immutable routing rules, inclusive threshold behavior, strict JSON-output validation, unchanged four-code reason vocabulary |
| `src/triage/evaluation/select_policy.py` | Exact score sweep, original constraints/tie breaks, validation-only selection, frozen policy hashes, unmet-target fallback, retained threshold/report evidence |
| `src/triage/service/{__init__,schemas,runtime,app}.py` | Strict API request/response/error contracts, trusted baseline loading, readiness/liveness, safe metadata, auth, rate/body/token limits, bounded asynchronous inference and timeout behavior |
| `src/triage/cli.py`, `configs/service.yaml` | `triage policy select` and `triage serve`, explicit policy/output paths, localhost defaults |
| `pyproject.toml`, `uv.lock` | Locked FastAPI/Pydantic/Uvicorn dependencies and httpx2 development client; type checks expanded to policy and API schemas |
| `tests/test_policy.py`, `tests/test_service.py` | 63 new tests; existing 33 tests retained. The existing CPU CI workflow automatically includes them |
| `scripts/verify_api.py` | Repeatable real-model validation parity through the ASGI API, with explicit evidence output and test-split rejection |
| `reports/baseline-c1-v1-policy/` | Frozen `policy.json`, all exact sweep points, and validation selection report |
| `reports/baseline-c1-v1-api/` | Genuine full-validation API parity and separate localhost HTTP smoke evidence |
| `README.md`, `docs/api.md`, `docs/decisions.md`, `docs/experiment_journal.md`, this file | Reproduction commands, API behavior/limits, ML decisions, measurements, and handoff |

### Measured acceptance results

All routing figures use the original validation mix: **3,100 requests = 3,000 supported + 100 oos**, model `baseline-c1-v1-551714a82382`. No test examples were predicted.

| Check | Observed result |
| --- | --- |
| Exact sweep | 3,068 thresholds, including every unique observed score plus zero/all-review endpoints |
| Frozen policy | `policy-v1-51f9c25a5c5c`, threshold **0.2088413160728636**, `targets_met`, automatic routing enabled |
| Routing coverage | **68.4194%**, 2,121 / 3,100 requests |
| Routing error | **4.7619%**, 101 / 2,121 routes; Wilson 95% **3.9345–5.7529%** |
| Oos recall | **90%**, 90 / 100; Wilson 95% **82.5634–94.4771%** |
| Supported review rate | **29.6333%**, 889 / 3,000 supported requests |
| Overall reviews | 979 / 3,100 requests; no gated infrastructure failures |
| API/offline parity | **3,100 actual baseline API calls, zero intent/decision/reason mismatches**; 2,121 routes and 979 reviews |
| Real HTTP smoke | `triage serve` on `127.0.0.1:8765`; live/ready/model endpoints returned 200; one genuine validation route and one review verified; server stopped afterward |
| Unmet-target behavior | Synthetic fixture freezes `targets_unmet`, disables routing, records null error/zero coverage, and returns review only while a healthy model is loaded |
| Failure contracts | 401 auth, 422 schema/body/token budget, 429 rate limit, and 503 missing/corrupt/unavailable/busy/timed-out inference verified |
| Runtime safety | Gate short-circuit and baseline prediction reuse verified; timeout retains occupied slot, liveness remains responsive, readiness recovers after successful completion, failed model stays unready |
| Input/log contracts | Instruction-like input remains data; raw messages and auth secrets absent from application logs; chunked oversized bodies rejected; no public `/metrics` |
| Tests and checks | **96 tests passed**, no warnings; lint, formatting, core type checks, and dependency-lock consistency passed |

These constraints apply to validation-selected point estimates. The Wilson intervals do not establish a <=5% unseen routing-error guarantee or >=90% unseen oos-recall guarantee. API parity and the five HTTP smoke calls are correctness checks, not throughput/latency benchmarks or production usage.

### Commands actually executed for Milestone 2

PowerShell, repository root (the local uv executable remains in `.tools/`):

```powershell
.\.tools\uv.exe sync --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.venv\Scripts\triage.exe policy select --predictions reports/baseline-c1-v1-val/predictions.jsonl --split val --output reports/baseline-c1-v1-policy
.\.venv\Scripts\ruff.exe check src tests scripts --fix
.\.venv\Scripts\ruff.exe format src tests scripts
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe scripts/verify_api.py --bundle artifacts/baseline-c1-v1 --policy reports/baseline-c1-v1-policy/policy.json --predictions reports/baseline-c1-v1-val/predictions.jsonl --output reports/baseline-c1-v1-api
.\.venv\Scripts\triage.exe serve --bundle artifacts/baseline-c1-v1 --policy reports/baseline-c1-v1-policy/policy.json --port 8765
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.tools\uv.exe lock --check --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
```

The live server was probed with Python `urllib.request`, saving `http_smoke.json`, then its known process was stopped. `uv sync` ran a second time after replacing deprecated httpx fallback with the installed Starlette version's supported httpx2 client. The first test run had a client deprecation warning (91 passed); after migration 93 passed without warnings; the final runtime/concurrency checks brought the suite to 96 passing tests. Early formatting violations were fixed before the final checks; no test failures occurred.

Final acceptance also updated the separate `.tools/acceptance-venv` via `UV_PROJECT_ENVIRONMENT=.tools/acceptance-venv` and `uv sync --locked --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache`. Its Ruff lint/format and mypy checks passed, and its pytest run reported **96 passed in 8.02 seconds**, without warnings. This verifies the updated lock independently of the development environment. Milestone 2 is committed locally with its lock and evidence; no push was performed.

Verified versions: FastAPI 0.141.1, Pydantic 2.13.5, Starlette 1.7.0, Uvicorn 0.54.0, httpx2 2.13.1; existing model dependencies remain unchanged. The full training lock hash in the original baseline metadata is historical; API evidence records the updated service dependency lock. No original benchmark artifact was rewritten.

### Limitations and next concrete task

No Milestone 2 blocker remains. The API is a single-process localhost demonstration with one inference slot and a global in-memory rate limiter. A native worker cannot be forcibly cancelled; a permanently stuck operation needs a process restart. Auth is optional for local use. `/metrics`, containers, remote deployment, release freezing, and production observability remain later work. Hosted CI/Linux execution is unverified. No GPU test or real SLM tokenizer was run or claimed verified; the current baseline has no prompt.

**Next: Milestone 3 — prompted model benchmark.** Inspect hardware and available resources, pin a feasible model/revision, implement the full fixed catalog and non-thinking prompt, verify real tokenizer budgeting, save every raw validation outcome, and attach this frozen baseline's gate by a checked one-to-one sample-ID join. Keep test prediction blocked and paid services disabled.

## Milestone 1 — historical completion record

## Delivered files

| Area | Files and behavior |
| --- | --- |
| Repository and environment | `.gitignore`, `.gitattributes`, `.python-version`, `pyproject.toml`, `uv.lock`, `README.md`; Python package, editable installation, CPU dependency lock, reproducible commands |
| Configuration | `configs/data.yaml`, `configs/baseline.yaml`; official commit/checksums and one fixed C=1.0 experiment |
| Package/CLI | `src/triage/__init__.py`, `cli.py`, `contracts.py`, `io.py`; four implemented commands, explicit outputs, clear errors, overwrite refusal |
| Data | `src/triage/data/{__init__,prepare,load}.py`; pinned downloads, schema/count/hash checks, deterministic IDs, catalog, canonical splits, duplicate/conflict audit, provenance manifest |
| Baseline | `src/triage/models/{__init__,baseline}.py`; supported-training-only TF-IDF/logistic regression, warning capture, saved pipeline, exact reload check, validation predictions and gate scores |
| Evaluation | `src/triage/evaluation/{__init__,metrics,report}.py`; fixed-catalog macro-F1, routing diagnostics, Wilson intervals, failure accounting, per-class CSV, confusion matrix, plot, saved-prediction completeness checks |
| Tests and CI | `tests/conftest.py`, `test_data.py`, `test_metrics.py`, `test_baseline.py`, `.github/workflows/cpu.yml`; 33 tests, pinned workflow actions, locked CPU install, lint/format/types/tests |
| Evidence | `reports/baseline-c1-v1-val/`: report, metrics, all 3,100 predictions and manifest, data manifest, duplicate audit, per-class report, confusion matrix, coverage/error curve and plot, 40 review samples, manual error analysis, source license |
| Decisions and learning | `docs/decisions.md`, `docs/experiment_journal.md`, this file |

Local ignored outputs remain available in `data/clinc150/`, `artifacts/baseline-c1-v1/`, and `artifacts/baseline-c1-v1-val/`. The model bundle is 37,762,940 bytes. `.tools/` contains the local uv/Python installation and a second acceptance environment; `.venv/` is the development environment. These are not repository dependencies or committed assets.

## Acceptance evidence

| Check | Observed result |
| --- | --- |
| Official source | `clinc/oos-eval@828f8093932c8fe6ca7936c3d2e52903b1c523de`; data, domain mapping, and license hashes match pinned values |
| Train split | 15,000 supported + 100 oos; classifier fit uses only the 15,000 supported samples |
| Validation split | 3,000 supported + 100 oos; every sample has a saved prediction |
| Test integrity | 4,500 supported + 1,000 oos; prepared and integrity-audited only; **no test predictions, tuning, or classification metrics** |
| Duplicate audit | 5 exact/normalized cross-split groups; 4 conflicting-label groups; 0 within-split groups; official membership preserved |
| Fresh installation | Second environment installed with `uv sync --locked --offline` using the already downloaded package cache |
| Fresh data path | Actual downloader fetched all three pinned files; the local-source option was also exercised earlier |
| Training | C=1.0, seed 42, word unigrams/bigrams, sublinear TF, one native math thread, float64 CPU; converged in 21 iterations, no warnings, 10.383 seconds |
| Artifact parity | Exact probability equality before/after save/reload on 32 training probes; fixture tests also verify labels/probabilities |
| Validation raw supported macro-F1 | **0.8826340782** over all 150 supported labels |
| Validation supported accuracy | **0.8833333333**; 350 errors / 3,000 supported samples |
| Invalid/infrastructure failures | 0 / 3,100 for each category |
| Ungated diagnostic | 3,100 / 3,100 routed, 450 errors / 3,100 routes (14.5161%), 0 / 100 oos recalled; no operating policy selected |
| Serial model latency | p50 0.6153 ms, p95 0.7844 ms, p99 1.0040 ms; per-record vectorization/classification, **not API latency or a load test** |
| Error inspection | 40 examples inspected and annotated: 30 supported errors + 10 oos; deterministic, class-order-biased sample |
| Final tests | **33 passed in 6.33 seconds**, no failed tests |
| Final code checks | Ruff lint and formatting passed; mypy passed on the 3 configured core-contract/helper modules |
| Lock consistency | `uv lock --check --offline` passed; no GPU dependencies installed |

Hardware: AMD Ryzen 5 5600H with Radeon Graphics, Windows x86-64. CPU name was read from the Windows registry; CIM inventory access was denied, so installed RAM was not measured. Runtime: Python 3.11.16, scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, joblib 1.6.0, uv 0.12.19. Exact runtime metadata is in the prediction manifest; all resolved dependencies are in `uv.lock`.

## Commands actually executed

The shell was PowerShell, from the repository root. Initial inspection read the specification, searched for applicable `AGENTS.md` files (none found), checked PATH and Git, and confirmed the folder initially contained only the specification. Network downloads required sandbox escalation; they succeeded. No paid resource was used.

Setup and first development checks:

```powershell
# Downloaded the uv Windows archive and resolved the official CLINC commit via HTTPS.
.\.tools\uv.exe python install 3.11 --install-dir .tools/python
.\.tools\uv.exe sync --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.venv\Scripts\triage.exe data prepare --config configs/data.yaml --source-dir .tools
.\.venv\Scripts\triage.exe train baseline --config configs/baseline.yaml
.\.venv\Scripts\ruff.exe check src tests --fix
.\.venv\Scripts\ruff.exe format src tests
.\.venv\Scripts\pytest.exe -q
git init
```

The first data/model development outputs were removed after checking that both resolved target directories were inside this workspace. The final acceptance sequence regenerated them using the current implementation. No existing user file was removed or rewritten.

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.tools/acceptance-venv'
.\.tools\uv.exe sync --locked --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.tools\acceptance-venv\Scripts\triage.exe data prepare --config configs/data.yaml
.\.tools\acceptance-venv\Scripts\triage.exe train baseline --config configs/baseline.yaml
.\.tools\acceptance-venv\Scripts\triage.exe predict --config configs/baseline.yaml --split val
.\.tools\acceptance-venv\Scripts\triage.exe evaluate --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --output reports/baseline-c1-v1-val
.\.tools\uv.exe lock --check --offline --python .tools/python/cpython-3.11.16-windows-x86_64-none/python.exe --cache-dir .uv-cache
.\.tools\acceptance-venv\Scripts\ruff.exe check .
.\.tools\acceptance-venv\Scripts\ruff.exe format --check .
.\.tools\acceptance-venv\Scripts\mypy.exe
.\.tools\acceptance-venv\Scripts\pytest.exe -q
```

Also executed: source-file SHA256 checks; `git ls-remote` for both pinned CI actions; inspection of all 40 saved error-review records; plot rendering and visual inspection; copying the data manifest, duplicate audit, and original license into the retained report. Early lint checks found line-length formatting issues, which were fixed before the final successful checks. There were no test failures. The original system `python`/`py` aliases were unusable; the local installation resolved this environment issue.

During the final Git review, Windows CRLF serialization was found to conflict with Git's LF normalization and recorded checksums. JSON/CSV serialization now explicitly writes LF, and deterministic-data tests assert this. The data preparation, training, prediction, and evaluation sequence was rerun after this correction, using `data prepare --config configs/data.yaml --source-dir .tools` with the same verified official source files. Classification metrics and the 40 reviewed errors were unchanged; the table above records the final run's timing. The retained report's prediction, data-manifest, duplicate-audit, and license hashes were checked against their Git-staged bytes.

Repository initialization succeeded. Staging required sandbox escalation and a command-local `safe.directory` setting because the sandbox and host user have different ownership identities; no global Git setting was changed. The milestone files and lockfile are included in the initial local commit. No remote push was performed.

## Limits and next task

No blocking Milestone 1 issue remains. Windows execution is verified; the GitHub Actions Windows/Linux matrix is authored but has not run on a hosted runner. Fresh-environment installation was verified on this machine with cached locked distributions, not on a different operating system. No GPU test was attempted or claimed verified; GPU work is outside this milestone.

The baseline is deliberately ungated. Its scores are not calibrated correctness probabilities, and raw oos rejection is zero. The report's threshold curve is exploratory only. No validation constraint, deployment readiness, cost saving, or real customer outcome is claimed. Only one C configuration was run. The error observations are assistant-authored and not an independent human annotation study.

The next task recorded at Milestone 1 completion was Milestone 2; it is now complete as documented above. The current next task is Milestone 3.
