# Implementation status

Last updated: **2026-09-25**. Milestone 1: **implemented, executed, and locally verified**. Milestones 2–6 have not been implemented. The original specification is preserved unchanged.

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

**Next concrete task: Milestone 2.** Implement exact validation-score threshold selection with the specified error/oos/coverage constraints and tie breaks, persist a frozen policy, and build the FastAPI routing/error/readiness contracts and tests. Reuse these saved validation predictions; keep test prediction blocked until the later release-freeze workflow. Do not add GPU or cloud work yet.
