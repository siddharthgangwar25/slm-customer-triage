# Implementation status

Updated **2026-10-04**. Milestones 1-6 are complete locally. The frozen final
results remain unchanged: no candidate qualified for automatic release.

| Area | Status | Evidence |
| --- | --- | --- |
| CPU baseline and API | Reproduced from a clean checkout; 3,100 validation/API matches | [Publication checks](../reports/publication-readiness-v1/README.md) |
| Prompted and QLoRA models | Paired validation and frozen final test complete | [Model card](model_card.md) |
| Transformers serving | CPU Docker and full GPU container parity/load checks complete | [GPU serving](../reports/milestone5-container-gpu-full-v1/README.md) |
| vLLM comparison | Completed; backend not adopted | [Comparison](../reports/vllm-validation-v1/README.md) |
| Demo and handoff | CPU reproduction, live demo and documentation verified | [Handoff](handoff.md) |
| Manual GPU fixture | Passed locally; GitHub execution unverified | [GPU smoke](gpu_smoke.md) |
| GitHub publication/CI | No remote configured; hosted execution unverified | [Publication checklist](publication.md) |
| Cloud deployment | Omitted; no spending authorized | [AWS runbook](../deployment/aws_runbook.md) |
| Independent challenge set | Optional, deferred | [Protocol](challenge_protocol.md) |
| Code license | Undecided at the owner's request | [README](../README.md#data-and-licensing) |

## Release cleanup

Reviewed the tracked source, scripts, tests, deployment files, environments,
documentation and evidence layout before editing. The package structure and
frozen model/service implementation are retained. The CPU/GPU dependencies have
runtime, test, tooling or compatibility uses; none was removed speculatively.

- Reorganized the README around features, results, setup, API usage, environment
  variables, architecture and development commands.
- Added a blank `.env.example` and excluded local credentials from Docker build
  contexts. Native Python still reads shell variables rather than loading dotenv.
- Improved reproduction preflight errors and failed-stage records, reused the
  CLI command prefix, and normalized new helper JSON files to LF.
- Removed assistant-specific completion messages from run scripts and current
  runbooks. Kept useful progress, failure and operational logs.
- Moved the chronological handoff log to [implementation history](implementation_history.md).

Validation: **165 tests passed in 23.19 seconds**, including six new reproduction
failure-path tests. Applied Ruff fixes/formatting; final lint, format and mypy
checks pass. The formatter resolved an initial line-length failure in the new
test file. The revised helper completed all eight CPU stages using verified
cached sources and a fresh environment. Its predictions and API decisions match
all **3,100** reference validation records; the authenticated demo passed and
stopped its process. `verify_handoff.py` confirms the frozen release and completed
test ledger still verify.

Commands run: `ruff check src scripts tests deployment --fix`,
`ruff format src scripts tests deployment`, `ruff check .`,
`ruff format --check .`, `mypy`, and `python -m pytest -q` in the locked CPU
environment; `scripts/reproduce_cpu.py --source-dir data/clinc150/raw --output
artifacts/cleanup-cpu-reproduction-v1`; and `scripts/verify_handoff.py` against
that reproduction with output `artifacts/cleanup-handoff-audit-v1`. Local command
logs and audit details are in `artifacts/cleanup-review-v1`.

The limited credential-pattern scan found no matches in tracked/new files.
Checked local documentation links and secret-file ignore rules. Existing
source, configs, locks and retained report files are unchanged in Git.
Historical per-candidate report headings have a known split-label error,
documented in the new [results index](../reports/README.md); the frozen files
remain intact. No GPU inference, final-test rerun, image rebuild, hosted CI or
cloud execution was needed for this cleanup.

## Next steps

Create an empty GitHub repository, push the reviewed local history, and inspect
hosted CPU CI before claiming it passed. See the [publication checklist](publication.md).
Do not rerun the completed final test, reset its ledger, retune thresholds or
activate the disqualified model while preparing the repository for publication.

The [historical log](implementation_history.md) retains earlier commands,
failures and stage-specific limitations. Its old pending-task instructions are
superseded by this page.
