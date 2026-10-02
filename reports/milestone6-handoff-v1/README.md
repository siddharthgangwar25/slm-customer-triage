# Milestone 6 local handoff acceptance

Executed 2 October 2026. Mandatory local handoff deliverables are implemented
and verified. Cloud execution is omitted/unverified; the optional independently
reviewed challenge set is deferred. No test inference, GPU inference, release
activation, paid provisioning or original model/source change occurred.

## Fresh CPU reproduction

A **new environment** used the root CPU lock, Python 3.11.16 and uv 0.12.19.
Packages and pinned raw source files came from local caches; this is not a claim
of a new internet download. No original model was loaded for fitting/prediction.

| Stage | Seconds | Result |
| --- | ---: | --- |
| Locked environment sync | 1.77 | Fresh CPU environment installed |
| Data preparation | 2.01 | Source hashes/counts checked; manifest identical |
| Training/reload | 23.43 | 15,000 supported records; converged in 21 iterations, no warnings |
| Validation prediction | 3.87 | 3,100 complete records |
| Report | 3.78 | Supported macro-F1 **0.8826340782** |
| Policy selection | 0.79 | Threshold **0.2088413160728636** |
| API parity | 24.51 | **3,100/3,100** matches; 2,121 routes, 979 reviews |
| Live demo | 4.30 | HTTP checks passed; service stopped |

Combined stages: **64.45 seconds**; model fitting alone: **10.57 seconds**.
These are local acceptance timings, not load-performance measurements.
Baseline fitting excluded the 100 training oos records. Every validation ID,
true/predicted label, gate score, parse status and error type matches the original
reference exactly. New output paths/configuration give the reproduced bundle
its own model/policy IDs. Historical frozen release inputs remain unchanged.

The first attempt stopped before preparation because the new helper invoked
`python -m triage`; the executable module is `triage.cli`. The failure log is
retained. Corrected the helper and reran in a fresh output directory. A new
real-HTTP integration test covers the demo entry point, cleanup and refusal to
overwrite evidence.

## Demo and verification

Both fresh and original CPU bundles were tested through actual authenticated
localhost HTTP. The authored account-balance prompt routed to `balance`.
The lunar telescope repair and instruction-like submarine replacement prompts
both returned review with `low_gate_score`. These are synthetic illustrations,
not independently reviewed accuracy evidence.

Health/model metadata passed. Missing credentials returned 401, blank text 422;
API credentials failed private metrics access while the metrics key succeeded.
Both servers stopped. The auditor verified that submitted text did not appear
in service logs. `demo.json`, metrics and logs are retained for both runs.

The CPU-only auditor checks full reproduction, 3,100 API outcomes, both demos,
local documentation links and frozen release integrity. `acceptance.json`
records the exact documentation hashes/link count. Source hash remains
`2a91de5d0e0e4625c3b3fcb22a5898d16b387f6259568b9bdd2b21df3f48a4c4`.
Release `release-a0bcea381506` still verifies, its ledger is unchanged and no
active pointer exists. Large weights/environments are excluded from Git.

Retained Windows log/command/config text copies use LF line endings so Git
checkout preserves their checksums. `line_endings.json` maps original source-byte
hashes to retained hashes; no original execution artifact was modified.

**159 tests passed in 15.08 seconds**; Ruff lint/format and configured mypy passed.
CPU/GPU Compose configuration checks passed with temporary credentials and no
containers started. This is syntax validation, not new GPU Compose execution.
Historical CPU Docker/GPU runner acceptance remains separate; hosted CI is
unverified. The [two-minute demo](../../docs/demo.md) has a tested live script,
but no video recording or timed human rehearsal is claimed.

## Executed commands

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_cpu.py --source-dir data/clinc150/raw --output artifacts/milestone6-cpu-reproduction-v2
.\.venv\Scripts\python.exe scripts/run_demo.py --output artifacts/milestone6-demo-v1
.\.venv\Scripts\python.exe scripts/verify_handoff.py --reproduction artifacts/milestone6-cpu-reproduction-v2 --demo artifacts/milestone6-demo-v1 --output artifacts/milestone6-acceptance-v2
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\pytest.exe -q
```

These are history, with existing output directories. Choose new names to repeat
the CPU workflow; [setup](../../docs/setup.md). No account/region/host/budget/session
was authorized for cloud execution. The optional challenge set needs 100+ fresh
independently reviewed requests; the three demo prompts do not qualify.
The failed release criterion remains a finding, not permission to retune.
[Developer handoff](../../docs/handoff.md).
