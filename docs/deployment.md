# Local deployment and recovery

These commands start a **local research/demo service**. None of the frozen
candidates qualified for automatic production release. No active release pointer
exists. Cloud hosting remains [unverified](../deployment/aws_runbook.md).

## CPU Python service

With a trusted original local bundle, generate separate temporary credentials
in PowerShell and run the server:

```powershell
$env:TRIAGE_API_KEY = [guid]::NewGuid().ToString('N')
$env:TRIAGE_METRICS_KEY = [guid]::NewGuid().ToString('N')
.\.venv\Scripts\triage.exe serve --bundle artifacts/baseline-c1-v1 --policy reports/baseline-c1-v1-policy/policy.json --config configs/service.yaml
```

For a fresh reproduced bundle, use its `bundle` and `policy/policy.json` paths
from [setup](setup.md). Keep credentials in the local shell, not source control,
screenshots or command-history literals. Open `/docs` on localhost and supply
the API bearer credential through Authorize. Ctrl+C stops this process.
`scripts/run_demo.py` automates authenticated requests and cleanup without
requiring manual credential handling.

Health endpoints: `/health/live` confirms the process; `/health/ready` confirms
usable inference. `/v1/model` reports the loaded model/policy identity.
`POST /v1/triage` uses the API key. `/metrics` uses the separate metrics key and
exists only when that key is configured. Inspect logs and readiness before
retrying a 503; repeated busy responses mean insufficient capacity, not success.

## CPU Docker Compose

For file-based configuration, copy `.env.example` to `.env` in the repository
root and fill in three distinct randomly generated values. Pass it explicitly:

```console
docker compose --env-file .env -f deployment/compose.yaml --profile cpu up --build
```

The shared Compose file requires all three keys during interpolation, including
the worker key when selecting the CPU profile. `.env` is ignored by Git and
excluded from Docker's build context. The Python CLI does not load this file;
use process environment variables for native Python commands.

Docker Desktop must run Linux containers. Prepare the original-layout CPU bundle
first, or use the alternate-bundle command below. Compose interpolates all
required environment fields, so set all three credentials even for CPU mode:

```powershell
$env:TRIAGE_API_KEY = [guid]::NewGuid().ToString('N')
$env:TRIAGE_METRICS_KEY = [guid]::NewGuid().ToString('N')
$env:TRIAGE_WORKER_KEY = [guid]::NewGuid().ToString('N')
docker compose -f deployment/compose.yaml --profile cpu config --quiet
docker compose -f deployment/compose.yaml --profile cpu up -d --build baseline
docker compose -f deployment/compose.yaml --profile cpu logs --tail 50 baseline
```

The default CPU image expects `artifacts/baseline-c1-v1` and the retained original
policy under `reports/baseline-c1-v1-policy`. A newly trained model must instead
use its own matched policy. Run this foreground alternative, without first
starting the default service:

```powershell
docker compose -f deployment/compose.yaml --profile cpu run --rm --service-ports baseline --bundle artifacts/reproduction-v1/bundle --policy artifacts/reproduction-v1/policy/policy.json
```

The Python and Compose examples both use port 8000; stop one before starting
the other. Artifacts/reports mount read-only; only localhost is published.
The CPU image uses the pinned CPU lock and imports no GPU library.

For background Compose cleanup, run this from the same project directory:

```powershell
docker compose -f deployment/compose.yaml --profile cpu down
Remove-Item Env:TRIAGE_API_KEY, Env:TRIAGE_METRICS_KEY, Env:TRIAGE_WORKER_KEY
```

This targets the project's Compose resources, not unrelated Docker containers
or images. The foreground `run --rm` container exits with Ctrl+C.

## Optional GPU research profile

Requires an NVIDIA GPU visible inside Linux Docker and the trusted original
base cache, checkpoint-944 adapter, baseline gate and reference reports. This
does not download missing project weights automatically. Original hardware was
GTX 1650 4 GiB; fit/driver compatibility elsewhere requires a smoke test.
GPU image construction involves several gigabytes of dependencies.

After setting the same three keys above:

```powershell
docker compose -f deployment/compose.yaml --profile gpu config --quiet
docker compose -f deployment/compose.yaml --profile gpu up -d --build worker gateway
docker compose -f deployment/compose.yaml --profile gpu logs --tail 50 worker gateway
docker compose -f deployment/compose.yaml --profile gpu down
```

The worker starts before the gateway, remains on an internal network and has no
published port. Startup failure must be resolved before a gateway is considered
ready. The equivalent runner topology passed full parity/load checks, but the
GPU Compose profile itself is **not separately executed**. Its commands are a
handoff path, not additional measured acceptance. See [serving runbook](release_benchmark.md).
The vLLM variant is separate and not adopted.

## Staging, activation and recovery

Before staging a future variant, verify trusted bundle hashes, exact package
versions, prompt/tokenizer identity, validation parity and HTTP contracts. Preserve
old manifests, weights, image digests and environments. A failed readiness check
must fail visibly; never fabricate an intent or silently replace an SLM label
with the baseline label.

The CLI exposes `release activate` and `release rollback`, but activation of this
completed candidate is expected to fail its final-test guard. Do not edit that
guard, reset the test ledger or create an active pointer by hand. There is no
previous accepted live release to roll back to. For this local demo, stop the
failed process and restart the documented trusted bundle/configuration; this is
process recovery, not evidence of production traffic migration.

Future cloud staging, restricted roles, HTTPS, secret injection, health checks
and teardown are specified in the [AWS plan](../deployment/aws_runbook.md).
Do not provision until the account/region/host/budget/session/cleanup decisions
are authorized. No cloud execution is needed to reproduce the CPU demo.
