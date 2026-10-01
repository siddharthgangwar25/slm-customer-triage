# Local CPU Docker acceptance — 1 October 2026

Docker Desktop 4.93.0, Linux engine 29.8.1, Compose 5.5.1, WSL kernel 6.18.40.1. The pinned CPU Dockerfile built successfully with the unchanged root lock. Image identity is in `identity.json` (tag `triage-m5-cpu:82ed8d7`). No model training or real test inference was run.

- The unchanged genuine baseline bundle matched **3,100/3,100 validation API decisions**, including intent/reason: **2,121 routes, 979 reviews**, zero mismatches.
- A synthetic fixture generated inside the Linux image matched **3/3** decisions. It is CI evidence, not model-quality evidence.
- Both services passed live/readiness/model checks, authentication, empty/oversized-input rejection and separate-key private metrics checks.
- Tests used non-root/read-only containers, read-only model/report mounts, a writable temporary directory and randomly assigned **localhost-only** gateway ports. All owned containers/networks were removed afterward.
- The corrected Compose CPU profile separately passed configuration validation, readiness and an authenticated triage request; its uniquely named project was removed afterward.
- A temporary container queried **NVIDIA GeForce GTX 1650, driver 617.14, 4,096 MiB**. This verifies GPU device visibility only. The GPU model image/worker and vLLM remain unexecuted.

Two development failures are retained/disclosed. First, an internal-only Docker network omitted published ports (`NetworkSettings.Ports` contained an empty list). Compose now attaches the gateway to a separate `frontend` network and the private worker only to `private`; the GPU gateway joins both. Second, a Windows-generated synthetic fixture differed on one threshold-boundary request: reference/threshold **0.5923113658354859**, Linux gate **0.5923113658354858**. The original failed result and numeric diagnostic are retained. The synthetic fixture is now fitted/selected inside its tested Linux runtime; no real model, threshold or saved prediction was altered. Genuine baseline parity passed before and after the fixture change.

Commands executed:

```powershell
docker build --progress plain -f deployment/Dockerfile.cpu -t triage-m5-cpu:82ed8d7 .
docker run --rm --gpus all --entrypoint nvidia-smi triage-m5-cpu:82ed8d7 -L
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage cpu --skip-build --output artifacts/milestone5-container-cpu-v4
```

The earlier v1/v2/v3 attempt directories remain under ignored `artifacts/`; v3 completes genuine baseline parity while preserving the synthetic mismatch. Compose was run with a temporary image-tag override and random environment secrets; no secrets are retained here. Ruff lint/format passed. The original frozen release still verifies and remains disqualified; no pointer was activated. Hosted CI is unverified. Next: the user runs the GPU container smoke command in `docs/release_benchmark.md`, followed by full validation-only parity/load if the smoke passes.
