# Milestone 5 serving and release runbook

The full native workflow and CPU/GPU container acceptance completed on 1 October 2026 and were audited from saved evidence. See [full GPU container results](../reports/milestone5-container-gpu-full-v1/README.md): 3,100 exact parity matches and 500/500 serial HTTP completions. The [original final report](../reports/milestone5-final-v1/README.md) still disqualifies C because its fixed-policy oos review recall is 89.6%, below 90%. **Do not repeat the completed test experiment or tune its thresholds.** The local Transformers/NF4/PEFT path is verified; Section 9's [vLLM smoke](../reports/vllm-smoke-v1/README.md) is executed and audited; full vLLM validation and hosted CI remain unexecuted. The commands below are completed-run history, not outstanding terminal tasks.

## Completed full GPU container validation

The six-output smoke and subsequent full run passed. The full run used the built images:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage gpu --full --skip-build --output artifacts/milestone5-container-gpu-full-v1
```

This checked all 3,100 validation outputs and 500 HTTP submissions per concurrency level. It skipped image builds and did not use the test set. Its saved results have been audited and retained. No additional run is needed for the existing Transformers delivery.

## Completed GPU smoke workflow

Keep Docker Desktop running. From the project root, without activating a venv:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage gpu --output artifacts/milestone5-container-gpu-smoke-v1
```

This builds both images from committed locks, starts an authenticated worker on an internal Docker network, publishes only the gateway on a random localhost port, waits for the worker before starting the gateway, checks API contracts, compares six real validation outputs and sends 24 HTTP requests at each concurrency. The initial GPU image downloads several gigabytes of dependencies. Build progress is written to `build-cpu.log` / `build-gpu.log` inside the output directory; runtime progress appears in the terminal. It creates random secrets in child environments and removes only its own containers/networks on exit. Model/report mounts are read-only; new parity evidence has a separate writable mount. It does not train, activate a deployment or access test records.

The smoke passed; the longer validation-only command is:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage gpu --full --skip-build --output artifacts/milestone5-container-gpu-full-v1
```

This performed 3,100 raw comparisons and 500 HTTP requests at concurrency 1/4/8. The Linux backend demonstrated full parity. A future backend difference must be retained, not used to justify changing thresholds or repeating final testing. Use a fresh output directory for any separately justified validation attempt; this wrapper does not resume interrupted runs. Both smoke and full acceptance are now verified.

CPU reproduction, including a Linux-generated synthetic fixture and every genuine baseline validation decision:

```powershell
.\.venv\Scripts\python.exe scripts/run_container_validation.py --stage cpu --output artifacts/milestone5-container-cpu-repeat
```

## Run the long workflow

From the repository root in PowerShell, no environment activation is needed:

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone5.py
```

The script owns a GPU worker on localhost:8011 and a CPU gateway on localhost:8010. It generates per-run authentication secrets in child-process environments, checks port availability, records logs/resources, and terminates only its own processes when it finishes or receives Ctrl+C. Other running applications are untouched. It uses cached model files with model-hub networking disabled.

The stages are:

1. Compare all **3,100 validation** raw outputs/token counts from the worker against frozen C reference predictions. Partial parity can resume from its checksummed prefix. Any mismatch blocks freeze.
2. Warm up the actual HTTP service, then submit **500 requests at each concurrency 1, 4 and 8**. Sampling uses seed 42, with replacement, and a validation-like supported/oos mix. The same workload runs at each concurrency. Every failure is counted, no retries are hidden, completed/all-request latency distributions are separate, and successful decisions are checked against the saved policy. Private metric deltas show actual gate short-circuiting.
3. Freeze all three candidates and select C using validation coverage. Its 80.35% coverage exceeds A's 68.42% by more than the 1-percentage-point tie margin, so no cost tie-break is needed. The release checks source/config/prompt/data/weights/tokenizer/lock/policy hashes and requires full serving evidence. The source must be committed before freezing.
4. Record test use **before opening test records**, then evaluate A, B and C on the official **4,500 supported + 1,000 oos test requests** using frozen validation thresholds. No training, checkpoint selection, prompt tuning or threshold sweep for selection happens here. Report raw quality, fixed-policy metrics, Wilson/bootstrap intervals and the separate operational/cost evidence.

Expect several hours; the actual duration depends on the machine. The initial smoke is not a reliable throughput guarantee. The script does not train again, provision paid resources, or activate a deployment automatically.

Individual stages:

```powershell
.\.venv\Scripts\python.exe scripts/run_milestone5.py --stage serving
.\.venv\Scripts\python.exe scripts/run_milestone5.py --stage freeze
.\.venv\Scripts\python.exe scripts/run_milestone5.py --stage final
```

Stop on any failure and inspect the saved log. Re-running the same script resumes unfinished parity and final predictions with unchanged source/configuration. A load test is not resumable mid-workload: preserve an interrupted load directory for inspection rather than mixing sessions or overwriting outcomes. Complete runs are reused; freeze independently rejects evidence from different source hashes. Do not edit source, locks, prompt, configuration or model files during a pending run.

The final output has a UTC timestamp, recorded in `artifacts/milestone5/release/test_use.json`. That ledger and `artifacts/test-use-registry/` prevent accidental duplicate test experiments. `active.lock` prevents concurrent final writers. Ctrl+C removes that lock; a hard process kill may leave it behind. Confirm the owning run is stopped before recovering a stale lock. Do not delete test-use records to restart. A new experiment must disclose all prior exposure through `prior_test_exposure`, use a new release identity/output, and must not become a hidden test-tuning loop.

Expected outputs under `artifacts/milestone5/`: `parity/`, `load/` (all outcomes), `cost.json`, GPU samples, worker resource metadata, and `release/release.json`. The ledger points to `artifacts/final-<UTC>/`, containing all three prediction files/reports and `final_report.json`. Ask Codex to audit these and update status after completion.

## Service design and known capacity

The gateway validates schema/length, requests a CPU token count from the worker, computes A's frozen gate, and only calls GPU generation if the gate accepts. Baseline output is never reused as the SLM classification. The worker loads C's exact evaluated NF4/FP16 adapter representation, greedy decoder and complete non-thinking catalog prompt. Request cache copies are independent. The same source can serve paired B by pointing at B's reference manifest; a backend/precision/export change requires a new full parity/validation run before release.

This first worker and gateway are **single-flight, batch size 1**, with immediate 503 busy responses rather than an unbounded queue. The smoke showed heavy rejection at concurrency 4/8. Those levels measure overload behavior, not useful capacity. The report includes completed throughput and completed-request percentiles so fast failures cannot masquerade as performance improvement. No service SLO or cost saving is claimed. Timeout/unavailable responses are failures, never completed reviews.

`/metrics` is absent unless `TRIAGE_METRICS_KEY` is set; when enabled it requires its own bearer key and is excluded from OpenAPI. Metrics use bounded names/model identities and contain counts, request/inference histograms, gate rejections, model calls, invalid outputs and in-flight work. Logs contain status, timing, model version and outcome/reason; no raw request, client ID, headers or arbitrary URL. The runner polls whole-GPU utilization/memory every two seconds via nvidia-smi when available, explicitly including unrelated desktop use. Worker allocator measurements are separate. Counters reset on restart; scrape before shutdown.

## Freeze, activation and rollback

The selected model is chosen on validation before test. A test failure of the original quality constraints disqualifies automatic release; it does not authorize retuning. The final report states the disqualification. Activation refuses a missing/incomplete/disqualified final report. Complete container/backend/API acceptance before using activation; this command changes only a local atomic pointer, then you restart the service deliberately:

```powershell
.\.venv\Scripts\triage.exe release activate --release artifacts/milestone5/release/release.json
# Start the matching authenticated worker, then start the gateway with the same env secrets:
.\.venv\Scripts\triage.exe serve --active artifacts/active-release.json --worker-url http://127.0.0.1:8001 --config configs/service-slm.yaml
```

Keep the previous known-working source/environment and model bundle. Roll back the pointer atomically, stop the affected service and restart with the previous source/environment:

```powershell
.\.venv\Scripts\triage.exe release rollback
.\.venv\Scripts\triage.exe serve --active artifacts/active-release.json --config configs/service.yaml
```

The first activation records the known baseline configuration as its previous state. Later activations retain the previous pointer. A pointer is not hot reload, an artifact signature or permission to load an untrusted pickle. Release paths are trusted project-controlled files.

## Docker compose and remaining GPU acceptance

Dockerfiles pin the official Python image by immutable digest and install committed locks; CPU/GPU environments remain separate. Large data/models are mounted read-only, not baked into images. The worker joins only the internal `private` network; the gateway also joins `frontend` so Docker can publish its localhost port. An internal-only network omitted port mappings on the tested Docker engine. Set three distinct secret environment values in your terminal without putting them in source control. Use Docker Desktop's Linux engine with NVIDIA support for the GPU profile:

```powershell
docker compose -f deployment/compose.yaml --profile cpu build
docker compose -f deployment/compose.yaml --profile cpu up -d
curl.exe --fail http://127.0.0.1:8000/health/ready
docker compose -f deployment/compose.yaml --profile cpu down
# GPU profile: worker has no published port; only gateway is bound to localhost.
docker compose -f deployment/compose.yaml --profile gpu build
docker compose -f deployment/compose.yaml --profile gpu up -d
curl.exe --fail http://127.0.0.1:8000/health/ready
docker compose -f deployment/compose.yaml --profile gpu down
```

CPU image/Compose checks and full GPU worker/gateway parity/load now pass locally; hosted CI remains unverified. The full GPU run uses the runner's equivalent worker/gateway topology; the GPU Compose profile itself has not been separately executed. Future backend changes require validation. None of these infrastructure checks changes the completed release's test disqualification.

## vLLM feasibility and cost scope

Rechecked on 1 October: the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) documents Linux/WSL and NVIDIA compute capability 7.5 or higher, and the [supported-model table](https://docs.vllm.ai/en/latest/models/supported_models/) lists Qwen3. The [BitsAndBytes guide](https://docs.vllm.ai/en/stable/features/quantization/bnb/) describes that quantization path. These are feasibility signals, not proof that this exact adapter, memory budget and decoder work together. Docker/WSL2 is now available, so the earlier native-Windows limitation no longer rules out a trial. The separate [vLLM 0.10.2 smoke](../reports/vllm-smoke-v1/README.md) subsequently completed on 1 October: 12/12 requests, B 5/6 and C 6/6 exact generation matches after auditing parser/inference error accounting. Full validation and vLLM load testing remain pending; see the [vLLM runbook](vllm_experiment.md). No weight conversion, adapter merging or constrained decoder was introduced.

`deployment/pricing/ec2-us-east-1-2026-09-29.json` records an official AWS quote retrieved 2026-09-29 (feed publication 2026-09-25): Linux On-Demand g4dn.xlarge **$0.526/hour**, us-east-1. The cost report is an explicit scenario using local measured active duration at that hourly price, **not measured EC2 throughput or a bill**. It assumes 100,000 submitted requests/month, 730 billed hours and a separately labeled $10/month allowance for storage/network/logging/registry support. It reports idle time, hosting per 1,000 submitted, active-only compute, training separately and one-month training amortization. Cloud training duration is also unmeasured. Local electricity/hardware cost was not measured. Price quote, workload assumptions and exclusions remain visible; no paid resource was used.
