# Milestone 5 native serving smoke — not a release benchmark

Executed 2026-09-29 on Windows/GTX 1650 using the completed C adapter. No real test records were accessed. The final command guard rejected this smoke as incomplete evidence. No deployment pointer was changed and no paid resource was created.

- Separate GPU worker and CPU gateway loaded successfully; combined measured startup-to-readiness was **17.14 seconds**.
- **Six** real validation raw outputs/token counts matched the offline reference exactly. This is a limited probe, not full 3,100-example serving parity.
- After warmup, **24** requests were submitted at each concurrency. At concurrency 1, **24/24** completed, all matched frozen offline policy decisions, **six** short-circuited at the gate and **18** invoked generation. Completed throughput **1.098/s**, p95 **1,473.88 ms** under this small workload.
- At concurrency 4 and 8, **1/24** completed at each level; **23/24** received explicit 503 busy failures. Every outcome is retained. This demonstrates single-flight overload, not high concurrent throughput. No retries masked failures.
- Private metric deltas, raw outcomes by sample ID, worker metadata/allocator counters and **32** whole-GPU utilization/memory samples are retained. GPU samples include unrelated desktop processes.
- `cost.json` is a labeled illustrative scenario using an official dated AWS us-east-1 g4dn.xlarge quote at $0.526/hour, 100,000 monthly submitted requests, 730 billed hours and an assumed $10/month support allowance. Hosting is $3.9398 per 1,000 submitted under those assumptions. It is **not measured cloud performance or cost**; active time here comes from this small local smoke. Training is separate.

The earlier development smoke also passed six worker comparisons and 24 serial requests, but this directory retains the later implementation's evidence. The source hash in parity/load matches the implementation at recording. No weight, prompt or decoder was altered. Full terminal parity/load/final-test runs, Docker build/runtime acceptance and the vLLM alternative remain pending/unverified. See `docs/release_benchmark.md` and `docs/implementation_status.md`.
