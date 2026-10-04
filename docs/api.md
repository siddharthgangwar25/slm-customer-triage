# Triage API

The gateway serves the CPU baseline or a separate private GPU worker using the same routing contract. Run from the repository root with the locked CPU environment. No model download occurs at import or request time.

## Policy selection

```console
uv run --locked triage policy select --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --split val --output reports/local-policy
```

This command verifies prediction hashes, one-to-one sample IDs, split identity, and baseline-gate identity. It sweeps every observed score, processing score ties together, plus zero and an all-review endpoint. It maximizes coverage subject to routing error <=5%, oos recall >=90%, and coverage >=20%. Ties prefer lower error and then higher threshold. A score equal to threshold passes.

If `--output` is omitted, the policy output directory is `artifacts/policy-v1`; the same overwrite protection applies.

`policy.json` records the selected threshold, validation counts and Wilson intervals, constraints, model/gate versions, taxonomy hash, pipeline hash, data-manifest hash, source prediction hashes, and a content-derived configuration hash/policy version. `threshold_sweep.json` preserves every candidate. Existing output directories are never overwritten. The service verifies the policy and model match before becoming ready; edit neither file in place. Changed data, predictions, or models require a newly identified selection run.

When no threshold qualifies, the command writes `status=targets_unmet`, `automatic_routing=false`, zero coverage, and null routing error. Valid requests then receive `human_review` with `reason=low_gate_score`. The response uses the four documented reason codes; `/v1/model` explicitly exposes that automatic routing is disabled. Disabled routing still requires the configured bundle to load successfully. No test split may be used for selection.

The measured baseline threshold is `0.2088413160728636`: 2,121 of 3,100 requests routed; 101 of those routes were wrong; 90 of 100 oos requests were reviewed. Routing error is 4.76% (Wilson 95%: 3.93–5.75%); oos recall is 90% (82.56–94.48%). Selection uses the specified point estimates, not interval bounds. The upper error interval exceeds 5%; this is not a guarantee of satisfying the targets on unseen traffic.

## Start and call the server

```console
uv run --locked triage serve --bundle artifacts/baseline-c1-v1 --policy reports/local-policy/policy.json --config configs/service.yaml
```

Default address: `127.0.0.1:8000`. `--host` changes the binding address, and `--port` changes the port. Docker uses `0.0.0.0` inside the container, with Compose publishing the gateway on host loopback. If `--policy` is omitted, the service looks for `policy.json` within the trusted bundle directory. A bad bundle leaves the process live but not ready; fix the configuration and restart. There is no reload or administration endpoint.

For optional authentication, set `TRIAGE_API_KEY` in the process environment before starting the server. Send `Authorization: Bearer <your-key>` to `/v1/triage` and `/v1/model`; Swagger's Authorize control supports this. Authentication is disabled when the variable is absent or empty. Health endpoints remain available without credentials. Do not put keys in YAML or commit them.

Example request:

```json
{"text":"what is my account balance","client_request_id":"local-demo-1"}
```

Successful response schema (values are illustrative, not a saved benchmark response):

```json
{
  "request_id":"server-generated-uuid",
  "intent":"balance",
  "decision":"route",
  "reason":"supported_intent",
  "model_version":"baseline-version",
  "policy_version":"policy-version",
  "latency_ms":1.2
}
```

For review, `intent` is null and `reason` is `low_gate_score`, `out_of_scope`, or `invalid_model_output`. `client_request_id` is accepted but neither logged nor used for idempotency; every request receives a fresh server UUID, also sent in `X-Request-ID`. Supported intent names come only from the frozen catalog. Model paths cannot be supplied by clients. Instruction-like text is input data and triggers no commands or tools.

Error response schema:

```json
{
  "request_id":"server-generated-uuid",
  "error":{"code":"model_unavailable","message":"Model or policy bundle is unavailable"}
}
```

| HTTP status | Conditions |
| --- | --- |
| 200 | Completed route or review decision; live/ready health check; version metadata |
| 401 | Missing/incorrect bearer token when authentication is configured |
| 422 | Malformed JSON, extra fields, non-string/empty text, text >2,000 characters, body >configured byte limit, or model token budget exceeded |
| 429 | Configured per-process triage rate limit; includes `Retry-After` |
| 503 | Missing/corrupt/incompatible model or policy, unavailable model, full inference slot, or inference deadline exceeded |
| 404 | Unsupported endpoint, including `/metrics` when no metrics key is configured |

The character cap applies to submitted text; valid text is trimmed before token counting and inference. The body cap also checks chunked bodies without trusting `Content-Length`. Pydantic errors are converted to the standard envelope without echoing submitted content. `/v1/model` returns only versions, model type, catalog size, automatic-routing state, and the fixture marker.

## Execution limits and telemetry

One bounded worker thread serves CPU inference. Concurrent work beyond its single slot receives `503 model_busy`; there is no unbounded queue. A baseline request computes class probabilities once and reuses the label returned with its gate score. A generic adapter below threshold is not called for a final prediction. Disabled routing skips gate/model calls after input validation and readiness checks.

The inference deadline includes token counting, gate computation, and model prediction. Python cannot terminate a running native sklearn operation when its awaiting request times out. The slot remains occupied and readiness is false until that operation finishes; no replacement work is queued. A late inference failure marks the runtime unhealthy until restart. A permanently stuck native operation requires stopping the process. The GPU path now uses a separate authenticated Transformers worker; measured concurrent load still mostly fails busy. See the serving report.

Rate limiting is global to this single local process, not distributed or per-account. Health checks are exempt. No remote deployment or multi-worker coordination is claimed. `latency_ms` measures the server handler path from body receipt to decision construction; it excludes network transfer and final response transmission.

Default application logs contain event name, HTTP status, duration, model version and outcome/reason; startup failures log an exception class, not messages or paths. Raw text, client request IDs, authentication headers, query strings, and secrets are not logged. Uvicorn access logging is disabled by the CLI. With `TRIAGE_METRICS_KEY`, `/metrics` exposes private Prometheus-compatible counters and bounded timing summaries. It requires its own bearer key; the API key does not grant metrics access. Without that configuration the endpoint is absent. See [architecture](architecture.md) and [deployment](deployment.md).

## Re-run API/offline parity

```console
uv run --locked python scripts/verify_api.py --bundle artifacts/baseline-c1-v1 --policy reports/local-policy/policy.json --predictions artifacts/baseline-c1-v1-val/predictions.jsonl --output reports/local-api-parity
```

The script calls the real loaded baseline through the ASGI API for every saved validation record and compares intent, decision, and reason. It disables rate limiting for this check and refuses test predictions or a policy bound to different predictions. It needs development dependencies. The retained run had zero mismatches on 3,100 records. This is correctness evidence, not network performance or throughput evidence.
