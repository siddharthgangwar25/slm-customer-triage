# Prompted model benchmark

Milestone 3 uses the pinned official Qwen3 model through Transformers. The completed configuration is Qwen/Qwen3-4B at revision `1cfa9a7208912126459214e8b04321603b3df60c`, with NF4 double quantization, FP16 compute and system-prefix caching on the local 4 GiB GTX 1650. All 3,100 validation outcomes and the [baseline comparison](../reports/baseline-prompted-v1/report.md) are retained. A software test suite or ten-request smoke alone is not a completed benchmark; see `implementation_status.md` for measurements and limitations.

## Isolated environment

The ordinary `uv sync --locked` environment remains CPU-only. Install the optional inference environment separately from the repository root:

```console
uv sync --project environments/prompted --locked --python 3.11
```

Its own `uv.lock` pins PyTorch 2.8.0 with CUDA 12.8, Transformers 4.57.6, Accelerate 1.15.0, bitsandbytes 0.50.2, and their dependencies. No API key, model inference provider, cloud machine, or payment is used. The first installation/model load downloads several GB. The cache lives in the ignored `artifacts/huggingface/` directory. The command examples below use Windows executable paths; on Linux use `environments/prompted/.venv/bin/triage` instead. Linux GPU execution is not claimed verified by Windows results.

```powershell
.\environments\prompted\.venv\Scripts\triage.exe predict --config configs/prompted.yaml --split val --limit 10 --output artifacts/prompted-smoke
.\environments\prompted\.venv\Scripts\triage.exe predict --config configs/prompted.yaml --split val
```

Run the full command only after the smoke demonstrates that the selected model actually loads, fits the complete prompt, generates non-thinking outputs, and is feasible on your hardware. No automatic model/precision/device fallback is allowed. A smaller smoke or a changed model must keep its actual identity and receive a separate configuration/experiment ID. If the paired research model changes, Milestone 4 must fine-tune that exact replacement model and revision.

## Prompt and decoding

`prompts/intent-v1.txt` supplies every supported label, sorted, plus `oos`. The catalog is verified against training labels and the pinned domain mapping. No validation examples enter the system prompt and no domain metadata is passed as a feature. The request is a separate user message, encoded as a JSON string. Angle brackets are escaped so user-supplied chat delimiter spellings cannot create additional messages; decoding the JSON recovers the unchanged request text. This is structural isolation, not a claim that a language model is immune to semantic prompt injection. The service performs no generated commands or account actions.

The runner explicitly calls `apply_chat_template(..., enable_thinking=False)` and checks the pinned template's empty thinking prefix. It tokenizes the complete catalog plus each validation message before loading weights. Any input that would exceed the configured budget causes the run to fail before inference; no catalog labels, user text, or target tokens are truncated. Generation reserves 32 additional tokens and uses deterministic greedy decoding, batch size one. Model defaults, precision, seed, revision, prompt/template hashes, and context limits are retained in metadata.

The optional `prefix_cache` setting precomputes only the fixed system message. Each request receives a deep copy of that KV cache, so request/response tokens cannot leak across requests. The full tokenized input must exactly match the cached prefix. `scripts/verify_prompt_cache.py --output reports/local-cache-smoke` runs a real cached/uncached parity probe on six predetermined validation examples; inspect its evidence before enabling caching on a different stack. Prefix caching can change floating-point execution, so a small parity probe is evidence for those samples, not a guarantee for every possible input. Cache use is part of the frozen configuration and timing scope.

Only the final EOS token is removed during decoding. Other special tokens, prose, duplicate JSON keys, unknown labels, and extra fields remain visible to the strict parser and count as invalid output. Generation that exhausts its budget without EOS is counted as truncated even if its partial text resembles JSON. There is no silent JSON repair and no model-generated confidence score.

## Evidence and resumption

Every attempted validation sample gets a raw output or explicit inference failure, plus sample ID, token counts, latency, model identity, and the frozen baseline gate score. The gate join requires exactly one match for every sample and identical source text, label, split, and dataset version. All requests reach the model before gating; policy evaluation reuses saved predictions afterward.

The output directory contains a frozen `run.json`, system prompt, model metadata, append-only predictions, and a checksum checkpoint after every record. Completed runs add a prediction manifest and a copied gate reference. Partial runs have no completed manifest and cannot be evaluated as a finished benchmark. `--limit` always marks a hardware smoke; evaluation and policy selection reject it.

```powershell
.\environments\prompted\.venv\Scripts\triage.exe predict --config configs/prompted.yaml --split val --resume
```

Resumption requires identical configuration, prompt, runner code, source data, gate predictions, dependency lock, and ordered partial sample IDs. Completed runs cannot be overwritten. If interrupted while a record/checkpoint is being written, an integrity mismatch fails safely; do not silently delete difficult cases or edit recorded predictions to force resumption. A new experiment/output directory is appropriate after a material change.

## Evaluate and compare

After full predictions exist, all analysis runs in the CPU environment without loading weights:

```console
uv run --locked triage evaluate --predictions artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --output reports/local-prompted-val
uv run --locked triage policy select --predictions artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --split val --output reports/local-prompted-policy
uv run --locked triage compare --baseline reports/baseline-c1-v1-val/predictions.jsonl --candidate artifacts/prompted-qwen3-4b-nf4-cache-v1-val/predictions.jsonl --output reports/local-baseline-prompted
```

The shared evaluator preserves invalid and failed requests in all accounting. Each candidate gets its own exact validation-selected threshold against the same baseline gate. A candidate that misses the specified constraints is reported with automatic routing disabled, zero coverage, and null routing error. Raw macro-F1 uses all 150 supported labels. A paired bootstrap estimates the macro-F1 difference on matched supported samples; Wilson intervals describe selected routing error and oos recall. These validation intervals do not eliminate selection bias or prove production quality.

Runtime metadata distinguishes model load time (including optional prefix prefill), summed per-request inference time, and the last process session's wall time. Cached inference timings exclude that one-time prefix prefill. Memory includes process RSS/Windows peak working set and peak CUDA allocated/reserved bytes; CUDA allocator figures do not include other applications or all driver overhead. CPU baseline and GPU prompted timings have different execution conditions and are not interchangeable with API latency, throughput, or cost.

The Milestone 2 API still loads only the baseline bundle. A prompted policy can be evaluated offline but cannot be substituted into that baseline service. GPU serving and backend parity belong to later milestones. Test prediction remains blocked.
