# Customer Request Triage Project Specification

Implementation plan and coding handoff for Siddharth Gangwar

Version 1.1 | 24 September 2026 | Status: proposed implementation specification

## 1 Objective and project boundaries

Build and evaluate a customer-request triage service that assigns a supported intent to an English message or recommends human review. Compare a classical classifier, a prompted small language model, and the same language model after parameter-efficient fine-tuning. Deploy the selected approach behind an API and measure quality, latency, throughput, and operating cost.

The central question is how much request volume can be handled automatically at an explicitly measured error rate. Fine-tuning is an experiment, not a predetermined winner. The deployed model may be the classical baseline if it offers the best measured trade-off.

### Career objective

Demonstrate skills that complement Siddharth's existing RAG, computer-vision, research, and MCP experience: PyTorch, Transformers, model adaptation, reproducible experiments, FastAPI, testing, inference serving, CI, and deployment. The deliverable is a portfolio engineering project; it must not be described as production customer usage without actual evidence.

### Required scope

- English, short, single-intent requests using the full CLINC150 benchmark.
- A fixed taxonomy of 150 supported intents plus the dataset's out-of-scope label, oos.
- Three reproducible experiments and a shared evaluation implementation.
- A configurable human-review policy, validated JSON responses, and versioned artifacts.
- CPU baseline, optional GPU training environment, local container deployment, and a cloud deployment runbook.
- Benchmark report, failure analysis, model card, data card, and a short demonstration.

### Deferred scope

Multilingual input, multi-intent decomposition, entity extraction, response generation, RAG, MCP integration, real ticket creation, a staffed review queue, Kubernetes, and reinforcement learning are extensions. The initial service recommends routing; it performs no customer-account actions. The first interface can be FastAPI's API documentation. A custom frontend is optional.

### Default assumptions

Work at your own pace and progress through the milestones when their completion criteria are met. Start locally on CPU. Hardware, operating system, GPU access, and budget are unknown. Keep GPU dependencies optional. Cloud spending is disabled by default until an actual budget and deployment authorization are supplied.

## 2 Data specification and provenance

Use the original CLINC150 repository [1], files data/data_full.json and data/domains.json. Pin a repository commit and record SHA256 checksums before processing. Retain the dataset license and attribution. Do not silently switch to a repackaged variant.

| Original split | Supported requests | Out of scope requests |
| --- | --- | --- |
| Train | 15000 | 100 |
| Validation | 3000 | 100 |
| Test | 4500 | 1000 |

These counts describe the documented full variant. Check downloaded contents against them and report discrepancies. The source keys include train, val, test, oos_train, oos_val, and oos_test. Combine corresponding supported and oos records into canonical splits without changing membership. CLINC150 covers English single-intent requests across domains; it is not a hospitality dataset [1].

### Canonical record

Each processed JSONL record contains sample_id, text, label, domain, split, source_index, and dataset_version. Use a deterministic ID from source version, split, and record index. Domain is metadata derived from the source mapping; it must never be supplied as an input feature at inference. Preserve raw text. Convert only the oos label consistently; retain supported intent names unchanged.

Create a manifest recording URLs, commit, hashes, counts, class frequencies, text-length distribution, preprocessing version, and source license. Check missing or empty text, unknown labels, conflicting labels, and exact or normalized duplicates within and across splits. Fail on corrupt records. Flag cross-split duplicates in the report; preserve the official benchmark and optionally report a separately named deduplicated sensitivity analysis. Never silently alter the test set.

### Split discipline

Fit text transformations and models on training data only. Use validation data for prompts, hyperparameters, checkpoint choice, and thresholds. Keep test labels out of all training and tuning code paths. Do not train on validation data for this project. Freeze configuration hashes before the final test run. A later change requires a new experiment version and disclosure of prior test exposure.

The 100 training oos examples are limited and cover only some unsupported requests. Report this limitation. The benchmark is public and old enough that language-model pretraining contamination cannot be ruled out. Its test score is benchmark evidence, not a guarantee of novel customer performance.

### Optional fresh challenge set

After the main pipeline works, write and independently review at least 100 fresh requests covering supported intents, unfamiliar topics, ambiguity, typos, and instruction-like text. Keep this set held out and report it separately. Hospitality requests absent from the taxonomy should be oos. A hospitality-specific taxonomy requires its own training, validation, and test data. Synthetic data must be labeled as synthetic and must not replace independent evaluation.

## 3 Experimental design and model selection

Implement three model adapters behind a common prediction interface. Every adapter returns a raw predicted label, parse status, model version, and timing metadata. Offline predictions also retain sample_id, token counts when applicable, and error type. Attach the frozen baseline's gate_score to every prediction by sample_id, and verify a one-to-one join before policy evaluation.

### Experiment A Classical baseline

Use a scikit-learn Pipeline containing TfidfVectorizer and LogisticRegression. Fit on the 150 supported training classes. Start with word unigrams and bigrams, sublinear term frequency, seed 42, C=1.0, and a sufficiently large iteration limit. Treat these as starting settings. Log convergence warnings and compare at most C values 0.1, 1.0, and 10.0 on validation data. An embedding classifier is a later ablation if error analysis justifies it.

The maximum predicted class probability becomes a gate score. It is a ranking signal, not a calibrated probability of being correct [6]. Keep the vectorizer and classifier in one saved pipeline. Load serialized artifacts only from trusted project-controlled locations.

### Experiment B Prompted small language model

Use Qwen/Qwen3-4B as the default candidate [2], with a pinned model revision and non-thinking mode. A smaller Qwen3 model may be used for hardware smoke tests; results must retain that model identity. If the final model changes, both prompted and fine-tuned experiments must use the same replacement model and revision.

The system prompt defines the task, supplies the fixed sorted intent catalog, permits oos, and asks for one JSON object containing only intent. Validate this exact schema and reject extra fields or surrounding prose; do not repair malformed output silently. User text is placed in a separate user message. The catalog comes from training labels and public domain mapping, not held-out examples. Start zero-shot; any few-shot variant must draw examples from training data only and receive its own experiment ID.

### Experiment C Fine tuned small language model

Fine-tune the exact model used in B using supervised examples and LoRA or QLoRA [3, 4]. Include the supported training data and oos_train. Use the same catalog, chat formatting, output schema, and evaluation decoder as B. Keep label semantics unchanged. Do not train the model to generate confidence scores or chain-of-thought explanations.

### Fair comparison rules

Report raw classification quality and policy-based routing quality separately. Raw evaluation runs every valid input through each model without a gate. Production-policy evaluation includes the shared gate described in section 6. Count malformed responses, unknown labels, and truncated generations as failures. Do not discard difficult cases. Record model precision, prompt version, decoding configuration, hardware, and serving backend for every result.

## 4 Fine tuning implementation

Use PyTorch, Hugging Face Transformers, TRL SFTTrainer, PEFT, and an appropriate quantization backend. TRL supports prompt-completion data and adapter training [3]. Keep training and serving dependency environments separate where their PyTorch or CUDA requirements conflict. Choose compatible versions through a smoke test, then lock them. This document does not prescribe unverified package-version combinations.

### Training data format

Use conversational prompt-completion records. The prompt contains a system message with the taxonomy and a user message containing the request. The completion is a single assistant JSON object, for example {"intent":"balance"}, using a label that exists in the pinned catalog. Train on completion tokens only. Verify the token mask directly: system and user tokens must not contribute to the supervised completion loss. Match non-thinking chat-template behavior at training and inference.

### Initial experiment settings

| Setting | Initial value or decision |
| --- | --- |
| Adaptation | QLoRA with a 4 bit base on a compatible GPU |
| LoRA | Rank 16, alpha 32, dropout 0.05 |
| Target layers | Supported linear layers, verified against the chosen model |
| Learning rate | 0.0001 as an initial setting |
| Training duration | Up to 3 epochs; retain each epoch checkpoint |
| Microbatch | 1; use accumulation toward effective batch 16 |
| Random seed | 42; repeat the selected setup if budget permits |
| Precision | BF16 if supported; explicitly record any fallback |
| Sequence length | Compute from tokenized examples and available memory |

These are starting points, not promised optimal settings. Use gradient checkpointing when needed. Measure token lengths for the full 150-label prompt plus completion before training. Fail on would-be truncation of the message or target. Choose a sequence limit that fits the data; if a long input is rejected, report it. Do not silently drop intent labels to fit memory.

### Execution sequence

First validate tokenization and loss masking on 10 examples. Then run a small training smoke test and verify adapter save and reload. Launch the full run only after these pass. Track training loss, validation classification metrics, GPU memory, elapsed time, and examples processed. Select the checkpoint using validation raw macro-F1, breaking ties by fewer invalid outputs and then earlier checkpoint.

Save the adapter, tokenizer information, base-model revision, configuration, environment lock, prompt hash, seed, and data manifest hash together. Run a reload parity check. Limit initial tuning to three clearly documented adapter configurations. Save checkpoints before ending a rented GPU session. Do not assume the QLoRA training representation can be served unchanged: validate supported adapter loading or export a separate serving artifact, then re-evaluate it.

## 5 Evaluation protocol and success criteria

The evaluator must produce metrics.json, predictions.jsonl, a per-class report, a confusion matrix, coverage versus error plots, and a Markdown report from saved predictions. Include sample counts and dataset mix with every table. No metric may depend on dropping parse failures or timeouts.

| Metric | Exact definition |
| --- | --- |
| Raw supported macro F1 | Unweighted mean F1 over the 150 supported labels on supported samples before gating; oos or invalid predictions count as misses |
| Out of scope recall | True oos requests assigned human_review divided by all true oos requests |
| Routing coverage | Requests assigned route divided by all evaluated requests |
| Routing error | Routed requests with the wrong intent, including any routed true oos request, divided by routed requests |
| Supported review rate | Supported requests assigned human_review divided by all supported requests |
| Invalid output rate | Malformed, unknown-label, or truncated model outputs divided by model calls |
| p95 latency | 95th percentile of end-to-end request durations for the stated workload |

When no requests are routed, routing error is undefined, not zero. Report a null value and zero coverage. Separate infrastructure failures from valid human-review outcomes; report their rate and include them in the overall request accounting. For raw offline classification, an inference failure counts as an incorrect prediction.

### Choosing a routing operating point

For each candidate model, sweep the shared gate threshold over its validation scores. Select the threshold giving highest coverage subject to validation routing error at most 5 percent and out-of-scope recall at least 90 percent. These are proposed operating targets, not measured achievements. Break ties by lower routing error, then higher threshold. A candidate also needs at least 20 percent coverage to qualify for automatic routing. If none qualifies, label the targets unmet and retain human-review-only demonstration mode. Do not weaken the constraints silently.

Select the deployment candidate using validation results before final testing. Among qualifying candidates, prefer higher coverage; if within 1 percentage point, prefer lower measured cost per request, then lower p95 latency. Save this decision and its configuration hash. Evaluate all three frozen candidates on test data once for the report. Test findings can disqualify an automatic-routing release, but must not become a hidden tuning loop.

Report 95 percent Wilson intervals for routing error and out-of-scope recall. Use paired bootstrap intervals for model differences in macro-F1 when feasible. Validation has limited oos examples, so threshold estimates may be unstable. Add selected-class error examples and a separate fresh-set result if available. A model that does not improve the baseline still constitutes a valid experiment.

## 6 Routing policy and service architecture

Keep the system small: a data pipeline creates datasets; experiment runners create model bundles and prediction files; a model adapter provides predictions; a policy module decides routing; FastAPI exposes that decision; telemetry records operational measurements.

### Shared gate design

For version 1, all production candidates use the gate score from Experiment A's frozen classifier. This avoids presenting an LLM's written confidence as a probability and makes escalation implementable without model-specific log-probability infrastructure. Each candidate has its own validation-selected threshold against this same score. The gate can limit LLM coverage; disclose this limitation and compare ungated predictions separately.

At inference, validate the request, compute the gate score, and return human_review if the score is below threshold. Otherwise obtain the selected model's label. A supported label produces route; oos produces human_review. Malformed or unknown model output produces human_review with invalid_model_output. Baseline inference reuses its gate computation. A timeout or unavailable model produces HTTP 503, not a fictional completed review. A later policy may add calibrated model-specific scores as a separately evaluated experiment.

The gate makes final outcomes dependent on the baseline classifier. Do not claim that this policy measures independent LLM uncertainty. In offline policy evaluation, reuse each model's saved raw predictions and simulate gating; in load tests, actually short-circuit rejected requests so latency reflects the deployed behavior.

### API contract

POST /v1/triage accepts a JSON object with text and optional client_request_id. Trim text and reject whitespace-only input. Initially limit text to 2000 characters, then apply a token-budget check that accounts for the complete prompt. Return HTTP 422 for invalid or oversized input. Model selection is server configuration, not an arbitrary client-supplied model path.

Successful responses contain request_id, intent, decision, reason, model_version, policy_version, and latency_ms. decision is route or human_review. intent is a supported label only when routed; otherwise it is null. reason is supported_intent, low_gate_score, out_of_scope, or invalid_model_output. These are machine-readable codes, not generated explanations. Do not expose a calibrated confidence field in version 1.

GET /health/live checks the API process; GET /health/ready checks the selected model and policy bundle. GET /v1/model returns non-sensitive version metadata. Keep /metrics private. Use 401 for failed authentication, 429 for configured rate limits, and 503 for unavailable or timed-out inference. Provide a standard error schema. A human_review decision recommends escalation; it does not claim that a human has received a ticket.

## 7 Repository layout and command contracts

Use a Python package with CLI entry point triage, a pyproject.toml, a committed dependency lock, and editable development installation. Prefer Python 3.11 initially, adjusting only for verified dependency compatibility. Group GPU training dependencies separately from CPU serving dependencies. Avoid import-time model downloads and GPU initialization.

| Path | Responsibility |
| --- | --- |
| src/triage/data/ | Download, verification, canonical records and manifests |
| src/triage/models/ | Classical, prompted and adapter-based model implementations |
| src/triage/training/ | SFT preparation, training and export |
| src/triage/evaluation/ | Metrics, bootstrap, policy selection and report generation |
| src/triage/service/ | API schemas, request handling and readiness |
| src/triage/policy.py | Shared gate and decision rules |
| configs/ and prompts/ | Versioned settings and model prompts |
| tests/ | Unit, integration, contract and smoke tests |
| data/ and artifacts/ | Ignored local data and generated model bundles |
| reports/ and docs/ | Benchmark results, model card, runbooks and decisions |
| deployment/ | Containers, local compose and optional cloud definitions |

Keep notebooks optional for exploration. Core training, evaluation, and inference must run from commands. The CLI contracts below are to be implemented; they are not claims of existing commands.

```text
triage data prepare --config configs/data.yaml
triage train baseline --config configs/baseline.yaml
triage predict --config configs/baseline.yaml --split val
triage evaluate --predictions <path> --output <directory>
triage policy select --predictions <path> --split val
triage train slm --config configs/qlora.yaml
triage predict --config configs/prompted.yaml --split val
triage predict --config configs/finetuned.yaml --split val
triage release freeze --config configs/release.yaml
triage benchmark final --release <manifest>
triage serve --bundle <trusted-bundle-path>
```

All commands accept explicit output locations and fail clearly on missing inputs. The final benchmark command requires a frozen release manifest, writes a timestamped report, and records test use. Refuse to overwrite an existing run directory. Offline evaluation must work from saved predictions without loading model weights. Use fixture data and fake adapters in CI; mark all such outputs as test fixtures.

## 8 Testing and continuous integration

Write tests for data integrity, evaluation correctness, routing behavior, and deployment contracts. These tests protect the evidence and externally visible behavior. Avoid tests that simply repeat internal implementation details.

### Required tests

- Dataset schema and counts for fixtures; checksum and split manifest validation for downloaded data. Fit operations must reject split=test and split=val.
- Label-catalog validation and deterministic preprocessing. Test-only text must not enter the vectorizer vocabulary through evaluation.
- Hand-calculated examples for macro-F1 accounting, coverage, routing error, oos recall, and confidence intervals, including zero routes and missing classes.
- Policy boundaries: below threshold, equal to threshold, valid intent, oos, malformed JSON, unknown label, and disabled automatic routing.
- API contracts: valid request, empty input, oversized body or token budget, invalid authentication, timeout, unavailable model, and readiness transitions.
- Prompt tests: user content cannot replace the system taxonomy; instruction-like text remains data. The service performs no generated commands or tool calls.
- Artifact reload parity: predictions from a saved baseline match the in-memory model. A selected adapter can be saved, reloaded, and used.
- Metric and log tests: raw request text and secrets are absent from default telemetry; metric labels have bounded cardinality.

### CI pipeline

On every pull request, install locked CPU dependencies; run formatting or lint checks, type checks on core contracts, and the fast unit suite. Run an API integration test with a fake adapter and a tiny baseline fixture. Build the CPU container and verify health endpoints. Do not download large models, require a GPU, or call paid APIs in normal CI.

Add a manually invoked GPU smoke workflow later. It checks tokenization, training, save/reload, and inference on a tiny fixture, with an explicit job time limit. Real benchmark runs remain separate from CI and preserve their manifests. Pin workflow dependencies or image digests where practical.

### Development reporting

After each milestone, Codex must report changed files, commands actually run, tests passed or failed, unresolved limitations, and the next milestone. If a GPU test cannot run, report it as unverified. Never replace an unavailable model with a mock and describe the result as a real benchmark. Persist progress in docs/implementation_status.md so another session can resume without guessing.

## 9 Serving deployment and observability

Begin with a single CPU FastAPI service loading the baseline bundle. Add GPU-backed inference as a separate process when the SLM is ready. Use Transformers for the initial offline reference path, then evaluate a supported vLLM serving path [5]. Validate the current vLLM documentation and chosen model support during implementation; do not assume library APIs or adapter combinations are stable.

### Release bundle

A release manifest identifies model type, model and tokenizer revisions, adapter or weight hashes, gate pipeline hash, selected threshold, taxonomy and prompt hashes, policy version, data manifest hash, dependency lock hash, source commit, and reference validation report. Stage the new bundle, run contract and parity checks, and switch the configured bundle atomically or by restarting the service. Keep the previous known-working bundle for rollback.

Compare reference and serving-backend outputs on the same validation examples before release. Quantization, adapter merging, constrained decoding, or a backend change creates a new evaluated variant. If JSON-constrained decoding is introduced, apply it consistently to B and C and disclose it in the report.

### Local and cloud environments

Provide a CPU container and an optional GPU compose profile. GPU inference listens on an internal network; only the API gateway is exposed. Bind to localhost for development. For cloud deployment, provide an AWS runbook using S3 for artifacts, ECR for images, and a suitable EC2 host. Define restricted access, HTTPS termination, secret injection, health checks, and cleanup. Provision infrastructure only after instance choice, region, and budget are confirmed. Terraform is optional; Kubernetes is out of scope.

### Operational measurements

Record request count, outcome and error counts, p50/p95/p99 latency, inference duration, gate rejection rate, invalid-output rate, in-flight requests, and model version. Track GPU memory and utilization when available. Use JSON logs and Prometheus-compatible metrics; add a simple dashboard only after metrics work. Do not log raw text by default. Operational distributions can indicate drift, but actual accuracy monitoring requires reviewed labels. Do not claim production quality from unlabeled telemetry.

### Performance and cost benchmark

Warm up the service, then run at least 500 requests for each feasible concurrency level such as 1, 4, and 8. Record input mix, token lengths, hardware, batching, precision, backend, timeouts, completed throughput, and cold-start time separately. Include all failed requests in accounting. Measure actual gate-short-circuit behavior. Estimate cost per 1000 submitted requests from dated regional prices, runtime, idle time, and supporting resources. Report training cost separately; use assumed demand explicitly when amortizing it. Never claim a cost saving from throughput alone.

## 10 Implementation milestones

Progress by acceptance criteria at your own pace. Complete each milestone's usable slice before adding the next infrastructure layer. Record your progress so you can pause and resume whenever needed.

### Milestone 1 Reproducible CPU baseline

Create the package, CLI, dependency lock, data downloader, manifests, baseline training, validation prediction, and core metric tests. Produce a validation report and inspect 30 to 50 errors. Completion: a fresh environment can prepare data, train, save, reload, predict, and evaluate using documented commands. No GPU dependency is required.

### Milestone 2 Routing policy and API

Implement threshold selection, frozen policy configuration, API schemas, error behavior, health checks, and request tests. Completion: routed and review responses match the contract; validation constraints and unmet-target behavior are reported correctly; missing models do not produce invented responses.

### Milestone 3 Prompted model benchmark

Inspect hardware, choose and pin a feasible model, build the prompt catalog, verify token budget and non-thinking behavior, and save validation predictions. Completion: every validation request has a result or explicit failure; raw and policy metrics compare with A. Track runtime and memory. Keep test data unused.

### Milestone 4 Fine tuning and analysis

Build completion-only SFT data, test masking, run adapter training, select a checkpoint using validation data, and evaluate C. Completion: training and adapter reload are reproducible, all three candidate reports exist, and the report explains which errors changed. Optional additional seeds or an embedding baseline must resolve a specific uncertainty.

### Milestone 5 Serving and release benchmark

Build containers, introduce the serving backend, check parity, add telemetry, load tests, release manifests, and rollback. Select the deployment candidate using validation only. Freeze all three candidates and produce the final test report. Completion: quality, coverage, latency, throughput, and cost are reported with clear conditions and no fabricated results.

### Milestone 6 Demonstration and handoff

Finish the model card, data card, architecture explanation, error analysis, setup instructions, deployment runbook, and two-minute demo. Add the fresh challenge set if feasible. Perform an authorized short cloud deployment and teardown if budget permits. Completion: another developer can reproduce the CPU baseline, inspect the SLM experiment evidence, run the service, and understand limitations. If cloud execution was omitted, label the cloud path unverified.

### Learning checkpoints

Siddharth should be able to explain why TF-IDF is a useful baseline, what LoRA changes, how completion masking works, why public test data can mislead, how threshold selection trades coverage against error, and how deployment changes affect model behavior. Keep a short experiment journal recording hypotheses and decisions in plain language.

## 11 Risks decisions and final acceptance

### Decisions already made

Use CLINC150 full official splits, English single-intent scope, a CPU classifier baseline, a shared gate for version 1, and a paired prompted versus fine-tuned model comparison. Start with Qwen3-4B subject to hardware validation. Use an API-first implementation and keep model benchmarking separate from business-system actions.

### Decisions to resolve when needed

Before GPU training, record operating system, GPU model and VRAM, CUDA compatibility, chosen model revision, and maximum training budget. Before cloud execution, record region, host, access method, session duration, and cleanup plan. These unknowns must not block the CPU milestone. Default all paid provisioning to disabled.

### Main risks and responses

- Hardware limits: run small smoke tests, reduce microbatch, use accumulation and checkpointing, or choose a smaller paired model. Preserve experiment identity and disclose substitutions.
- Fine-tuning adds no value: report the result, inspect failure categories, and select the better measured approach. Do not keep searching the test set for wins.
- Gate rejects useful LLM predictions: publish ungated and gated results and the shared-gate limitation. A model-specific gate is future work.
- Benchmark is too clean: add a separately reported fresh challenge set and avoid claims about real customer traffic.
- Overengineering: complete the local service before optional dashboards, cloud infrastructure, or additional models.
- Privacy and operational scope: use public or independently authored data; do not upload employer messages without permission. No live reservation or financial actions are part of this project.

### Definition of done

The project is complete when all required milestone artifacts exist; the CPU path is reproducible; all three model experiments have genuine results; the API contract and policy tests pass; and the final report states which approach won under which conditions. A hardware-blocked SLM experiment is an incomplete research deliverable, even if the CPU service is usable.

The final repository includes a README, environment locks, configurations, data provenance, evaluation code, model and data cards, release metadata, benchmark tables and plots, error examples, CI, local deployment instructions, and a demonstration. Large weights and raw data are excluded from Git and referenced through manifests. Distinguish implemented, executed, and verified features.

Resume evidence should report actual model size, data volume, held-out metrics, coverage at a measured routing error, deployment environment, latency, and cost under a named workload. Do not write target numbers as achieved results or claim real users when only a demo exists.

## 12 Instructions for the coding assistant

The Markdown version can be placed in the repository as PROJECT_SPEC.md. This specification describes the desired project; its example command names are interfaces to implement.

```text
Implement the customer-request triage project described in
PROJECT_SPEC.md. Start with Milestone 1 and produce a working,
reproducible CPU baseline before adding GPU or deployment work.

First inspect the repository and applicable AGENTS.md instructions.
Preserve unrelated files. If starting fresh, create the package,
configuration files, CLI, and meaningful tests described in the spec.
Use ordinary engineering judgment for reversible choices. Record
material deviations in docs/decisions.md.

Keep training, validation, and test responsibilities separate.
Do not tune on the test set, silently replace the dataset, fabricate
metrics, or label mocked runs as genuine experiments. Keep paid
services disabled until a budget and authorization are supplied.

Keep the CPU path independent of optional GPU imports. Verify current
library APIs and dependency compatibility, then commit a lockfile.
Use small fixtures for CI and real data for benchmark evidence.

Run the milestone acceptance checks. Report changed files, commands
actually executed, test outcomes, measured results, and blockers.
Update docs/implementation_status.md with the next concrete task.
Explain key ML decisions so I can understand and defend the project.
```

### Source references

Primary documentation checked on 24 September 2026. These sources ground the dataset and tool capabilities; implementation must verify the installed versions. Project targets, architecture, and initial hyperparameters are proposed design decisions, not external facts or achieved results.

1. CLINC original dataset and evaluation repository. https://github.com/clinc/oos-eval
2. Qwen3 4B model card and usage. https://huggingface.co/Qwen/Qwen3-4B
3. Hugging Face TRL SFTTrainer. https://huggingface.co/docs/trl/sft_trainer
4. Hugging Face PEFT quantization guide. https://huggingface.co/docs/peft/developer_guides/quantization
5. vLLM documentation and serving reference. https://docs.vllm.ai/
6. Scikit-learn probability calibration. https://scikit-learn.org/stable/modules/calibration.html

### First action

Create the repository and implement Milestone 1. The first reviewable output is a reproducible validation baseline report with saved predictions, data provenance, and clear error analysis.
