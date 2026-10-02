# Two-minute demonstration

Use the real CPU service for the live portion and saved GPU evidence for the
model comparison. No cloud, GPU inference, retraining or test rerun is needed
during the presentation. The examples are authored illustrations, not a reviewed
challenge set. Do not claim every unfamiliar phrase will be rejected.

## Rehearse

With the original local baseline bundle available:

```powershell
.\.venv\Scripts\python.exe scripts/run_demo.py --output artifacts/demo-rehearsal-v1
```

For another developer's fresh reproduction:

```console
uv run --locked python scripts/run_demo.py --bundle artifacts/reproduction-v1/bundle --policy artifacts/reproduction-v1/policy/policy.json --output artifacts/demo-rehearsal-v1
```

Use a fresh output name each time. The runner starts a real localhost HTTP
service on an available port, supplies temporary API/metrics keys, submits three
requests, checks unauthorized/invalid input and private metrics, writes the
responses and stops only its own process, including on failure. It does not
print credentials, activate a release or alter thresholds. Inspect `demo.json`,
`service.log` and `metrics.txt`; do not substitute invented responses if it fails.

## Presentation script

| Time | Show | Say |
| --- | --- | --- |
| 0:00-0:20 | README result and architecture diagram | "This recommends an intent route or human review. It performs no business action. We compared a CPU baseline, a prompted model and a fine-tuned version of the same model." |
| 0:20-0:55 | Run the live demo and read its three responses | "This is the real CPU API. A familiar request can route; unfamiliar or instruction-like requests may go to review. These examples are illustrative, not accuracy evidence." |
| 0:55-1:15 | Demo checks and `/metrics` output | "Invalid requests and missing authentication fail explicitly. Metrics use a separate key. Logs omit submitted text. Human review recommends escalation; it does not create a ticket." |
| 1:15-1:40 | Model card's final-test table | "Fine tuning improved supported macro-F1 to 0.9568. But the selected candidate reviewed only 89.6% of out-of-scope test requests, missing our fixed 90% requirement. No candidate was approved for automatic release." |
| 1:40-2:00 | Container load table and setup guide | "The verified GPU service completed 500 serial requests, with a 1.117-second p95, but mostly rejected concurrent load. Cloud remains unverified. Another developer can reproduce the CPU path with the pinned lock and inspect all retained evidence." |

Keep the README, [model card](model_card.md), [architecture](architecture.md) and
[container evidence](../reports/milestone5-container-gpu-full-v1/README.md) open
before presenting. The two minutes is a speaking plan; no video recording or
timed human rehearsal is claimed. API startup depends on the local machine.

If asked about vLLM: its full validation experiment ran successfully, but five
C outputs changed, including three routed labels. It was not adopted and has
no measured warmed load benchmark. If asked about costs: the hosting figure is
an explicitly assumed demand/price scenario, not an AWS bill or demonstrated
saving. If asked about real customers: none were involved.
