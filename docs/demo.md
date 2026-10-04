# Local API demonstration

Run the CPU service with three example requests and inspect its responses.
The examples demonstrate the API; they are not an independent evaluation set.
No GPU or cloud service is required.

## Run the demo

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
responses and stops its own process, including on failure. Inspect `demo.json`,
`service.log` and `metrics.txt` for results. Credentials are generated temporarily
and are not printed. The demo does not change the release or routing thresholds.

For the model comparison, see the [model card](model_card.md) and
[serving results](../reports/milestone5-container-gpu-full-v1/README.md).
