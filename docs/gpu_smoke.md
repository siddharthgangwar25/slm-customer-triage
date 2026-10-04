# Manual GPU integration check

Section 8's optional-to-run GPU CI is defined in
[gpu-smoke.yml](../.github/workflows/gpu-smoke.yml). Normal push/PR CI remains
CPU-only. This workflow does not provision hardware, call paid APIs, download a
pretrained model, or read benchmark splits. Installing the locked training
dependencies can still download several gigabytes on a new runner.

The fixture constructs a tiny random Qwen3 model, a local tokenizer and two
synthetic sequences. It verifies tokenization and tokenizer reload, completion
masking (including EOS when EOS equals padding), two real CUDA optimizer steps,
finite gradients/loss and changed LoRA parameters. It then saves/reloads the
base and adapter and compares logits and greedy generation. It fails without
CUDA and refuses an existing output directory. FP32 LoRA exercises the training
stack; this is not a quantized 0.6B model check or quality/performance benchmark.
Original QLoRA benchmark and smoke evidence remain separate.

## Run locally

From the project root, using the already installed locked training environment:

```powershell
.\environments\training\.venv\Scripts\python.exe scripts/verify_gpu_fixture.py --output artifacts/gpu-fixture-manual-v1
```

Choose a fresh output path each time. The result is `verification.json`; other
outputs are tiny fixture weights and trainer records under the same directory.
The program does not start a service or change the release. Training has two
steps; initial Python/GPU imports can take longer than the training itself.

## Run through GitHub Actions

This is a deliberately opt-in workflow, not an instruction to register the
personal development laptop as a public-repository runner. Use an isolated,
disposable Linux x64 GPU runner with the custom label `triage-gpu`, a compatible
CUDA driver, and no personal credentials or benchmark artifacts. Configure a
runner only when needed; remove it after the job. GitHub documents the risks of
[self-hosted runners](https://docs.github.com/en/actions/reference/security/secure-use).

1. Review the workflow on the repository's default branch. Provisioning a paid
   machine requires a separate explicit budget; this project creates none.
2. With a suitable runner available, set the repository Actions variable
   `ENABLE_GPU_SMOKE` to `true`.
3. Select **Actions -> Manual GPU fixture -> Run workflow**, choosing the default
   branch. Other branches are skipped. Without the variable the job is skipped;
   without a matching runner an enabled job queues rather than using a CPU runner.
4. Inspect the step logs and JSON in the job summary. Missing result JSON is a
   failure, not a passing smoke. Keep the run URL/commit with the result.
5. Remove the disposable runner and unset `ENABLE_GPU_SMOKE` when finished.

The job has a 20-minute execution limit including setup, and the GPU step has a
five-minute limit. Actions are pinned to commits, token permissions are
read-only, checkout credentials are not retained, and model/data hub offline
flags apply throughout. No pull-request trigger can invoke this workflow as
written. See GitHub's [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
for manual dispatch and timeout behavior. Queue time is not a GPU execution
measurement. Local success does not establish that this hosted orchestration
has run; hosted execution must be recorded separately.
