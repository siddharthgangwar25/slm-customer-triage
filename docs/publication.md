# GitHub and CV publication checklist

The six milestone deliverables are complete locally. This page separates local
checks from steps that require a GitHub repository. It does not approve the
failed automatic-routing release.

## Local preparation

- Section 8's [manual GPU fixture](gpu_smoke.md) and opt-in workflow are added.
  The workflow requires a separately configured GPU runner; normal CI is CPU-only.
- The README CPU reproduction is checked in a separate local Git clone with a
  new environment, dependency cache and newly downloaded pinned dataset. The
  system's existing Python 3.11 and uv executables bootstrap the new environment;
  no old model, data or environment is copied into the checkout.
- `.gitignore` excludes local environment secrets, AWS credentials and private
  key files in addition to data, artifacts and environments. Ignore rules do not
  remove files already present in history.
- A basic scan of reachable Git blobs found no common credential patterns;
  inspect changes before publishing. This is not a comprehensive security audit.
- Licensing is **undecided at the owner's request**. No root code license is
  added. Existing dataset/model attribution remains unchanged.
- The [status page](implementation_status.md) links to a separate historical
  log. Older handoff commands are not current tasks.

Retained check results and exact limitations are in
[publication evidence](../reports/publication-readiness-v1/README.md).

## GitHub steps

1. Review and commit the publication changes locally.
2. Create an empty GitHub repository. Leave generated README, license and
   `.gitignore` options unchecked because this project already has history and
   those files are managed locally. Choose the intended visibility explicitly.
3. Add that repository URL as `origin` and push the chosen branch. The actual
   URL and branch must be supplied/checked before issuing those commands.
4. In **Actions**, inspect **CPU checks**. Both Windows/Linux checks and the CPU
   container job must pass before marking hosted CI verified. Record the run URL
   and commit in the status log. Local tests do not substitute for this evidence.
5. The GPU workflow remains separately opt-in. A local fixture pass is useful
   evidence even without a hosted runner; label hosted execution unverified
   until it is actually run. Follow the [GPU guide](gpu_smoke.md) if needed.
6. Add a concise repository description and topics; link the repo from your CV.
   Rehearse the [two-minute demo](demo.md). A recording is optional.

No remote repository is created, code uploaded, runner registered or paid
resource provisioned by these local preparation changes.

## Claims supported by the evidence

- Fine-tuned Qwen3-0.6B using QLoRA on CLINC150, improving supported-intent test
  macro-F1 from 0.8866 to 0.9568 over TF-IDF on a 4 GB GPU.
- Built a Dockerized FastAPI service with Pydantic validation, authentication,
  a separate GPU worker, health checks and Prometheus metrics; verified 3,100
  serving outputs and measured 1.12-second p95 over 500 serial requests.

The model's 89.6% out-of-scope review recall missed the fixed 90% criterion.
Keep this result visible. Do not claim production approval, scalable concurrent
serving, cloud deployment, real customer accuracy or a vLLM speedup. The test
split is already consumed; publication does not justify another test run.

Optional future work remains actual GPU Compose execution, an independently
reviewed fresh challenge set and separately authorized cloud deployment. These
are not prerequisites for presenting the completed local project.
