# AWS deployment plan — not provisioned

Milestone 6 status, 2 October 2026: **cloud execution is omitted and unverified**.
No account, region, paid host, budget or session duration has been authorized.
No S3 bucket, ECR repository, EC2 instance, certificate or cloud secret was
created. The completed release is disqualified, so even an approved future
infrastructure smoke would be a research demo, not permission to bypass the
release guard. [Local deployment commands](../docs/deployment.md) are available.

Paid provisioning is disabled. The prices in `pricing/` are research quotes, not an authorization, region selection or budget. Before any launch, obtain the user's chosen AWS account/region, host, maximum spend/session duration and cleanup ownership. This runbook creates no resource automatically.

Candidate topology for a future approved experiment: one Linux g4dn.xlarge (T4 16 GiB) running a private worker plus CPU gateway. Actual fit, throughput, driver compatibility and costs must be measured there; local GTX 1650 results do not establish them. The local vLLM compatibility/validation experiment completed but changed outputs and was not adopted. This plan retains the verified Transformers path.

1. Use a dedicated private S3 prefix for versioned manifests/weights and a private ECR repository for immutable image digests. Upload only public benchmark/project artifacts; never customer messages or secrets. Enable S3 block-public-access, encryption and versioning. Verify downloaded artifact hashes against the frozen release.
2. Give the instance role only ECR pull permissions, S3 read access to that prefix, SSM session access and narrowly scoped log writes. Use a separate upload role. Avoid static access keys, public S3 policies and broad administrative instance permissions. Require IMDSv2. Account for EBS/ECR/S3/logging/network costs separately from compute.
3. Use SSM for administration, with no public SSH port. Permit HTTPS only from intended client/test networks. Keep the worker/metrics ports internal. Terminate HTTPS at a configured reverse proxy on the host, forwarding to the loopback gateway. Confirm hostname/certificate ownership first. Keep health endpoints scoped; do not expose the worker or metrics via the public proxy.
4. Inject separate API, worker and metrics secrets at runtime through approved secret storage or protected environment files; never bake them into images or commit them. Disable access logging of request bodies/query strings. Use the private metrics endpoint and bounded JSON telemetry.
5. Stage images and bundles without replacing the known-working pointer. Check liveness/readiness, authentication, invalid-request behavior, CPU gate short-circuit, complete validation parity, and the stated HTTP workload. Freeze image digests, dependencies, policies and measured cost assumptions. Do not use test outcomes to tune deployment parameters or silently replace the frozen model variant.
6. Switch the configured pointer only after acceptance; restart worker then gateway and verify `/v1/model` and `/health/ready`. Retain the previous digest/bundle/source/environment and restore it on failures. No traffic migration is authorized by this document.
7. Before the approved session ends, save required run/checkpoint manifests to the private prefix. Stop/terminate the instance, remove unneeded EBS snapshots/volumes, release unused public addresses, clean temporary ECR/S3 artifacts under the agreed retention policy and verify billing/resource inventory. Stopping compute does not end all storage/address charges. Record cleanup completion and actual billed cost; set budget alerts before launch.

Primary references: [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/), [AWS Price List documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/price-changes.html), and [G4 instance information](https://aws.amazon.com/ec2/instance-types/g4/). Recheck dated quotes and selected-region ancillary prices before provisioning.

## Execution record required for any future approved session

Record approved account ID/profile, region, AMI, instance type, maximum spend,
start/stop deadline, operator, cleanup owner, domain/certificate owner and artifact
retention decision. Resolve resource names and ARNs before creating anything.
Use immutable ECR image digests and exact manifests for the base cache, adapter,
baseline gate and policy. GPU images do not contain those large mounted weights;
include them in the approved private S3 artifact inventory. Do not upload an
entire working directory or include local credentials.

Use ECR password-stdin authentication rather than credentials in image layers;
the selected instance role needs scoped pull access. Use SSM instead of opening
SSH. Require IMDSv2, keep the worker unpublished, and terminate HTTPS with a
valid certificate at a restricted reverse proxy. Validate health, authentication,
invalid-input behavior and validation-only requests, retaining all failures.
Do not point the proxy at private metrics or the worker.

Record resource IDs at creation. Before the deadline, stop traffic, save approved
evidence and hashes, then terminate the named instance and inspect associated
volumes, snapshots, public IPs, images, objects and log retention. Record actual
retained/deleted resources and cost separately; do not equate instance shutdown
with zero remaining charges. The operator must verify completion in the approved
account/region. No teardown command was executed in this project because no cloud
resources were created.

Operational references checked 2 October 2026:
[ECR authentication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html),
[Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html),
[S3 Block Public Access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html),
and [EC2 instance metadata](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html).
These ground the plan; they are not proof of a deployed or tested AWS system.
