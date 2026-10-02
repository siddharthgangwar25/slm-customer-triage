# Full paired vLLM validation experiment

Completed and audited on 2 October 2026. **6,200/6,200 requests completed**, with
zero inference failures. This completes the separate Section 9 backend
comparison. It does **not** establish exact parity or approve a backend change.
Keep the verified Transformers service as the existing implementation.

| Validation result | B: Transformers | B: vLLM | C: Transformers | C: vLLM |
| --- | ---: | ---: | ---: | ---: |
| Supported macro-F1 | 0.272377 | 0.269796 | 0.966217 | 0.966535 |
| Supported accuracy | 22.4000% | 22.1333% | 96.5333% | 96.5667% |
| Invalid outputs | 654 | 632 | 1 | 1 |
| Frozen-threshold routes | 0 | 0 | 2,491 | 2,491 |
| Routed errors | 0 | 0 | 56 | 54 |
| Coverage | 0% | 0% | 80.3548% | 80.3548% |
| Routing error | Undefined | Undefined | 2.2481% | 2.1678% |
| Oos review recall | 100%, review all | 100%, review all | 90/100 | 90/100 |

Each candidate used the same **3,100 validation examples**: 3,000 supported and
100 oos. Policy numbers apply the original thresholds to saved gate scores;
these are offline diagnostics, not a newly selected policy or a live gateway
load test. All 6,200 prompts reached the model; this run did not measure gate
short-circuit savings. B's disabled automatic routing is not a successful
routing outcome.

## Output parity and changed errors

| Candidate | Exact generation matches | Raw differences | Output-token differences | Input-token / truncation differences |
| --- | ---: | ---: | ---: | ---: |
| B | **3,005/3,100 (96.9355%)** | 95 | 72 | 0 / 0 |
| C | **3,095/3,100 (99.8387%)** | 5 | 3 | 0 / 0 |

Both candidates therefore fail exact generation parity. B has 82 changed parsed
labels, but its frozen review-all policy produces identical responses. C has
five changed labels and three changed routed responses. No C request changed
between route and review, so unchanged coverage alone would conceal the changes.

| True label | Transformers label | vLLM label | Frozen-policy effect |
| --- | --- | --- | --- |
| replacement_card_duration | replacement_card_duration | expiration_date | Both reviewed by low gate score; raw correctness regresses |
| user_name | change_user_name | user_name | Routed error corrected |
| min_payment | bill_due | min_payment | Routed error corrected |
| text | change_ai_name | change_user_name | Both incorrect and reviewed by low gate score |
| oos | meaning_of_life | what_song | Both incorrectly routed, to different labels |

All changed examples, raw outputs and policy decisions are retained in
`B-output-changes.json` and `C-output-changes.json`. Two corrected routes do not
justify a post-test backend substitution or a claim of statistically established
improvement. The original C final-test **89.6%** oos-recall disqualification remains
unchanged. No test examples were rerun and no threshold was retuned.

## Execution conditions

- Image `sha256:a34b46ec3436ed26dfbe32970bb10f2aa945d0adee6e2aea00353887203c3115`,
  matching the smoke and the currently installed image.
- Variant `vllm-b1ac55ec8d4cf3c0`; its identity differs from the initial smoke
  because the comparison runner's parser/inference error accounting was fixed.
  The worker, dependency lock, base weights and adapter are unchanged.
- vLLM 0.10.2, PyTorch 2.8.0+cu128, Transformers 4.57.6, bitsandbytes 0.50.2,
  PEFT 0.17.1, Qwen3-0.6B pinned revision and checkpoint-944 LoRA.
- Linux/WSL2, GTX 1650 4 GiB, NF4/FP16, V1 FlexAttention/Punica, greedy
  non-thinking output, prefix cache, one sequence, eager execution. Logs again
  show chunked prefill enabled despite the requested disabled flag.
- B comparison: **2,415.43 seconds (40.26 minutes)**; C: **2,976.11 seconds
  (49.60 minutes)**. Combined comparisons: **89.86 minutes**. These include
  first-use work and report generation, not a warmed serving benchmark.
- Worker load **58.09 seconds**; host-observed HTTP readiness **80.56 seconds**,
  separately measured. About **91.2 minutes** including readiness and both
  comparisons, excluding earlier preflight work.
- Whole-GPU snapshots before/after: **3,823 / 3,726 MiB**, including other
  processes; not process-specific memory or workload peaks.
- Auth rejection passed; the temporary container was removed. A fresh Docker
  query confirmed no `triage-vllm-*` containers remained.

## Audit and scope decision

The CPU auditor verified all **14 original file checksums**, all 6,200 canonical
validation joins, outputs, token/EOS/truncation accounting, parsing, classification
and frozen-threshold metrics. Additional analysis checked original text, labels,
split, gate score, source index and dataset version against both references,
then enumerated every changed policy response. All four experiment-file hashes
match the executed code. Cached base-model file hashes match the smoke.
The frozen release still loads successfully and its release/test-ledger hashes
are unchanged. Derived analysis is in `comparison_analysis.json`; original run
records remain unmodified. `retained_checksums.json` covers the retained report.

Executed audit command:

```powershell
.\.venv\Scripts\python.exe scripts/verify_vllm_experiment.py --input artifacts/vllm-validation-v1 --output artifacts/vllm-validation-audit-v1
```

The vLLM compatibility/validation experiment is complete and the alternative is
**not adopted**. No additional GPU run is needed to close this comparison.
A vLLM gateway integration, warmed 500-request concurrency/load benchmark and
backend-specific cost report were not performed. The existing Transformers
benchmark remains the serving-performance evidence; no vLLM speedup, production
readiness or new final-test result is claimed. Future adoption would require
those separate checks. Milestone 6 remains the next unimplemented milestone.
