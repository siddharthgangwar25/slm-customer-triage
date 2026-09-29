# Qwen3-0.6B CLINC150 QLoRA adapter

## Identity and purpose

Research candidate C for English, single-intent request classification into 150 CLINC150 intents or `oos`. The raw adapter returns an intent JSON object; the separate frozen baseline gate recommends route or human review. It does not execute business actions. It is not yet deployed by the project's API.

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, 596,049,920 parameters.
- Adapter: `artifacts/finetuned-qwen3-06b-qlora-v1/checkpoint-944`; model version `finetuned-qwen3-06b-qlora-v1-step944-bfe54e26195f`.
- Provenance and individual artifact checksums: [selected bundle manifest](../reports/finetuning-run-v1/selected_bundle.json). Large model/tokenizer/optimizer files remain local and ignored by Git.
- Validation policy: `policy-v1-2fe7bafe54be`; threshold 0.11417805060646902 against A's unchanged gate. A gate score is not calibrated LLM correctness confidence.

## Data and training

Official CLINC150 full source commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`; 15,000 supported and 100 oos training examples. No validation examples enter fitting. Test predictions remain unused. Original data attribution and CC BY 3.0 license are retained with the [baseline data report](../reports/baseline-c1-v1-val/CLINC_LICENSE.txt).

Completion-only conversational SFT, all 150 labels in the system prompt, JSON+EOS targets, explicit Qwen non-thinking behavior. All records fit within the 1,024-token training limit without truncation. Ten masks and the installed TRL collator were inspected before training. QLoRA rank 16, alpha 32, dropout 0.05 on attention/feed-forward linear projections, learning rate 1e-4, microbatch 1, accumulation 16, seed 42, one epoch/944 steps. This is one configuration and seed, with a single eligible epoch checkpoint.

Windows GTX 1650 4 GiB; NF4 base and FP32 training compute because native BF16 is unavailable and the prior 4B FP16 attempt had non-finite gradients. Training took 11.69 hours. Training and inference precision are explicitly different: both paired B and C validation use NF4/FP16. The adapter was evaluated in that inference representation, with exact shared prompt/decoder/gate. Training/serving locks remain separate.

## Measured validation behavior

On 3,000 supported plus 100 oos requests, C achieves 0.966217 supported macro-F1 and 96.5333% supported accuracy. There is one invalid output and no infrastructure failure. Its selected policy routes 2,491 requests, makes 56 routed errors, and reviews 90/100 oos requests. Policy point estimates meet the original constraints; they are validation-selected and not guaranteed on future data.

The paired prompted base scores 0.272377 macro-F1 and the CPU baseline 0.882634. Raw oos prediction falls from 67/100 correct for prompted B to 55/100 for C. The shared gate accounts for the stronger policy review recall. See the [complete comparison](../reports/three-way-qwen3-06b-v1/report.md) and [40-example review](../reports/three-way-qwen3-06b-v1/error_analysis.md).

## Limits and release status

Only 100 training oos examples, public benchmark contamination risk, known source duplicates, ambiguous labels, one seed/epoch, and validation-based checkpoint/threshold selection limit generalization. The adapter can invent unknown labels and overassign supported intents to unsupported requests. Strict parsing and the frozen policy must be retained; human review is a recommendation, not a submitted ticket.

C has not passed service-backend/export parity, API load testing, final frozen test evaluation, cost measurement or a deployment release. Its offline p95 serial latency is 1,306.89 ms, excluding loading/prefix prefill/gate/API overhead. A remains the current serving adapter. Revalidate any changed model representation, decoding or prompt under Milestone 5 before release.
