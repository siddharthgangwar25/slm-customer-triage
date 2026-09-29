# Completed local QLoRA run and acceptance audit

The user ran `scripts/run_milestone4.ps1` on 28–29 September 2026. The assistant verified the resulting artifacts on 29 September using:

```powershell
.\.venv\Scripts\python.exe scripts/verify_milestone4.py --output artifacts/milestone4-acceptance-v1
```

The audit is CPU-only: no model execution or test records. It verifies the current source/config/prompt/data/lock identity, all ten saved training checkpoints plus the final adapter (11 bundles), complete finite training logs, final/selected adapter equality, all canonical validation joins, recomputed raw metrics and exact policies, paired B/C inference settings, and byte-identical recomputation of three-way reports/bootstrap/error files.

The run trained **15,100 examples in one epoch, 944 optimizer steps**, with **10,092,544** trainable LoRA parameters on the pinned **596,049,920-parameter** Qwen3-0.6B base. Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`. Rank 16, alpha 32, dropout 0.05, learning rate 1e-4, accumulation 16, seed 42, NF4 base with explicit FP32 training compute. Validation inference uses NF4/FP16 for both B and C.

Training took **42,072.27 seconds (11.69 hours)**, within the 24-hour ceiling. Result elapsed scope including loading/save/reload is 42,080.26 seconds. Mean trainer loss is **0.0586019859**; all 944 logged losses/gradient norms are finite. Saved/reloaded probe logits differ by **0.0**. The selected epoch checkpoint is `checkpoint-944`; its adapter weights exactly match the final adapter used in that reload check. Full C validation separately loaded the checkpoint in a fresh process.

GPU: GTX 1650, approximately 4 GiB dedicated VRAM. Peak PyTorch allocated memory is **2,375,820,800 bytes (2.21 GiB)**; peak reserved memory is **5,135,925,248 bytes (4.78 GiB)**. Reservation exceeds physical VRAM, so these counters must not be presented as proof the entire job stayed within dedicated memory; Windows may use shared memory and the run did not independently measure paging. Process RSS at completion is 1,135,386,624 bytes, not a peak measurement.

Retained files contain original result, run identity, hardware, collator audit, selected checkpoint manifest, checkpoint selection and acceptance verification. `training_log.jsonl` preserves every parsed log record with normalized LF JSON serialization. Large adapter/optimizer/tokenizer files remain in ignored `artifacts/`, referenced by the saved bundle hashes; copying this repository alone does not copy those large artifacts. Do not discard them if you need to resume or serve the selected model.

Genuine held-out results and limitations are in [the three-way report](../three-way-qwen3-06b-v1/report.md) and [reviewed error analysis](../three-way-qwen3-06b-v1/error_analysis.md). These are validation-selected results, with one epoch/seed. Test evaluation and serving release are deferred to Milestone 5; paid cloud remains unused.
