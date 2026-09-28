# Milestone 4 prerequisite evidence

Recorded 2026-09-28 on the local Windows GTX 1650. **These are hardware/training smoke checks, not complete validation benchmark results.** Original CLINC150 source/license attribution is retained in `reports/baseline-c1-v1-val/`.

- `data_manifest.json`: all 15,100 training examples, immutable inputs, tokenizer hashes and token lengths.
- `mask_audit.json`: ten directly inspected token/label arrays; `-100` excludes system/user tokens from loss. Includes longest request and oos.
- `collator_audit.json`: exact agreement with the installed TRL collator.
- `hardware.json`, `training_log.jsonl`, `training_result.json`: real 16-example QLoRA update, finite loss/gradients, memory, time, changed weights and zero-difference saved adapter reload.
- `prompted_cache_parity.json`: six predetermined cached/uncached comparisons on the smaller paired model.
- `inference/`: fresh-process adapter NF4/FP16 inference, all three outcomes including two invalid completions. The manifest marks these incomplete smoke predictions.
- `cpu_fixture_verification.json`: explicitly synthetic random tiny Qwen model; verifies padding with EOS and exact optimizer/RNG checkpoint resume using the installed TRL/PEFT stack.

Large weights, optimizer checkpoints and full prepared training records stay in ignored `artifacts/`. Their hashes/provenance are retained in the manifests. The smoke's adapter is not eligible for full C evaluation. Full training and B/C validation are pending the user's terminal run; see `docs/finetuning.md`.
