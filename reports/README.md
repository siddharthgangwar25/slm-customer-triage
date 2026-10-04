# Experiment results

Start with the final comparison for model quality and the container report for
serving performance. Each report states the dataset split, hardware and limits
of its measurements.

| Question | Report |
| --- | --- |
| How do A, B and C compare on the frozen test? | [Final test comparison](milestone5-final-v1/README.md) |
| Which errors changed after fine-tuning? | [Paired validation review](three-way-qwen3-06b-v1/error_analysis.md) |
| How was the adapter trained? | [Training run](finetuning-run-v1/README.md) |
| Does the GPU container match offline inference? How fast is it? | [Transformers parity and load tests](milestone5-container-gpu-full-v1/README.md) |
| Does the CPU Docker service work? | [CPU container acceptance](milestone5-container-cpu-v1/README.md) |
| What happened with vLLM? | [Separate backend comparison](vllm-validation-v1/README.md) |
| Can the CPU project be reproduced? | [Clean-checkout verification](publication-readiness-v1/README.md) |

Candidate directories contain saved predictions, metrics, per-intent CSVs,
confusion matrices and coverage/error plots. Earlier smoke runs, failure logs
and the original 4B experiment are historical evidence, not additional paired
candidates or current setup instructions.

Retained reports and their checksums are immutable records of the original runs.
Some historical per-candidate report headings incorrectly say "validation" for
test data; the explicit `split` field, sample counts and final comparison identify
the actual split. The original files are preserved rather than silently edited.
Current results and limitations are summarized in the [model card](../docs/model_card.md).

Historical reports also reference documentation that has since been consolidated
or removed. Demo and artifact instructions now live in [setup](../docs/setup.md);
the unexecuted AWS deployment plan was removed. Document paths and hashes in
the original audit records describe the repository at the time of each run.
