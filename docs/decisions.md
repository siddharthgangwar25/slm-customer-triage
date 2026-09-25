# Implementation decisions

## 2026-09-25 — Milestone 1

- **CPU first.** Python 3.11.16, uv 0.12.19, scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, and joblib 1.6.0 were installed and exercised together on Windows. `uv.lock` includes exact versions and distribution hashes. GPU packages are absent. The local Python installation was needed because PATH exposed only unusable Windows app aliases.
- **Baseline settings.** One `C=1.0` experiment, seed 42, word `(1,2)` n-grams, sublinear term frequency, default L2 regularization, `lbfgs`, and `max_iter=2000`. No C search has been conducted. This satisfies the initial setting without introducing validation-driven tuning. Native math libraries use one thread to avoid small-matrix oversubscription; this setting is recorded in the configuration.
- **Data pin.** Use the official repository commit `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Pin all three source hashes in `configs/data.yaml` before canonicalization. IDs hash the commit, original source key (including `oos_` where applicable), and zero-based source index; thus supported and oos source indices cannot collide.
- **Conflicting duplicates.** Five duplicate groups are present across official splits, with four label conflicts. These are source annotation findings rather than structurally corrupt records; report them and preserve membership as required. Invalid record structure, empty text, unknown labels, checksum mismatches, or count discrepancies fail preparation. Normalization is only for the duplicate audit, never a rewrite of raw text.
- **Held-out discipline.** Training reads only `train.jsonl`, catalog, and manifest. The catalog is checked against training labels and public domain mapping. Test records are parsed during data integrity preparation only; no test prediction or error analysis is permitted by the Milestone 1 CLI. The manifest contains aggregate test integrity metadata, never test input features.
- **Metric definitions.** Macro-F1 averages all 150 supported labels on supported validation records. Missing classes receive zero F1; oos, invalid outputs, and infrastructure failures are misses. Routing errors are undefined (`null`) at zero routes. Failure accounting distinguishes invalid outputs from infrastructure failures. No samples are silently removed.
- **Exploratory curve.** The report sweeps 101 evenly spaced thresholds plus an explicit all-review point for plotting only. It does not choose an operating point. Milestone 2 must implement the exact observed-score threshold sweep, constraints, tie breaks, and disabled-routing outcome. No target is claimed achieved in this milestone.
- **Error review.** Inspect the first 30 supported errors in official order and first 10 oos examples. This deterministic sample gives 40 concrete observations, but is biased toward early classes and must not be treated as a representative category-frequency estimate.
- **Fresh environment check.** A second virtual environment installed exclusively from the lock, then downloaded the pinned data, trained, saved/reloaded, predicted, and evaluated. The first exploratory training artifact was removed before the final acceptance run; configuration did not change and no test metrics were exposed.
- **Scope.** API, policy selection, containers, GPU dependencies, bootstrap model comparisons, and release-freeze commands remain in later milestones. The CI file covers the implemented CPU slice only. There is no material change to Milestone 1's dataset or model requirements.

Portability correction: generated JSON, JSONL, and CSV use explicit LF line endings. Git attributes preserve LF for evidence and lockfiles so Windows checkout normalization cannot invalidate hashes. A regression assertion covers serialized data and manifest bytes. Final evidence was regenerated after this correction.

## Verified primary API references

Checked on 2026-09-25 and confirmed by executing the installed library APIs:

- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Official CLINC dataset](https://github.com/clinc/oos-eval)
