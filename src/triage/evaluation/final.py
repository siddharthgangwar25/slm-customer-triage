"""One frozen test experiment with append-only outcomes and unchanged validation policies."""

import gc
import os
import shutil
import sys
import time
from pathlib import Path

from triage.data.load import load_split
from triage.evaluation.bootstrap import paired_macro_f1
from triage.evaluation.metrics import classification, routing
from triage.evaluation.report import evaluate, load_predictions
from triage.io import (
    new_directory,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
)
from triage.models.prompting import system_prompt
from triage.policy import parse_model_json
from triage.release import claim_test_use, load_release
from triage.service.runtime import BaselineAdapter


def run(release_path, output, *, resume=False):
    release = load_release(release_path)  # Verify everything before reading any test record.
    output = Path(output)
    if not resume:
        new_directory(output)
    elif not output.is_dir():
        raise ValueError("Missing partial final run")
    lock = output / "active.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        ledger = claim_test_use(release_path, release, output, resume)
        frozen = {"release_sha256": release["release_sha256"]}
        if resume:
            if read_json(output / "run.json") != frozen:
                raise ValueError("Final resume identity mismatch")
        else:
            write_json(output / "run.json", frozen)
        rows, catalog, data = load_split(Path(release["config"]["data_dir"]), "test")
        for name in ("baseline", "prompted", "finetuned"):
            directory = output / name
            if (directory / "prediction_manifest.json").exists():
                _, _, completed = load_predictions(directory / "predictions.jsonl")
                if completed["release_sha256"] != release["release_sha256"]:
                    raise ValueError("Completed candidate belongs to another release")
                continue
            directory.mkdir(exist_ok=True)
            path = directory / "predictions.jsonl"
            saved = read_jsonl(path) if path.exists() else []
            if saved:
                verify_hash(path, read_json(directory / "progress.json")["predictions_sha256"])
            if [r["sample_id"] for r in saved] != [r["sample_id"] for r in rows[: len(saved)]]:
                raise ValueError("Partial final predictions are not an ordered prefix")
            meta = release["validation_manifests"][name]["model"]
            cfg = meta["config"]
            if name == "baseline":
                adapter = BaselineAdapter(
                    Path(release["config"]["baseline_bundle"]), release["policies"][name]
                )
                gates = None
            else:
                from triage.models.finetuned import FinetunedAdapter
                from triage.models.prompted import TransformersAdapter

                factory = FinetunedAdapter if name == "finetuned" else TransformersAdapter
                adapter = factory(
                    cfg, system_prompt(catalog, Path(cfg["prompt_file"])), [r["text"] for r in rows]
                )
                gates = {
                    r["sample_id"]: r for r in read_jsonl(output / "baseline" / "predictions.jsonl")
                }
            started = time.perf_counter()
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                for row in rows[len(saved) :]:
                    start = time.perf_counter()
                    if name == "baseline":
                        gate = adapter.gate(row["text"])
                        result = {
                            "predicted_label": gate.cached_output.label,
                            "parse_status": "ok",
                            "error_type": None,
                            "gate_score": gate.score,
                            "token_count": None,
                        }
                    else:
                        try:
                            raw = adapter.generate(row["text"])
                            parsed = parse_model_json(
                                raw["raw_output"], catalog, truncated=raw["truncated"]
                            )
                            result = {
                                **raw,
                                "predicted_label": parsed.label,
                                "parse_status": "ok" if parsed.valid else "invalid",
                                "error_type": raw["error_type"]
                                or (None if parsed.valid else "invalid_model_output"),
                                "token_count": raw["input_tokens"] + raw["output_tokens"],
                            }
                        except RuntimeError:
                            result = {
                                "predicted_label": None,
                                "parse_status": "error",
                                "error_type": "inference_error",
                                "raw_output": "",
                                "token_count": None,
                                "input_tokens": None,
                                "output_tokens": None,
                                "truncated": False,
                            }
                        result["gate_score"] = gates[row["sample_id"]]["gate_score"]
                    prediction = {
                        **row,
                        **result,
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "model_version": meta["model_version"],
                    }
                    import json

                    stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    saved.append(prediction)
                    write_json(
                        directory / "progress.json",
                        {"count": len(saved), "predictions_sha256": sha256(path)},
                    )
                    if len(saved) % 100 == 0:
                        print(
                            {"candidate": name, "completed": len(saved), "total": len(rows)},
                            flush=True,
                        )
            manifest = {
                "schema_version": 1,
                "split": "test",
                "sample_count": len(rows),
                "catalog": catalog,
                "sample_ids": [r["sample_id"] for r in rows],
                "fixture": data["fixture"],
                "benchmark_complete": True,
                "release_sha256": release["release_sha256"],
                "predictions_sha256": sha256(path),
                "dataset_version": data["commit"],
                "data_manifest_sha256": release["validation_manifests"][name][
                    "data_manifest_sha256"
                ],
                "model": meta,
                "gate_model_version": release["policies"][name]["gate_model_version"],
                "timing_scope": "serial raw model inference, no API or gate short circuit; "
                "excludes load/prefix prefill",
                "execution": {
                    "last_session_seconds": time.perf_counter() - started,
                    **(adapter.memory() if name != "baseline" else {}),
                },
            }
            if name != "baseline":
                source = output / "baseline" / "predictions.jsonl"
                shutil.copyfile(source, directory / "gate_predictions.jsonl")
                manifest["gate_reference"] = {
                    "predictions_sha256": sha256(source),
                    "source_predictions_sha256": sha256(source),
                }
                manifest["gate_model"] = release["validation_manifests"]["baseline"]["model"]
            write_json(directory / "prediction_manifest.json", manifest)
            del adapter
            gc.collect()
            if name != "baseline":
                if "torch" in sys.modules:
                    sys.modules["torch"].cuda.empty_cache()
        report = summarize(release, output)
        write_json(
            ledger,
            {
                **read_json(ledger),
                "status": "complete",
                "report_sha256": sha256(output / "final_report.json"),
            },
        )
        return report
    finally:
        lock.unlink()


def summarize(release, output):
    results, all_rows = {}, {}
    for name in ("baseline", "prompted", "finetuned"):
        path = output / name / "predictions.jsonl"
        rows, catalog, meta = load_predictions(path)
        all_rows[name] = rows
        metrics = classification(rows, catalog)
        policy = release["policies"][name]
        fixed = routing(rows, catalog, policy["threshold"])
        results[name] = {
            "raw": {k: v for k, v in metrics.items() if k != "per_class"},
            "fixed_policy": fixed,
            "policy_version": policy["policy_version"],
            "automatic_routing_frozen": policy["automatic_routing"],
        }
        if not (output / (name + "-report")).exists():
            evaluate(path, output / (name + "-report"))
    selected = results[release["selected"]]["fixed_policy"]
    passes = (
        selected["routing_error"] is not None
        and selected["routing_error"] <= 0.05
        and selected["oos_recall"] >= 0.90
        and selected["coverage"] >= 0.20
    )
    report = {
        "release_id": release["release_id"],
        "split": "test",
        "fixture": meta["fixture"],
        "operational": release.get("operational"),
        "selected_before_test": release["selected"],
        "candidates": results,
        "automatic_release_disqualified": not passes,
        "action": "Proceed only after serving/container acceptance"
        if passes
        else "Disable automatic release; no test-driven tuning",
        "c_minus_a": paired_macro_f1(all_rows["baseline"], all_rows["finetuned"], catalog),
        "c_minus_b": paired_macro_f1(all_rows["prompted"], all_rows["finetuned"], catalog),
    }
    write_json(output / "final_report.json", report)
    lines = [
        "# Frozen final test report",
        "",
        f"Release: {release['release_id']}. Selected on validation: {release['selected']}.",
        "",
        f"Test mix: {results['baseline']['raw']['supported_count']} supported / "
        f"{results['baseline']['raw']['oos_count']} oos. "
        "All failures retained; no thresholds retuned.",
        "TEST FIXTURE; not benchmark evidence." if meta["fixture"] else "Genuine frozen test run.",
        "",
        "| Candidate | Macro-F1 | Coverage | Routing error | Oos recall | Invalid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in results.items():
        raw, policy = item["raw"], item["fixed_policy"]
        lines.append(
            f"| {name} | {raw['raw_supported_macro_f1']:.6f} | {policy['coverage']:.6f} | "
            f"{policy['routing_error']} | {policy['oos_recall']} | {raw['invalid_output_count']} |"
        )
    lines += [
        "",
        report["action"],
        "",
        "Wilson intervals and paired bootstrap intervals are in final_report.json. "
        "The test mix differs from validation; coverage is not directly comparable without "
        "accounting for that mix. Public pretraining contamination and source duplicates remain "
        "limitations.",
        "",
    ]
    if release.get("operational"):
        operations = release["operational"]
        lines += [
            "## Serving workload",
            "",
            "All failed submissions are counted. "
            "Concurrent busy rejections are not successful throughput.",
            "",
            "| Concurrency | Submitted | Completed | Completed/s | All p95 ms | Completed p95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for measured in operations["load"]["runs"]:
            completed_p95 = (
                measured["latency_completed_ms"]["p95"]
                if measured["latency_completed_ms"]
                else None
            )
            lines.append(
                f"| {measured['concurrency']} | {measured['submitted']} | "
                f"{measured['completed']} | {measured['completed_per_second']:.3f} | "
                f"{measured['latency_all_ms']['p95']:.3f} | {completed_p95} |"
            )
        cost = operations["cost"]["scenario"]
        lines += [
            "",
            f"Cold start: {operations['load']['cold_start_seconds']} seconds.",
            "",
            f"Illustrative hosting: ${cost['hosting_usd_per_1000_submitted']:.4f}/1,000 "
            f"submitted at {cost['monthly_submitted_requests_assumed']} requests/month and "
            f"{cost['billed_monthly_hours_assumed']} billed hours, including the explicitly "
            "assumed supporting-service allowance. This is not measured cloud cost.",
            "",
            cost["limitation"],
            "",
        ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "output": str(output),
        "release_id": release["release_id"],
        "automatic_release_disqualified": not passes,
    }
