"""Real HTTP parity and closed-loop load benchmarks; no hidden retries or dropped failures."""

import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import load_policy
from triage.io import (
    new_directory,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
    write_jsonl,
)
from triage.policy import ModelOutput
from triage.release import source_hash
from triage.service.remote import call_worker


def parity(predictions, worker_url, worker_key, output, *, resume=False, limit=None):
    rows, _, manifest = load_predictions(predictions)
    if manifest["split"] != "val" or manifest["fixture"]:
        raise ValueError("Parity requires genuine validation predictions")
    if limit is not None:
        if not 0 < limit < len(rows):
            raise ValueError("Smoke limit must be smaller than validation")
        rows = rows[:limit]
    info = call_worker(worker_url, worker_key, "/health/ready")
    frozen = {
        "predictions_sha256": sha256(predictions),
        "source_sha256": source_hash(),
        "reference_metadata_sha256": info["reference_metadata_sha256"],
        "limit": limit,
    }
    if info["model_version"] != manifest["model"]["model_version"]:
        raise ValueError("Worker differs from reference")
    if resume:
        if read_json(output / "run.json") != frozen:
            raise ValueError("Serving parity resume changed")
        saved = read_jsonl(output / "outcomes.jsonl")
        verify_hash(output / "outcomes.jsonl", read_json(output / "progress.json")["sha256"])
    else:
        new_directory(output)
        write_json(output / "run.json", frozen)
        saved = []
    if [r["sample_id"] for r in saved] != [r["sample_id"] for r in rows[: len(saved)]]:
        raise ValueError("Parity outcomes are not an ordered prefix")
    with (output / "outcomes.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows[len(saved) :]:
            actual = call_worker(worker_url, worker_key, "/generate", {"text": row["text"]})
            equal = all(
                actual[k] == row[k]
                for k in ("raw_output", "input_tokens", "output_tokens", "truncated")
            )
            equal = equal and actual["error_type"] is None
            record = {"sample_id": row["sample_id"], "equal": equal, "actual": actual}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            saved.append(record)
            write_json(
                output / "progress.json",
                {"sha256": sha256(output / "outcomes.jsonl"), "count": len(saved)},
            )
            if len(saved) % 100 == 0:
                print({"parity_completed": len(saved), "total": len(rows)}, flush=True)
    result = {
        **frozen,
        "sample_count": len(saved),
        "complete": limit is None,
        "model_version": info["model_version"],
        "passed": all(r["equal"] for r in saved),
        "mismatches": sum(not r["equal"] for r in saved),
        "worker": info,
        "scope": "same raw outputs and token accounting through separate worker process",
    }
    write_json(output / "parity.json", result)
    if not result["passed"]:
        raise ValueError("Serving parity failed; do not freeze or run test")
    return result


def http_call(url, key, body=None, timeout=15):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw, status = response.read().decode(), response.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode(), exc.code
    except OSError:
        raw, status = "{}", 0
    elapsed = time.perf_counter() - started
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {"text": raw}
    return status, parsed, elapsed


def metrics_snapshot(url, key):
    status, parsed, _ = http_call(url + "/metrics", key)
    if status != 200:
        raise ValueError("Private metrics endpoint unavailable")
    return {
        line.split()[0]: float(line.split()[1])
        for line in parsed["text"].splitlines()
        if line and not line.startswith("#") and "{" not in line
    }


def load_test(
    predictions,
    policy_path,
    url,
    api_key,
    metrics_key,
    output,
    *,
    count=500,
    cold_start_seconds=None,
):
    rows, catalog, manifest = load_predictions(predictions)
    policy, frozen = load_policy(policy_path)
    if manifest["split"] != "val" or frozen["predictions_sha256"] != sha256(predictions):
        raise ValueError("Load benchmark requires matching validation reference/policy")
    if count < 1:
        raise ValueError("Positive sample count required")
    rng = random.Random(42)
    supported = [r for r in rows if r["label"] != "oos"]
    oos = [r for r in rows if r["label"] == "oos"]
    oos_count = max(1, round(count * len(oos) / len(rows)))
    workload = rng.choices(supported, k=count - oos_count) + rng.choices(oos, k=oos_count)
    rng.shuffle(workload)
    new_directory(output)
    before_ready = time.perf_counter()
    status, model, _ = http_call(url + "/v1/model", api_key)
    if status != 200 or model["model_version"] != manifest["model"]["model_version"]:
        raise ValueError("API/reference model mismatch")
    # Twenty warmups are excluded from throughput and metric deltas.
    for row in workload[:20]:
        status, _, _ = http_call(url + "/v1/triage", api_key, {"text": row["text"]})
        if status != 200:
            raise ValueError("Service warmup failed")
    warmup_seconds = time.perf_counter() - before_ready
    runs = []
    for concurrency in (1, 4, 8):
        before = metrics_snapshot(url, metrics_key)

        def request(row):
            status, body, elapsed = http_call(url + "/v1/triage", api_key, {"text": row["text"]})
            expected = policy.decide(
                row["gate_score"],
                ModelOutput(row["predicted_label"], row["parse_status"] == "ok"),
                catalog,
            )
            mismatch = status == 200 and any(
                body.get(k) != getattr(expected, k) for k in ("intent", "decision", "reason")
            )
            return {
                "sample_id": row["sample_id"],
                "label": row["label"],
                "status": status,
                "latency_ms": elapsed * 1000,
                "decision": body.get("decision"),
                "reason": body.get("reason"),
                "error_code": body.get("error", {}).get("code"),
                "decision_mismatch": mismatch,
                "reference_input_tokens": row.get("input_tokens"),
            }

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(request, workload))
        seconds = time.perf_counter() - started
        after = metrics_snapshot(url, metrics_key)
        completed = sum(r["status"] == 200 for r in outcomes)
        stats = {
            "concurrency": concurrency,
            "submitted": count,
            "completed": completed,
            "failures": count - completed,
            "elapsed_seconds": seconds,
            "submitted_per_second": count / seconds,
            "completed_per_second": completed / seconds,
            "latency_all_ms": dict(
                zip(
                    ("p50", "p95", "p99"),
                    np.percentile([r["latency_ms"] for r in outcomes], [50, 95, 99]).tolist(),
                    strict=True,
                )
            ),
            "latency_completed_ms": dict(
                zip(
                    ("p50", "p95", "p99"),
                    np.percentile(
                        [r["latency_ms"] for r in outcomes if r["status"] == 200], [50, 95, 99]
                    ).tolist(),
                    strict=True,
                )
            )
            if completed
            else None,
            "statuses": dict(Counter(str(r["status"]) for r in outcomes)),
            "decision_mismatches": sum(r["decision_mismatch"] for r in outcomes),
            "metrics_delta": {
                k: after.get(k, 0) - before.get(k, 0) for k in after if k.endswith("_total")
            },
        }
        runs.append(stats)
        write_jsonl(output / f"concurrency-{concurrency}.jsonl", outcomes)
        print(stats, flush=True)
    result = {
        "model_version": model["model_version"],
        "policy_version": model["policy_version"],
        "source_sha256": source_hash(),
        "predictions_sha256": sha256(predictions),
        "split": "val",
        "fixture": manifest["fixture"],
        "complete": count >= 500,
        "backend": "HTTP gateway + private transformers worker",
        "batch_size": 1,
        "mix": {"supported": count - oos_count, "oos": oos_count},
        "workload": "seed 42 sampling with replacement; same requests at each concurrency; "
        "closed loop, no retries",
        "cold_start_seconds": cold_start_seconds,
        "ready_and_warmup_seconds": warmup_seconds,
        "model_metadata": manifest["model"],
        "runs": runs,
        "note": "Failures remain in submitted throughput and all-request percentiles; "
        "consult completed percentiles separately.",
    }
    write_json(output / "load.json", result)
    return result
