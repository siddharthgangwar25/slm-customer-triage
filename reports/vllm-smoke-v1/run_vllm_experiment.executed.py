"""Validation-only B/C HTTP comparison. No training, test split, or release writes."""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from triage.evaluation.metrics import classification, routing
from triage.io import new_directory, read_json, read_jsonl, sha256, write_json
from triage.policy import parse_model_json
from triage.release import source_hash

BASE_ID = "sha256:031350c8fef591b8e2129056cb96a94e1bf5516a636a9c691fa69964bf9d7dbc"
SOURCE = "2a91de5d0e0e4625c3b3fcb22a5898d16b387f6259568b9bdd2b21df3f48a4c4"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
STEMS = {"B": "prompted-qwen3-06b-nf4-v1", "C": "finetuned-qwen3-06b-qlora-v1"}
FIELDS = ("raw_output", "input_tokens", "output_tokens", "truncated", "error_type")


def select_rows(rows, full):
    if len(rows) != 3100 or any(r["split"] != "val" for r in rows):
        raise ValueError("Only the complete 3,100-row validation reference is allowed")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate validation IDs")
    return (
        rows
        if full
        else (
            [r for r in rows if r["label"] != "oos"][:3]
            + [r for r in rows if r["label"] == "oos"][:3]
        )
    )


def compare(reference, response):
    return [k for k in FIELDS if reference[k] != response[k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    if source_hash() != SOURCE:
        raise ValueError("Frozen reference source changed")
    frozen_paths = [
        Path("artifacts/milestone5/release/release.json"),
        Path("artifacts/milestone5/release/test_use.json"),
    ]
    frozen_before = {p.as_posix(): sha256(p) for p in frozen_paths}
    output = new_directory(args.output.resolve())
    inputs = new_directory(output / "inputs")
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        fallback = Path(os.environ.get("LOCALAPPDATA", "")) / (
            "Programs/DockerDesktop/resources/bin/docker.exe"
        )
        docker_bin = str(fallback) if fallback.is_file() else "docker"

    def docker(*cmd, **kwargs):
        return subprocess.check_output([docker_bin, *cmd], text=True, **kwargs).strip()

    refs, manifests, policies = {}, {}, {}
    for candidate, stem in STEMS.items():
        folder = root / "reports" / (stem + "-val")
        manifest = read_json(folder / "prediction_manifest.json")
        if (
            manifest["split"] != "val"
            or not manifest["benchmark_complete"]
            or sha256(folder / "predictions.jsonl") != manifest["predictions_sha256"]
        ):
            raise ValueError("Reference identity mismatch")
        refs[candidate] = select_rows(read_jsonl(folder / "predictions.jsonl"), args.full)
        manifests[candidate] = manifest
        policies[candidate] = read_json(root / "reports" / (stem + "-policy") / "policy.json")
        if policies[candidate]["predictions_sha256"] != manifest["predictions_sha256"]:
            raise ValueError("Policy/reference mismatch")
    if [r["sample_id"] for r in refs["B"]] != [r["sample_id"] for r in refs["C"]]:
        raise ValueError("Paired reference IDs differ")
    c = manifests["C"]["model"]
    for field in ("model_id", "revision", "system_prompt_sha256", "chat_template_sha256"):
        if manifests["B"]["model"][field] != c[field]:
            raise ValueError("Paired settings differ: " + field)
    for field in (
        "dtype",
        "quantization",
        "max_input_tokens",
        "max_new_tokens",
        "seed",
        "do_sample",
        "enable_thinking",
        "batch_size",
        "prefix_cache",
    ):
        if manifests["B"]["model"]["config"][field] != c["config"][field]:
            raise ValueError("Paired decoding differs: " + field)
    if c["revision"] != REVISION or c["config"]["quantization"] != "nf4":
        raise ValueError("This experiment requires the pinned Qwen3 NF4 reference")
    adapter = root / c["config"]["adapter"].replace("\\", "/")
    bundle = read_json(adapter / "bundle.json")
    if sha256(adapter / "bundle.json") != c["adapter_bundle_sha256"]:
        raise ValueError("Adapter bundle changed")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        if sha256(adapter / name) != bundle["files"][name]:
            raise ValueError("Adapter file changed: " + name)
    model = root / "artifacts/huggingface/models--Qwen--Qwen3-0.6B/snapshots" / REVISION
    identity = {
        "catalog": manifests["C"]["catalog"],
        "source_sha256": SOURCE,
        "prompt_metadata": {
            k: c[k] for k in ("system_prompt_sha256", "chat_template_sha256", "enable_thinking")
        },
        "references": {k: m["predictions_sha256"] for k, m in manifests.items()},
        "policies": policies,
        "adapter_bundle_sha256": c["adapter_bundle_sha256"],
        "model_id": c["model_id"],
        "revision": REVISION,
        "model_files": {p.name: sha256(p) for p in model.iterdir() if p.is_file()},
        "full_validation": args.full,
        "test_inference": False,
        "experiment_files": {
            p: sha256(root / p)
            for p in (
                "deployment/Dockerfile.vllm",
                "deployment/vllm_worker.py",
                "scripts/run_vllm_experiment.py",
                "environments/vllm/requirements.lock",
            )
        },
    }
    write_json(inputs / "identity.json", identity)
    shutil.copyfile(root / c["config"]["prompt_file"], inputs / "prompt.txt")
    if docker("info", "--format", "{{.OSType}}") != "linux":
        raise ValueError("Linux Docker engine required")
    if docker("image", "inspect", "triage-m5-gpu:82ed8d7", "--format", "{{.Id}}") != BASE_ID:
        raise ValueError("Audited base image ID changed")
    if not args.skip_build:
        with (output / "build.log").open("w", encoding="utf-8") as log:
            print(f"Building isolated image; see {log.name}", flush=True)
            subprocess.run(
                [
                    docker_bin,
                    "build",
                    "--progress",
                    "plain",
                    "-f",
                    "deployment/Dockerfile.vllm",
                    "-t",
                    "triage-vllm:0.10.2",
                    ".",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    image_id = docker("image", "inspect", "triage-vllm:0.10.2", "--format", "{{.Id}}")
    write_json(
        output / "host.json",
        {
            "base_image_id": BASE_ID,
            "image_id": image_id,
            "docker": json.loads(docker("version", "--format", "{{json .}}")),
        },
    )
    key = secrets.token_hex(32)
    env = {**os.environ, "VLLM_EXPERIMENT_KEY": key}
    name = "triage-vllm-" + secrets.token_hex(5)
    status = {"status": "starting", "test_inference": False, "candidates": {}}
    write_json(output / "summary.json", status)
    owned = False
    try:
        docker(
            "create",
            "--name",
            name,
            "--gpus",
            "all",
            "--read-only",
            "--tmpfs",
            "/tmp:exec,size=2g",
            "--shm-size",
            "1g",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "-p",
            "127.0.0.1::8001",
            "-e",
            "VLLM_EXPERIMENT_KEY",
            "-v",
            f"{model}:/model:ro",
            "-v",
            f"{adapter}:/adapter:ro",
            "-v",
            f"{inputs}:/inputs:ro",
            "-v",
            f"{root / 'deployment/vllm_worker.py'}:/experiment/worker.py:ro",
            image_id,
            "/experiment/worker.py",
            env=env,
        )
        owned = True
        docker("start", name)
        port = docker("port", name, "8001/tcp").split(":")[-1]
        url = "http://127.0.0.1:" + port

        def request(path, body=None, *, authenticated=True):
            req = urllib.request.Request(
                url + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": "Bearer " + key} if authenticated else {}),
                },
            )
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)

        started = time.perf_counter()
        while time.perf_counter() - started < 1200:
            try:
                ready = request("/health/ready")
                break
            except (OSError, ValueError):
                if docker("inspect", name, "--format", "{{.State.Running}}") != "true":
                    raise RuntimeError("Worker exited during startup; see worker.log") from None
                print(f"Waiting for vLLM startup: {time.perf_counter() - started:.0f}s", flush=True)
                time.sleep(10)
        else:
            raise TimeoutError("vLLM startup exceeded 20 minutes")
        write_json(
            output / "runtime.json", {**ready, "http_ready_seconds": time.perf_counter() - started}
        )
        try:
            request("/health/ready", authenticated=False)
        except urllib.error.HTTPError as exc:
            if exc.code != 401:
                raise
        else:
            raise ValueError("Worker allowed unauthenticated access")
        status["auth_check"] = "passed"
        write_json(output / "resources-start.json", request("/resources"))
        for candidate in ("B", "C"):
            rows, mismatches, failures = [], Counter(), 0
            started = time.perf_counter()
            with (output / f"{candidate}-comparison.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                for index, reference in enumerate(refs[candidate]):
                    tick = time.perf_counter()
                    try:
                        result = request(
                            "/generate", {"candidate": candidate, "text": reference["text"]}
                        )
                        if (
                            result["candidate"] != candidate
                            or result["variant_id"] != ready["variant_id"]
                        ):
                            raise ValueError("Worker identity changed")
                    except (OSError, ValueError) as exc:
                        failures += 1
                        result = {
                            "raw_output": "",
                            "input_tokens": reference["input_tokens"],
                            "output_tokens": 0,
                            "truncated": False,
                            "error_type": "inference_error",
                            "exception": str(exc),
                        }
                    latency = (time.perf_counter() - tick) * 1000
                    differences = compare(reference, result)
                    mismatches.update(differences)
                    parsed = parse_model_json(
                        result["raw_output"], identity["catalog"], truncated=result["truncated"]
                    )
                    rows.append(
                        {
                            **reference,
                            **{k: result[k] for k in FIELDS},
                            "predicted_label": parsed.label,
                            "parse_status": "ok" if parsed.valid else "invalid",
                            "model_version": ready["variant_id"] + "-" + candidate,
                            "latency_ms": latency,
                            "token_count": result["input_tokens"] + result["output_tokens"],
                        }
                    )
                    stream.write(
                        json.dumps(
                            {
                                "sample_id": reference["sample_id"],
                                "reference": {k: reference[k] for k in FIELDS},
                                "observed": result,
                                "differences": differences,
                                "latency_ms": latency,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    stream.flush()
                    if (index + 1) % 100 == 0 or not args.full:
                        print(
                            f"{candidate}: {index + 1}/{len(refs[candidate])}; errors={failures}",
                            flush=True,
                        )
            from triage.io import write_jsonl

            write_jsonl(output / f"{candidate}-predictions.jsonl", rows)
            policy = policies[candidate]
            summary = {
                "submitted": len(rows),
                "completed": len(rows) - failures,
                "infrastructure_failures": failures,
                "mismatch_fields": dict(mismatches),
                "exact_matches": sum(
                    not compare(r, p) for r, p in zip(refs[candidate], rows, strict=True)
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "classification": classification(rows, identity["catalog"]),
                "frozen_threshold_diagnostic": routing(
                    rows, identity["catalog"], policy["threshold"]
                ),
                "policy_automatic_routing": policy["automatic_routing"],
                "scope": "Validation comparison including first-request warmup; not load testing",
            }
            status["candidates"][candidate] = summary
            write_json(output / "summary.json", status)
        write_json(output / "resources-end.json", request("/resources"))
        status["status"] = (
            "completed"
            if all(x["infrastructure_failures"] == 0 for x in status["candidates"].values())
            else "inference_failed"
        )
    except BaseException as exc:
        status.update(status="failed", error=str(exc), error_class=type(exc).__name__)
        raise
    finally:
        frozen_after = {p.as_posix(): sha256(p) for p in frozen_paths}
        write_json(
            output / "frozen_release_check.json",
            {
                "before": frozen_before,
                "after": frozen_after,
                "unchanged": frozen_before == frozen_after,
            },
        )
        if frozen_before != frozen_after:
            status.update(status="failed", error="Frozen release/ledger changed during experiment")
        write_json(output / "summary.json", status)
        if owned:
            write_json(
                output / "container_state.json",
                json.loads(docker("inspect", name, "--format", "{{json .State}}")),
            )
            with (output / "worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [docker_bin, "logs", name], stdout=log, stderr=subprocess.STDOUT, check=False
                )
            docker("rm", "-f", name)
        write_json(
            output / "checksums.json",
            {
                str(p.relative_to(output)).replace("\\", "/"): sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "checksums.json"
            },
        )
    print(f"vLLM experiment {status['status']}: {output}", flush=True)
    if status["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
