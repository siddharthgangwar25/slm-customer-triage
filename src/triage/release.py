"""Validation-only release selection, immutable inputs and explicit test-use tracking."""

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from triage.evaluation.report import load_predictions
from triage.evaluation.select_policy import load_policy
from triage.evaluation.three_way import validate_pair
from triage.io import new_directory, object_hash, read_json, sha256, verify_hash, write_json
from triage.training.runner import verify_bundle


def implementation_files():
    return sorted(Path("src/triage").rglob("*.py"))


def source_hash():
    return object_hash({p.as_posix(): sha256(p) for p in implementation_files()})


def previous_test_use():
    registry = Path("artifacts/test-use-registry")
    return sorted(read_json(p)["release_sha256"] for p in registry.glob("*.json"))


def choose_candidate(policies, costs=None):
    eligible = [name for name, policy in policies.items() if policy["automatic_routing"]]
    if not eligible:
        return "baseline", "No eligible candidate; review-only release"
    best = max(policies[n]["validation"]["coverage"] for n in eligible)
    near = [n for n in eligible if best - policies[n]["validation"]["coverage"] <= 0.01]
    if len(near) == 1:
        return near[0], "Highest eligible validation coverage, more than one percentage point ahead"
    if not costs or any(n not in costs for n in near):
        raise ValueError("Coverage tie requires measured cost and p95 for every tied candidate")
    return min(
        near, key=lambda n: (costs[n]["per_request"], costs[n]["p95_ms"])
    ), "Coverage tie: cost then p95"


def freeze(cfg, output):
    exposure = previous_test_use()
    if cfg.get("prior_test_exposure", []) != exposure:
        raise ValueError(
            "Prior test use must be explicitly disclosed in the new release configuration"
        )
    if cfg["paid_cloud_enabled"] is not False:
        raise ValueError("Paid resources are disabled")
    if set(cfg["candidates"]) != {"baseline", "prompted", "finetuned"}:
        raise ValueError("Freeze exactly A/B/C")
    policies, manifests, inputs = {}, {}, set(implementation_files())
    for name, candidate in cfg["candidates"].items():
        path = Path(candidate["predictions"])
        rows, catalog, meta = load_predictions(path)
        if meta["split"] != "val" or meta["fixture"] or len(rows) != 3100:
            raise ValueError("Release requires genuine complete validation evidence")
        _, policy = load_policy(Path(candidate["policy"]))
        if policy["predictions_sha256"] != sha256(path):
            raise ValueError("Validation/policy mismatch")
        verify_hash(path.parent / "prediction_manifest.json", policy["prediction_manifest_sha256"])
        policies[name], manifests[name] = policy, meta
        inputs.update([path, path.parent / "prediction_manifest.json", Path(candidate["policy"])])
        if name != "baseline":
            inputs.add(path.parent / "gate_predictions.jsonl")
    validate_pair(manifests["prompted"]["model"], manifests["finetuned"]["model"])
    gate_version = manifests["baseline"]["model"]["model_version"]
    baseline_hash = sha256(Path(cfg["candidates"]["baseline"]["predictions"]))
    for name in ("prompted", "finetuned"):
        if (
            policies[name]["gate_model_version"] != gate_version
            or manifests[name]["gate_reference"]["source_predictions_sha256"] != baseline_hash
        ):
            raise ValueError("Candidates must share the frozen baseline gate")
    if len({m["data_manifest_sha256"] for m in manifests.values()}) != 1:
        raise ValueError("Candidate dataset mismatch")
    selected, reason = choose_candidate(policies, cfg.get("measured_cost_tiebreak"))
    for path in (
        Path(cfg["parity_evidence"]),
        Path(cfg["load_evidence"]),
        Path(cfg["cost_evidence"]),
    ):
        inputs.add(path)
    parity, load = read_json(Path(cfg["parity_evidence"])), read_json(Path(cfg["load_evidence"]))
    selected_version = manifests[selected]["model"]["model_version"]
    if not (
        parity["complete"]
        and parity["passed"]
        and parity["sample_count"] == 3100
        and parity["model_version"] == selected_version
        and parity["source_sha256"] == source_hash()
    ):
        raise ValueError("Require complete matching serving/reference parity before freeze")
    if (
        load["model_version"] != selected_version
        or load["source_sha256"] != source_hash()
        or any(r["submitted"] < 500 for r in load["runs"])
        or {r["concurrency"] for r in load["runs"]} != {1, 4, 8}
    ):
        raise ValueError("Require 500 real HTTP requests at concurrency 1, 4, 8")
    if any(r["decision_mismatches"] for r in load["runs"]):
        raise ValueError("API/reference decisions differ")
    if load["runs"][0]["completed"] == 0:
        raise ValueError("No successful service requests")
    data = Path(cfg["data_dir"])
    verify_hash(data / "manifest.json", manifests[selected]["data_manifest_sha256"])
    inputs.update(
        [
            data / "manifest.json",
            data / "catalog.json",
            Path("uv.lock"),
            Path("environments/training/uv.lock"),
            Path(cfg["service_config"]),
        ]
    )
    gate = Path(cfg["baseline_bundle"])
    inputs.update([gate / "pipeline.joblib", gate / "metadata.json"])
    verify_hash(gate / "pipeline.joblib", policies[selected]["pipeline_sha256"])
    adapter = Path(cfg["adapter_bundle"])
    verify_bundle(adapter)
    c_meta = manifests["finetuned"]["model"]
    if (
        sha256(adapter / "bundle.json") != c_meta["adapter_bundle_sha256"]
        or adapter.resolve() != Path(c_meta["config"]["adapter"].replace("\\", "/")).resolve()
    ):
        raise ValueError("Selected adapter differs from evaluated C")
    inputs.update(p for p in adapter.iterdir() if p.is_file())
    paired = manifests["prompted"]["model"]["config"]
    inputs.add(Path(paired["prompt_file"]))
    snapshot = (
        Path(paired["cache_dir"])
        / ("models--" + paired["model_id"].replace("/", "--"))
        / "snapshots"
        / paired["revision"]
    )
    weights = list(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("Pinned base weights must be cached before release")
    inputs.update(p for p in snapshot.iterdir() if p.is_file())
    git = ["git", "-c", f"safe.directory={Path.cwd().as_posix()}"]
    commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        [*git, "diff", "HEAD", "--", "src", "configs", "deployment", "uv.lock", "environments"],
        text=True,
    ).strip():
        raise ValueError("Commit release source/configuration before freezing")
    new_directory(output)
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_commit": commit,
        "source_sha256": source_hash(),
        "config": cfg,
        "selected": selected,
        "selection_reason": reason,
        "policies": policies,
        "validation_manifests": manifests,
        "catalog": catalog,
        "test_use": "not yet accessed; ledger is created before first test read",
        "prior_test_exposure": exposure,
        "operational": {"load": load, "cost": read_json(Path(cfg["cost_evidence"]))},
        "input_sha256": {p.as_posix(): sha256(p) for p in sorted(inputs)},
    }
    digest = object_hash(payload)
    result = {**payload, "release_sha256": digest, "release_id": "release-" + digest[:12]}
    write_json(output / "release.json", result)
    return {
        "release": str(output / "release.json"),
        "selected": selected,
        "release_id": result["release_id"],
    }


def load_release(path):
    manifest = read_json(path)
    payload = {k: v for k, v in manifest.items() if k not in ("release_sha256", "release_id")}
    digest = object_hash(payload)
    if digest != manifest["release_sha256"] or manifest["release_id"] != "release-" + digest[:12]:
        raise ValueError("Release manifest changed")
    for name, expected in manifest["input_sha256"].items():
        verify_hash(Path(name), expected)
    if manifest["source_sha256"] != source_hash():
        raise ValueError("Release implementation changed; do not rerun test under this release")
    return manifest


def claim_test_use(release_path, manifest, output, resume):
    """Exclusive ledger creation prevents concurrent/different-output repeat test experiments."""
    ledger = release_path.parent / "test_use.json"
    expected = {"release_sha256": manifest["release_sha256"], "output": str(output.resolve())}
    registry = Path("artifacts/test-use-registry")
    registry.mkdir(parents=True, exist_ok=True)
    others = [v for v in previous_test_use() if v != manifest["release_sha256"]]
    if others != manifest.get("prior_test_exposure", []):
        raise ValueError("New test exposure occurred after release freeze")
    marker = registry / (manifest["release_id"] + ".json")
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            import json

            json.dump(expected, stream)
    elif read_json(marker) != expected:
        raise ValueError("Release already used test in another output")
    if ledger.exists():
        value = read_json(ledger)
        if (
            not resume
            or any(value[k] != v for k, v in expected.items())
            or value["status"] == "complete"
        ):
            raise ValueError("Test already used; only unchanged incomplete run may resume")
    else:
        if resume:
            raise ValueError("No test run to resume")
        with ledger.open("x", encoding="utf-8", newline="\n") as stream:
            import json

            json.dump(
                {**expected, "status": "started", "started_utc": datetime.now(UTC).isoformat()},
                stream,
            )
    return ledger


def switch_pointer(pointer, value):
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(".new")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        import json

        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, pointer)


def activate(release_path, pointer):
    release = load_release(release_path)
    ledger = read_json(release_path.parent / "test_use.json")
    if ledger["status"] != "complete":
        raise ValueError("Final test acceptance required before activation")
    report_path = Path(ledger["output"]) / "final_report.json"
    verify_hash(report_path, ledger["report_sha256"])
    if read_json(report_path)["automatic_release_disqualified"]:
        raise ValueError("Test disqualified automatic release; no silent threshold retuning")
    previous = (
        read_json(pointer)
        if pointer.exists()
        else {
            "mode": "baseline",
            "bundle": release["config"]["baseline_bundle"],
            "policy": release["config"]["candidates"]["baseline"]["policy"],
        }
    )
    switch_pointer(
        pointer,
        {
            "mode": "release",
            "release": str(release_path.resolve()),
            "release_sha256": release["release_sha256"],
            "previous": previous,
        },
    )
    return {
        "pointer": str(pointer),
        "action": "Restart worker/gateway to use the activated manifest",
    }


def rollback(pointer):
    current = read_json(pointer)
    previous = current.get("previous")
    if previous is None:
        raise ValueError("No previous release recorded")
    switch_pointer(pointer, previous)
    return {
        "pointer": str(pointer),
        "action": "Restart the service with the previous source/environment and bundle",
    }


def active_bundle(pointer):
    value = read_json(pointer)
    if value["mode"] == "baseline":
        return Path(value["bundle"]), Path(value["policy"]), False
    release = load_release(Path(value["release"]))
    if release["release_sha256"] != value["release_sha256"]:
        raise ValueError("Active pointer identity mismatch")
    return (
        Path(release["config"]["baseline_bundle"]),
        Path(release["config"]["candidates"][release["selected"]]["policy"]),
        release["selected"] != "baseline",
    )
