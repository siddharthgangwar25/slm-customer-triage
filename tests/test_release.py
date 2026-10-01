"""Release/test-use and remote-serving contracts; all generated inputs are synthetic."""

import copy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_service import FakeAdapter

from triage.io import object_hash, read_json, write_json
from triage.policy import Policy
from triage.release import choose_candidate, claim_test_use, load_release, source_hash
from triage.service.app import create_app
from triage.service.remote import RemoteAdapter
from triage.service.runtime import Runtime


def test_selection_uses_validation_coverage_then_explicit_cost_tie():
    policies = {
        "baseline": {"automatic_routing": True, "validation": {"coverage": 0.68}},
        "prompted": {"automatic_routing": False, "validation": {"coverage": 0}},
        "finetuned": {"automatic_routing": True, "validation": {"coverage": 0.80}},
    }
    assert choose_candidate(policies)[0] == "finetuned"
    policies["baseline"]["validation"]["coverage"] = 0.795
    with pytest.raises(ValueError, match="tie"):
        choose_candidate(policies)
    costs = {
        "baseline": {"per_request": 0.001, "p95_ms": 5},
        "finetuned": {"per_request": 0.002, "p95_ms": 1},
    }
    assert choose_candidate(policies, costs)[0] == "baseline"
    for p in policies.values():
        p["automatic_routing"] = False
    assert "review-only" in choose_candidate(policies)[1]


def test_test_ledger_blocks_duplicate_and_wrong_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    release = tmp_path / "release.json"
    manifest = {"release_sha256": "a" * 64, "release_id": "release-a", "prior_test_exposure": []}
    output = tmp_path / "final"
    ledger = claim_test_use(release, manifest, output, False)
    assert read_json(ledger)["status"] == "started"
    with pytest.raises(ValueError, match="already used"):
        claim_test_use(release, manifest, output, False)
    claim_test_use(release, manifest, output, True)
    with pytest.raises(ValueError):
        claim_test_use(release, manifest, tmp_path / "elsewhere", True)
    write_json(ledger, {**read_json(ledger), "status": "complete"})
    with pytest.raises(ValueError, match="already used"):
        claim_test_use(release, manifest, output, True)


def test_new_test_exposure_after_freeze_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = {"release_sha256": "a", "release_id": "release-a", "prior_test_exposure": []}
    claim_test_use(tmp_path / "a.json", manifest, tmp_path / "out", False)
    other = {**manifest, "release_sha256": "b", "release_id": "release-b"}
    with pytest.raises(ValueError, match="after release freeze"):
        claim_test_use(tmp_path / "b.json", other, tmp_path / "out2", False)


def test_frozen_inputs_and_manifest_detect_mutation(tmp_path):
    path = tmp_path / "release.json"
    payload = {"source_sha256": source_hash(), "input_sha256": {}}
    digest = object_hash(payload)
    frozen = {**payload, "release_sha256": digest, "release_id": "release-" + digest[:12]}
    write_json(path, frozen)
    assert load_release(path) == frozen
    changed = copy.deepcopy(frozen)
    changed["input_sha256"]["injected"] = "hash"
    write_json(path, changed)
    with pytest.raises(ValueError, match="manifest changed"):
        load_release(path)


def test_private_metrics_measure_short_circuit_without_request_content():
    adapter = FakeAdapter()
    adapter.score = 0.1
    runtime = Runtime(adapter, Policy(0.5, True, "policy"))
    with TestClient(create_app(runtime=runtime, metrics_key="metrics-secret")) as client:
        client.post("/v1/triage", json={"text": "private raw request 123"})
        assert client.get("/metrics").status_code == 401
        response = client.get("/metrics", headers={"Authorization": "Bearer metrics-secret"})
        assert response.status_code == 200
        assert "triage_gate_rejections_total 1" in response.text
        assert "triage_http_2xx_total 1" in response.text
        assert "triage_request_seconds_count 1" in response.text
        assert "private raw" not in response.text and "metrics-secret" not in response.text
        assert adapter.model_calls == 0


def test_remote_adapter_never_reuses_baseline_classification(monkeypatch):
    from triage.policy import ModelOutput
    from triage.service.runtime import GateResult

    class Gate:
        def gate(self, text):
            return GateResult(0.8, ModelOutput("wrong_baseline_label"))

    remote = object.__new__(RemoteAdapter)
    remote.baseline = Gate()
    assert remote.gate("request").cached_output is None


def test_final_rejects_unfrozen_input_before_reading_test(tmp_path, monkeypatch):
    from triage.evaluation import final

    touched = []
    monkeypatch.setattr(final, "load_split", lambda *a, **k: touched.append(True))
    path = tmp_path / "release.json"
    write_json(path, {"release_sha256": "fake", "release_id": "fake"})
    with pytest.raises(ValueError, match="manifest changed"):
        final.run(path, tmp_path / "final")
    assert not touched


@pytest.mark.parametrize("interrupt", [False, True])
def test_three_candidate_final_fixture_keeps_fixed_policies_and_blocks_repeat(
    baseline_config, tmp_path, monkeypatch, interrupt
):
    from triage.evaluation import final
    from triage.evaluation.select_policy import select_policy
    from triage.io import read_jsonl
    from triage.models import finetuned, prompted
    from triage.models.baseline import predict, train

    cfg = baseline_config
    meta = train(cfg, Path(cfg["bundle"]))
    predict(cfg, "val", Path(cfg["prediction_output"]))
    policy = select_policy(
        Path(cfg["prediction_output"]) / "predictions.jsonl", "val", tmp_path / "policy"
    )
    prompt_file = str(Path("prompts/intent-v1.txt").resolve())
    manifest = {
        "release_id": "fixture-release",
        "release_sha256": "fixture-sha",
        "prior_test_exposure": [],
        "config": {"data_dir": cfg["data_dir"], "baseline_bundle": cfg["bundle"]},
        "selected": "baseline",
        "policies": {n: policy for n in ("baseline", "prompted", "finetuned")},
        "validation_manifests": {
            n: {
                "model": meta
                if n == "baseline"
                else {
                    "model_version": "FIXTURE-" + n,
                    "backend": "fixture",
                    "config": {"prompt_file": prompt_file},
                },
                "data_manifest_sha256": meta["data_manifest_sha256"],
            }
            for n in ("baseline", "prompted", "finetuned")
        },
    }

    calls = [0]

    class Generator:
        def __init__(self, *args):
            pass

        def generate(self, text):
            calls[0] += 1
            if interrupt and calls[0] == 2:
                raise KeyboardInterrupt("synthetic interruption after one saved B outcome")
            label = "bill" if "bill" in text else "weather" if "weather" in text else "oos"
            return {
                "raw_output": '{"intent":"' + label + '"}',
                "input_tokens": 10,
                "output_tokens": 7,
                "truncated": False,
                "error_type": None,
            }

        def memory(self):
            return {"fixture": True}

    monkeypatch.setattr(final, "load_release", lambda p: manifest)
    monkeypatch.setattr(prompted, "TransformersAdapter", Generator)
    monkeypatch.setattr(finetuned, "FinetunedAdapter", Generator)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "final"
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            final.run(tmp_path / "release.json", output)
        assert not (output / "active.lock").exists()
        assert len(read_jsonl(output / "prompted/predictions.jsonl")) == 1
        final.run(tmp_path / "release.json", output, resume=True)
    else:
        final.run(tmp_path / "release.json", output)
    report = read_json(output / "final_report.json")
    assert report["fixture"] is True
    for name, candidate in report["candidates"].items():
        assert candidate["fixed_policy"]["threshold"] == policy["threshold"]
        assert candidate["raw"]["sample_count"] == 3
        assert len(read_jsonl(output / name / "predictions.jsonl")) == 3
    with pytest.raises(ValueError, match="already used"):
        final.run(tmp_path / "release.json", output, resume=True)


def test_cost_accounts_for_idle_support_and_training_separately():
    from triage.evaluation.cost import estimate

    load = {"runs": [{"concurrency": 1, "elapsed_seconds": 500, "submitted": 500}]}
    report = estimate(
        load,
        0.5,
        monthly_requests=100000,
        monthly_hours=730,
        supporting_monthly_usd=10,
        training_hours=12,
    )
    assert report["hosting_usd_per_1000_submitted"] == 3.75
    assert report["training_usd_at_scenario_rate"] == 6
    assert report["idle_hours_at_local_measured_rate"] > 700
    assert report["measured_cloud_cost"] is False


def test_rollback_atomically_restores_known_previous_configuration(tmp_path):
    from triage.release import rollback, switch_pointer

    pointer = tmp_path / "active.json"
    previous = {"mode": "baseline", "bundle": "trusted", "policy": "trusted-policy"}
    switch_pointer(pointer, {"mode": "release", "release": "new", "previous": previous})
    rollback(pointer)
    assert read_json(pointer) == previous
    assert not pointer.with_suffix(".new").exists()
    with pytest.raises(ValueError, match="No previous"):
        rollback(pointer)


def test_frozen_artifact_mutation_rejected(tmp_path):
    from triage.io import sha256

    artifact = tmp_path / "adapter"
    artifact.write_text("verified", encoding="utf-8")
    payload = {"source_sha256": source_hash(), "input_sha256": {str(artifact): sha256(artifact)}}
    digest = object_hash(payload)
    path = tmp_path / "release.json"
    write_json(path, {**payload, "release_sha256": digest, "release_id": "release-" + digest[:12]})
    load_release(path)
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_release(path)
