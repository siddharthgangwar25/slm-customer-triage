"""Verify saved baseline gates and attach them by a strict one-to-one sample-ID join."""

from pathlib import Path

from triage.io import read_jsonl, sha256


def join_gate(rows, gate_path: Path, catalog, dataset_hash):
    from triage.evaluation.report import load_predictions

    gates, gate_catalog, manifest = load_predictions(gate_path)
    if (
        manifest["split"] != "val"
        or manifest["model"]["backend"] != "scikit-learn CPU"
        or manifest["gate_model_version"] != manifest["model"]["model_version"]
    ):
        raise ValueError("Expected frozen baseline validation predictions")
    if gate_catalog != catalog or manifest["data_manifest_sha256"] != dataset_hash:
        raise ValueError("Gate catalog/data manifest mismatch")
    ids = [row["sample_id"] for row in rows]
    by_id = {row["sample_id"]: row for row in gates}
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("Gate join must be one-to-one over the complete validation split")
    for row in rows:
        gate = by_id[row["sample_id"]]
        for key in ("text", "label", "split", "dataset_version"):
            if row[key] != gate[key]:
                raise ValueError(f"Gate record differs in {key}")
    return by_id, manifest


def verify_saved_gate(rows, manifest, directory: Path):
    """The evaluator needs the copied gate file, never model weights or mutable outside paths."""
    if "gate_reference" not in manifest:
        if manifest["model"]["backend"] != "scikit-learn CPU":
            raise ValueError("Non-baseline predictions require a verified gate reference")
        return
    ref = manifest["gate_reference"]
    path = directory / "gate_predictions.jsonl"
    if sha256(path) != ref["predictions_sha256"]:
        raise ValueError("Gate reference checksum mismatch")
    gates = read_jsonl(path)
    by_id = {r["sample_id"]: r for r in gates}
    if len(gates) != len(by_id) or set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("Gate reference is not a one-to-one join")
    for row in rows:
        gate = by_id[row["sample_id"]]
        if gate["model_version"] != manifest["gate_model_version"]:
            raise ValueError("Gate model version mismatch")
        for key in ("text", "label", "split", "dataset_version", "gate_score"):
            if row[key] != gate[key]:
                raise ValueError(f"Saved gate differs in {key}")
