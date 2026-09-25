import copy

import pytest

from triage.data.load import load_split
from triage.data.prepare import audit_duplicates, canonicalize, prepare
from triage.io import read_json, read_jsonl, sha256, write_json


def test_deterministic_preparation_and_raw_text(source, tmp_path):
    directory, cfg, raw, domains = source
    raw["train"][0][0] = "  PAY my bill  "
    first, catalog = canonicalize(raw, domains, cfg["commit"], cfg["expected_counts"], 2)
    second, _ = canonicalize(raw, domains, cfg["commit"], cfg["expected_counts"], 2)
    assert first == second
    assert first["train"][0]["text"] == "  PAY my bill  "
    assert catalog == ["bill", "weather"]
    assert len({row["sample_id"] for rows in first.values() for row in rows}) == 11
    assert first["train"][-1]["domain"] is None
    prepare(cfg, tmp_path / "a", directory)
    prepare(cfg, tmp_path / "b", directory)
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
        assert b"\r\n" not in (tmp_path / "a" / name).read_bytes()


@pytest.mark.parametrize("corruption", ["empty", "unknown", "shape", "count", "oos_label"])
def test_corrupt_sources_fail(source, corruption):
    _, cfg, raw, domains = source
    raw = copy.deepcopy(raw)
    if corruption == "empty":
        raw["train"][0][0] = "  "
    elif corruption == "unknown":
        raw["train"][0][1] = "unknown"
    elif corruption == "shape":
        raw["train"][0] = ["missing label"]
    elif corruption == "count":
        raw["train"].pop()
    else:
        raw["oos_train"][0][1] = "weather"
    with pytest.raises(ValueError):
        canonicalize(raw, domains, cfg["commit"], cfg["expected_counts"], 2)


def test_duplicate_conflicts_are_flagged_without_removal(source):
    _, cfg, raw, domains = source
    raw["val"][0] = [raw["train"][2][0].upper(), "bill"]
    rows, _ = canonicalize(raw, domains, cfg["commit"], cfg["expected_counts"], 2)
    report = audit_duplicates(rows)
    assert report["exact"]["group_count"] == 0
    assert report["normalized"]["cross_split_groups"] == 1
    assert report["normalized"]["conflicting_label_groups"] == 1
    assert len(rows["val"]) == 3


def test_checksum_and_overwrite_protection(source, tmp_path):
    directory, cfg, _, _ = source
    cfg["files"]["data_full.json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        prepare(cfg, tmp_path / "bad", directory)
    with pytest.raises(FileExistsError):
        prepare(cfg, tmp_path / "bad", directory)


@pytest.mark.parametrize("split", ["val", "test"])
def test_fit_rejects_held_out_splits(dataset, split):
    with pytest.raises(ValueError, match="split=train"):
        load_split(dataset, split, for_fit=True)


def test_split_hash_and_manifest_counts(dataset):
    path = dataset / "val.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_split(dataset, "val")
    manifest = read_json(dataset / "manifest.json")
    manifest["splits"]["train"]["count"] += 1
    write_json(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError, match="count"):
        load_split(dataset, "train")


def test_catalog_integrity(dataset):
    write_json(dataset / "catalog.json", ["bill", "unknown"])
    with pytest.raises(ValueError, match="SHA256"):
        load_split(dataset, "train")


def test_membership_validation_even_with_updated_hash(dataset):
    rows = read_jsonl(dataset / "train.jsonl")
    rows[0]["split"] = "test"
    from triage.io import write_jsonl

    write_jsonl(dataset / "train.jsonl", rows)
    manifest = read_json(dataset / "manifest.json")
    manifest["splits"]["train"]["sha256"] = sha256(dataset / "train.jsonl")
    write_json(dataset / "manifest.json", manifest)
    with pytest.raises(ValueError, match="split/version"):
        load_split(dataset, "train")
