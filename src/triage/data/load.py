"""Load only the requested split. Training never opens validation or test records."""

from pathlib import Path

from triage.io import read_json, read_jsonl, verify_hash


def load_split(directory: Path, split: str, *, for_fit: bool = False):
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")
    if for_fit and split != "train":
        raise ValueError("Fit operations require split=train")
    manifest = read_json(directory / "manifest.json")
    verify_hash(directory / "catalog.json", manifest["catalog_sha256"])
    catalog = read_json(directory / "catalog.json")
    if not catalog or catalog != sorted(set(catalog)) or "oos" in catalog:
        raise ValueError("Invalid supported-label catalog")
    path = directory / f"{split}.jsonl"
    verify_hash(path, manifest["splits"][split]["sha256"])
    rows = read_jsonl(path)
    if len(rows) != manifest["splits"][split]["count"]:
        raise ValueError("Split count does not match manifest")
    ids = set()
    for row in rows:
        if row["split"] != split or row["dataset_version"] != manifest["commit"]:
            raise ValueError("Record split/version does not match manifest")
        if row["label"] not in catalog + ["oos"] or not row["text"].strip():
            raise ValueError("Invalid canonical record")
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
    return rows, catalog, manifest
