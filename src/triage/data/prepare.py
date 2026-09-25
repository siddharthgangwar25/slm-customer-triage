"""Pinned official-source download; no benchmark samples are dropped or rewritten."""

import hashlib
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from triage.io import new_directory, read_json, sha256, verify_hash, write_json, write_jsonl

SPLITS = ("train", "val", "test")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonicalize(raw, domains, version, expected_counts, expected_intents):
    if not isinstance(raw, dict) or set(raw) != set(expected_counts):
        raise ValueError("Source split keys differ from the expected six official splits")
    if not isinstance(domains, dict) or not domains:
        raise ValueError("Invalid domain mapping")
    label_domains = {}
    for domain, labels in domains.items():
        if not isinstance(domain, str) or not isinstance(labels, list):
            raise ValueError("Invalid domain mapping")
        for label in labels:
            if not isinstance(label, str) or not label or label == "oos" or label in label_domains:
                raise ValueError(f"Invalid or repeated domain label: {label}")
            label_domains[label] = domain
    if len(label_domains) != expected_intents:
        raise ValueError("Unexpected supported intent count")
    result = {split: [] for split in SPLITS}
    for split in SPLITS:
        for source_key in (split, f"oos_{split}"):
            rows = raw[source_key]
            if not isinstance(rows, list) or len(rows) != expected_counts[source_key]:
                actual = len(rows) if isinstance(rows, list) else "not a list"
                raise ValueError(
                    f"Count discrepancy for {source_key}: {actual}, "
                    f"expected {expected_counts[source_key]}"
                )
            for index, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"Corrupt record: {source_key}:{index}")
                text, label = row
                if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
                    raise ValueError(f"Missing/empty text or label: {source_key}:{index}")
                if source_key.startswith("oos_"):
                    if label != "oos":
                        raise ValueError(f"Unsupported oos label: {label}")
                elif label not in label_domains:
                    raise ValueError(f"Unknown supported label: {label}")
                sample_id = hashlib.sha256(f"{version}:{source_key}:{index}".encode()).hexdigest()
                result[split].append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "label": label,
                        "domain": label_domains.get(label),
                        "split": split,
                        "source_index": index,
                        "dataset_version": version,
                    }
                )
    catalog = sorted({row["label"] for row in result["train"] if row["label"] != "oos"})
    if catalog != sorted(label_domains):
        raise ValueError("Training catalog does not match the public domain mapping")
    for split, rows in result.items():
        if {r["label"] for r in rows if r["label"] != "oos"} != set(catalog):
            raise ValueError(f"Missing supported labels in {split}")
    return result, catalog


def audit_duplicates(splits):
    """Report IDs only, including conflicts. Preserve the official split membership."""
    report = {}
    for mode, transform in (("exact", lambda text: text), ("normalized", normalize)):
        groups = defaultdict(list)
        for rows in splits.values():
            for row in rows:
                groups[transform(row["text"])].append(row)
        duplicates = []
        for rows in groups.values():
            if len(rows) > 1:
                duplicates.append(
                    {
                        "sample_ids": [r["sample_id"] for r in rows],
                        "splits": sorted({r["split"] for r in rows}),
                        "conflicting_labels": len({r["label"] for r in rows}) > 1,
                        "cross_split": len({r["split"] for r in rows}) > 1,
                        "within_split": len(rows) > len({r["split"] for r in rows}),
                    }
                )
        report[mode] = {
            "groups": duplicates,
            "group_count": len(duplicates),
            "cross_split_groups": sum(d["cross_split"] for d in duplicates),
            "within_split_groups": sum(d["within_split"] for d in duplicates),
            "conflicting_label_groups": sum(d["conflicting_labels"] for d in duplicates),
        }
    return report


def prepare(cfg, output: Path, source_dir: Path | None = None):
    if not re.fullmatch(r"[0-9a-f]{40}", cfg["commit"]):
        raise ValueError("Dataset source must pin a full Git commit")
    destination = new_directory(output)
    raw_dir = new_directory(destination / "raw")
    sources = {}
    for name, info in cfg["files"].items():
        if Path(name).name != name:
            raise ValueError("Source filenames must be simple basenames")
        url = f"https://raw.githubusercontent.com/clinc/oos-eval/{cfg['commit']}/{info['path']}"
        target = raw_dir / name
        if source_dir is not None:
            shutil.copyfile(source_dir / name, target)
        else:
            with urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
        verify_hash(target, info["sha256"])
        sources[name] = {"url": url, "sha256": sha256(target)}
    splits, catalog = canonicalize(
        read_json(raw_dir / "data_full.json"),
        read_json(raw_dir / "domains.json"),
        cfg["commit"],
        cfg["expected_counts"],
        cfg["expected_intents"],
    )
    duplicates = audit_duplicates(splits)
    write_json(destination / "duplicates.json", duplicates)
    write_json(destination / "catalog.json", catalog)
    shutil.copyfile(raw_dir / "LICENSE", destination / "LICENSE")
    details = {}
    for split, rows in splits.items():
        path = destination / f"{split}.jsonl"
        write_jsonl(path, rows)
        lengths = [len(r["text"]) for r in rows]
        details[split] = {
            "sha256": sha256(path),
            "count": len(rows),
            "class_frequencies": dict(sorted(Counter(r["label"] for r in rows).items())),
            "text_length_characters": dict(
                zip(
                    ("min", "median", "p95", "max"),
                    map(float, np.percentile(lengths, [0, 50, 95, 100])),
                    strict=True,
                )
            ),
        }
    manifest = {
        "dataset": "CLINC150 full",
        "fixture": cfg.get("fixture", False),
        "repository": cfg["repository"],
        "commit": cfg["commit"],
        "license": cfg["license"],
        "attribution": "Larson et al. (2019), CLINC / clinc/oos-eval",
        "preprocessing_version": cfg["preprocessing_version"],
        "sources": sources,
        "expected_source_counts": cfg["expected_counts"],
        "splits": details,
        "catalog_sha256": sha256(destination / "catalog.json"),
        "duplicates_sha256": sha256(destination / "duplicates.json"),
        "duplicate_summary": {
            mode: {k: v for k, v in value.items() if k != "groups"}
            for mode, value in duplicates.items()
        },
        "normalization_for_audit_only": "Unicode NFKC, casefold, collapse whitespace",
        "raw_text_preserved": True,
        "membership_preserved": True,
        "limitations": [
            "Only 100 training oos examples.",
            "Public benchmark; pretraining contamination cannot be ruled out.",
            "Test data inspected only for preparation integrity, not model evaluation.",
        ],
    }
    write_json(destination / "manifest.json", manifest)
    return manifest
