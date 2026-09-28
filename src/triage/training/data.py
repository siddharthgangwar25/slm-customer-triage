"""Training-only conversational records with explicit completion-token supervision."""

import json
from pathlib import Path

from triage.data.load import load_split
from triage.io import (
    config,
    new_directory,
    object_hash,
    read_json,
    read_jsonl,
    sha256,
    verify_hash,
    write_json,
    write_jsonl,
)
from triage.models.prompted import validate_config
from triage.models.prompting import messages, prompt_metadata, system_prompt, tokenize


def tokenizer_for(pair):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = snapshot_download(
        pair["model_id"],
        revision=pair["revision"],
        cache_dir=pair["cache_dir"],
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.model"],
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def encode_record(row, system, tokenizer, max_length):
    if row["split"] != "train":
        raise ValueError("Only training records may enter SFT")
    target = json.dumps({"intent": row["label"]}, separators=(",", ":"))
    prompt_ids = tokenize(tokenizer, system, row["text"])
    target_ids = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        raise ValueError(f"Would truncate prompt/target: {row['sample_id']} ({len(ids)} tokens)")
    return {
        "sample_id": row["sample_id"],
        "split": "train",
        "label": row["label"],
        "prompt": messages(system, row["text"]),
        "completion": [{"role": "assistant", "content": target}],
        "input_ids": ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * len(target_ids),
    }


def audit_record(record, tokenizer):
    ids, mask = record["input_ids"], record["completion_mask"]
    if len(ids) != len(mask) or 0 not in mask or 1 not in mask:
        raise ValueError("Missing prompt/completion mask")
    boundary = mask.index(1)
    if mask != [0] * boundary + [1] * (len(ids) - boundary):
        raise ValueError("Completion mask must be a contiguous suffix")
    expected = record["completion"][0]["content"]
    if ids[-1] != tokenizer.eos_token_id or tokenizer.decode(ids[boundary:-1]) != expected:
        raise ValueError("Supervision must be exactly assistant JSON plus EOS")
    return {
        "sample_id": record["sample_id"],
        "label": record["label"],
        "prompt_tokens": boundary,
        "completion_tokens": len(ids) - boundary,
        "supervised_text": expected,
        "eos_supervised": True,
        "input_ids": ids,
        "labels": [-100] * boundary + ids[boundary:],
    }


def source_hash():
    root = Path(__file__).parents[1]
    paths = list((root / "training").glob("*.py")) + [
        root / "models" / name for name in ("prompting.py", "prompted.py", "finetuned.py")
    ]
    return object_hash({str(p.relative_to(root)): sha256(p) for p in sorted(paths) if p.exists()})


def identity(cfg):
    pair = config(Path(cfg["paired_config"]))
    validate_config(pair)
    return {
        "config": cfg,
        "paired_config": pair,
        "data_manifest_sha256": sha256(Path(pair["data_dir"]) / "manifest.json"),
        "prompt_file_sha256": sha256(Path(pair["prompt_file"])),
        "environment_lock_sha256": sha256(Path(cfg["environment_lock"])),
        "implementation_sha256": source_hash(),
    }


def prepare(cfg, output, tokenizer=None):
    frozen = identity(cfg)
    pair = frozen["paired_config"]
    rows, catalog, manifest = load_split(Path(pair["data_dir"]), "train", for_fit=True)
    system = system_prompt(catalog, Path(pair["prompt_file"]))
    tokenizer = tokenizer or tokenizer_for(pair)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires an EOS token")
    records = [encode_record(r, system, tokenizer, cfg["max_seq_length"]) for r in rows]
    audits = [audit_record(r, tokenizer) for r in records]
    longest = max(range(len(records)), key=lambda i: len(records[i]["input_ids"]))
    oos = next(i for i, r in enumerate(rows) if r["label"] == "oos")
    indices = list(dict.fromkeys([longest, oos, *range(len(rows))]))[:10]
    new_directory(output)
    write_jsonl(output / "train.jsonl", records)
    write_json(output / "mask_audit.json", [audits[i] for i in indices])
    (output / "system_prompt.txt").write_text(system, encoding="utf-8", newline="\n")
    tokenizer.save_pretrained(output / "tokenizer")
    result = {
        "identity": frozen,
        "split": "train",
        "fixture": manifest["fixture"],
        "catalog": catalog,
        "sample_count": len(records),
        "supported_count": sum(r["label"] != "oos" for r in rows),
        "oos_count": sum(r["label"] == "oos" for r in rows),
        "min_tokens": min(len(r["input_ids"]) for r in records),
        "max_tokens": max(len(r["input_ids"]) for r in records),
        "train_sha256": sha256(output / "train.jsonl"),
        "mask_audit_sha256": sha256(output / "mask_audit.json"),
        "tokenizer_files": {p.name: sha256(p) for p in (output / "tokenizer").iterdir()},
        **prompt_metadata(system, tokenizer),
    }
    write_json(output / "manifest.json", result)
    return {k: v for k, v in result.items() if k != "identity"}


def load_prepared(cfg):
    directory = Path(cfg["prepared_data"])
    manifest = read_json(directory / "manifest.json")
    if manifest["identity"] != identity(cfg) or manifest["split"] != "train":
        raise ValueError("Prepared data identity changed; create a new preparation")
    verify_hash(directory / "train.jsonl", manifest["train_sha256"])
    verify_hash(directory / "mask_audit.json", manifest["mask_audit_sha256"])
    for name, digest in manifest["tokenizer_files"].items():
        verify_hash(directory / "tokenizer" / name, digest)
    records = read_jsonl(directory / "train.jsonl")
    source, catalog, _ = load_split(
        Path(manifest["identity"]["paired_config"]["data_dir"]), "train", for_fit=True
    )
    if len(records) != len(source) or catalog != manifest["catalog"]:
        raise ValueError("Prepared training split mismatch")
    for record, row in zip(records, source, strict=True):
        if (record["sample_id"], record["label"], record["split"]) != (
            row["sample_id"],
            row["label"],
            "train",
        ):
            raise ValueError("Prepared training IDs/labels mismatch")
    return records, manifest
