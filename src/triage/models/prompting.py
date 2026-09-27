"""Catalog and prompt contracts; no torch/transformers import in the CPU environment."""

import json
from pathlib import Path

from triage.io import object_hash


def system_prompt(catalog: list[str], path: Path) -> str:
    if not catalog or catalog != sorted(set(catalog)) or "oos" in catalog:
        raise ValueError("Require a sorted unique supported catalog")
    template = path.read_text(encoding="utf-8")
    if template.count("{catalog}") != 1:
        raise ValueError("Prompt must contain one catalog placeholder")
    return template.replace("{catalog}", "\n".join(catalog))


def messages(system: str, text: str) -> list[dict[str, str]]:
    content = json.dumps({"request": text}, ensure_ascii=False)
    content = content.replace("<", "\\u003c").replace(">", "\\u003e")
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def tokenize(tokenizer, system: str, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(system, text),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_metadata(system: str, tokenizer) -> dict:
    return {
        "system_prompt_sha256": object_hash(system),
        "chat_template_sha256": object_hash(tokenizer.chat_template),
        "enable_thinking": False,
    }
