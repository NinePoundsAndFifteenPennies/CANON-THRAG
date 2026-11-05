"""Recover presentation order from extraction provenance, not hash/label order."""

from __future__ import annotations

from typing import Any

from canon.core.io import stable_id


def occurrence_order(blocks: list[dict[str, Any]]) -> dict[str, int]:
    order: dict[str, int] = {}
    for block in blocks:
        for triple_position, item in enumerate(block["triples"]):
            sentence = item["sentence"]
            count = len(sentence) if isinstance(sentence, list) else 1
            for sentence_position in range(count):
                identity = stable_id("occurrence", block["chunk_id"], triple_position, sentence_position)
                if identity in order:
                    raise ValueError(f"Duplicate extraction occurrence: {identity}")
                order[identity] = len(order)
    return order
