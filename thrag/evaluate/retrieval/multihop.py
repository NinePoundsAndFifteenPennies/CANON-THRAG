"""MultiHop-RAG evidence tied to the prepared source documents."""

from __future__ import annotations

from functools import cache
from typing import Any
from unicodedata import normalize

from .evidence import CorpusChunks, GoldQuestion, Span


def document_regions(metadata: list[dict[str, Any]], text: str) -> dict[int, Span]:
    titles: dict[int, str] = {}
    for row in metadata:
        for fact in row["evidence"]:
            index = fact["context_document_index"]
            if index is not None:
                if titles.setdefault(index, fact["title"]) != fact["title"]:
                    raise ValueError(f"Inconsistent MultiHop title at document {index}")

    # Document bodies can contain blank lines. Locate the ordered headers
    # written by prepare_multihop_1000.py rather than splitting the bodies.
    headers: list[tuple[int, int, int]] = []
    search_start = 0
    for index, title in sorted(titles.items()):
        separator = "\n\n" if headers else ""
        title = normalize("NFC", title).replace("\r\n", "\n").replace("\r", "\n")
        heading = separator + title + "\n"
        start = text.find(heading, search_start)
        if start < 0 or (not headers and start != 0):
            raise ValueError(f"MultiHop document {index} does not align with contexts.txt")
        search_start = start + len(heading)
        headers.append((index, start, search_start))

    region_ends = [start for _, start, _ in headers[1:]] + [len(text)]
    return {index: (body_start, end) for (index, _, body_start), end in zip(headers, region_ends)}


def build_gold(metadata: list[dict[str, Any]], corpus: CorpusChunks) -> list[GoldQuestion]:
    regions = document_regions(metadata, corpus.text)

    @cache
    def matching_chunks(index: int | None, text: str) -> frozenset[str]:
        return corpus.containing_evidence(text, [regions[index]]) if index in regions else frozenset()

    questions = []
    for row in metadata:
        facts = dict.fromkeys(
            (fact["context_document_index"], fact["fact"]) for fact in row["evidence"]
        )
        questions.append(GoldQuestion(
            tuple(matching_chunks(index, text) for index, text in facts),
            row.get("evidence_complete", True),
        ))
    return questions
