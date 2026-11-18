"""HotpotQA supporting facts in the prepared corpus."""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from typing import Any
from unicodedata import normalize

from .evidence import CorpusChunks, GoldQuestion, Span


def build_gold(metadata: list[dict[str, Any]], corpus: CorpusChunks) -> list[GoldQuestion]:
    paragraphs: dict[str, list[Span]] = defaultdict(list)
    start = 0
    for paragraph in corpus.text.split("\n\n"):
        title, separator, _ = paragraph.partition("\n")
        end = start + len(paragraph)
        if separator:
            paragraphs[title].append((start + len(title) + 1, end))
        start = end + 2

    @cache
    def matching_chunks(title: str, text: str) -> frozenset[str]:
        return corpus.containing_evidence(text, paragraphs.get(normalize("NFC", title), ()))

    questions = []
    for row in metadata:
        facts = dict.fromkeys(
            (fact["title"], fact["sent_id"], fact["text"]) for fact in row["supporting_facts"]
        )
        questions.append(GoldQuestion(
            tuple(matching_chunks(title, text) for title, _, text in facts),
            row.get("supporting_facts_complete", True),
        ))
    return questions
