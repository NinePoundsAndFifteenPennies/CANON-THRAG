"""Match annotated evidence to the text blocks from the saved index."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from unicodedata import normalize

Span = tuple[int, int]


@dataclass(frozen=True)
class GoldQuestion:
    # Each evidence item can occur in more than one overlapping chunk.
    evidence_chunks: tuple[frozenset[str], ...]
    complete: bool

    @property
    def aligned(self) -> bool:
        return self.complete and bool(self.evidence_chunks) and all(self.evidence_chunks)


class CorpusChunks:
    def __init__(self, text: str, kv_store: dict[str, dict[str, str]]) -> None:
        self.text = normalize("NFC", text)
        self.ids: list[str] = []
        self.starts: list[int] = []
        self.ends: list[int] = []

        # The KV store preserves the order of the original sliding windows.
        previous_start = -1
        previous_end = 0
        previous_split = False
        for chunk_id, record in kv_store.items():
            content = record["content"]
            # A token window may cut through the bytes of a Unicode character.
            piece = normalize("NFC", content.strip("\ufffd"))
            earliest = max(previous_start + 1, previous_end - len(piece) + 1)
            start = self.text.find(piece, earliest)
            if not piece or start < 0:
                raise ValueError(f"{chunk_id} does not align with the saved contexts.txt")
            split = content.startswith("\ufffd") or previous_split
            gap = self.text[previous_end:start]
            if gap.strip() and not (split and len(gap) == 1):
                raise ValueError(f"{chunk_id} does not align with the saved contexts.txt")
            previous_start, previous_end = start, start + len(piece)
            previous_split = content.endswith("\ufffd")
            self.ids.append(chunk_id)
            self.starts.append(previous_start)
            self.ends.append(previous_end)

        if not self.ids or self.text[previous_end:].strip("\ufffd").strip():
            raise ValueError("The KV store does not cover the saved contexts.txt")

    def containing_evidence(self, evidence: str, regions: Iterable[Span]) -> frozenset[str]:
        words = normalize("NFC", evidence).split()
        if not words:
            return frozenset()
        expression = r"\s+".join(map(re.escape, words))
        if re.match(r"\w", words[0][0]):
            expression = r"(?<!\w)" + expression
        if re.match(r"\w", words[-1][-1]):
            expression += r"(?!\w)"
        pattern = re.compile(expression)
        matches = set()
        for start, end in regions:
            for match in pattern.finditer(self.text, start, end):
                first = bisect_left(self.ends, match.end())
                last = bisect_right(self.starts, match.start())
                matches.update(self.ids[first:last])
        return frozenset(matches)
