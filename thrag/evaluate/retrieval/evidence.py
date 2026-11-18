"""Match annotated evidence to the text blocks from the saved index."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from unicodedata import normalize

Span = tuple[int, int]

QUOTE_CHARS = "\"'“”‘’"
DASH_CHARS = "-‐‑‒–—−"


def tolerant_expression(text: str) -> str:
    """Encode only known graph-sanitization differences as regex alternatives."""
    parts: list[str] = []
    previous_whitespace = False
    for char in normalize("NFC", text):
        if char.isspace():
            if not previous_whitespace:
                parts.append(r"\s+")
            previous_whitespace = True
        elif char in QUOTE_CHARS:
            parts.append(r"""["'“”‘’]""")
            previous_whitespace = False
        elif char in DASH_CHARS:
            parts.append(r"[-‐‑‒–—−]")
            previous_whitespace = False
        else:
            parts.append(re.escape(char))
            previous_whitespace = False
    return "".join(parts)


def tolerant_find(text: str, piece: str, start: int = 0) -> re.Match[str] | None:
    if not piece:
        return None
    return re.compile(tolerant_expression(piece)).search(text, start)


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
            match = tolerant_find(self.text, piece, earliest)
            if match is None:
                raise ValueError(f"{chunk_id} does not align with the saved contexts.txt")
            start, end = match.span()
            split = content.startswith("\ufffd") or previous_split
            gap = self.text[previous_end:start]
            if gap.strip() and not (split and len(gap) == 1):
                raise ValueError(f"{chunk_id} does not align with the saved contexts.txt")
            previous_start, previous_end = start, end
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
        expression = r"\s+".join(tolerant_expression(word) for word in words)
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
