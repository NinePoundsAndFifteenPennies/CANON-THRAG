"""ID-based contracts shared by independently replaceable stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

Branch = tuple[str, str]


@dataclass(frozen=True)
class Occurrence:
    id: str
    chunk_id: str
    source_id: str
    target_id: str
    source: str
    target: str
    label: str
    subject_topic: str
    subject_subtopic: str
    object_topic: str
    object_subtopic: str


@dataclass(frozen=True)
class SentenceRecord:
    id: str
    sentence: str
    occurrences: tuple[Occurrence, ...]

    @property
    def entity_ids(self) -> frozenset[str]:
        return frozenset(entity for item in self.occurrences for entity in (item.source_id, item.target_id))


@dataclass
class QueryContext:
    query: str
    vector: np.ndarray
    sentence_scores: np.ndarray


@dataclass
class RoutingResult:
    paths: list[Branch]
    scores: dict[Branch, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSelection:
    indices: list[int]
    chunks: list[str]
    scores: dict[int, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Router(Protocol):
    def route(self, context: QueryContext) -> RoutingResult: ...


class EvidenceSelector(Protocol):
    def select(
        self, context: QueryContext, routing: RoutingResult, indices: list[int], k1: int, k2: int,
    ) -> EvidenceSelection: ...
