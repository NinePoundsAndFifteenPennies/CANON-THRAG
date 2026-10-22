"""Compose independent hierarchy, routing, and evidence-allocation stages."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from canon.allocation import Allocator
from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext
from canon.core.evidence import EvidenceStore
from canon.core.io import unit_rows
from canon.core.telemetry import record_calls
from canon.hierarchy import HierarchyArtifacts
from canon.routing import make_router


class RetrievalFailure(RuntimeError):
    def __init__(self, cause: Exception, diagnostics: dict[str, Any]) -> None:
        super().__init__(f"Routing failed: {type(cause).__name__}")
        self.error_type = type(cause).__name__
        self.diagnostics = diagnostics


class CanonRetriever:
    def __init__(
        self, store: EvidenceStore, hierarchy: HierarchyArtifacts, encoder: Any,
        config: CanonConfig, runtime: Any, client: Any = None,
    ) -> None:
        self.store, self.hierarchy, self.encoder = store, hierarchy, encoder
        self.config, self.runtime = config, runtime
        self.router = make_router(store, hierarchy, config, runtime, client)
        self.allocator = Allocator(store, hierarchy, config)

    def retrieve(self, query: str, top_k1: int | None = None, top_k2: int | None = None) -> dict[str, Any]:
        k1 = self.runtime.top_k1 if top_k1 is None else top_k1
        k2 = self.runtime.top_k2 if top_k2 is None else top_k2
        if not query.strip() or not 1 <= k2 <= k1:
            raise ValueError("A nonempty query and 1 <= K2 <= K1 are required")
        started = perf_counter()
        vector = unit_rows(self.encoder.encode_queries([query]))[0]
        if vector.shape != (self.store.dimension,):
            raise ValueError("Query embedding dimension differs from the prepared evidence")
        # One query embedding and one global score vector are reused by all stages.
        context = QueryContext(query, vector, np.asarray(self.store.vectors @ vector))
        scored_at = perf_counter()
        calls: list[dict[str, Any]] = []
        try:
            with record_calls(calls, "routing"):
                routing = self.router.route(context)
        except Exception as exc:
            raise RetrievalFailure(exc, {
                "routing": {"mode": self.config.routing, "failed": True},
                "routing_calls": len(calls), "calls": calls,
                "seconds": {"retrieval": perf_counter() - started},
            }) from exc
        routed_at = perf_counter()
        candidates = self.store.candidates(self.hierarchy.entities_for(routing.paths))
        fallback = None
        if not candidates:
            candidates = list(range(len(self.store.records)))
            fallback = "empty_routed_pool_global_evidence"
        selection = self.allocator.select(context, routing, candidates, k1, k2)
        finished = perf_counter()
        topics = list(dict.fromkeys(self.hierarchy.nodes[topic]["label"] for topic, _ in routing.paths))
        subtopics: dict[str, list[str]] = {}
        for topic, subtopic in routing.paths:
            label = self.hierarchy.nodes[topic]["label"]
            values = subtopics.setdefault(label, [])
            child_label = self.hierarchy.nodes[subtopic]["label"]
            if child_label not in values:
                values.append(child_label)
        return {
            "chunks": selection.chunks,
            "edges": [self.store.edge_result(index, selection.scores[index], rank) for rank, index in enumerate(selection.indices, start=1)],
            "topics": topics, "subtopics": subtopics,
            "paths": [{"topic_id": topic, "subtopic_id": subtopic, "score": routing.scores.get((topic, subtopic))} for topic, subtopic in routing.paths],
            "diagnostics": {
                "routing": routing.diagnostics, "allocation": selection.diagnostics, "fallback": fallback,
                "routing_calls": len(calls), "calls": calls,
                "seconds": {"embedding_and_scoring": scored_at - started, "routing": routed_at - scored_at,
                            "allocation": finished - routed_at, "retrieval": finished - started},
                "entity_filter_size": len(self.hierarchy.entities_for(routing.paths)),
            },
        }
