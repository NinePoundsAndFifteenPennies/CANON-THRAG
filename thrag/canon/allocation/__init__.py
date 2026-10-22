"""Independently switch bridge reranking and branch-balanced chunk allocation."""

from __future__ import annotations

from canon.allocation.bridge import bridge_scores
from canon.allocation.coverage import choose_chunks
from canon.core.config import CanonConfig
from canon.core.contracts import EvidenceSelection, QueryContext, RoutingResult
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


class Allocator:
    def __init__(self, store: EvidenceStore, hierarchy: HierarchyArtifacts, config: CanonConfig) -> None:
        self.store, self.hierarchy, self.config = store, hierarchy, config

    def select(self, context: QueryContext, routing: RoutingResult, indices: list[int], k1: int, k2: int) -> EvidenceSelection:
        if not 1 <= k2 <= k1:
            raise ValueError("Retrieval budgets must satisfy 1 <= K2 <= K1")
        scores, bridge_entities = bridge_scores(context, routing, indices, self.store, self.hierarchy, self.config)
        ranked = sorted(indices, key=lambda index: (-scores[index], self.store.records[index].id))
        covered = None
        if self.config.balance:
            chunks, mandatory, covered = choose_chunks(
                indices, scores, routing.paths, self.store, self.hierarchy, k2,
                self.config.coverage_weight, 1 + (self.config.bridge_weight if self.config.bridge else 0),
            )
            selected = set(mandatory)
            for index in ranked:
                if len(selected) >= k1:
                    break
                selected.add(index)
            ordered = [index for index in ranked if index in selected]
        else:
            ordered = ranked[:k1]
            chunks = []
            seen = set()
            for index in ordered:
                for chunk in sorted({item.chunk_id for item in self.store.records[index].occurrences}):
                    if chunk not in seen and len(chunks) < k2:
                        seen.add(chunk)
                        chunks.append(chunk)
        return EvidenceSelection(ordered, chunks, scores, {
            "bridge": self.config.bridge, "balance": self.config.balance,
            "bridge_entities": bridge_entities, "covered_paths": covered,
            "candidate_sentences": len(indices), "unique_chunks": len(chunks),
        })
