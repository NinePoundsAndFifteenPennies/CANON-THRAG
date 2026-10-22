"""Quality-weighted, diminishing-return coverage of selected hierarchy paths."""

from __future__ import annotations

from collections import defaultdict

from canon.core.contracts import Branch
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


def choose_chunks(
    indices: list[int], scores: dict[int, float], paths: list[Branch], store: EvidenceStore,
    hierarchy: HierarchyArtifacts, k2: int, coverage_weight: float, score_scale: float,
) -> tuple[list[str], set[int], int]:
    selected_paths = set(paths)
    relevance: dict[str, float] = {}
    support: dict[str, dict[Branch, float]] = defaultdict(dict)
    representative: dict[str, int] = {}
    for index in indices:
        quality = max(0.0, min(1.0, scores[index] / score_scale))
        for occurrence in store.records[index].occurrences:
            chunk = occurrence.chunk_id
            relevance[chunk] = max(relevance.get(chunk, 0.0), quality)
            previous = representative.get(chunk)
            if previous is None or (-scores[index], store.records[index].id) < (-scores[previous], store.records[previous].id):
                representative[chunk] = index
            for path in selected_paths.intersection(hierarchy.occurrence_paths[occurrence.id]):
                support[chunk][path] = max(support[chunk].get(path, 0.0), quality)
    chosen: list[str] = []
    covered: dict[Branch, float] = defaultdict(float)
    remaining = set(relevance)
    for _ in range(min(k2, len(remaining))):
        def gain(chunk: str) -> float:
            return relevance[chunk] + coverage_weight * sum(
                max(0.0, value - covered[path]) for path, value in support[chunk].items()
            )
        best = min(remaining, key=lambda chunk: (-gain(chunk), chunk))
        chosen.append(best)
        remaining.remove(best)
        for path, value in support[best].items():
            covered[path] = max(covered[path], value)
    return chosen, {representative[chunk] for chunk in chosen}, sum(value > 0 for value in covered.values())
