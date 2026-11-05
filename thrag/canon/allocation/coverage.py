"""Quality-weighted, diminishing-return coverage of selected hierarchy paths."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from canon.core.contracts import Branch
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


def choose_chunks(
    indices: list[int], scores: dict[int, float], paths: list[Branch], store: EvidenceStore,
    hierarchy: HierarchyArtifacts, k2: int, coverage_weight: float, score_scale: float,
    *, raw_scores: dict[int, float] | None = None, anchor_fraction: float = 0.5,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[str], set[int], int]:
    raw_scores = scores if raw_scores is None else raw_scores
    selected_paths = set(paths)
    relevance: dict[str, float] = {}
    support: dict[str, dict[Branch, float]] = defaultdict(dict)
    representative: dict[str, int] = {}
    raw_representative: dict[str, int] = {}
    branch_strength: dict[Branch, float] = defaultdict(float)
    for index in indices:
        quality = max(0.0, min(1.0, scores[index] / score_scale))
        for occurrence in store.records[index].occurrences:
            chunk = occurrence.chunk_id
            relevance[chunk] = max(relevance.get(chunk, 0.0), quality)
            previous = representative.get(chunk)
            if previous is None or (-scores[index], store.records[index].id) < (-scores[previous], store.records[previous].id):
                representative[chunk] = index
            raw_previous = raw_representative.get(chunk)
            if raw_previous is None or (-raw_scores[index], store.records[index].id) < (-raw_scores[raw_previous], store.records[raw_previous].id):
                raw_representative[chunk] = index
            for path in selected_paths.intersection(hierarchy.occurrence_paths[occurrence.id]):
                support[chunk][path] = max(support[chunk].get(path, 0.0), quality)
                branch_strength[path] = max(branch_strength[path], max(0.0, min(1.0, raw_scores[index])))
    total_strength = sum(branch_strength.values())
    weights = {path: strength / total_strength for path, strength in branch_strength.items()} if total_strength else {}
    chosen: list[str] = []
    covered: dict[Branch, float] = defaultdict(float)
    remaining = set(relevance)

    def include(chunk: str) -> None:
        chosen.append(chunk)
        remaining.remove(chunk)
        for path, value in support[chunk].items():
            covered[path] = max(covered[path], value)

    anchor_count = min(k2, len(remaining), math.ceil(k2 * anchor_fraction))
    anchors = sorted(remaining, key=lambda chunk: (
        -raw_scores[raw_representative[chunk]], store.records[raw_representative[chunk]].id, chunk,
    ))[:anchor_count]
    for chunk in anchors:
        include(chunk)
    for _ in range(min(k2 - len(chosen), len(remaining))):
        def gain(chunk: str) -> float:
            return (1 - coverage_weight) * relevance[chunk] + coverage_weight * sum(
                weights.get(path, 0.0) * max(0.0, value - covered[path]) for path, value in support[chunk].items()
            )
        best = min(remaining, key=lambda chunk: (-gain(chunk), chunk))
        include(best)
    if diagnostics is not None:
        diagnostics.update(
            protected_chunks=anchors, coverage_normalized=True,
            coverage_path_weight_sum=sum(weights.values()), active_coverage_paths=len(weights),
        )
    # At most one required sentence per chunk, including its original top match for anchors.
    required = {raw_representative[chunk] if chunk in anchors else representative[chunk] for chunk in chosen}
    return chosen, required, sum(value > 0 for value in covered.values())
