"""One-hop bonuses from actual fact endpoints, never sentence-level Cartesian products."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext, RoutingResult
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


def _words(text: str) -> str:
    return " " + " ".join(re.findall(r"\w+", text.casefold())) + " "


def bridge_scores(
    context: QueryContext, routing: RoutingResult, indices: list[int], store: EvidenceStore,
    hierarchy: HierarchyArtifacts, config: CanonConfig,
) -> tuple[dict[int, float], list[str]]:
    scores = {index: float(context.sentence_scores[index]) for index in indices}
    if not config.bridge or config.bridge_weight == 0:
        return scores, []
    selected = set(routing.paths)
    anchors = sorted(indices, key=lambda index: (-scores[index], store.records[index].id))[:config.bridge_seed_sentences]
    support: dict[str, list[tuple[int, float, frozenset[str]]]] = defaultdict(list)
    query = _words(context.query)
    for index in anchors:
        strength = max(0.0, min(1.0, scores[index]))
        if strength == 0:
            continue
        for occurrence in store.records[index].occurrences:
            branches = selected.intersection(hierarchy.occurrence_paths[occurrence.id])
            if not branches:
                continue
            subtopics = frozenset(path[1] for path in branches)
            for entity in (occurrence.source_id, occurrence.target_id):
                if any(_words(name).strip() and _words(name) in query for name in store.entity_labels[entity]):
                    continue
                degree = len(store.entity_neighbors[entity])
                support[entity].append((index, strength / math.log2(2 + degree), subtopics))
    for index in indices:
        bonus = 0.0
        for occurrence in store.records[index].occurrences:
            branches = selected.intersection(hierarchy.occurrence_paths[occurrence.id])
            for entity in (occurrence.source_id, occurrence.target_id):
                for seed_index, strength, seed_subtopics in support.get(entity, ()):
                    if index == seed_index:
                        continue
                    if any(path[1] != previous for path in branches for previous in seed_subtopics):
                        bonus = max(bonus, strength)
        scores[index] += config.bridge_weight * bonus
    return scores, sorted(support)
