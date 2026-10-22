"""Shared scores and ID-based path budgets; no dependency on a route mode."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from canon.core.config import CanonConfig
from canon.core.contracts import Branch, QueryContext
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts


def activation_scores(
    context: QueryContext, store: EvidenceStore, hierarchy: HierarchyArtifacts, config: CanonConfig,
) -> dict[str, float]:
    labels = hierarchy.label_vectors @ context.vector
    scores: dict[str, float] = {}
    for position, node_id in enumerate(hierarchy.order):
        indices = hierarchy.nodes[node_id]["sentences"]
        values = context.sentence_scores[indices]
        count = min(config.activation_top_n, len(values))
        evidence = float(np.partition(values, len(values) - count)[-count:].mean()) if count else 0.0
        scores[node_id] = config.activation_label_weight * float(labels[position]) + (1 - config.activation_label_weight) * evidence
    return scores


def ranked_paths(hierarchy: HierarchyArtifacts, scores: dict[str, float], topics: int, subtopics: int) -> list[Branch]:
    selected_topics = sorted(hierarchy.children, key=lambda node: (-scores.get(node, 0.0), node))[:topics]
    children = {
        topic: sorted(hierarchy.children[topic], key=lambda node: (-scores.get(node, 0.0), node))[:subtopics]
        for topic in selected_topics
    }
    # Interleave by child rank so prompt trimming preserves strong paths from multiple topics.
    return [(topic, children[topic][rank]) for rank in range(subtopics) for topic in selected_topics if rank < len(children[topic])]


def bounded_paths(paths: list[Branch], topic_limit: int, subtopic_limit: int) -> list[Branch]:
    selected: dict[str, set[str]] = defaultdict(set)
    result: list[Branch] = []
    for topic, subtopic in paths:
        if topic not in selected and len(selected) >= topic_limit:
            continue
        if subtopic not in selected[topic] and len(selected[topic]) < subtopic_limit:
            selected[topic].add(subtopic)
            result.append((topic, subtopic))
    return result


def path_scores(paths: list[Branch], scores: dict[str, float]) -> dict[Branch, float]:
    return {path: (scores.get(path[0], 0.0) + scores.get(path[1], 0.0)) / 2 for path in paths}


def limits(runtime: Any) -> tuple[int, int]:
    topics, subtopics = runtime.topic_choice_max, runtime.subtopic_choice_max
    if topics < 1 or subtopics < 1:
        raise ValueError("Topic/subtopic selection maxima must be positive")
    return topics, subtopics
