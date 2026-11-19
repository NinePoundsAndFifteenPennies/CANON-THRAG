"""Shared scores and ID-based path budgets; no dependency on a route mode."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from canon.core.config import CanonConfig
from canon.core.contracts import Branch, QueryContext
from canon.hierarchy import HierarchyArtifacts


@dataclass
class RouteScores:
    topics: dict[str, float]
    paths: dict[Branch, float]


def activation_scores(
    context: QueryContext, hierarchy: HierarchyArtifacts, config: CanonConfig,
) -> RouteScores:
    semantic = hierarchy.semantic_scores(context.vector)
    paths: dict[Branch, float] = {}
    topics: dict[str, float] = {}
    weight = config.activation_label_weight
    for path, score in zip(hierarchy.branches, semantic, strict=True):
        indices = hierarchy.branch_sentences[path]
        if len(indices) == 1:
            evidence = float(context.sentence_scores[indices[0]])
        else:
            values = context.sentence_scores[indices]
            count = min(config.activation_top_n, len(values))
            evidence = float(np.partition(values, len(values) - count)[-count:].mean())
        paths[path] = weight * float(score) + (1 - weight) * evidence
        topics[path[0]] = max(topics.get(path[0], -np.inf), paths[path])
    return RouteScores(topics, paths)


def ranked_paths(hierarchy: HierarchyArtifacts, scores: RouteScores, topics: int, subtopics: int) -> list[Branch]:
    selected_topics = sorted(hierarchy.children, key=lambda node: (-scores.topics[node], node))[:topics]
    children = {
        topic: sorted(hierarchy.children[topic], key=lambda node: (-scores.paths[topic, node], node))[:subtopics]
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


def path_scores(paths: list[Branch], scores: RouteScores) -> dict[Branch, float]:
    return {path: scores.paths[path] for path in paths}


def limits(runtime: Any) -> tuple[int, int]:
    topics, subtopics = runtime.topic_choice_max, runtime.subtopic_choice_max
    if topics < 1 or subtopics < 1:
        raise ValueError("Topic/subtopic selection maxima must be positive")
    return topics, subtopics
