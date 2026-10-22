"""A: dual-seed personalized diffusion over a typed, sparse transition graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext, RoutingResult
from canon.core.evidence import EvidenceStore
from canon.hierarchy import HierarchyArtifacts
from canon.routing.common import limits, path_scores, ranked_paths


class DiffusionRouter:
    def __init__(self, store: EvidenceStore, hierarchy: HierarchyArtifacts, config: CanonConfig, runtime: Any) -> None:
        self.store, self.hierarchy, self.config, self.runtime = store, hierarchy, config, runtime
        self.order = sorted(hierarchy.graph.nodes)
        self.positions = {node: index for index, node in enumerate(self.order)}
        groups: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for source, target, relation, weight in hierarchy.edges:
            if weight <= 0:
                continue
            if relation == "soft":
                if not config.soft_links or config.soft_transition_weight == 0:
                    continue
                forward = backward = "soft"
            elif relation == "predicate":
                forward = backward = "predicate"
            else:
                forward, backward = "down", "up"
            groups[source][forward].append((target, weight))
            if source != target:
                groups[target][backward].append((source, weight))
        sources, targets, weights = [], [], []
        for source, typed_neighbors in sorted(groups.items()):
            masses = {kind: config.soft_transition_weight if kind == "soft" else 1.0 for kind in typed_neighbors}
            denominator = sum(masses.values())
            for kind, neighbors in sorted(typed_neighbors.items()):
                total = sum(weight for _, weight in neighbors)
                for target, weight in sorted(neighbors):
                    sources.append(self.positions[source])
                    targets.append(self.positions[target])
                    weights.append(masses[kind] / denominator * weight / total)
        self.sources = np.asarray(sources, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.dangling = np.ones(len(self.order), dtype=bool)
        self.dangling[self.sources] = False

    @staticmethod
    def _positive(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=np.float64), 0)
        return values / values.sum() if values.sum() else np.full(len(values), 1 / len(values))

    def seeds(self, context: QueryContext) -> np.ndarray:
        bottom, top = np.zeros(len(self.order)), np.zeros(len(self.order))
        indices = sorted(range(len(self.store.records)), key=lambda i: (-float(context.sentence_scores[i]), self.store.records[i].id))[:self.config.seed_sentences]
        if indices:
            masses = self._positive(context.sentence_scores[indices])
            for index, mass in zip(indices, masses, strict=True):
                entities = sorted(self.store.records[index].entity_ids)
                for entity in entities:
                    bottom[self.positions[entity]] += mass / len(entities)
        labels = self.hierarchy.label_vectors @ context.vector
        for kind in ("topic", "subtopic"):
            candidates = [node for node in self.hierarchy.order if self.hierarchy.nodes[node]["kind"] == kind]
            candidates.sort(key=lambda node: (-float(labels[self.hierarchy.positions[node]]), node))
            candidates = candidates[:self.config.seed_labels]
            if candidates:
                masses = self._positive(np.asarray([labels[self.hierarchy.positions[node]] for node in candidates]))
                for node, mass in zip(candidates, masses, strict=True):
                    top[self.positions[node]] += mass / 2
        seed = self.config.bottom_up_weight * bottom + (1 - self.config.bottom_up_weight) * top
        if seed.sum() == 0:
            raise ValueError("No nonzero diffusion seeds")
        return seed / seed.sum()

    def score(self, context: QueryContext) -> tuple[dict[str, float], dict[str, Any]]:
        seed = self.seeds(context)
        current = seed.copy()
        converged = False
        residual = float("inf")
        for iteration in range(1, self.config.diffusion_iterations + 1):
            incoming = np.bincount(self.targets, weights=current[self.sources] * self.weights, minlength=len(seed))
            incoming += float(current[self.dangling].sum()) * seed
            following = self.config.restart * seed + (1 - self.config.restart) * incoming
            residual = float(np.abs(following - current).sum())
            current = following
            if residual < self.config.diffusion_tolerance:
                converged = True
                break
        current /= current.sum()
        return (
            {node: float(current[self.positions[node]]) for node in self.hierarchy.order},
            {"iterations": iteration, "converged": converged, "residual_l1": residual,
             "soft_links_used": self.config.soft_links and self.config.soft_transition_weight > 0},
        )

    def route(self, context: QueryContext) -> RoutingResult:
        scores, diagnostics = self.score(context)
        topic_limit, subtopic_limit = limits(self.runtime)
        paths = ranked_paths(self.hierarchy, scores, topic_limit, subtopic_limit)
        return RoutingResult(paths, path_scores(paths, scores), {"mode": "a", **diagnostics})
