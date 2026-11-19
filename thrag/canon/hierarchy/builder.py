"""Original hierarchy and factual transitions used by the routing index."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from canon.core.contracts import Branch
from canon.core.evidence import EvidenceStore
from canon.core.io import atomic_array, atomic_json, read_json
from canon.hierarchy.enrich import branch_material
from index.json_to_gexf import clean_id


def factual_transitions(
    members: dict[str, set[str]], memberships: dict[str, set[str]], specificity: dict[str, float],
    entity_neighbors: dict[str, set[str]],
) -> dict[str, list[tuple[str, float]]]:
    """Project shared entities and unique predicate neighbors, one sparse row at a time."""
    transitions = {}
    for subtopic, entities in members.items():
        strengths: dict[str, float] = defaultdict(float)
        for entity in sorted(entities):
            for other in memberships[entity]:
                strengths[other] += specificity[entity]
            neighbors = entity_neighbors.get(entity, ())
            for neighbor in sorted(neighbors):
                weight = 1 / math.sqrt(len(neighbors) * len(entity_neighbors[neighbor]))
                for other in memberships[neighbor]:
                    strengths[other] += weight
        strengths.pop(subtopic, None)
        transitions[subtopic] = sorted(strengths.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    return transitions


class HierarchyArtifacts:
    def __init__(
        self, nodes: dict, edges: list, occurrence_paths: dict[str, list[Branch]], branch_rows: list[dict],
        transitions: dict, profile_vectors: np.ndarray, store: EvidenceStore,
    ) -> None:
        self.nodes, self.edges, self.occurrence_paths = nodes, edges, occurrence_paths
        self.branch_rows, self.transitions = branch_rows, transitions
        self.order = sorted(nodes)
        self.positions = {node: i for i, node in enumerate(self.order)}
        self.branches = [(row["topic_id"], row["subtopic_id"]) for row in branch_rows]
        self.branch_synopses = {path: row["synopsis"] for path, row in zip(self.branches, branch_rows, strict=True)}
        if profile_vectors.shape != (len(self.branches), 3, store.dimension):
            raise ValueError("Branch vectors must contain a synopsis and two aspects per path")
        self.profile_vectors = profile_vectors
        self.label_vectors = np.stack([store.text_vector(nodes[node]["label"]) for node in self.order])
        self.branch_node_indices = np.asarray([(self.positions[t], self.positions[s]) for t, s in self.branches])

        self.graph = nx.Graph()
        for node in sorted(nodes, key=lambda key: nodes[key]["order"]):
            self.graph.add_node(node, type=nodes[node]["kind"], label=nodes[node]["label"])
        for entity, names in store.entity_labels.items():
            self.graph.add_node(entity, type="entity", label=min(names))
        self.children: dict[str, list[str]] = defaultdict(list)
        members: dict[str, set[str]] = defaultdict(set)
        for source, target, relation, weight in edges:
            self.graph.add_edge(source, target, relation_type=relation, weight=weight)
            if relation == "hierarchy":
                self.children[source].append(target)
            elif relation == "membership":
                members[source].add(target)
        shared_members = {subtopic: frozenset(entities) for subtopic, entities in members.items()}
        self.branch_entities = {path: shared_members[path[1]] for path in self.branches}
        branch_sentences: dict[Branch, set[int]] = defaultdict(set)
        for i, record in enumerate(store.records):
            for item in record.occurrences:
                for path in occurrence_paths[item.id]:
                    branch_sentences[path].add(i)
        self.branch_sentences = {path: sorted(indices) for path, indices in branch_sentences.items()}

    def semantic_scores(self, vector: np.ndarray) -> np.ndarray:
        labels = (self.label_vectors @ vector)[self.branch_node_indices].mean(axis=1)
        views = (self.profile_vectors.reshape(-1, vector.size) @ vector).reshape(-1, 3)
        return (labels + views[:, 0] + views[:, 1:].max(axis=1)) / 3

    def node_maxima(self, branch_scores: np.ndarray) -> np.ndarray:
        maxima = np.full(len(self.order), -np.inf, dtype=branch_scores.dtype)
        for endpoints in self.branch_node_indices.T:
            np.maximum.at(maxima, endpoints, branch_scores)
        return maxima

    def entities_for(self, paths: list[Branch]) -> set[str]:
        return {entity for path in paths for entity in self.branch_entities[path]}

    def save(self, directory: Path) -> None:
        atomic_array(directory / "branch_views.npy", self.profile_vectors)
        atomic_json(directory / "hierarchy.json", {
            "nodes": self.nodes, "edges": self.edges, "occurrence_paths": self.occurrence_paths,
            "branches": self.branch_rows, "transitions": self.transitions,
        })

    @classmethod
    def load(cls, directory: Path, store: EvidenceStore) -> HierarchyArtifacts:
        data = read_json(directory / "hierarchy.json")
        return cls(
            data["nodes"], data["edges"],
            {key: [tuple(path) for path in paths] for key, paths in data["occurrence_paths"].items()},
            data["branches"], data["transitions"], np.load(directory / "branch_views.npy", mmap_mode="r"), store,
        )


def build_hierarchy(
    store: EvidenceStore, encoder: Any, distill: Callable[[list[dict]], dict[int, dict]], *,
    source_order: dict[str, int] | None = None,
) -> HierarchyArtifacts:
    nodes: dict[str, dict] = {}
    occurrence_paths: dict[str, list[Branch]] = {}
    members: dict[str, set[str]] = defaultdict(set)
    memberships: dict[str, set[str]] = defaultdict(set)
    predicates: set[tuple[str, str]] = set()
    occurrences = [item for record in store.records for item in record.occurrences]
    if source_order is not None:
        occurrences.sort(key=lambda item: source_order[item.id])
    for rank, item in enumerate(occurrences):
        paths = []
        endpoints = ((item.subject_topic, item.subject_subtopic, item.source_id),
                     (item.object_topic, item.object_subtopic, item.target_id))
        for side, (topic, subtopic, entity) in enumerate(endpoints):
            order = 2 * rank + side
            topic_id, child_id = "topic_" + clean_id(topic), "subtopic_" + clean_id(subtopic)
            for node_id, kind, label in ((topic_id, "topic", topic), (child_id, "subtopic", subtopic)):
                nodes.setdefault(node_id, {"kind": kind, "order": order})["label"] = label
            path = topic_id, child_id
            paths.append(path)
            members[child_id].add(entity)
            memberships[entity].add(child_id)
        occurrence_paths[item.id] = paths
        predicates.add(tuple(sorted((item.source_id, item.target_id))))

    path_order = dict.fromkeys(path for paths in occurrence_paths.values() for path in paths)
    branches = sorted(path_order)
    profiles = distill(branch_material(branches, nodes, occurrence_paths, store))
    rows = [
        {"topic_id": topic, "subtopic_id": subtopic, **profiles[i]}
        for i, (topic, subtopic) in enumerate(branches)
    ]
    views = [text for row in rows for text in (row["synopsis"], *row["aspects"])]
    vectors = encoder.encode_passages(views, show_progress=True).reshape(len(rows), 3, store.dimension)
    specificity = {e: 1 / math.log2(2 + len(parents)) for e, parents in memberships.items()}
    edges = [(topic, child, "hierarchy", 1.0) for topic, child in path_order]
    edges.extend((s, e, "membership", specificity[e]) for s in sorted(members) for e in sorted(members[s]))
    edges.extend((u, v, "predicate", 1.0) for u, v in sorted(predicates))
    transitions = factual_transitions(members, memberships, specificity, store.entity_neighbors)
    return HierarchyArtifacts(nodes, edges, occurrence_paths, rows, transitions, vectors, store)
