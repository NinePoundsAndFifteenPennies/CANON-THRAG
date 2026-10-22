"""Scoped splitting, complete-link prototype merging, and optional lateral edges."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from index.json_to_gexf import clean_id
from canon.core.config import CanonConfig
from canon.core.contracts import Branch
from canon.core.evidence import EvidenceStore
from canon.core.io import atomic_array, atomic_json, centroid, read_json, stable_id


@dataclass
class _Node:
    id: str
    kind: str
    labels: Counter[str] = field(default_factory=Counter)
    sentences: set[int] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    parents: set[str] = field(default_factory=set)
    members: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        return min(self.labels, key=lambda value: (-self.labels[value], value))


def _prototypes(nodes: dict[str, _Node], store: EvidenceStore) -> dict[str, tuple[np.ndarray, ...]]:
    result = {}
    for node_id, node in nodes.items():
        names = sorted({name for entity in node.entities for name in store.entity_labels[entity]})
        result[node_id] = (
            centroid(np.asarray([store.text_vector(label) for label in sorted(node.labels)]), store.dimension),
            centroid(store.vectors[sorted(node.sentences)], store.dimension),
            centroid(np.asarray([store.text_vector(name) for name in names]), store.dimension),
        )
    return result


def _similarity(left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...], config: CanonConfig) -> tuple[float, float]:
    label = float(left[0] @ right[0])
    weight = config.sentence_prototype_weight
    context = weight * float(left[1] @ right[1]) + (1 - weight) * float(left[2] @ right[2])
    return label, context


def _near_pairs(ids: list[str], prototypes: dict[str, tuple[np.ndarray, ...]], k: int) -> set[tuple[str, str]]:
    if len(ids) < 2:
        return set()
    import faiss

    matrix = np.ascontiguousarray([prototypes[node][0] for node in ids], dtype=np.float32)
    if len(ids) <= 256:
        index = faiss.IndexFlatIP(matrix.shape[1])
    else:
        # On unit vectors L2/cosine neighbor order is equivalent. Large groups
        # use HNSW candidate recall, followed by exact prototype compatibility.
        index = faiss.IndexHNSWFlat(matrix.shape[1], 32)
        index.hnsw.efConstruction = 128
        index.hnsw.efSearch = max(64, 4 * (k + 1))
    previous_threads = faiss.omp_get_max_threads()
    try:
        faiss.omp_set_num_threads(1)
        index.add(matrix)
        _distances, neighbors = index.search(matrix, min(k + 1, len(ids)))
    finally:
        faiss.omp_set_num_threads(previous_threads)
    return {
        tuple(sorted((ids[row], ids[int(column)])))
        for row, columns in enumerate(neighbors) for column in columns
        if column >= 0 and column != row
    }


def _merge(nodes: dict[str, _Node], store: EvidenceStore, config: CanonConfig) -> tuple[dict[str, _Node], dict[str, str]]:
    if not nodes:
        return {}, {}
    prototypes = _prototypes(nodes, store)
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for node_id, node in sorted(nodes.items()):
        groups[tuple(sorted(node.parents))].append(node_id)
    candidates: set[tuple[str, str]] = set()
    for ids in groups.values():
        candidates.update(_near_pairs(ids, prototypes, config.merge_neighbors))

    def compatible(a: str, b: str) -> bool:
        label, context = _similarity(prototypes[a], prototypes[b], config)
        return label >= config.merge_label_threshold and context >= config.merge_context_threshold

    def pair_key(pair: tuple[str, str]) -> tuple[float, tuple[str, str]]:
        return (-sum(_similarity(prototypes[pair[0]], prototypes[pair[1]], config)), pair)

    owners = {node_id: node_id for node_id in nodes}
    clusters = {node_id: [node_id] for node_id in nodes}
    for left, right in sorted(candidates, key=pair_key):
        a, b = owners[left], owners[right]
        if a == b or not all(compatible(u, v) for u in clusters[a] for v in clusters[b]):
            continue
        keep, drop = sorted((a, b))
        clusters[keep] = sorted(clusters[keep] + clusters.pop(drop))
        for member in clusters[keep]:
            owners[member] = keep

    result: dict[str, _Node] = {}
    mapping: dict[str, str] = {}
    for members in clusters.values():
        kind = nodes[members[0]].kind
        canonical_id = members[0] if len(members) == 1 else stable_id(kind, members)
        combined = _Node(canonical_id, kind)
        for member in members:
            node = nodes[member]
            combined.labels.update(node.labels)
            combined.sentences.update(node.sentences)
            combined.entities.update(node.entities)
            combined.parents.update(node.parents)
            combined.members.update(node.members)
            mapping[member] = canonical_id
        result[canonical_id] = combined
    return result, mapping


class HierarchyArtifacts:
    def __init__(
        self, nodes: dict[str, dict[str, Any]], label_vectors: np.ndarray,
        edges: list[tuple[str, str, str, float]], occurrence_paths: dict[str, list[Branch]],
        canonical_map: dict[str, Any], store: EvidenceStore,
    ) -> None:
        self.nodes = nodes
        self.order = sorted(nodes)
        if label_vectors.shape != (len(self.order), store.dimension) or not np.isfinite(label_vectors).all():
            raise ValueError("Hierarchy label vectors are not aligned with nodes")
        self.label_vectors = label_vectors
        self.label_vectors.flags.writeable = False
        self.positions = {node_id: index for index, node_id in enumerate(self.order)}
        self.edges = edges
        self.occurrence_paths = occurrence_paths
        self.canonical_map = canonical_map
        self.graph = nx.Graph()
        for node_id, node in nodes.items():
            self.graph.add_node(node_id, type=node["kind"], label=node["label"])
        for entity_id, names in sorted(store.entity_labels.items()):
            self.graph.add_node(entity_id, type="entity", label=min(names))
        self.children: dict[str, list[str]] = defaultdict(list)
        sub_entities: dict[str, set[str]] = defaultdict(set)
        for source, target, relation, weight in edges:
            self.graph.add_edge(source, target, relation_type=relation, weight=weight)
            if relation == "hierarchy":
                self.children[source].append(target)
            elif relation == "membership":
                sub_entities[source].add(target)
        for children in self.children.values():
            children.sort()
        self.branches: list[Branch] = sorted((topic, child) for topic, children in self.children.items() for child in children)
        # Without splitting, a shared Subtopic keeps its original union of entities.
        self.branch_entities = {path: frozenset(sub_entities[path[1]]) for path in self.branches}
        self.branch_sentences: dict[Branch, set[int]] = defaultdict(set)
        for index, record in enumerate(store.records):
            for occurrence in record.occurrences:
                for path in occurrence_paths[occurrence.id]:
                    if path not in self.branch_entities:
                        raise ValueError(f"Occurrence refers to a nonexistent path: {path}")
                    self.branch_sentences[path].add(index)

    def entities_for(self, paths: list[Branch]) -> set[str]:
        return {entity for path in paths for entity in self.branch_entities[path]}

    def save(self, directory: Path) -> None:
        atomic_json(directory / "hierarchy.json", {
            "nodes": self.nodes, "edges": self.edges,
            "occurrence_paths": self.occurrence_paths,
        })
        atomic_json(directory / "canonical_map.json", self.canonical_map)
        atomic_array(directory / "hierarchy_labels.npy", self.label_vectors)

    @classmethod
    def load(cls, directory: Path, store: EvidenceStore) -> HierarchyArtifacts:
        data = read_json(directory / "hierarchy.json")
        return cls(
            data["nodes"], np.load(directory / "hierarchy_labels.npy", allow_pickle=False, mmap_mode="r"),
            [tuple(edge) for edge in data["edges"]],
            {key: [tuple(path) for path in paths] for key, paths in data["occurrence_paths"].items()},
            read_json(directory / "canonical_map.json"), store,
        )


def build_hierarchy(store: EvidenceStore, config: CanonConfig) -> HierarchyArtifacts:
    topics: dict[str, _Node] = {}
    subtopics: dict[str, _Node] = {}
    original_paths: dict[str, list[Branch]] = {}
    old_topics: dict[str, set[str]] = defaultdict(set)
    old_subtopics: dict[str, set[str]] = defaultdict(set)

    def add_path(topic: str, subtopic: str, entity: str, index: int) -> Branch:
        if config.hierarchy == "none":
            topic_id, subtopic_id = f"topic_{clean_id(topic)}", f"subtopic_{clean_id(subtopic)}"
        else:
            topic_id = stable_id("topic", " ".join(topic.casefold().split()))
            subtopic_id = stable_id("subtopic", topic_id, " ".join(subtopic.casefold().split()))
        for collection, node_id, kind, label in (
            (topics, topic_id, "topic", topic), (subtopics, subtopic_id, "subtopic", subtopic),
        ):
            node = collection.setdefault(node_id, _Node(node_id, kind))
            node.labels[label] += 1
            node.sentences.add(index)
            node.entities.add(entity)
            node.members.add(node_id)
        subtopics[subtopic_id].parents.add(topic_id)
        old_topics[f"topic_{clean_id(topic)}"].add(topic_id)
        old_subtopics[f"subtopic_{clean_id(subtopic)}"].add(subtopic_id)
        return topic_id, subtopic_id

    for index, record in enumerate(store.records):
        for item in record.occurrences:
            original_paths[item.id] = [
                add_path(item.subject_topic, item.subject_subtopic, item.source_id, index),
                add_path(item.object_topic, item.object_subtopic, item.target_id, index),
            ]
    before = {"topics": len(topics), "subtopics": len(subtopics)}
    topic_map = {key: key for key in topics}
    subtopic_map = {key: key for key in subtopics}
    if config.hierarchy == "split_merge":
        topics, topic_map = _merge(topics, store, config)
        for node in subtopics.values():
            node.parents = {topic_map[parent] for parent in node.parents}
        subtopics, subtopic_map = _merge(subtopics, store, config)
    nodes = {**topics, **subtopics}
    prototypes = _prototypes(nodes, store)
    paths = {
        occurrence_id: [(topic_map[topic], subtopic_map[subtopic]) for topic, subtopic in pairs]
        for occurrence_id, pairs in original_paths.items()
    }
    edge_set: set[tuple[str, str, str, float]] = set()
    for record in store.records:
        for item in record.occurrences:
            for entity, (topic, subtopic) in zip((item.source_id, item.target_id), paths[item.id], strict=True):
                edge_set.add((topic, subtopic, "hierarchy", 1.0))
                edge_set.add((subtopic, entity, "membership", 1.0))
            source, target = sorted((item.source_id, item.target_id))
            edge_set.add((source, target, "predicate", 1.0))

    if config.soft_links:
        def soft_compatible(left: str, right: str) -> bool:
            label, context = _similarity(prototypes[left], prototypes[right], config)
            return label >= config.soft_label_threshold and context >= config.soft_context_threshold

        neighbors: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for collection in (topics, subtopics):
            for left, right in sorted(_near_pairs(sorted(collection), prototypes, max(config.merge_neighbors, config.soft_neighbors))):
                if not soft_compatible(left, right):
                    continue
                if nodes[left].kind == "subtopic" and not all(
                    a == b or soft_compatible(a, b)
                    for a in nodes[left].parents for b in nodes[right].parents
                ):
                    continue
                weight = max(0.0, min(1.0, sum(_similarity(prototypes[left], prototypes[right], config)) / 2))
                neighbors[left].append((weight, right))
                neighbors[right].append((weight, left))
        retained = {
            node: dict((neighbor, weight) for weight, neighbor in sorted(values, key=lambda pair: (-pair[0], pair[1]))[:config.soft_neighbors])
            for node, values in neighbors.items()
        }
        for left, values in sorted(retained.items()):
            for right, weight in values.items():
                if left < right and left in retained.get(right, {}):
                    edge_set.add((left, right, "soft", weight))

    serialized = {
        key: {
            "kind": node.kind, "label": node.label, "aliases": sorted(node.labels),
            "members": sorted(node.members), "sentences": sorted(node.sentences),
            "entities": sorted(node.entities), "parents": sorted(node.parents),
        }
        for key, node in sorted(nodes.items())
    }
    canonical_map = {
        "hierarchy": config.hierarchy,
        "old_topic_ids": {key: sorted({topic_map[node] for node in values}) for key, values in sorted(old_topics.items())},
        "old_subtopic_ids": {key: sorted({subtopic_map[node] for node in values}) for key, values in sorted(old_subtopics.items())},
        "stats": {"scoped_before_merge": before, "topics": len(topics), "subtopics": len(subtopics),
                  "soft_edges": sum(edge[2] == "soft" for edge in edge_set)},
    }
    return HierarchyArtifacts(
        serialized, np.asarray([prototypes[key][0] for key in sorted(nodes)], dtype=np.float32),
        sorted(edge_set), paths, canonical_map, store,
    )
