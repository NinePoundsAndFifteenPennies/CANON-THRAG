"""Meaning-preserving normalization plus explicitly retained legacy variants."""

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
    order: int = 2**63 - 1
    last_order: int = -1
    last_label: str = ""

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


def _combine(nodes: dict[str, _Node], clusters: Any) -> tuple[dict[str, _Node], dict[str, str]]:
    result: dict[str, _Node] = {}
    mapping: dict[str, str] = {}
    for members in clusters:
        members = sorted(members)
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
            combined.order = min(combined.order, node.order)
            if node.last_order >= combined.last_order:
                combined.last_order, combined.last_label = node.last_order, node.last_label
            mapping[member] = canonical_id
        result[canonical_id] = combined
    return result, mapping


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

    return _combine(nodes, clusters.values())


class HierarchyArtifacts:
    def __init__(
        self, nodes: dict[str, dict[str, Any]], label_vectors: np.ndarray,
        edges: list[tuple[str, str, str, float]], occurrence_paths: dict[str, list[Branch]],
        canonical_map: dict[str, Any], store: EvidenceStore,
        branch_order: dict[Branch, int] | None = None,
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
        self.branch_order = branch_order or {}
        self.presentation_order = sorted(nodes, key=lambda node: (nodes[node].get("order", 2**63 - 1), node))
        self.graph = nx.Graph()
        for node_id in self.presentation_order:
            node = nodes[node_id]
            self.graph.add_node(node_id, type=node["kind"], label=node["label"])
        for entity_id, names in sorted(store.entity_labels.items()):
            self.graph.add_node(entity_id, type="entity", label=min(names))
        self.children: dict[str, list[str]] = defaultdict(list)
        sub_entities: dict[str, set[str]] = defaultdict(set)
        graph_edges = sorted(edges, key=lambda edge: (
            0 if edge[2] == "hierarchy" else 1,
            self.branch_order.get((edge[0], edge[1]), 2**63 - 1), edge[:3],
        ))
        for source, target, relation, weight in graph_edges:
            self.graph.add_edge(source, target, relation_type=relation, weight=weight)
            if relation == "hierarchy":
                self.children[source].append(target)
            elif relation == "membership":
                sub_entities[source].add(target)
        for topic, children in self.children.items():
            children.sort(key=lambda child: (self.branch_order.get((topic, child), 2**63 - 1), child))
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
            "branch_order": [[*path, order] for path, order in sorted(self.branch_order.items())],
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
            {(topic, subtopic): order for topic, subtopic, order in data.get("branch_order", [])},
        )


def build_hierarchy(
    store: EvidenceStore, config: CanonConfig, *, source_order: dict[str, int] | None = None,
    synonym_reviewer: Any = None,
) -> HierarchyArtifacts:
    topics: dict[str, _Node] = {}
    subtopics: dict[str, _Node] = {}
    original_paths: dict[str, list[Branch]] = {}
    old_topics: dict[str, set[str]] = defaultdict(set)
    old_subtopics: dict[str, set[str]] = defaultdict(set)
    original_branch_order: dict[Branch, int] = {}

    def add_path(topic: str, subtopic: str, entity: str, index: int, order: int) -> Branch:
        if config.hierarchy == "none":
            topic_id, subtopic_id = f"topic_{clean_id(topic)}", f"subtopic_{clean_id(subtopic)}"
        elif config.hierarchy == "normalize":
            topic_id = stable_id("topic", " ".join(topic.casefold().split()))
            subtopic_id = stable_id("subtopic", " ".join(subtopic.casefold().split()))
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
            node.order = min(node.order, order)
            if order >= node.last_order:
                node.last_order, node.last_label = order, label
        subtopics[subtopic_id].parents.add(topic_id)
        old_topics[f"topic_{clean_id(topic)}"].add(topic_id)
        old_subtopics[f"subtopic_{clean_id(subtopic)}"].add(subtopic_id)
        original_branch_order.setdefault((topic_id, subtopic_id), order)
        return topic_id, subtopic_id

    occurrences = [(index, item) for index, record in enumerate(store.records) for item in record.occurrences]
    if source_order is not None:
        if set(source_order) != {item.id for _, item in occurrences}:
            raise ValueError("Extraction order does not match the evidence snapshot")
        occurrences.sort(key=lambda pair: source_order[pair[1].id])
    for rank, (index, item) in enumerate(occurrences):
        original_paths[item.id] = [
            add_path(item.subject_topic, item.subject_subtopic, item.source_id, index, rank * 2),
            add_path(item.object_topic, item.object_subtopic, item.target_id, index, rank * 2 + 1),
        ]
    before = {"topics": len(topics), "subtopics": len(subtopics)}
    topic_map = {key: key for key in topics}
    subtopic_map = {key: key for key in subtopics}
    normalization_audit: dict[str, Any] | None = None
    reviewed_soft: list[tuple[str, str, float]] = []
    if config.hierarchy == "normalize":
        from canon.hierarchy.normalize import normalize_nodes

        prototypes = _prototypes({**topics, **subtopics}, store)
        topics, subtopics, topic_map, subtopic_map, normalization_audit, reviewed_soft = normalize_nodes(
            topics, subtopics, store, config, synonym_reviewer, prototypes, _near_pairs, _combine,
        )
    elif config.hierarchy == "split_merge":
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
    branch_order: dict[Branch, int] = {}
    for (topic, subtopic), order in original_branch_order.items():
        path = topic_map[topic], subtopic_map[subtopic]
        branch_order[path] = min(branch_order.get(path, order), order)
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
        if config.hierarchy == "normalize":
            for left, right, weight in reviewed_soft:
                neighbors[left].append((weight, right))
                neighbors[right].append((weight, left))
        for collection in (() if config.hierarchy == "normalize" else (topics, subtopics)):
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
            "kind": node.kind, "label": node.last_label if config.hierarchy == "none" else node.label,
            "aliases": sorted(node.labels), "order": node.order,
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
        "normalization": normalization_audit,
    }
    return HierarchyArtifacts(
        serialized, np.asarray([prototypes[key][0] for key in sorted(nodes)], dtype=np.float32),
        sorted(edge_set), paths, canonical_map, store, branch_order,
    )
