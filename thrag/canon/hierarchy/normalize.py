"""Merge differently named equivalent concepts without splitting shared labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from itertools import zip_longest
import re
from typing import Any, Callable

import numpy as np

from canon.core.config import CanonConfig
from canon.core.evidence import EvidenceStore


def scope_conflict(left: str, right: str) -> str | None:
    """Reject explicit contradictions; relatedness is left to semantic review."""
    numbers = lambda text: {str(Decimal(value).normalize()) for value in re.findall(r"\d+(?:\.\d+)?", text)}
    a, b = numbers(left), numbers(right)
    if a and b and a != b:
        return "different_numeric_anchors"
    aa, bb = set(re.findall(r"\w+", left.casefold())), set(re.findall(r"\w+", right.casefold()))
    negators = {"not", "non"}
    if not a and not b:
        negators.add("no")
    if bool(aa & negators) != bool(bb & negators) and aa - negators == bb - negators:
        return "opposite_polarity"
    for first, second in (("before", "after"), ("increase", "decrease"), ("positive", "negative"), ("legal", "illegal")):
        pair = {first, second}
        if (first in aa and second in bb or second in aa and first in bb) and aa - pair == bb - pair:
            return "opposite_scope"
    return None


def _node_card(
    node: Any, nodes: dict[str, Any], store: EvidenceStore, config: CanonConfig,
    prototype: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    buckets: dict[str, list[int]] = defaultdict(list)
    name_counts: Counter[str] = Counter()
    aliases = set(node.labels)
    indices = sorted(node.sentences)
    centrality = dict(zip(indices, (store.vectors[indices] @ prototype[1]).tolist(), strict=True))
    for index in indices:
        groups = set()
        for occurrence in store.records[index].occurrences:
            for entity, name in ((occurrence.source_id, occurrence.source), (occurrence.target_id, occurrence.target)):
                if entity in node.entities:
                    name_counts[name] += 1
            if node.kind == "topic":
                if occurrence.subject_topic in aliases or occurrence.object_topic in aliases:
                    groups.add(occurrence.label)
            else:
                if occurrence.subject_subtopic in aliases:
                    groups.add(occurrence.subject_topic)
                if occurrence.object_subtopic in aliases:
                    groups.add(occurrence.object_topic)
        for group in groups or {"unknown"}:
            buckets[group].append(index)
    chosen: list[int] = []
    ordered_groups = sorted(buckets, key=lambda group: (-len(buckets[group]), group))
    for group in ordered_groups:
        available = [index for index in buckets[group] if index not in chosen]
        candidate = min(available, key=lambda index: (-centrality[index], index)) if available else None
        if candidate is not None:
            chosen.append(candidate)
        if len(chosen) >= config.synonym_contexts:
            break
    for index in sorted(node.sentences, key=lambda item: (-centrality[item], item)):
        if len(chosen) >= config.synonym_contexts:
            break
        if index not in chosen:
            chosen.append(index)
    names = sorted(name_counts, key=lambda name: (-name_counts[name], name))
    return {
        # Never clip a label's qualifier/polarity to make a semantic judgment fit.
        "label": node.label, "aliases": sorted(node.labels),
        "parents": [nodes[parent].label for parent in sorted(node.parents, key=lambda p: nodes[p].order)[:6]],
        "parent_count": len(node.parents), "context_group_count": len(buckets),
        "entities": [name[:80] for name in names[:4]],
        "evidence": [store.records[index].sentence[:config.synonym_evidence_chars] for index in chosen],
        "evidence_count": len(node.sentences),
        "evidence_truncated": any(len(store.records[index].sentence) > config.synonym_evidence_chars for index in chosen),
    }


def normalize_nodes(
    topics: dict[str, Any], subtopics: dict[str, Any], store: EvidenceStore, config: CanonConfig,
    reviewer: Any, prototypes: dict[str, tuple[np.ndarray, ...]], near_pairs: Callable[..., Any],
    combine: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str], dict[str, str], dict[str, Any], list[tuple[str, str, float]]]:
    nodes = {**topics, **subtopics}
    candidates: dict[str, list[tuple[str, str, float, float]]] = {"topic": [], "subtopic": []}
    guarded: Counter[str] = Counter()
    guard_examples: list[dict[str, Any]] = []
    for kind, group in (("topic", topics), ("subtopic", subtopics)):
        # Subtopics can be synonyms across compatible domains; no parent-equality gate.
        for left, right in sorted(near_pairs(sorted(group), prototypes, config.merge_neighbors)):
            lv, rv = prototypes[left], prototypes[right]
            label_score = float(lv[0] @ rv[0])
            if label_score < config.merge_label_threshold:
                continue
            conflict = next((reason for a in nodes[left].labels for b in nodes[right].labels
                             if (reason := scope_conflict(a, b)) is not None), None)
            if conflict:
                guarded[conflict] += 1
                if len(guard_examples) < 30:
                    guard_examples.append({"left": nodes[left].label, "right": nodes[right].label, "reason": conflict})
                continue
            weight = config.sentence_prototype_weight
            context_score = weight * float(lv[1] @ rv[1]) + (1 - weight) * float(lv[2] @ rv[2])
            # Context helps review/prioritization; sparse or different instances are not a veto.
            candidates[kind].append((left, right, label_score, context_score))
        candidates[kind].sort(key=lambda pair: (-pair[2], -pair[3], min(nodes[pair[0]].order, nodes[pair[1]].order), pair[:2]))
    interleaved = [pair for row in zip_longest(candidates["topic"], candidates["subtopic"]) for pair in row if pair is not None]
    requested = interleaved[:config.synonym_max_pairs] if config.synonym_max_pairs else interleaved
    if requested and reviewer is None:
        raise ValueError("hierarchy=normalize requires an offline synonym reviewer; similarity-only merging is not a fallback")
    cards: dict[str, dict[str, Any]] = {}
    payloads = []
    for left, right, label_score, context_score in requested:
        for node_id in (left, right):
            if node_id not in cards:
                cards[node_id] = _node_card(nodes[node_id], nodes, store, config, prototypes[node_id])
        payloads.append({
            "pair": [left, right], "layer": nodes[left].kind, "left": cards[left], "right": cards[right],
            "label_cosine": round(label_score, 6), "context_cosine": round(context_score, 6),
        })
    print(f"H normalization: candidates={len(interleaved)}, review_budget={len(requested)}; unreviewed pairs remain separate", flush=True)
    verdicts = reviewer.review(payloads) if payloads else {}
    # Missing/uncertain judgments cannot authorize either a hard merge or a soft seed.
    verdicts = {tuple(sorted(pair)): value for pair, value in verdicts.items()}

    def equivalent(left: str, right: str) -> bool:
        return verdicts.get(tuple(sorted((left, right)))) == "equivalent"

    def merge_layer(group: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        owners = {node: node for node in group}
        clusters = {node: [node] for node in group}
        for left, right, _label, _context in requested:
            if left not in group or right not in group or not equivalent(left, right):
                continue
            a, b = owners[left], owners[right]
            if a == b or not all(equivalent(u, v) for u in clusters[a] for v in clusters[b]):
                continue
            keep, drop = sorted((a, b), key=lambda node: (group[node].order, node))
            clusters[keep] = sorted(clusters[keep] + clusters.pop(drop))
            for member in clusters[keep]:
                owners[member] = keep
        return combine(group, clusters.values())

    merged_topics, topic_map = merge_layer(topics)
    merged_subtopics, subtopic_map = merge_layer(subtopics)
    for node in merged_subtopics.values():
        node.parents = {topic_map[parent] for parent in node.parents}
    combined = {**merged_topics, **merged_subtopics}
    mapping = {**topic_map, **subtopic_map}
    soft_candidates: dict[tuple[str, str], float] = {}
    for left, right, label_score, context_score in requested:
        a, b = mapping[left], mapping[right]
        if a == b or not equivalent(left, right):
            continue
        if any(verdicts.get(tuple(sorted((u, v)))) in {"different_sense", "broader_narrower", "related"}
               or scope_conflict(nodes[u].label, nodes[v].label) is not None
               for u in combined[a].members for v in combined[b].members):
            continue
        if label_score >= config.soft_label_threshold and context_score >= config.soft_context_threshold:
            pair = tuple(sorted((a, b)))
            soft_candidates[pair] = max(soft_candidates.get(pair, 0.0), min(1.0, (label_score + context_score) / 2))
    audit = {
        "protocol": "context_checked_synonyms_v1", "candidate_pairs": {kind: len(rows) for kind, rows in candidates.items()},
        "reviewed_pairs": len(requested), "unreviewed_pairs": len(interleaved) - len(requested),
        "guard_rejections": dict(guarded), "guard_examples": guard_examples,
        "review_stats": dict(getattr(reviewer, "stats", {})), "calls": list(getattr(reviewer, "calls", [])),
        "review_cache": str(getattr(reviewer, "cache_path", "")),
        "cached_total_calls": len(getattr(reviewer, "previous_calls", [])) + len(getattr(reviewer, "calls", [])),
        "decisions": [{"left_id": a, "right_id": b, "left": nodes[a].label, "right": nodes[b].label,
                       "label_cosine": score, "context_cosine": context,
                       "relation": verdicts.get((a, b), "uncertain")}
                      for a, b, score, context in requested],
    }
    return merged_topics, merged_subtopics, topic_map, subtopic_map, audit, [(*pair, weight) for pair, weight in sorted(soft_candidates.items())]
