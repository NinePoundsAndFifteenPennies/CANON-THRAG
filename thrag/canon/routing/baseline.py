"""Existing label selectors adapted to path IDs for module-off runs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

import networkx as nx

from index.topic_choice import choose_topics_from_graph
from index.subtopic_choice import choose_subtopics_for_topic
from canon.core.contracts import QueryContext, RoutingResult
from canon.hierarchy import HierarchyArtifacts
from canon.routing.common import limits


class BaselineRouter:
    """This is a selector adapter, not an author-baseline experiment preset."""

    def __init__(self, hierarchy: HierarchyArtifacts, runtime: Any, client: Any) -> None:
        self.hierarchy, self.runtime, self.client = hierarchy, runtime, client
        self.graph = hierarchy.graph.copy()
        counts = Counter(node["label"] for node in hierarchy.nodes.values() if node["kind"] == "topic")
        ordinals: Counter[str] = Counter()
        self.ids_by_display: dict[str, str] = {}
        for node_id in hierarchy.presentation_order:
            node = hierarchy.nodes[node_id]
            if node["kind"] != "topic":
                continue
            label = node["label"]
            ordinals[label] += 1
            display = f"{label} [variant {ordinals[label]}]" if counts[label] > 1 else label
            self.graph.nodes[node_id]["label"] = display
            self.ids_by_display[display] = node_id

    def route(self, context: QueryContext) -> RoutingResult:
        topic_limit, subtopic_limit = limits(self.runtime)
        topics = choose_topics_from_graph(
            context.query, self.graph, self.client, model=self.runtime.default_model,
            min_topics=self.runtime.topic_choice_min, max_topics=topic_limit, max_retries=self.runtime.max_retries,
        )

        def select(display: str) -> list[tuple[str, str]]:
            topic = self.ids_by_display[display]
            # A request-local view avoids changing shared labels in concurrent topics.
            # Equal names under different parents require no disambiguation at all.
            view = nx.Graph()
            view.add_node(topic, type="topic", label=display)
            local_counts = Counter(self.hierarchy.nodes[child]["label"] for child in self.hierarchy.children[topic])
            local_ordinals: Counter[str] = Counter()
            children: dict[str, str] = {}
            for child in self.hierarchy.children[topic]:
                label = self.hierarchy.nodes[child]["label"]
                local_ordinals[label] += 1
                shown = f"{label} [variant {local_ordinals[label]}]" if local_counts[label] > 1 else label
                view.add_node(child, type="subtopic", label=shown)
                view.add_edge(topic, child)
                children[shown] = child
            chosen = choose_subtopics_for_topic(
                question=context.query, topic_nid=topic, graph=view, client=self.client,
                model=self.runtime.default_model, min_subtopics=self.runtime.subtopic_choice_min,
                max_subtopics=subtopic_limit,
            )
            return [(topic, children[label]) for label in chosen]

        paths = []
        if topics:
            with ThreadPoolExecutor(max_workers=max(1, min(self.runtime.max_workers, len(topics)))) as executor:
                futures = [executor.submit(copy_context().run, select, display) for display in topics]
                for future in futures:
                    paths.extend(future.result())
        return RoutingResult(paths, diagnostics={"mode": "baseline"})
