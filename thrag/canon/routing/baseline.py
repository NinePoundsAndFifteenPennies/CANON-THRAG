"""Existing label selectors adapted to path IDs for module-off runs."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

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
        counts = Counter((node["kind"], node["label"]) for node in hierarchy.nodes.values())
        self.ids_by_display: dict[str, str] = {}
        for node_id, node in hierarchy.nodes.items():
            label = node["label"]
            display = f"{label} [{node_id}]" if counts[(node["kind"], label)] > 1 else label
            self.graph.nodes[node_id]["label"] = display
            # Separate types; equal topic/subtopic display text is legal.
            self.ids_by_display[f"{node['kind']}:{display}"] = node_id

    def route(self, context: QueryContext) -> RoutingResult:
        topic_limit, subtopic_limit = limits(self.runtime)
        topics = choose_topics_from_graph(
            context.query, self.graph, self.client, model=self.runtime.default_model,
            min_topics=self.runtime.topic_choice_min, max_topics=topic_limit, max_retries=self.runtime.max_retries,
        )

        def select(display: str) -> list[tuple[str, str]]:
            topic = self.ids_by_display[f"topic:{display}"]
            chosen = choose_subtopics_for_topic(
                question=context.query, topic_nid=topic, graph=self.graph, client=self.client,
                model=self.runtime.default_model, min_subtopics=self.runtime.subtopic_choice_min,
                max_subtopics=subtopic_limit,
            )
            children = {self.graph.nodes[child]["label"]: child for child in self.hierarchy.children[topic]}
            return [(topic, children[label]) for label in chosen]

        paths = []
        if topics:
            with ThreadPoolExecutor(max_workers=max(1, min(self.runtime.max_workers, len(topics)))) as executor:
                futures = [executor.submit(copy_context().run, select, display) for display in topics]
                for future in futures:
                    paths.extend(future.result())
        return RoutingResult(paths, diagnostics={"mode": "baseline"})
