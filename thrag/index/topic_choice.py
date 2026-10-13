"""Topic selection helpers for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import List

import networkx as nx
from openai import OpenAI

from config import get_config
from index.label_selection import format_numbered_candidates, select_labels
from prompt.topic_choice import TOPIC_CHOICE_PROMPT

config = get_config()

DEFAULT_MODEL = config.default_model
TOPIC_CHOICE_MIN = config.topic_choice_min
TOPIC_CHOICE_MAX = config.topic_choice_max
MAX_RETRIES = config.max_retries


def extract_graph_topic_labels(graph: nx.Graph) -> List[str]:
    """Return unique topic labels in graph iteration order."""

    seen: set[str] = set()
    labels: list[str] = []
    for _node_id, data in graph.nodes(data=True):
        if data.get("type") != "topic":
            continue
        label = str(data.get("label", "")).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def choose_topics_from_graph(
    question: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_topics: int = TOPIC_CHOICE_MAX,
    min_topics: int = TOPIC_CHOICE_MIN,
    max_retries: int = MAX_RETRIES,
) -> List[str]:
    """Ask the LLM to select relevant topic labels from the graph."""

    topic_labels = extract_graph_topic_labels(graph)
    if not topic_labels:
        raise ValueError("The graph does not contain any topic nodes.")

    min_topics = max(1, min(min_topics, len(topic_labels)))
    max_topics = max(min_topics, min(max_topics, len(topic_labels)))

    prompt = (
        TOPIC_CHOICE_PROMPT
        .replace("{{TOPIC_LIST}}", format_numbered_candidates(topic_labels))
        .replace("{{question}}", question)
        .replace("{min_topics}", str(min_topics))
        .replace("{max_topics}", str(max_topics))
    )

    return select_labels(
        client=client,
        model=model,
        prompt=prompt,
        key="topics",
        labels=topic_labels,
        max_choices=max_topics,
        max_retries=max_retries,
        description=f"Topic selection question={question[:120]!r}",
    )

