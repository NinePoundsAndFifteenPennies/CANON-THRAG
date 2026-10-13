"""Subtopic selection helpers for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import List, Tuple

import networkx as nx
from openai import OpenAI

from config import get_config
from index.label_selection import format_numbered_candidates, select_labels
from prompt.subtopic_choice import SUBTOPIC_CHOICE_PROMPT

config = get_config()

SUBTOPIC_CHOICE_MIN = config.subtopic_choice_min
SUBTOPIC_CHOICE_MAX = config.subtopic_choice_max
DEFAULT_MODEL = config.default_model
MAX_RETRIES = config.max_retries


def extract_subtopics_for_topic(graph: nx.Graph, topic_node_id: str) -> List[Tuple[str, str]]:
    """Return the direct subtopic neighbors for a topic node."""

    return [
        (neighbor, str(graph.nodes[neighbor].get("label", "")).strip())
        for neighbor in graph.neighbors(topic_node_id)
        if graph.nodes[neighbor].get("type") == "subtopic"
    ]


def choose_subtopics_for_topic(
    *,
    question: str,
    topic_nid: str,
    graph: nx.Graph,
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    max_subtopics: int = SUBTOPIC_CHOICE_MAX,
    min_subtopics: int = SUBTOPIC_CHOICE_MIN,
) -> list[str]:
    """Return the ordered list of subtopics chosen by the LLM."""

    if graph.nodes[topic_nid].get("type") != "topic":
        raise ValueError(f"Node {topic_nid} is not a topic node.")

    subtopics = extract_subtopics_for_topic(graph, topic_nid)
    if not subtopics:
        return []

    subtopic_labels = list(dict.fromkeys(label for _node_id, label in subtopics if label))
    if not subtopic_labels:
        return []

    min_subtopics = max(1, min(min_subtopics, len(subtopic_labels)))
    max_subtopics = max(min_subtopics, min(max_subtopics, len(subtopic_labels)))

    prompt = (
        SUBTOPIC_CHOICE_PROMPT
        .replace("{{TOPIC_LABEL}}", str(graph.nodes[topic_nid].get("label", "")))
        .replace("{{SUBTOPIC_LIST}}", format_numbered_candidates(subtopic_labels))
        .replace("{question}", question)
        .replace("{min_subtopics}", str(min_subtopics))
        .replace("{max_subtopics}", str(max_subtopics))
    )

    return select_labels(
        client=client,
        model=model,
        prompt=prompt,
        key="subtopics",
        labels=subtopic_labels,
        max_choices=max_subtopics,
        max_retries=MAX_RETRIES,
        description=f"Subtopic selection topic={str(graph.nodes[topic_nid].get('label', ''))[:80]!r} "
                    f"question={question[:120]!r}",
    )

