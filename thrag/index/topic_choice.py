"""Topic selection helpers for TH-RAG."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from typing import Callable, List

import networkx as nx
from openai import OpenAI

from config import get_config
from index.label_selection import (
    format_compact_candidates,
    format_numbered_candidates,
    select_labels,
    selection_messages,
    selection_output_budget,
)
from model_services import get_chat_tokenizer
from prompt.topic_choice import TOPIC_CHOICE_PROMPT

config = get_config()

DEFAULT_MODEL = config.default_model
TOPIC_CHOICE_MIN = config.topic_choice_min
TOPIC_CHOICE_MAX = config.topic_choice_max
MAX_RETRIES = config.max_retries

# Reserve room for retry instructions and small server/local template differences.
CONTEXT_MARGIN = 512


def partition_topic_candidates(
    labels: list[str], fits: Callable[[list[str]], bool],
) -> list[list[str]]:
    """Partition by actual prompt size without discarding candidate labels."""
    batches: list[list[str]] = []
    start = 0
    while start < len(labels):
        low, high = start + 1, len(labels)
        best = start
        while low <= high:
            end = (low + high) // 2
            if fits(labels[start:end]):
                best = end
                low = end + 1
            else:
                high = end - 1
        if best == start:
            raise ValueError(
                "A single topic candidate and the question exceed the input budget: "
                f"{labels[start][:120]!r}. Check the question/label length and tokenizer configuration."
            )
        batches.append(labels[start:best])
        start = best
    return batches


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

    request_config = get_config()
    tokenizer = get_chat_tokenizer(request_config.chat_tokenizer_path)
    # Reserve the existing output allowance; do not shrink it to squeeze in a prompt.
    input_budget = (
        request_config.max_context_length
        - selection_output_budget(len(topic_labels), max_topics)
        - CONTEXT_MARGIN
    )
    if input_budget <= 0:
        raise ValueError("MAX_CONTEXT_LENGTH leaves no input budget for topic selection.")
    template_kwargs = request_config.llm_extra_body().get("chat_template_kwargs", {})
    compact = False

    def make_prompt(labels: list[str]) -> str:
        local_min = min(min_topics, len(labels))
        local_max = min(max_topics, len(labels))
        formatted_candidates = (
            format_compact_candidates(labels) if compact else format_numbered_candidates(labels)
        )
        return (
            TOPIC_CHOICE_PROMPT
            .replace("{{TOPIC_LIST}}", formatted_candidates)
            .replace("{{question}}", question)
            .replace("{min_topics}", str(local_min))
            .replace("{max_topics}", str(local_max))
        )

    def prompt_tokens(labels: list[str]) -> int:
        return tokenizer.count_messages(
            selection_messages(make_prompt(labels)), chat_template_kwargs=template_kwargs,
        )

    def fits(labels: list[str]) -> bool:
        return prompt_tokens(labels) <= input_budget

    def select_batch(labels: list[str], suffix: str = "") -> list[str]:
        return select_labels(
            client=client,
            model=model,
            prompt=make_prompt(labels),
            key="topics",
            labels=labels,
            max_choices=min(max_topics, len(labels)),
            max_retries=max_retries,
            description=f"Topic selection question={question[:120]!r}{suffix}",
        )

    candidates = topic_labels
    original_token_count = prompt_tokens(candidates)
    if original_token_count <= input_budget:
        return select_batch(candidates)

    # Keep the existing request for graphs that already fit the service window.
    # Only oversized candidate lists switch to the compact representation.
    compact = True
    print(
        f"Topic selection question={question[:120]!r}: prompt_tokens={original_token_count}, "
        f"input_budget={input_budget}, candidates={len(candidates)}; compacting candidate IDs.",
        flush=True,
    )
    round_number = 0
    while True:
        token_count = prompt_tokens(candidates)
        if token_count <= input_budget:
            print(
                f"Topic selection question={question[:120]!r}: using compact candidates="
                f"{len(candidates)}, prompt_tokens={token_count}, input_budget={input_budget}.",
                flush=True,
            )
            return select_batch(candidates)

        # Every candidate participates in batch selection before the final selection.
        batches = partition_topic_candidates(candidates, fits)
        round_number += 1
        print(
            f"Topic selection question={question[:120]!r}: prompt_tokens={token_count}, "
            f"input_budget={input_budget}, candidates={len(candidates)}, "
            f"batches={len(batches)}, round={round_number}.",
            flush=True,
        )
        selected: list[str] = []
        for batch_number, batch in enumerate(batches, start=1):
            selected.extend(select_batch(batch, f" round={round_number} batch={batch_number}/{len(batches)}"))
        selected = list(dict.fromkeys(selected))
        if len(selected) <= max_topics:
            return selected
        if len(selected) >= len(candidates):
            raise ValueError(
                "Topic selection could not reduce candidates enough to fit MAX_CONTEXT_LENGTH. "
                "The question or individual topic labels consume too much of the context window."
            )
        candidates = selected

