"""Offline regression cases for bounded topic/subtopic selection."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

from index import label_selection
from index.topic_choice import choose_topics_from_graph
from index.subtopic_choice import choose_subtopics_for_topic


def completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(completion_tokens=20),
    )


@pytest.fixture
def client(monkeypatch) -> MagicMock:
    config = SimpleNamespace(
        answer_temperature=0.3,
        retry_backoff=0.0,
        llm_extra_body=lambda: {"chat_template_kwargs": {"enable_thinking": False}},
    )
    monkeypatch.setattr(label_selection, "get_config", lambda: config)
    mocked = MagicMock()
    mocked.with_options.return_value = mocked
    return mocked


def select(client, *, max_retries: int = 10) -> list[str]:
    return label_selection.select_labels(
        client=client,
        model="mock-model",
        prompt="Select IDs.",
        key="subtopics",
        labels=["First", "Second", "Third"],
        max_choices=2,
        max_retries=max_retries,
        description="Subtopic selection",
    )


@pytest.mark.parametrize("chosen", [[], [0], [4], [True], [1.0], ["1"], [1, 2, 3]])
def test_rejects_invalid_or_excessive_ids(chosen) -> None:
    with pytest.raises(ValueError):
        label_selection.parse_selected_ids(
            json.dumps({"subtopics": chosen}), key="subtopics", candidate_count=3, max_choices=2,
        )


def test_topic_ids_map_back_to_exact_labels_in_graph_order(client) -> None:
    labels = ["Research \"A\"", "Other", "系统 / Hierarchy"]
    graph = nx.Graph()
    for index, label in enumerate(labels):
        graph.add_node(str(index), type="topic", label=label)
    client.chat.completions.create.return_value = completion('{"topics": [3, 1]}')

    assert choose_topics_from_graph("Question", graph, client, min_topics=1, max_topics=2) == [labels[0], labels[2]]
    request = client.chat.completions.create.call_args.kwargs
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["schema"]["properties"]["topics"]["maxItems"] == 2
    assert 0 < request["max_tokens"] <= 512
    client.with_options.assert_called_once_with(max_retries=0)


def test_subtopic_ids_are_scoped_to_the_selected_topic(client) -> None:
    graph = nx.Graph()
    graph.add_node("t1", type="topic", label="Topic 1")
    graph.add_node("t2", type="topic", label="Topic 2")
    graph.add_node("s1", type="subtopic", label="First")
    graph.add_node("s2", type="subtopic", label="Second")
    graph.add_node("s3", type="subtopic", label="Other topic's subtopic")
    graph.add_edges_from([("t1", "s1"), ("t1", "s2"), ("t2", "s3")])
    client.chat.completions.create.return_value = completion('{"subtopics": [2]}')

    assert choose_subtopics_for_topic("Question", "t1", graph, client, min_subtopics=1, max_subtopics=2) == ["Second"]
    request = client.chat.completions.create.call_args.kwargs
    assert request["response_format"]["json_schema"]["schema"]["properties"]["subtopics"]["items"]["enum"] == [1, 2]


def test_malformed_response_retries_then_recovers(client) -> None:
    client.chat.completions.create.side_effect = [completion('{"subtopics": ['), completion('{"subtopics": [3]}')]

    assert select(client) == ["Third"]
    assert client.chat.completions.create.call_count == 2


def test_repeated_malformed_responses_have_a_finite_request_budget(client) -> None:
    client.chat.completions.create.return_value = completion('{"subtopics": [')

    assert select(client, max_retries=10) == ["First", "Second"]
    assert client.chat.completions.create.call_count == 3
    client.with_options.assert_called_once_with(max_retries=0)


def test_length_truncation_is_rejected_even_if_content_parses(client) -> None:
    client.chat.completions.create.return_value = completion('{"subtopics": [3]}', finish_reason="length")

    assert select(client, max_retries=1) == ["First", "Second"]


def test_oversized_response_is_not_parsed(client, monkeypatch) -> None:
    client.chat.completions.create.return_value = completion("x" * 184529)
    parser = MagicMock()
    monkeypatch.setattr(label_selection, "parse_selected_ids", parser)

    assert select(client, max_retries=1) == ["First", "Second"]
    parser.assert_not_called()


def test_permanent_api_error_is_not_retried_or_hidden_by_fallback(client, monkeypatch) -> None:
    class PermanentAPIError(Exception):
        status_code = 400

    monkeypatch.setattr(label_selection, "APIStatusError", PermanentAPIError)
    client.chat.completions.create.side_effect = PermanentAPIError("Unsupported schema")

    with pytest.raises(PermanentAPIError):
        select(client)
    assert client.chat.completions.create.call_count == 1
