"""Server-only regression cases; clients and embeddings are local test doubles."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from canon.allocation import Allocator
from canon.allocation.bridge import bridge_scores
from canon.core.artifacts import finish_cache, valid_cache
from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext, RoutingResult
from canon.core.evidence import EvidenceStore
from canon.core.io import atomic_json, unit_rows
from canon.core.sentence_views import build_sentence_vectors
from canon.core.telemetry import MeteredClient, record_calls
from canon.hierarchy import build_hierarchy
from canon.hierarchy.builder import HierarchyArtifacts
from canon.hierarchy.source_order import occurrence_order
from canon.pipeline import load_questions, run_questions
from canon.retriever import CanonRetriever
from canon.routing.baseline import BaselineRouter
from canon.routing.diffusion import DiffusionRouter
from canon.routing.hybrid import HybridRouter
from canon.routing.preview import EvidencePreviewRouter
from canon.routing.replay import ReplayOnlyRouter, RoutingReplay


class Encoder:
    def __init__(self, values=None):
        self.values = values or {}

    def encode_passages(self, texts, show_progress=False):
        return unit_rows(np.asarray([self.values.get(text, [1.0, 0.0, 0.0]) for text in texts]))

    def encode_queries(self, texts):
        return self.encode_passages(texts)


def fact(source, target, sentence, topic="Research", subtopic="Methods", object_topic=None, object_subtopic=None):
    return {
        "triple": [source, "related_to", target], "sentence": sentence,
        "subject": {"main_topic": topic, "subtopic": subtopic},
        "object": {"main_topic": object_topic or topic, "subtopic": object_subtopic or subtopic},
    }


def make_store(chunks, values=None):
    chunk_map = {chunk: f"Content for {chunk}" for chunk in chunks}
    blocks = [{"chunk_id": chunk, "content": chunk_map[chunk], "triples": triples} for chunk, triples in chunks.items()]
    return EvidenceStore.build(blocks, chunk_map, Encoder(values))


def make_hierarchy(store, *, source_order=None):
    def profiles(rows):
        return {row["id"]: {"synopsis": row["evidence"][0], "aspects": [row["topic"], row["subtopic"]]}
                for row in rows}

    return build_hierarchy(store, Encoder(), profiles, source_order=source_order)


def method(**kwargs):
    return replace(CanonConfig(routing="a", bridge=False, balance=False), **kwargs)


def runtime(**kwargs):
    values = dict(
        topic_choice_min=1, topic_choice_max=2, subtopic_choice_min=1, subtopic_choice_max=2,
        max_retries=3, retry_backoff=0.0, max_workers=2, default_model="mock", answer_temperature=0.0,
        max_context_length=10000, chat_tokenizer_path="mock", top_k1=2, top_k2=2,
        llm_extra_body=lambda: {}, llm_base_url="http://mock.invalid",
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def query(store, scores=None, text="Which fact?"):
    return QueryContext(text, np.asarray([1, 0, 0], dtype=np.float32), np.asarray(scores if scores is not None else [1.0] * len(store.records)))


def path_named(hierarchy, topic, subtopic):
    return next(path for path in hierarchy.branches if hierarchy.nodes[path[0]]["label"] == topic and hierarchy.nodes[path[1]]["label"] == subtopic)


class Tokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)

    def count_messages(self, messages, **kwargs):
        return sum(len(message["content"]) for message in messages) // 4 + 24


def completion(content, reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=reason)],
                           usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5, total_tokens=25))


def client_for(content):
    client = MagicMock()
    client.with_options.return_value = client
    client.chat.completions.create.return_value = completion(content)
    return client


def test_sentence_keeps_all_fact_endpoints_and_separate_chunk_associations(tmp_path):
    text = "A bought B / C funded A."
    store = make_store({
        "c1": [fact("A", "B", text), fact("C", "A", text)],
        "c2": [fact("D", "E", text)],
    })
    assert len(store.records) == 1
    assert store.candidates({"entity_c"}) == [0]
    assert store.records[0].sentence == text
    payload = store.edge_result(0, 0.9, 1)
    assert payload["chunk_ids"] == ["c1", "c2"]
    assert {(item["chunk_id"], item["source_id"], item["target_id"]) for item in payload["occurrences"]} == {
        ("c1", "entity_a", "entity_b"), ("c1", "entity_c", "entity_a"), ("c2", "entity_d", "entity_e"),
    }
    store.save(tmp_path)
    loaded = EvidenceStore.load(tmp_path, store.chunk_map)
    assert loaded.edge_result(0, 0.9, 1) == payload


def test_exact_candidate_pool_retains_sentence_below_global_prefix():
    store = make_store({f"c{i}": [fact(f"A{i}", f"B{i}", f"Sentence {i}", subtopic=f"Branch {i}")] for i in range(12)})
    config = method()
    hierarchy = make_hierarchy(store)
    desired = next(index for index, record in enumerate(store.records) if record.sentence == "Sentence 9")
    scores = np.ones(len(store.records))
    scores[desired] = 0.01
    indices = store.candidates({"entity_a9"})
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), indices, 1, 1)
    assert result.indices == [desired]
    assert result.chunks == ["c9"]


def test_shared_subtopic_keeps_path_evidence_and_uses_the_best_probe():
    vectors = {label: [0, 1, 0] for label in ("Biology", "Film", "Adaptation", "Overview", "Miss")}
    store = make_store({"bio": [fact("Species", "Trait", "Evolution", "Biology", "Adaptation")],
                        "film": [fact("Novel", "Movie", "Screenplay", "Film", "Adaptation")]}, vectors)

    def distill(rows):
        by_topic = {row["topic"]: row for row in rows}
        assert by_topic["Biology"]["evidence"] == ["Evolution"]
        assert set(by_topic["Film"]["entities"]) == {"Novel", "Movie"}
        return {r["id"]: {"synopsis": "Overview", "aspects": ["Hit" if r["topic"] == "Biology" else "Miss", "Miss"]}
                for r in rows}

    hierarchy = build_hierarchy(store, Encoder(vectors), distill)
    bio = path_named(hierarchy, "Biology", "Adaptation")
    film = path_named(hierarchy, "Film", "Adaptation")
    assert bio[1] == film[1]
    assert hierarchy.branch_entities[bio] == hierarchy.branch_entities[film]
    scores = dict(zip(hierarchy.branches, hierarchy.semantic_scores(query(store).vector), strict=True))
    assert scores[bio] == pytest.approx(1 / 3)
    assert scores[film] == pytest.approx(0)


def test_sentence_views_rank_original_evidence_and_reuse_chunk_prefixes(tmp_path, monkeypatch):
    sentence, shared = "She joined the board.", "He graduated."
    values = {sentence: [0, 1, 0], shared: [0, 1, 0]}
    encoder = Encoder(values)
    store = make_store({"board": [fact("Ada", "Board", sentence)], "x": [fact("Evan", "X", shared)],
                        "y": [fact("Omar", "Y", shared)]}, values)
    requested = []

    def complete(client, **kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        requested.append(payload)
        return completion(json.dumps({"1": {"prefix": "Ada"}}))

    monkeypatch.setattr("canon.core.offline.create_chat_client", lambda _: MagicMock())
    monkeypatch.setattr("canon.core.offline.get_chat_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr("canon.core.offline.create_answer_completion", complete)
    repeated = next(i for i, record in enumerate(store.records) if record.sentence == shared)
    for mode in ("triple", "llm", "llm"):
        vectors = build_sentence_vectors(store, encoder, mode, runtime(), tmp_path, force=False)
        assert np.array_equal(vectors[repeated], store.vectors[repeated])
        retriever = CanonRetriever(store, make_hierarchy(store), encoder, method(), runtime(), sentence_vectors=vectors)
        result = retriever.retrieve("Who joined the board?", 1, 1)
        assert result["edges"][0]["sentence"] == sentence
        assert result["chunks"] == ["board"]
    assert len(requested) == 1
    assert requested[0]["source_chunk"] == store.chunk_map["board"]
    assert [row["sentence"] for row in requested[0]["rows"]] == [sentence]


def test_diffusion_conserves_mass_and_exposes_truncation():
    store = make_store({"a": [fact("A", "B", "One")], "b": [fact("B", "C", "Two", subtopic="Other")]})
    config = method(bottom_up_weight=1.0, seed_sentences=1, diffusion_iterations=1, diffusion_tolerance=1e-12)
    hierarchy = make_hierarchy(store)
    router = DiffusionRouter(store, hierarchy, config, runtime())
    first, second = (path[1] for path in hierarchy.branches)
    assert not hierarchy.graph.has_edge(first, second)
    assert dict(hierarchy.transitions[first])[second] > 0
    prob = {(router.order[s], router.order[t]): w for s, t, w in zip(router.sources, router.targets, router.weights)}
    methods = path_named(hierarchy, "Research", "Methods")[1]
    assert prob[methods, "entity_a"] > prob[methods, "entity_b"]
    assert prob["entity_a", methods] > prob["entity_b", first] + prob["entity_b", second]
    assert prob[first, second] > 0
    assert router.seeds(query(store)).sum() == pytest.approx(1.0)
    outgoing = np.bincount(router.sources, weights=router.weights, minlength=len(router.order))
    assert np.allclose(outgoing[~router.dangling], 1)
    _scores, diagnostics = router.score(query(store))
    assert diagnostics["iterations"] == 1
    assert diagnostics["converged"] is False


def test_a_uses_the_prepared_index_without_online_chat():
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method()
    retriever = CanonRetriever(store, make_hierarchy(store), Encoder(), config, runtime(), client=None)
    result = retriever.retrieve("Question")
    assert result["chunks"] == ["a"]
    assert result["diagnostics"]["routing_calls"] == 0


def test_b_uses_one_joint_call_and_validates_path_ids(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One", "Biology", "History")],
                        "b": [fact("C", "D", "Two", "Film", "History")]})
    config = method(routing="b")
    hierarchy = make_hierarchy(store)
    client = client_for('{"paths":[2]}')
    retriever = CanonRetriever(store, hierarchy, Encoder(), config, runtime(), MeteredClient(client))
    result = retriever.retrieve("Question")
    assert len(result["paths"]) == 1
    assert result["diagnostics"]["routing_calls"] == 1
    assert client.chat.completions.create.call_count == 1
    payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"].split("\n", 1)[1])
    selected_topic = next(topic for topic in payload["topics"] if any(card["id"] == 2 for card in topic["paths"]))
    assert hierarchy.nodes[result["paths"][0]["topic_id"]]["label"] == selected_topic["topic"]
    assert all(card["preview"] and card["synopsis"] for topic in payload["topics"] for card in topic["paths"])


def test_preview_failures_have_finite_budget_and_deterministic_fallback(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method(routing="b")
    hierarchy = make_hierarchy(store)
    client = client_for('{"paths":[true]}')
    router = EvidencePreviewRouter(store, hierarchy, config, runtime(), client)
    result = router.route(query(store))
    assert client.chat.completions.create.call_count == 3
    assert result.diagnostics["fallback"] == "candidate_scores"
    assert result.paths == hierarchy.branches


def test_hybrid_composes_diffusion_with_exactly_one_joint_call(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One")], "b": [fact("C", "D", "Two", subtopic="Other")]})
    config = method(routing="hybrid")
    client = client_for('{"paths":[1]}')
    result = HybridRouter(store, make_hierarchy(store), config, runtime(), client).route(query(store))
    assert result.diagnostics["mode"] == "hybrid"
    assert "iterations" in result.diagnostics
    assert client.chat.completions.create.call_count == 1


def test_preview_trimming_keeps_ids_valid_with_a_bounded_input(monkeypatch):
    class BudgetTokenizer(Tokenizer):
        def count_messages(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"].split("\n", 1)[1])
            return 200 + 100 * sum(len(topic["paths"]) for topic in payload["topics"])

    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: BudgetTokenizer())
    store = make_store({str(i): [fact(f"A{i}", f"B{i}", f"Sentence {i}", subtopic=f"Branch {i}")] for i in range(6)})
    config = method(routing="b", preview_input_tokens=500)
    client = client_for('{"paths":[3]}')
    result = EvidencePreviewRouter(store, make_hierarchy(store), config, runtime(topic_choice_max=1), client).route(query(store))
    assert result.diagnostics["candidate_paths"] == 4
    assert result.diagnostics["prompt_paths"] == 3
    assert result.diagnostics["input_tokens_estimate"] <= 500
    assert result.diagnostics["previews_removed"] is False
    assert result.diagnostics["retained_previews"] == 3
    assert len(result.paths) == 1


def test_permanent_routing_error_keeps_usage_without_retry_or_fallback(monkeypatch):
    class PermanentAPIError(Exception):
        status_code = 401

    monkeypatch.setattr("canon.routing.preview.APIStatusError", PermanentAPIError)
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method(routing="b")
    raw = client_for("")
    raw.chat.completions.create.side_effect = PermanentAPIError("private response body")
    retriever = CanonRetriever(store, make_hierarchy(store), Encoder(), config, runtime(), MeteredClient(raw))
    with pytest.raises(RuntimeError) as captured:
        retriever.retrieve("Question")
    assert raw.chat.completions.create.call_count == 1
    assert captured.value.diagnostics["routing_calls"] == 1
    assert "private response body" not in str(captured.value.diagnostics)


def test_unfit_preview_is_rejected_before_remote_call(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method(routing="b", preview_input_tokens=1)
    client = client_for('{"paths":[1]}')
    with pytest.raises(ValueError, match="input budget"):
        EvidencePreviewRouter(store, make_hierarchy(store), config, runtime(), client).route(query(store))
    client.chat.completions.create.assert_not_called()


def test_baseline_adapter_resolves_same_label_within_its_parent(monkeypatch):
    store = make_store({"a": [fact("A", "B", "One", "Biology", "History")], "b": [fact("C", "D", "Two", "Film", "History")]})
    hierarchy = make_hierarchy(store)
    monkeypatch.setattr("canon.routing.baseline.choose_topics_from_graph", lambda *args, **kwargs: ["Film"])
    monkeypatch.setattr("canon.routing.baseline.choose_subtopics_for_topic", lambda **kwargs: [
        kwargs["graph"].nodes[child]["label"] for child in hierarchy.children[kwargs["topic_nid"]]
    ])
    result = BaselineRouter(hierarchy, runtime(), object()).route(query(store))
    assert result.paths == [path_named(hierarchy, "Film", "History")]


def test_branch_balance_protects_other_branch_before_k1_truncation():
    store = make_store({"a": [fact("A", "B", "One", subtopic="First")],
                        "b": [fact("C", "D", "Two", subtopic="First")],
                        "c": [fact("E", "F", "Three", subtopic="Second")]})
    config = method(balance=True)
    hierarchy = make_hierarchy(store)
    values = {"One": 0.99, "Two": 0.98, "Three": 0.95}
    context = query(store, [values[record.sentence] for record in store.records])
    routes = RoutingResult(hierarchy.branches)
    plain = Allocator(store, hierarchy, replace(config, balance=False)).select(context, routes, list(range(3)), 2, 2)
    assert plain.chunks == ["a", "b"]
    balanced = Allocator(store, hierarchy, config).select(context, routes, list(range(3)), 2, 2)
    assert balanced.chunks == ["a", "c"]
    assert len(balanced.indices) == 2
    assert all(any(occ.chunk_id == chunk for index in balanced.indices for occ in store.records[index].occurrences) for chunk in balanced.chunks)


def test_bridge_bonus_uses_real_shared_entity_in_a_different_branch():
    store = make_store({"a": [fact("Anchor", "Relay", "First", subtopic="Hop1")],
                        "b": [fact("Relay", "Answer", "Second", subtopic="Hop2")],
                        "c": [fact("Noise", "Other", "Third", subtopic="Hop2")]})
    config = method(bridge=True, bridge_seed_sentences=1)
    hierarchy = make_hierarchy(store)
    raw = {"First": 0.9, "Second": 0.5, "Third": 0.6}
    context = query(store, [raw[record.sentence] for record in store.records], "Tell me about Anchor")
    scores, entities = bridge_scores(context, RoutingResult(hierarchy.branches), list(range(3)), store, hierarchy, config)
    assert "entity_anchor" not in entities
    assert "entity_relay" in entities
    by_text = {record.sentence: scores[index] for index, record in enumerate(store.records)}
    assert by_text["Second"] > 0.5
    assert by_text["Third"] == 0.6


def test_invalid_k_budgets_fail_before_routing():
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method()
    retriever = CanonRetriever(store, make_hierarchy(store), Encoder(), config, runtime())
    retriever.router = MagicMock()
    with pytest.raises(ValueError):
        retriever.retrieve("Question", top_k1=1, top_k2=2)
    retriever.router.route.assert_not_called()


def test_usage_counts_stay_in_their_query_scope():
    raw = client_for('{"paths":[1]}')
    client = MeteredClient(raw)
    first, second = [], []
    with record_calls(first, "routing"):
        client.with_options(max_retries=0).chat.completions.create(model="mock")
    with record_calls(second, "answer"):
        client.chat.completions.create(model="mock")
    assert first[0]["stage"] == "routing" and second[0]["stage"] == "answer"
    assert first[0]["total_tokens"] == 25


def test_cache_identity_and_corruption_are_rejected(tmp_path):
    atomic_json(tmp_path / "data.json", {"value": 1})
    finish_cache(tmp_path, {"method": "a"}, ("data.json",), 0.0)
    assert valid_cache(tmp_path, {"method": "a"}, ("data.json",))
    with pytest.raises(ValueError, match="identity"):
        valid_cache(tmp_path, {"method": "b"}, ("data.json",))
    atomic_json(tmp_path / "data.json", {"value": 2})
    with pytest.raises(ValueError, match="corrupt"):
        valid_cache(tmp_path, {"method": "a"}, ("data.json",))


def test_duplicate_queries_keep_distinct_ids_and_positions(tmp_path):
    atomic_json(tmp_path / "qa.json", [{"query": "Same", "answer": "A"}, {"query": "Same", "answer": "B"}])
    config = SimpleNamespace(dataset_name="demo", get_questions_file=lambda: tmp_path / "qa.json", get_dataset_dir=lambda: tmp_path)
    questions, _ = load_questions(config, None, None)
    assert len({question["qid"] for question in questions}) == 2
    assert [question["row_index"] for question in questions] == [0, 1]


def test_checkpoints_resume_successes_without_retrieval(tmp_path):
    questions = [{"qid": "demo:hash:0", "query": "Q", "row_index": 0, "source_id": None}]
    directory = tmp_path / "retrieve"
    atomic_json(directory / "questions/000000.json", {
        **questions[0], "run_fingerprint": "fingerprint", "status": "ok", "retrieval": {"chunks": ["c"]},
    })
    retriever = MagicMock()
    result = run_questions("retrieve", directory, "fingerprint", questions, runtime(), None, retriever, None, False)
    assert result["failed"] == 0
    retriever.retrieve.assert_not_called()


def test_answerer_injection_keeps_all_source_chunks_without_constructing_old_retriever(monkeypatch):
    from generate import graph_rag

    original_constructor = MagicMock(side_effect=AssertionError("Old retriever must not be created"))
    monkeypatch.setattr(graph_rag, "Retriever", original_constructor)
    monkeypatch.setattr(graph_rag, "get_chat_tokenizer", lambda path: SimpleNamespace(count_text=lambda text: len(text)))
    client = client_for("Answer")
    config = runtime(chat_model="mock", dataset_name="demo")
    retrieval = {"chunks": ["a", "b"], "edges": [{
        "source": "A", "target": "B", "label": "related_to", "sentence": "Evidence",
        "chunk_id": "a", "chunk_ids": ["a", "b"],
    }]}
    retriever = SimpleNamespace(retrieve=lambda *args, **kwargs: retrieval)
    rag = graph_rag.GraphRAG(
        dataset_name="demo", answer_prompt="{{question}}\n{{context}}", system_prompt="Use evidence",
        default_top_k1=2, default_top_k2=2, temperature=0, max_output_tokens=64,
        retriever=retriever, client=client, config=config, chunk_map={"a": "First chunk", "b": "Second chunk"},
    )
    answer, _, _ = rag.answer("Question")
    assert answer == "Answer"
    assert rag.all_sentence_chunk_ids == ["a", "b"]
    assert rag.last_retrieval is retrieval
    original_constructor.assert_not_called()


def test_source_order_survives_sentence_sorting_and_artifact_round_trip(tmp_path):
    blocks = [
        {"chunk_id": "c0", "content": "First", "triples": [fact("A", "B", "Zebra evidence", "Business", "Zulu role")]},
        {"chunk_id": "c1", "content": "Second", "triples": [fact("C", "D", "Alpha evidence", "Business", "Alpha role"),
                                                               fact("E", "F", "Other evidence", "Academia", "Shared")]},
    ]
    chunks = {block["chunk_id"]: block["content"] for block in blocks}
    store = EvidenceStore.build(blocks, chunks, Encoder())
    assert store.records[0].sentence == "Alpha evidence"
    hierarchy = make_hierarchy(store, source_order=occurrence_order(blocks))
    hierarchy.save(tmp_path)
    restored = HierarchyArtifacts.load(tmp_path, store)
    for graph in (hierarchy, restored):
        assert [data["label"] for _, data in graph.graph.nodes(data=True) if data["type"] == "topic"] == ["Business", "Academia"]
        assert [graph.nodes[child]["label"] for child in graph.children["topic_business"]] == ["Zulu role", "Alpha role"]


def test_raw_node_keeps_last_display_label_from_extraction():
    blocks = [{"chunk_id": "c", "content": "Content", "triples": [
        fact("A", "B", "One", "Business"), fact("C", "D", "Two", "Business"),
        fact("E", "F", "Three", "business"),
    ]}]
    store = EvidenceStore.build(blocks, {"c": "Content"}, Encoder())
    hierarchy = make_hierarchy(store, source_order=occurrence_order(blocks))
    assert hierarchy.nodes["topic_business"]["label"] == "business"


def test_coverage_cannot_win_just_by_accumulating_many_low_quality_branches():
    noise = [fact(f"N{i}", f"M{i}", "Noise", subtopic=f"Noise branch {i}") for i in range(20)]
    store = make_store({"relevant": [fact("A", "B", "Relevant", subtopic="Target")], "noise": noise})
    config = method(balance=True, coverage_anchor_fraction=0)
    hierarchy = make_hierarchy(store)
    scores = [0.95 if record.sentence == "Relevant" else 0.3 for record in store.records]
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), [0, 1], 1, 1)
    assert result.chunks == ["relevant"]
    assert result.diagnostics["coverage_path_weight_sum"] == pytest.approx(1)


def test_half_budget_protects_raw_relevance_and_keeps_k1_k2_limits():
    store = make_store({str(i): [fact(f"A{i}", f"B{i}", f"Sentence {i}", subtopic=f"Branch {i}")] for i in range(6)})
    config = method(balance=True)
    hierarchy = make_hierarchy(store)
    values = {f"Sentence {i}": 1 - i * 0.05 for i in range(6)}
    scores = [values[record.sentence] for record in store.records]
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), list(range(6)), 4, 4)
    assert result.diagnostics["protected_chunks"] == ["0", "1"]
    assert len(result.chunks) == 4 and len(result.indices) == 4


def test_route_replay_is_per_question_and_makes_no_routing_call(tmp_path):
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method()
    hierarchy = make_hierarchy(store)
    topic, subtopic = hierarchy.branches[0]
    question = {"qid": "demo:hash:0", "row_index": 0, "query": "Question"}
    atomic_json(tmp_path / "manifest.json", {"fingerprint": "source", "identity": {"routing_signature": "signature"}})
    atomic_json(tmp_path / "generate_short/questions/000000.json", {
        **question, "run_fingerprint": "source", "status": "ok", "result": "Not used by replay",
        "retrieval": {"paths": [{"topic_id": topic, "subtopic_id": subtopic, "score": 1.0}],
                      "diagnostics": {"routing": {"mode": "a"}, "routing_calls": 0}},
    })
    replay = RoutingReplay(tmp_path, "signature")
    base = CanonRetriever(store, hierarchy, Encoder(), config, runtime(), router=ReplayOnlyRouter())
    fixed = base.with_routing(question["query"], replay.load(question))
    result = fixed.retrieve(question["query"])
    assert result["diagnostics"]["routing_calls"] == 0
    assert result["diagnostics"]["routing"]["replayed"] is True
    assert isinstance(base.router, ReplayOnlyRouter)
    with pytest.raises(ValueError, match="matching question"):
        replay.load({**question, "qid": "another-question"})
    with pytest.raises(ValueError, match="same routing index"):
        RoutingReplay(tmp_path, "other-signature")
    source_path = tmp_path / "generate_short/questions/000000.json"
    rewritten = json.loads(source_path.read_text(encoding="utf-8"))
    rewritten["retrieval"]["paths"][0]["score"] = 0.5
    atomic_json(source_path, rewritten)
    new_snapshot = RoutingReplay(tmp_path, "signature")
    assert replay.load(question).scores[(topic, subtopic)] == 1.0
    assert new_snapshot.load(question).scores[(topic, subtopic)] == 0.5
    assert replay.metadata()["routes_sha256"] != new_snapshot.metadata()["routes_sha256"]


def test_failed_replay_cannot_reuse_previous_questions_retrieval(tmp_path, monkeypatch):
    questions = [{"qid": f"demo:hash:{i}", "query": f"Question {i}", "row_index": i} for i in range(2)]
    replay = MagicMock()
    replay.load.side_effect = [RoutingResult([]), ValueError("Missing replay checkpoint")]
    retriever = MagicMock()
    rag = SimpleNamespace(last_retrieval={}, retriever=retriever)

    def answer(_query):
        rag.last_retrieval = {"chunks": ["first-question-only"], "diagnostics": {"seconds": {"retrieval": 0.1}}}
        return "Answer", 0.1, 1

    rag.answer = MagicMock(side_effect=answer)
    monkeypatch.setattr("canon.pipeline.make_answerer", lambda *args, **kwargs: rag)
    result = run_questions("generate_long", tmp_path, "target", questions, runtime(max_workers=1),
                           None, retriever, None, False, replay=replay)
    failed = json.loads((tmp_path / "questions/000001.json").read_text(encoding="utf-8"))
    assert result["failed"] == 1 and failed["status"] == "error"
    assert failed["retrieval"] == {}
    assert rag.answer.call_count == 1


def test_e_switches_do_not_change_routing_replay_settings():
    base = CanonConfig()
    changed = replace(base, bridge=False, balance=False, coverage_weight=0.1, coverage_anchor_fraction=0)
    assert base.routing_settings() == changed.routing_settings()


def test_null_queries_remain_regular_questions(tmp_path):
    atomic_json(tmp_path / "qa.json", [{"query": "Known", "answer": "A"},
                                      {"query": "Unanswerable", "answer": "Insufficient information."}])
    atomic_json(tmp_path / "metadata.json", [{"query": "Known", "question_type": "inference_query"},
                                            {"query": "Unanswerable", "question_type": "null_query"}])
    config = SimpleNamespace(dataset_name="demo", get_questions_file=lambda: tmp_path / "qa.json", get_dataset_dir=lambda: tmp_path)
    questions, _ = load_questions(config, None, None)
    assert [question["query"] for question in questions] == ["Known", "Unanswerable"]
