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
from canon.core.telemetry import MeteredClient, record_calls
from canon.hierarchy import build_hierarchy
from canon.hierarchy.builder import HierarchyArtifacts
from canon.hierarchy.source_order import occurrence_order
from canon.hierarchy.synonym_review import SynonymReviewer
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


def method(**kwargs):
    return replace(CanonConfig(hierarchy="split", soft_links=False, routing="a", bridge=False, balance=False), **kwargs)


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
    hierarchy = build_hierarchy(store, config)
    desired = next(index for index, record in enumerate(store.records) if record.sentence == "Sentence 9")
    scores = np.ones(len(store.records))
    scores[desired] = 0.01
    indices = store.candidates({"entity_a9"})
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), indices, 1, 1)
    assert result.indices == [desired]
    assert result.chunks == ["c9"]


def test_scoped_subtopics_do_not_mix_same_name_across_parents():
    store = make_store({"biology": [fact("Species", "Trait", "Evolution", "Biology", "Adaptation")],
                        "film": [fact("Novel", "Movie", "Screenplay", "Film", "Adaptation")]})
    scoped = build_hierarchy(store, method())
    biology = path_named(scoped, "Biology", "Adaptation")
    film = path_named(scoped, "Film", "Adaptation")
    assert biology[1] != film[1]
    assert scoped.branch_entities[biology].isdisjoint(scoped.branch_entities[film])
    raw = build_hierarchy(store, method(hierarchy="none"))
    assert len({path[1] for path in raw.branches}) == 1
    assert all(len(raw.branch_entities[path]) == 4 for path in raw.branches)


def test_dense_prototypes_merge_synonyms_without_shared_entities():
    store = make_store({"a": [fact("A", "B", "First evidence", subtopic="Film director")],
                        "b": [fact("C", "D", "Second evidence", subtopic="Movie director")]})
    hierarchy = build_hierarchy(store, method(hierarchy="split_merge"))
    subtopics = [node for node in hierarchy.nodes.values() if node["kind"] == "subtopic"]
    assert len(subtopics) == 1
    assert set(subtopics[0]["aliases"]) == {"Film director", "Movie director"}
    assert set(subtopics[0]["entities"]) == {"entity_a", "entity_b", "entity_c", "entity_d"}
    assert sum(len(record.occurrences) for record in store.records) == 2


def test_complete_link_rejects_transitive_merge_chain():
    vectors = {"A": [1, 0, 0], "B": [0.8660254, 0.5, 0], "C": [0.5, 0.8660254, 0]}
    store = make_store({str(i): [fact(f"X{i}", f"Y{i}", f"Fact {i}", subtopic=name)] for i, name in enumerate(vectors)}, vectors)
    hierarchy = build_hierarchy(store, method(hierarchy="split_merge", merge_label_threshold=0.8))
    assert len([node for node in hierarchy.nodes.values() if node["kind"] == "subtopic"]) == 2


def test_soft_links_preserve_nodes_and_check_parent_compatibility():
    vectors = {"Biology": [1, 0, 0], "Film": [0, 1, 0], "Evolution": [1, 0, 0], "Screenplay": [0, 1, 0]}
    store = make_store({"a": [fact("A", "B", "Evolution", "Biology", "Adaptation")],
                        "b": [fact("C", "D", "Screenplay", "Film", "Adaptation")]}, vectors)
    hierarchy = build_hierarchy(store, method(soft_links=True, soft_context_threshold=0.0))
    assert len(hierarchy.nodes) == 4
    assert not any(edge[2] == "soft" for edge in hierarchy.edges)
    compatible = make_store({"a": [fact("A", "B", "One", subtopic="First")], "b": [fact("C", "D", "Two", subtopic="Second")]})
    linked = build_hierarchy(compatible, method(soft_links=True))
    assert len(linked.nodes) == 3
    assert any(edge[2] == "soft" for edge in linked.edges)


def test_diffusion_conserves_mass_and_exposes_truncation():
    store = make_store({"a": [fact("A", "B", "One")], "b": [fact("B", "C", "Two", subtopic="Other")]})
    config = method(bottom_up_weight=1.0, seed_sentences=1, diffusion_iterations=1, diffusion_tolerance=1e-12)
    hierarchy = build_hierarchy(store, config)
    router = DiffusionRouter(store, hierarchy, config, runtime())
    assert router.seeds(query(store)).sum() == pytest.approx(1.0)
    outgoing = np.bincount(router.sources, weights=router.weights, minlength=len(router.order))
    assert np.allclose(outgoing[~router.dangling], 1)
    _scores, diagnostics = router.score(query(store))
    assert diagnostics["iterations"] == 1
    assert diagnostics["converged"] is False


def test_a_needs_no_chat_client_when_hierarchy_module_is_off():
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method(hierarchy="none")
    retriever = CanonRetriever(store, build_hierarchy(store, config), Encoder(), config, runtime(), client=None)
    result = retriever.retrieve("Question")
    assert result["chunks"] == ["a"]
    assert result["diagnostics"]["routing_calls"] == 0


def test_b_uses_one_joint_call_and_validates_path_ids(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One", "Biology", "History")],
                        "b": [fact("C", "D", "Two", "Film", "History")]})
    config = method(routing="b")
    hierarchy = build_hierarchy(store, config)
    client = client_for('{"paths":[2]}')
    retriever = CanonRetriever(store, hierarchy, Encoder(), config, runtime(), MeteredClient(client))
    result = retriever.retrieve("Question")
    assert len(result["paths"]) == 1
    assert result["diagnostics"]["routing_calls"] == 1
    assert client.chat.completions.create.call_count == 1
    payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"].split("\n", 1)[1])
    selected_topic = next(topic for topic in payload["topics"] if any(card["id"] == 2 for card in topic["paths"]))
    assert hierarchy.nodes[result["paths"][0]["topic_id"]]["label"] == selected_topic["topic"]
    assert all(card.get("preview") for topic in payload["topics"] for card in topic["paths"])


def test_preview_failures_have_finite_budget_and_deterministic_fallback(monkeypatch):
    monkeypatch.setattr("canon.routing.preview.get_chat_tokenizer", lambda path: Tokenizer())
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method(routing="b")
    hierarchy = build_hierarchy(store, config)
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
    result = HybridRouter(store, build_hierarchy(store, config), config, runtime(), client).route(query(store))
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
    result = EvidencePreviewRouter(store, build_hierarchy(store, config), config, runtime(topic_choice_max=1), client).route(query(store))
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
    retriever = CanonRetriever(store, build_hierarchy(store, config), Encoder(), config, runtime(), MeteredClient(raw))
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
        EvidencePreviewRouter(store, build_hierarchy(store, config), config, runtime(), client).route(query(store))
    client.chat.completions.create.assert_not_called()


def test_baseline_adapter_resolves_same_label_within_its_parent(monkeypatch):
    store = make_store({"a": [fact("A", "B", "One", "Biology", "History")], "b": [fact("C", "D", "Two", "Film", "History")]})
    hierarchy = build_hierarchy(store, method())
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
    hierarchy = build_hierarchy(store, config)
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
    hierarchy = build_hierarchy(store, config)
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
    retriever = CanonRetriever(store, build_hierarchy(store, config), Encoder(), config, runtime())
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


class SemanticReviewStub:
    def __init__(self, equivalents=(), verdict="related"):
        self.equivalents = {frozenset(pair) for pair in equivalents}
        self.verdict = verdict
        self.seen = []
        self.stats = {}
        self.calls = []

    def review(self, candidates):
        self.seen.extend(candidates)
        return {tuple(item["pair"]): (
            "equivalent" if frozenset((item["left"]["label"], item["right"]["label"])) in self.equivalents else self.verdict
        ) for item in candidates}


def test_source_order_survives_sentence_sorting_and_artifact_round_trip(tmp_path):
    blocks = [
        {"chunk_id": "c0", "content": "First", "triples": [fact("A", "B", "Zebra evidence", "Business", "Zulu role")]},
        {"chunk_id": "c1", "content": "Second", "triples": [fact("C", "D", "Alpha evidence", "Business", "Alpha role"),
                                                               fact("E", "F", "Other evidence", "Academia", "Shared")]},
    ]
    chunks = {block["chunk_id"]: block["content"] for block in blocks}
    store = EvidenceStore.build(blocks, chunks, Encoder())
    assert store.records[0].sentence == "Alpha evidence"
    hierarchy = build_hierarchy(store, method(hierarchy="none"), source_order=occurrence_order(blocks))
    hierarchy.save(tmp_path)
    restored = HierarchyArtifacts.load(tmp_path, store)
    for graph in (hierarchy, restored):
        assert [data["label"] for _, data in graph.graph.nodes(data=True) if data["type"] == "topic"] == ["Business", "Academia"]
        assert [graph.nodes[child]["label"] for child in graph.children["topic_business"]] == ["Zulu role", "Alpha role"]


def test_none_keeps_last_original_display_label_without_merging_by_frequency():
    blocks = [{"chunk_id": "c", "content": "Content", "triples": [
        fact("A", "B", "One", "Business"), fact("C", "D", "Two", "Business"),
        fact("E", "F", "Three", "business"),
    ]}]
    store = EvidenceStore.build(blocks, {"c": "Content"}, Encoder())
    hierarchy = build_hierarchy(store, method(hierarchy="none"), source_order=occurrence_order(blocks))
    assert hierarchy.nodes["topic_business"]["label"] == "business"


def test_new_h_merges_verified_synonyms_across_parents_and_disjoint_entities():
    store = make_store({"a": [fact("A", "B", "First film", "Cinema", "Film director")],
                        "b": [fact("C", "D", "Second film", "Employment", "Movie director")]})
    reviewer = SemanticReviewStub([("Film director", "Movie director")])
    hierarchy = build_hierarchy(store, method(hierarchy="normalize"), synonym_reviewer=reviewer)
    children = [node for node in hierarchy.nodes.values() if node["kind"] == "subtopic"]
    assert len(children) == 1
    assert set(children[0]["aliases"]) == {"Film director", "Movie director"}
    assert len(children[0]["parents"]) == 2
    assert set(children[0]["entities"]) == {"entity_a", "entity_b", "entity_c", "entity_d"}
    assert len(hierarchy.occurrence_paths) == 2


def test_new_h_keeps_shared_labels_instead_of_splitting_every_parent():
    store = make_store({"a": [fact("A", "B", "One", "Business", "Company")],
                        "b": [fact("C", "D", "Two", "Technology", "Company")]})
    hierarchy = build_hierarchy(store, method(hierarchy="normalize"), synonym_reviewer=SemanticReviewStub())
    assert len({path[1] for path in hierarchy.branches}) == 1
    assert all(len(hierarchy.branch_entities[path]) == 4 for path in hierarchy.branches)


@pytest.mark.parametrize("verdict", ["related", "broader_narrower", "different_sense", "uncertain"])
def test_high_vector_similarity_cannot_override_negative_or_uncertain_semantics(verdict):
    # All fixture vectors are identical; the old cosine rule alone would merge these.
    store = make_store({"a": [fact("A", "B", "One", "Legal Defense")],
                        "b": [fact("C", "D", "Two", "Legal System")]})
    hierarchy = build_hierarchy(store, method(hierarchy="normalize", soft_links=True),
                                synonym_reviewer=SemanticReviewStub(verdict=verdict))
    assert len([node for node in hierarchy.nodes.values() if node["kind"] == "topic"]) == 2
    assert not any(edge[2] == "soft" for edge in hierarchy.edges)


def test_numeric_and_polarity_conflicts_are_rejected_before_semantic_review():
    from canon.hierarchy.normalize import scope_conflict

    assert scope_conflict("Revenue 2023", "Revenue 2024") == "different_numeric_anchors"
    assert scope_conflict("Guilty", "Not guilty") == "opposite_polarity"
    assert scope_conflict("Before tax", "After tax") == "opposite_scope"
    assert scope_conflict("Not guilty", "Innocent") is None
    store = make_store({"a": [fact("A", "B", "One", subtopic="Revenue 2023")],
                        "b": [fact("C", "D", "Two", subtopic="Revenue 2024")]})
    reviewer = SemanticReviewStub(verdict="equivalent")
    hierarchy = build_hierarchy(store, method(hierarchy="normalize"), synonym_reviewer=reviewer)
    assert reviewer.seen == []
    assert len(hierarchy.branches) == 2


def test_semantic_cluster_requires_every_cross_member_pair_and_blocks_conflicting_soft_link():
    store = make_store({str(i): [fact(f"E{i}", f"F{i}", f"Sentence {i}", subtopic=label)]
                        for i, label in enumerate(("Alpha", "Beta", "Gamma"))})
    reviewer = SemanticReviewStub([("Alpha", "Beta"), ("Beta", "Gamma")])
    hierarchy = build_hierarchy(store, method(hierarchy="normalize", soft_links=True), synonym_reviewer=reviewer)
    assert len([node for node in hierarchy.nodes.values() if node["kind"] == "subtopic"]) == 2
    assert not any(edge[2] == "soft" for edge in hierarchy.edges)


def test_review_budget_does_not_turn_unreviewed_pairs_into_merges():
    store = make_store({str(i): [fact(f"E{i}", f"F{i}", f"Sentence {i}", subtopic=label)]
                        for i, label in enumerate(("Alpha", "Beta", "Gamma"))})
    reviewer = SemanticReviewStub(verdict="equivalent")
    hierarchy = build_hierarchy(store, method(hierarchy="normalize", synonym_max_pairs=1), synonym_reviewer=reviewer)
    audit = hierarchy.canonical_map["normalization"]
    assert audit["reviewed_pairs"] == 1 and audit["unreviewed_pairs"] == 2
    assert len([node for node in hierarchy.nodes.values() if node["kind"] == "subtopic"]) == 2


def test_semantic_review_keeps_long_labels_including_scope_qualifiers():
    prefix = "A descriptive category " * 12
    labels = (prefix + "with full control", prefix + "with limited control")
    store = make_store({str(i): [fact(f"E{i}", f"F{i}", f"Sentence {i}", subtopic=label)]
                        for i, label in enumerate(labels)})
    reviewer = SemanticReviewStub()
    build_hierarchy(store, method(hierarchy="normalize"), synonym_reviewer=reviewer)
    shown = {side["label"] for item in reviewer.seen for side in (item["left"], item["right"])}
    assert set(labels).issubset(shown)


def test_offline_synonym_decisions_are_cached_by_context_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr("canon.hierarchy.synonym_review.get_chat_tokenizer", lambda path: Tokenizer())
    raw = client_for('{"decisions":[{"id":1,"relation":"equivalent"}]}')
    candidates = [{"pair": ["a", "b"], "layer": "subtopic", "left": {"label": "Film director", "evidence": ["First"]},
                   "right": {"label": "Movie director", "evidence": ["Second"]}}]
    config = method(hierarchy="normalize")
    cache = tmp_path / "decisions.json"
    first = SynonymReviewer(runtime(), config, cache, client=MeteredClient(raw))
    assert first.review(candidates)[("a", "b")] == "equivalent"
    assert raw.chat.completions.create.call_count == 1
    assert len(json.loads(cache.read_text(encoding="utf-8"))["calls"]) == 1
    second_client = client_for('{"decisions":[{"id":1,"relation":"different_sense"}]}')
    second = SynonymReviewer(runtime(), config, cache, client=second_client)
    assert second.review(candidates)[("a", "b")] == "equivalent"
    assert len(second.previous_calls) == 1 and not second.calls
    second_client.chat.completions.create.assert_not_called()
    changed_context = [{**candidates[0], "left": {"label": "Film director", "evidence": ["Changed evidence"]}}]
    assert second.review(changed_context)[("a", "b")] == "different_sense"
    assert second_client.chat.completions.create.call_count == 1
    assert len(json.loads(cache.read_text(encoding="utf-8"))["calls"]) == 2
    with pytest.raises(ValueError, match="identity"):
        SynonymReviewer(runtime(default_model="different-model"), config, cache, client=second_client)


def test_malformed_synonym_review_never_falls_back_to_embedding_merge(tmp_path, monkeypatch):
    monkeypatch.setattr("canon.hierarchy.synonym_review.get_chat_tokenizer", lambda path: Tokenizer())
    raw = client_for('{"decisions":[{"id":true,"relation":"equivalent"}]}')
    reviewer = SynonymReviewer(runtime(), method(hierarchy="normalize"), tmp_path / "decisions.json", client=raw)
    result = reviewer.review([{"pair": ["a", "b"], "left": {"label": "A"}, "right": {"label": "B"}}])
    assert result[("a", "b")] == "uncertain"
    assert raw.chat.completions.create.call_count == 3
    assert len(reviewer.calls) == 3


def test_coverage_cannot_win_just_by_accumulating_many_low_quality_branches():
    noise = [fact(f"N{i}", f"M{i}", "Noise", subtopic=f"Noise branch {i}") for i in range(20)]
    store = make_store({"relevant": [fact("A", "B", "Relevant", subtopic="Target")], "noise": noise})
    config = method(balance=True, coverage_anchor_fraction=0)
    hierarchy = build_hierarchy(store, config)
    scores = [0.95 if record.sentence == "Relevant" else 0.3 for record in store.records]
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), [0, 1], 1, 1)
    assert result.chunks == ["relevant"]
    assert result.diagnostics["coverage_path_weight_sum"] == pytest.approx(1)


def test_half_budget_protects_raw_relevance_and_keeps_k1_k2_limits():
    store = make_store({str(i): [fact(f"A{i}", f"B{i}", f"Sentence {i}", subtopic=f"Branch {i}")] for i in range(6)})
    config = method(balance=True)
    hierarchy = build_hierarchy(store, config)
    values = {f"Sentence {i}": 1 - i * 0.05 for i in range(6)}
    scores = [values[record.sentence] for record in store.records]
    result = Allocator(store, hierarchy, config).select(query(store, scores), RoutingResult(hierarchy.branches), list(range(6)), 4, 4)
    assert result.diagnostics["protected_chunks"] == ["0", "1"]
    assert len(result.chunks) == 4 and len(result.indices) == 4


def test_route_replay_is_per_question_and_makes_no_routing_call(tmp_path):
    store = make_store({"a": [fact("A", "B", "One")]})
    config = method()
    hierarchy = build_hierarchy(store, config)
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
    with pytest.raises(ValueError, match="identical V2"):
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
