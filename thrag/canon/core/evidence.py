"""One vector per exact sentence, with lossless occurrence-level provenance."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from index.json_to_gexf import clean_id
from canon.core.contracts import Occurrence, SentenceRecord
from canon.core.io import atomic_array, atomic_json, read_json, stable_id, unit_rows


def collect_records(blocks: list[dict[str, Any]], chunk_map: dict[str, str]) -> tuple[SentenceRecord, ...]:
    """Never recover associations by splitting aggregated GEXF strings."""
    grouped: dict[str, list[Occurrence]] = defaultdict(list)
    seen_chunks: set[str] = set()
    for block in blocks:
        chunk_id = block["chunk_id"]
        if chunk_id in seen_chunks or chunk_id not in chunk_map or block["content"] != chunk_map[chunk_id]:
            raise ValueError(f"Extraction/KV mismatch or duplicate chunk: {chunk_id}")
        seen_chunks.add(chunk_id)
        for position, item in enumerate(block["triples"]):
            triple = item["triple"]
            if len(triple) != 3 or any(not isinstance(text, str) or not text.strip() for text in triple):
                raise ValueError(f"Invalid triple in {chunk_id}, position {position}")
            source, relation, target = (text.strip() for text in triple)
            value = item["sentence"]
            sentences = [value] if isinstance(value, str) else value
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(f"Invalid source sentence in {chunk_id}, position {position}")
            for sentence_position, sentence in enumerate(sentences):
                if not isinstance(sentence, str) or not sentence.strip():
                    raise ValueError(f"Empty/non-text source sentence in {chunk_id}")
                grouped[sentence].append(Occurrence(
                    id=stable_id("occurrence", chunk_id, position, sentence_position),
                    chunk_id=chunk_id,
                    source_id=f"entity_{clean_id(source)}", target_id=f"entity_{clean_id(target)}",
                    source=source, target=target, label=relation,
                    subject_topic=item["subject"]["main_topic"].strip(),
                    subject_subtopic=item["subject"]["subtopic"].strip(),
                    object_topic=item["object"]["main_topic"].strip(),
                    object_subtopic=item["object"]["subtopic"].strip(),
                ))
    if seen_chunks != set(chunk_map):
        raise ValueError("Extraction is incomplete; finish the original extraction/repair stage first")
    if not grouped:
        raise ValueError("No evidence sentences in the extraction snapshot")
    return tuple(
        SentenceRecord(stable_id("sentence", sentence), sentence, tuple(sorted(items, key=lambda item: item.id)))
        for sentence, items in sorted(grouped.items())
    )


class EvidenceStore:
    def __init__(
        self, records: tuple[SentenceRecord, ...], vectors: np.ndarray,
        vocabulary: list[str], text_vectors: np.ndarray, chunk_map: dict[str, str],
    ) -> None:
        if vectors.ndim != 2 or len(vectors) != len(records) or not len(records):
            raise ValueError("Sentence vectors and records are not aligned")
        if text_vectors.shape != (len(vocabulary), vectors.shape[1]):
            raise ValueError("Label/entity vocabulary vectors are not aligned")
        for matrix in (vectors, text_vectors):
            if not np.isfinite(matrix).all():
                raise ValueError("Non-finite embeddings")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.all(np.isclose(norms, 1, atol=1e-4) | (norms == 0)):
                raise ValueError("Expected normalized embeddings")
        self.records = records
        self.vectors = vectors
        self.vectors.flags.writeable = False
        self.text_vectors = text_vectors
        self.text_vectors.flags.writeable = False
        self.vocabulary = vocabulary
        self.text_index = {text: index for index, text in enumerate(vocabulary)}
        self.dimension = vectors.shape[1]
        self.chunk_map = chunk_map
        self.entity_to_indices: dict[str, set[int]] = defaultdict(set)
        self.entity_labels: dict[str, set[str]] = defaultdict(set)
        self.entity_neighbors: dict[str, set[str]] = defaultdict(set)
        for index, record in enumerate(records):
            for item in record.occurrences:
                if item.chunk_id not in chunk_map:
                    raise ValueError(f"Missing source chunk: {item.chunk_id}")
                for entity_id, label in ((item.source_id, item.source), (item.target_id, item.target)):
                    self.entity_to_indices[entity_id].add(index)
                    self.entity_labels[entity_id].add(label)
                if item.source_id != item.target_id:
                    self.entity_neighbors[item.source_id].add(item.target_id)
                    self.entity_neighbors[item.target_id].add(item.source_id)

    @classmethod
    def build(cls, blocks: list[dict[str, Any]], chunk_map: dict[str, str], encoder: Any) -> EvidenceStore:
        records = collect_records(blocks, chunk_map)
        vocabulary = sorted({
            text for record in records for item in record.occurrences for text in (
                item.source, item.target, item.subject_topic, item.subject_subtopic,
                item.object_topic, item.object_subtopic,
            )
        })
        vectors = unit_rows(encoder.encode_passages([record.sentence for record in records], show_progress=True))
        text_vectors = unit_rows(encoder.encode_passages(vocabulary, show_progress=True))
        return cls(records, vectors, vocabulary, text_vectors, chunk_map)

    def text_vector(self, text: str) -> np.ndarray:
        return self.text_vectors[self.text_index[text]]

    def candidates(self, entity_ids: set[str]) -> list[int]:
        """Exact entity-scoped pool; no global Top-K pre-truncation."""
        return sorted({index for entity in entity_ids for index in self.entity_to_indices.get(entity, ())})

    def save(self, directory: Path) -> None:
        atomic_json(directory / "sentences.json", [asdict(record) for record in self.records])
        atomic_json(directory / "vocabulary.json", self.vocabulary)
        atomic_array(directory / "sentences.npy", self.vectors)
        atomic_array(directory / "vocabulary.npy", self.text_vectors)

    @classmethod
    def load(cls, directory: Path, chunk_map: dict[str, str]) -> EvidenceStore:
        records = tuple(SentenceRecord(
            item["id"], item["sentence"], tuple(Occurrence(**entry) for entry in item["occurrences"]),
        ) for item in read_json(directory / "sentences.json"))
        return cls(
            records, np.load(directory / "sentences.npy", allow_pickle=False, mmap_mode="r"),
            read_json(directory / "vocabulary.json"),
            np.load(directory / "vocabulary.npy", allow_pickle=False, mmap_mode="r"), chunk_map,
        )

    def edge_result(self, index: int, score: float, rank: int) -> dict[str, Any]:
        record = self.records[index]
        representative = record.occurrences[0]
        chunks = sorted({item.chunk_id for item in record.occurrences})
        return {
            "sentence_id": record.id, "sentence": record.sentence,
            "source": representative.source, "target": representative.target, "label": representative.label,
            "source_id": representative.source_id, "target_id": representative.target_id,
            "entity_ids": sorted(record.entity_ids), "chunk_ids": chunks, "chunk_id": chunks[0],
            "occurrences": [asdict(item) for item in record.occurrences], "score": score, "rank": rank,
        }
