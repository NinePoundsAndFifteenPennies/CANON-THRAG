"""Local-only helpers shared by the four UltraDomain preparation entry points.

Reads the released JSONL files only. This module does not download data, call
models, or modify the source. Exactly 100 QA records are selected: a file that
already has 100 records is used in file order, and larger files are sampled
without replacement with a fixed seed. Passages are deduplicated by exact
context text in first-occurrence order, and each QA record points to its
passage by index. The full released document set stays in the corpus even when
a sampled QA record does not reference every document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from _prepare_common import (
    PreparationError,
    absolute_path,
    check_output_directory,
    print_summary,
    require_input_file,
    warn,
    write_outputs,
)


DATA_ROOT = Path("/data/panzheng/canon-thrag/datasets/UltraDomain")
OUTPUT_ROOT = Path("/data/panzheng/canon-thrag/data")
SAMPLE_SIZE = 100
DEFAULT_SEED = 42

DOMAINS: dict[str, dict[str, Any]] = {
    "agriculture": {
        "input": DATA_ROOT / "agriculture.jsonl",
        "output": OUTPUT_ROOT / "ultradomain_agriculture",
        "paper_label": "Agri",
    },
    "cs": {
        "input": DATA_ROOT / "cs.jsonl",
        "output": OUTPUT_ROOT / "ultradomain_cs",
        "paper_label": "CS",
    },
    "legal": {
        "input": DATA_ROOT / "legal.jsonl",
        "output": OUTPUT_ROOT / "ultradomain_legal",
        "paper_label": "Legal",
    },
    "mix": {
        "input": DATA_ROOT / "mix.jsonl",
        "output": OUTPUT_ROOT / "ultradomain_mix",
        "paper_label": "Mix",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read every non-empty line as one JSON object."""
    require_input_file(path)
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise PreparationError(f"{path}: blank line at line {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PreparationError(f"{path}: invalid JSON at line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise PreparationError(f"{path}: line {line_number} must be a JSON object")
                records.append(record)
    except OSError as exc:
        raise PreparationError(f"Cannot read {path}: {exc}") from exc
    if not records:
        raise PreparationError(f"{path} has no records")
    return records


def validate_record(record: dict[str, Any], source_index: int, path: Path) -> tuple[str, str]:
    """Return query and answer without changing their strings."""
    label = f"{path.name} record {source_index}"
    query = record.get("input")
    if not isinstance(query, str) or not query.strip():
        raise PreparationError(f"{label}: input must be a nonempty string")

    answers = record.get("answers")
    if not isinstance(answers, list) or len(answers) != 1:
        raise PreparationError(f"{label}: answers must be a one-element list")
    answer = answers[0]
    if not isinstance(answer, str) or not answer.strip():
        raise PreparationError(f"{label}: answers[0] must be a nonempty string")

    context = record.get("context")
    if not isinstance(context, str) or not context.strip():
        raise PreparationError(f"{label}: context must be a nonempty string")

    record_id = record.get("_id")
    if (
        isinstance(record_id, bool)
        or not isinstance(record_id, (str, int))
        or (isinstance(record_id, str) and not record_id.strip())
    ):
        raise PreparationError(f"{label}: _id must be a nonempty string or an integer")
    return query, answer


def build_documents(
    records: list[dict[str, Any]], path: Path
) -> tuple[list[str], list[str], list[int]]:
    """Deduplicate passages by exact context text, in first-occurrence order."""
    documents: list[str] = []
    hashes: list[str] = []
    record_document_indices: list[int] = []
    position_by_hash: dict[str, int] = {}
    context_ids_by_hash: dict[str, set[Any]] = defaultdict(set)
    hashes_by_context_id: dict[Any, set[str]] = defaultdict(set)
    lengths_by_hash: dict[str, set[Any]] = defaultdict(set)

    for source_index, record in enumerate(records):
        context = record["context"]
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if digest not in position_by_hash:
            position_by_hash[digest] = len(documents)
            documents.append(context)
            hashes.append(digest)
        record_document_indices.append(position_by_hash[digest])

        context_id = record.get("context_id")
        if context_id is not None:
            context_ids_by_hash[digest].add(context_id)
            hashes_by_context_id[context_id].add(digest)
        if "length" in record:
            lengths_by_hash[digest].add(record["length"])

    for digest, context_ids in context_ids_by_hash.items():
        if len(context_ids) > 1:
            warn(
                f"{path.name}: identical context text maps to multiple context_id values "
                f"{sorted(context_ids, key=str)!r}; the exact text is used as the document key"
            )
    for context_id, digests in hashes_by_context_id.items():
        if len(digests) > 1:
            warn(
                f"{path.name}: context_id {context_id!r} maps to {len(digests)} different "
                "context texts; the exact text is used as the document key"
            )
    for digest, lengths in lengths_by_hash.items():
        if len(lengths) > 1:
            warn(
                f"{path.name}: identical context text has multiple source length values "
                f"{sorted(lengths, key=str)!r}; per-record values are preserved in metadata"
            )
    return documents, hashes, record_document_indices


def build_parser(domain: str) -> argparse.ArgumentParser:
    config = DOMAINS[domain]
    parser = argparse.ArgumentParser(
        description=f"Prepare the UltraDomain {domain} subset from its released JSONL."
    )
    parser.add_argument(
        "--input", type=absolute_path, default=config["input"], help=f"{domain}.jsonl path"
    )
    parser.add_argument(
        "--output-dir", type=absolute_path, default=config["output"], help="Destination directory"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Sampling seed (default: 42)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace the three output files in an existing directory"
    )
    return parser


def prepare(domain: str, args: argparse.Namespace) -> None:
    check_output_directory(args.output_dir, args.force)
    records = read_jsonl(args.input)
    print(
        f"UltraDomain subset: {domain}\nInput: {args.input}\nSource records: {len(records)}",
        flush=True,
    )

    queries: list[str] = []
    answers: list[str] = []
    seen_ids: dict[tuple[str, Any], int] = {}
    for source_index, record in enumerate(records):
        query, answer = validate_record(record, source_index, args.input)
        record_id = record["_id"]
        identity = (type(record_id).__name__, record_id)
        if identity in seen_ids:
            raise PreparationError(
                f"{args.input.name}: duplicate _id {record_id!r} at records "
                f"{seen_ids[identity]} and {source_index}"
            )
        seen_ids[identity] = source_index
        queries.append(query)
        answers.append(answer)

    documents, hashes, record_document_indices = build_documents(records, args.input)

    if len(records) < SAMPLE_SIZE:
        raise PreparationError(
            f"{args.input.name} has {len(records)} records; {SAMPLE_SIZE} are required"
        )
    if len(records) == SAMPLE_SIZE:
        selected_indices = list(range(len(records)))
        sampling = "all released records (source has exactly 100); file order preserved"
    else:
        selected_indices = random.Random(args.seed).sample(range(len(records)), SAMPLE_SIZE)
        sampling = "without replacement; preserve random.sample order"
    selected_set = set(selected_indices)

    query_positions: dict[str, list[int]] = defaultdict(list)
    for source_index, query in enumerate(queries):
        query_positions[query].append(source_index)
    duplicate_groups = {
        query: positions for query, positions in query_positions.items() if len(positions) > 1
    }
    for query, positions in duplicate_groups.items():
        warn(
            f"{args.input.name}: duplicate input at source indices {positions}: {query!r}; "
            "selected records are flagged in metadata"
        )
    sample_duplicate_groups = {
        query: [position for position in positions if position in selected_set]
        for query, positions in duplicate_groups.items()
        if sum(position in selected_set for position in positions) > 1
    }

    qa: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for source_index in selected_indices:
        record = records[source_index]
        query = queries[source_index]
        answer = answers[source_index]
        document_index = record_document_indices[source_index]
        positions = query_positions[query]
        details: dict[str, Any] = {
            "domain": domain,
            "query": query,
            "answer": answer,
            "source_index": source_index,
            "source_record_id": record["_id"],
            "context_document_index": document_index,
            "context_sha256": hashes[document_index],
            "context_chars": len(documents[document_index]),
            "answers_raw": record["answers"],
            "sampling_seed": args.seed,
            "sample_retained": True,
            "duplicate_query_in_source": len(positions) > 1,
            "duplicate_query_source_indices": [
                position for position in positions if position != source_index
            ],
            "duplicate_query_in_sample": any(
                position in selected_set for position in positions if position != source_index
            ),
            "preparation_issues": [],
        }
        for field in ("context_id", "length", "label", "dataset", "meta"):
            if field in record:
                details[field] = record[field]
        qa.append({"query": query, "answer": answer})
        metadata.append(details)

    outputs = write_outputs(
        args.output_dir,
        documents,
        qa,
        metadata,
        force=args.force,
        expected_records=SAMPLE_SIZE,
    )

    unique_lengths: dict[str, Any] = {}
    for source_index, record in enumerate(records):
        digest = hashes[record_document_indices[source_index]]
        if digest not in unique_lengths and "length" in record:
            unique_lengths[digest] = record["length"]
    summary: dict[str, Any] = {
        "dataset": f"ultradomain_{domain}",
        "domain": domain,
        "paper_label": DOMAINS[domain]["paper_label"],
        "input": str(args.input),
        "sampling": sampling,
        "seed": args.seed,
        "sample_size": SAMPLE_SIZE,
        "source_records": len(records),
        "qa_written": len(qa),
        "context_documents_written": len(documents),
        "sample_referenced_context_documents": len(
            {record_document_indices[index] for index in selected_indices}
        ),
        "source_records_with_duplicate_context": len(records) - len(documents),
        "source_duplicate_query_groups": len(duplicate_groups),
        "source_records_in_duplicate_query_groups": sum(
            len(positions) for positions in duplicate_groups.values()
        ),
        "sample_duplicate_query_groups": len(sample_duplicate_groups),
        "sample_records_in_duplicate_query_groups": sum(
            1
            for index in selected_indices
            if any(
                position in selected_set
                for position in query_positions[queries[index]]
                if position != index
            )
        ),
        "source_unique_context_ids": len(
            {record["context_id"] for record in records if record.get("context_id") is not None}
        ),
        "source_unique_context_tokens": (
            sum(unique_lengths.values()) if len(unique_lengths) == len(documents) else None
        ),
        "source_unique_context_chars": sum(len(document) for document in documents),
        "qa_metadata_order_consistent": True,
        "outputs": outputs,
    }
    print_summary(summary)


def run_domain(domain: str) -> int:
    if domain not in DOMAINS:
        raise PreparationError(f"Unknown UltraDomain subset: {domain}")
    parser = build_parser(domain)
    args = parser.parse_args()
    try:
        prepare(domain, args)
    except (PreparationError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0
