"""Prepare 1,000 MultiHop-RAG queries using local JSON files only.

Default seed: 42. Sample from all source queries without replacement. Queries
with empty or unresolved evidence are retained, reported, and marked in metadata;
there is no resampling. Match exact titles; use exact URLs only to disambiguate
duplicate titles. Deduplicate usable documents by corpus row, in first-use order.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _prepare_common import (
    DEFAULT_SEED,
    SAMPLE_SIZE,
    PreparationError,
    absolute_path,
    check_output_directory,
    print_summary,
    read_json_records,
    require_text,
    sample_indices,
    validate_questions,
    warn,
    write_outputs,
)


DEFAULT_QUERIES = Path("/data/panzheng/canon-thrag/datasets/MultiHop/MultiHopRAG.json")
DEFAULT_CORPUS = Path("/data/panzheng/canon-thrag/datasets/MultiHop/corpus.json")
DEFAULT_OUTPUT = Path("/data/panzheng/canon-thrag/data/multihop_1000")
EVIDENCE_FIELDS = ("title", "source", "url", "author", "published_at", "category", "fact")


def index_corpus(corpus: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, int]]:
    if not corpus:
        raise PreparationError("corpus.json has no documents")
    by_title: dict[str, list[int]] = {}
    unusable_bodies = 0
    for index, document in enumerate(corpus):
        title = require_text(document.get("title"), f"Corpus record {index} title")
        by_title.setdefault(title, []).append(index)
        body = document.get("body")
        if not isinstance(body, str) or not body.strip():
            unusable_bodies += 1
            warn(f"Corpus record {index}, title={title!r}: missing, empty or non-string body")
    duplicate_titles = {title: indices for title, indices in by_title.items() if len(indices) > 1}
    for title, indices in duplicate_titles.items():
        warn(
            f"Duplicate corpus title {title!r} at indices {indices}; "
            "only a unique exact URL within these candidates can disambiguate it"
        )
    return by_title, {
        "corpus_documents": len(corpus),
        "corpus_unique_titles": len(by_title),
        "corpus_duplicate_titles": len(duplicate_titles),
        "corpus_documents_with_unusable_body": unusable_bodies,
    }


def match_evidence(
    evidence: dict[str, Any],
    corpus: list[dict[str, Any]],
    by_title: dict[str, list[int]],
) -> dict[str, Any]:
    # Copy every original field, including any fields outside the documented schema.
    result = dict(evidence)
    for field in EVIDENCE_FIELDS:
        result.setdefault(field, None)
    title = evidence.get("title")
    candidates: list[int] = []
    matched_index = None
    method = None
    status = "invalid_evidence_title"
    if isinstance(title, str) and title.strip():
        candidates = by_title.get(title, [])
        if not candidates:
            status = "not_found"
        elif len(candidates) == 1:
            matched_index = candidates[0]
            method = "exact_title"
        else:
            url = evidence.get("url")
            url_matches = [
                index for index in candidates
                if isinstance(url, str) and url.strip() and corpus[index].get("url") == url
            ]
            if len(url_matches) == 1:
                matched_index = url_matches[0]
                method = "exact_title_and_url"
            else:
                status = "ambiguous_title"

    body_nonempty = False
    if matched_index is not None:
        body = corpus[matched_index].get("body")
        body_nonempty = isinstance(body, str) and bool(body.strip())
        status = "matched" if body_nonempty else "unusable_body"
    result.update({
        "corpus_found": matched_index is not None,
        "corpus_record_index": matched_index,
        "corpus_candidate_indices": list(candidates),
        "corpus_match_method": method,
        "corpus_match_status": status,
        "corpus_body_nonempty": body_nonempty,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", type=absolute_path, default=DEFAULT_QUERIES, help="MultiHopRAG.json path"
    )
    parser.add_argument("--corpus", type=absolute_path, default=DEFAULT_CORPUS, help="corpus.json path")
    parser.add_argument(
        "--output-dir", type=absolute_path, default=DEFAULT_OUTPUT, help="Destination directory"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed (default: 42)")
    parser.add_argument(
        "--force", action="store_true", help="Replace the three output files in an existing directory"
    )
    return parser


def prepare(args: argparse.Namespace) -> None:
    check_output_directory(args.output_dir, args.force)
    rows = read_json_records(args.queries)
    corpus = read_json_records(args.corpus)
    print(
        f"MultiHop queries: {args.queries}\nSource rows: {len(rows)}\n"
        f"Corpus: {args.corpus}\nCorpus documents: {len(corpus)}",
        flush=True,
    )
    summary: dict[str, Any] = validate_questions(
        rows, "query", require_unique_queries=True, require_unique_records=True
    )
    by_title, corpus_summary = index_corpus(corpus)
    summary.update(corpus_summary)

    matches_by_row: list[list[dict[str, Any]]] = []
    issues_by_row: list[list[dict[str, Any]]] = []
    source_match_counts: Counter[str] = Counter()
    for source_index, row in enumerate(rows):
        if "question_type" not in row:
            raise PreparationError(f"Query record {source_index} is missing question_type")
        evidence_list = row.get("evidence_list")
        if not isinstance(evidence_list, list):
            raise PreparationError(f"Query record {source_index}: evidence_list must be a list")
        matches: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        if not evidence_list:
            issues.append({"reason": "empty_evidence_list"})
        for evidence_index, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                raise PreparationError(
                    f"Query record {source_index}, evidence {evidence_index} must be an object"
                )
            matched = match_evidence(evidence, corpus, by_title)
            matches.append(matched)
            status = matched["corpus_match_status"]
            source_match_counts[status] += 1
            if status != "matched":
                issues.append({
                    "evidence_index": evidence_index,
                    "title": evidence.get("title"),
                    "reason": status,
                    "corpus_candidate_indices": matched["corpus_candidate_indices"],
                })
        matches_by_row.append(matches)
        issues_by_row.append(issues)
        if issues:
            warn(
                f"MultiHop source_index={source_index}, query={row['query']!r}: "
                f"{json.dumps(issues, ensure_ascii=False)}; retained if sampled, without replacement"
            )

    selected_indices = sample_indices(len(rows), args.seed)
    qa: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    contexts: list[str] = []
    document_positions: dict[int, int] = {}
    sample_match_counts: Counter[str] = Counter()
    usable_document_references = 0
    for source_index in selected_indices:
        row = rows[source_index]
        evidence_output: list[dict[str, Any]] = []
        for matched in matches_by_row[source_index]:
            result = dict(matched)
            sample_match_counts[result["corpus_match_status"]] += 1
            corpus_index = result["corpus_record_index"]
            result["context_document_index"] = None
            if result["corpus_found"] and result["corpus_body_nonempty"]:
                usable_document_references += 1
                if corpus_index not in document_positions:
                    document = corpus[corpus_index]
                    document_positions[corpus_index] = len(contexts)
                    contexts.append(document["title"] + "\n" + document["body"])
                result["context_document_index"] = document_positions[corpus_index]
            evidence_output.append(result)

        qa.append({"query": row["query"], "answer": row["answer"]})
        details = {
            "query": row["query"],
            "answer": row["answer"],
            "question_type": row["question_type"],
            "source_index": source_index,
            "sampling_seed": args.seed,
            "sample_retained": True,
            "evidence": evidence_output,
            # Keep an unmodified copy even if a future source schema uses the
            # same field names as our added matching-result fields.
            "evidence_list_raw": row["evidence_list"],
            "evidence_complete": not issues_by_row[source_index],
            "preparation_issues": issues_by_row[source_index],
        }
        if "id" in row:
            details["id"] = row["id"]
        metadata.append(details)

    outputs = write_outputs(args.output_dir, contexts, qa, metadata, force=args.force)
    summary.update({
        "dataset": "multihop_rag",
        "queries_input": str(args.queries),
        "corpus_input": str(args.corpus),
        "seed": args.seed,
        "sample_size": SAMPLE_SIZE,
        "sampling": "without replacement; preserve random.sample order; no replacement of error rows",
        "source_queries_with_evidence_issues": sum(bool(issues) for issues in issues_by_row),
        "source_queries_with_empty_evidence": sum(not row["evidence_list"] for row in rows),
        "source_evidence_match_counts": dict(source_match_counts),
        "sample_queries_with_evidence_issues": sum(bool(row["preparation_issues"]) for row in metadata),
        "sample_evidence_match_counts": dict(sample_match_counts),
        "sample_usable_document_references": usable_document_references,
        "context_documents_written": len(contexts),
        "duplicate_document_references_removed": usable_document_references - len(contexts),
        "context_document_order": "first appearance in sampled query/evidence order; unique corpus row",
        "qa_metadata_order_consistent": True,
        "outputs": outputs,
    })
    print_summary(summary)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        prepare(args)
    except (PreparationError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
