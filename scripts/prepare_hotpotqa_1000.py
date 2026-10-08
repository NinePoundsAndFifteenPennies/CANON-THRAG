"""Prepare 1,000 HotpotQA Distractor rows without calling models or downloading data.

Default seed: 42. All source rows are checked, then sampled without replacement.
Supporting-fact errors do not trigger dropping or replacement sampling: the row
is retained if sampled, with all extractable facts and explicit metadata issues.
Malformed QA/context structures and duplicate IDs stop before output is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _prepare_common import (
    DEFAULT_SEED,
    SAMPLE_SIZE,
    PreparationError,
    absolute_path,
    check_output_directory,
    print_summary,
    require_input_file,
    require_text,
    sample_indices,
    validate_questions,
    warn,
    write_outputs,
)


DEFAULT_INPUT = Path(
    "/data/panzheng/canon-thrag/datasets/distractor/validation-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = Path("/data/panzheng/canon-thrag/data/hotpotqa_1000")
INPUT_COLUMNS = ("id", "question", "answer", "type", "level", "supporting_facts", "context")


def read_hotpotqa(path: Path) -> list[dict[str, Any]]:
    require_input_file(path)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise PreparationError(
            "HotpotQA Parquet reading requires pyarrow in the server environment; "
            "this script does not install dependencies"
        ) from exc
    try:
        return parquet.read_table(path, columns=list(INPUT_COLUMNS)).to_pylist()
    except Exception as exc:
        raise PreparationError(f"Cannot read HotpotQA Parquet {path}: {exc}") from exc


def validate_context(row: dict[str, Any], source_index: int) -> None:
    label = f"HotpotQA record {source_index}, id={row['id']!r}"
    context = row.get("context")
    if not isinstance(context, dict):
        raise PreparationError(f"{label}: context must be an object")
    titles = context.get("title")
    sentence_groups = context.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentence_groups, list):
        raise PreparationError(f"{label}: context.title and context.sentences must be lists")
    if len(titles) != len(sentence_groups):
        raise PreparationError(
            f"{label}: context has {len(titles)} titles but {len(sentence_groups)} sentence groups"
        )
    if not titles:
        raise PreparationError(f"{label}: context has no candidate paragraphs")
    for paragraph_index, (title, sentences) in enumerate(zip(titles, sentence_groups)):
        require_text(title, f"{label} paragraph {paragraph_index} title")
        if not isinstance(sentences, list):
            raise PreparationError(f"{label}: paragraph {paragraph_index} sentences must be a list")
        for sent_id, sentence in enumerate(sentences):
            if not isinstance(sentence, str):
                raise PreparationError(
                    f"{label}: paragraph {paragraph_index} sentence {sent_id} must be a string"
                )


def extract_supporting_facts(
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = row["context"]
    title_positions: dict[str, list[int]] = {}
    for paragraph_index, title in enumerate(context["title"]):
        title_positions.setdefault(title, []).append(paragraph_index)

    raw_facts = row.get("supporting_facts")
    facts: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    if not isinstance(raw_facts, dict):
        return facts, [{"reason": "supporting_facts is not an object"}]
    titles = raw_facts.get("title")
    sent_ids = raw_facts.get("sent_id")
    if not isinstance(titles, list) or not isinstance(sent_ids, list):
        return facts, [{"reason": "supporting_facts.title and sent_id must be lists"}]
    if len(titles) != len(sent_ids):
        issues.append({
            "reason": "supporting_facts title/sent_id length mismatch",
            "title_count": len(titles),
            "sent_id_count": len(sent_ids),
        })
    if not titles and not sent_ids:
        issues.append({"reason": "supporting_facts is empty"})

    for fact_index in range(max(len(titles), len(sent_ids))):
        title = titles[fact_index] if fact_index < len(titles) else None
        sent_id = sent_ids[fact_index] if fact_index < len(sent_ids) else None
        reason = None
        if not isinstance(title, str) or not title.strip():
            reason = "supporting title is missing or invalid"
        elif title not in title_positions:
            reason = "supporting title is absent from context"
        elif len(title_positions[title]) != 1:
            reason = "supporting title matches multiple context paragraphs"
        elif isinstance(sent_id, bool) or not isinstance(sent_id, int):
            reason = "sent_id is missing or is not an integer"
        else:
            sentences = context["sentences"][title_positions[title][0]]
            if sent_id < 0 or sent_id >= len(sentences):
                reason = "sent_id is outside the paragraph's zero-based sentence range"
            else:
                facts.append({"title": title, "sent_id": sent_id, "text": sentences[sent_id]})
        if reason is not None:
            issues.append({
                "fact_index": fact_index,
                "title": title,
                "sent_id": sent_id,
                "reason": reason,
            })
    return facts, issues


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=absolute_path, default=DEFAULT_INPUT, help="Source Parquet path")
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
    rows = read_hotpotqa(args.input)
    print(f"HotpotQA input: {args.input}\nSource rows: {len(rows)}", flush=True)
    summary: dict[str, Any] = validate_questions(rows, "question", require_ids=True)

    extracted: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for source_index, row in enumerate(rows):
        validate_context(row, source_index)
        facts, issues = extract_supporting_facts(row)
        extracted.append((facts, issues))
        if issues:
            warn(
                f"HotpotQA source_index={source_index}, id={row['id']!r}: "
                f"{json.dumps(issues, ensure_ascii=False)}; retained if sampled, without replacement"
            )

    selected_indices = sample_indices(len(rows), args.seed)
    qa: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    contexts: list[str] = []
    for source_index in selected_indices:
        row = rows[source_index]
        facts, issues = extracted[source_index]
        qa.append({"query": row["question"], "answer": row["answer"]})
        metadata.append({
            "id": row["id"],
            "query": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "source_index": source_index,
            "sampling_seed": args.seed,
            "sample_retained": True,
            "supporting_facts": facts,
            "supporting_facts_raw": row["supporting_facts"],
            "supporting_facts_complete": not issues,
            "supporting_fact_issues": issues,
        })
        # Do not deduplicate HotpotQA paragraphs: every sampled question keeps
        # all its distractor candidates, in their original paragraph order.
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            contexts.append("\n".join([title, *sentences]))

    outputs = write_outputs(args.output_dir, contexts, qa, metadata, force=args.force)
    summary.update({
        "dataset": "hotpotqa_distractor",
        "input": str(args.input),
        "seed": args.seed,
        "sample_size": SAMPLE_SIZE,
        "sampling": "without replacement; preserve random.sample order; no replacement of error rows",
        "source_rows_with_supporting_fact_issues": sum(bool(issues) for _, issues in extracted),
        "source_supporting_fact_issues": sum(len(issues) for _, issues in extracted),
        "sample_rows_with_supporting_fact_issues": sum(
            bool(row["supporting_fact_issues"]) for row in metadata
        ),
        "sample_supporting_facts_extracted": sum(len(row["supporting_facts"]) for row in metadata),
        "context_paragraphs_written": len(contexts),
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
