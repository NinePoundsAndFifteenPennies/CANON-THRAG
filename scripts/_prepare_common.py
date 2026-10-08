"""Local-only utilities for the two dataset preparation entry points."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_SEED = 42
SAMPLE_SIZE = 1_000
OUTPUT_NAMES = ("contexts.txt", "qa.json", "metadata.json")


class PreparationError(ValueError):
    """An input or output condition that prevents preparing a dataset."""


def absolute_path(value: str) -> Path:
    """Require absolute paths so input/output locations never depend on cwd."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"Use an absolute path: {value!r}")
    return path


def require_input_file(path: Path) -> None:
    if not path.is_file():
        raise PreparationError(f"Input file does not exist or is not a file: {path}")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    require_input_file(path)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PreparationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(records, list):
        raise PreparationError(f"Expected a JSON list in {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PreparationError(f"{path}: record {index} must be an object")
    return records


def require_text(value: Any, label: str) -> str:
    # strip() is only a check: the original string is returned without changes.
    if not isinstance(value, str) or not value.strip():
        raise PreparationError(f"{label} must be a nonempty string")
    return value


def validate_questions(
    records: list[dict[str, Any]],
    query_field: str,
    *,
    require_ids: bool = False,
    require_unique_queries: bool = False,
    require_unique_records: bool = False,
) -> dict[str, int]:
    """Check the entire source, before selecting any rows."""
    if len(records) < SAMPLE_SIZE:
        raise PreparationError(
            f"Source has {len(records)} rows; {SAMPLE_SIZE} distinct rows are required"
        )
    seen_ids: dict[tuple[str, Any], int] = {}
    seen_queries: dict[str, int] = {}
    seen_records: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PreparationError(f"Source record {index} must be an object")
        query = require_text(record.get(query_field), f"Record {index} {query_field}")
        require_text(record.get("answer"), f"Record {index} answer")
        if require_unique_queries and query in seen_queries:
            raise PreparationError(
                f"Duplicate query at records {seen_queries[query]} and {index}: {query!r}"
            )
        seen_queries.setdefault(query, index)

        if require_ids or "id" in record:
            record_id = record.get("id")
            if (
                isinstance(record_id, bool)
                or not isinstance(record_id, (str, int))
                or (isinstance(record_id, str) and not record_id.strip())
            ):
                raise PreparationError(f"Record {index} has a missing or invalid id")
            identity = (type(record_id).__name__, record_id)
            if identity in seen_ids:
                raise PreparationError(
                    f"Duplicate id {record_id!r} at records {seen_ids[identity]} and {index}"
                )
            seen_ids[identity] = index

        if require_unique_records:
            fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if fingerprint in seen_records:
                raise PreparationError(
                    f"Duplicate source records {seen_records[fingerprint]} and {index}"
                )
            seen_records[fingerprint] = index

    return {
        "source_rows": len(records),
        "source_unique_queries": len(seen_queries),
        "source_records_with_unique_ids": len(seen_ids),
    }


def sample_indices(total: int, seed: int) -> list[int]:
    """Sample without replacement; preserve the RNG's returned order."""
    if total < SAMPLE_SIZE:
        raise PreparationError(f"Cannot sample {SAMPLE_SIZE} rows from {total}")
    return random.Random(seed).sample(range(total), SAMPLE_SIZE)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr, flush=True)


def check_output_directory(output_dir: Path, force: bool) -> None:
    if output_dir.is_symlink():
        raise PreparationError(f"Output directory must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise PreparationError(f"Output path is not a directory: {output_dir}")
        if not force:
            raise PreparationError(
                f"Output directory already exists: {output_dir}; use --force to replace "
                "contexts.txt, qa.json and metadata.json"
            )
    for name in OUTPUT_NAMES:
        target = output_dir / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise PreparationError(f"Output target must be a regular file: {target}")


def write_outputs(
    output_dir: Path,
    contexts: list[str],
    qa: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    *,
    force: bool,
) -> dict[str, str]:
    """Serialize all three files before replacing their destination files."""
    if len(qa) != SAMPLE_SIZE or len(metadata) != SAMPLE_SIZE:
        raise PreparationError(f"qa and metadata must each contain exactly {SAMPLE_SIZE} rows")
    for index, (question, details) in enumerate(zip(qa, metadata)):
        if set(question) != {"query", "answer"}:
            raise PreparationError(f"QA record {index} must contain only query and answer")
        if any(question[key] != details.get(key) for key in ("query", "answer")):
            raise PreparationError(f"QA/metadata order mismatch at record {index}")
    check_output_directory(output_dir, force)
    try:
        qa_text = json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"Output contains a value that cannot be serialized: {exc}") from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-prepare-", dir=output_dir.parent
    ) as staging_name:
        staging = Path(staging_name)
        with (staging / "contexts.txt").open("w", encoding="utf-8", newline="\n") as handle:
            for index, block in enumerate(contexts):
                if index:
                    handle.write("\n\n")
                handle.write(block)
            if contexts:
                handle.write("\n")
        for name, content in (("qa.json", qa_text), ("metadata.json", metadata_text)):
            with (staging / name).open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)

        # With no --force, exclusive mkdir also prevents replacing a directory
        # that appeared after the initial output check, even if it is empty.
        check_output_directory(output_dir, force)
        output_dir.mkdir(exist_ok=force)
        for name in OUTPUT_NAMES:
            os.replace(staging / name, output_dir / name)
    return {name: str(output_dir / name) for name in OUTPUT_NAMES}


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
