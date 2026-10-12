"""Repair only failed extraction chunks using their original saved KV contents."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from config import THRAGConfig, get_config
from model_services import create_chat_client
from index.graph_construction import (
    ExtractionIncompleteError,
    call_model,
    get_failure_file,
    load_existing_blocks,
    load_failures,
    load_kv_store,
    invalidate_downstream_steps,
    make_failure,
    mark_extraction_state,
    save_blocks,
    save_failures,
    validate_snapshot,
)


def repair_graph(config: THRAGConfig) -> str:
    """Perform one repair pass; preserve unresolved records and successful chunks."""

    output_path = config.get_graph_json_file()
    failures = load_failures(config)
    if not failures:
        return str(output_path)

    # Invalidate a previously completed stage before inspecting any repair input.
    mark_extraction_state(config, completed=False, chunks=0)
    kv_store = load_kv_store(config)
    chunk_ids = list(kv_store)
    positions = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    blocks = load_existing_blocks(output_path)
    if set(blocks) - set(kv_store):
        raise ValueError("Graph JSON contains chunks absent from the original KV store")

    # Only invalid blocks explicitly listed as failed may be replaced. All other
    # successful blocks, including those with stale failure records, stay intact.
    for chunk_id, block in list(blocks.items()):
        if chunk_id in failures:
            try:
                validate_snapshot({chunk_id: block}, kv_store)
            except ValueError:
                del blocks[chunk_id]
                continue
        validate_snapshot({chunk_id: block}, kv_store)

    def checkpoint() -> None:
        validate_snapshot(blocks, kv_store)
        save_blocks(output_path, [blocks[chunk_id] for chunk_id in chunk_ids if chunk_id in blocks])
        if not failures and len(blocks) != len(kv_store):
            # Keep the on-disk log if unlogged missing chunks prevent completion.
            raise ExtractionIncompleteError(
                f"{config.dataset_name}: graph JSON still has "
                f"{len(kv_store) - len(blocks)} missing chunks without failure records"
            )
        # Never clear a record before its successful block is on disk.
        save_failures(config, failures)

    def record_failure(chunk_id: str, record: dict[str, Any], exc: Exception) -> None:
        content = kv_store.get(chunk_id, {}).get("content", "")
        diagnostic = make_failure(config, chunk_id, positions.get(chunk_id, -1), content, exc)
        updated = dict(record)
        # Preserve original identity fields even if locating the chunk failed.
        # Replacing its hash or index could silently accept the wrong original.
        for field in ("error_type", "reason", "finish_reason", "response_excerpt"):
            updated.pop(field, None)
            if field in diagnostic:
                updated[field] = diagnostic[field]
        failures[chunk_id] = updated

    def locate_chunk(chunk_id: str, record: dict[str, Any]) -> tuple[int, str]:
        if record.get("dataset") != config.dataset_name or record.get("chunk_id") != chunk_id:
            raise ValueError(f"{chunk_id}: failure record has a different dataset or chunk identity")
        if chunk_id not in positions:
            raise ValueError(f"{chunk_id}: original chunk is absent from the saved KV store")
        index = positions[chunk_id]
        recorded_index = record.get("chunk_index")
        if type(recorded_index) is not int or recorded_index != index:
            raise ValueError(f"{chunk_id}: failure record chunk_index does not match the original KV order")
        recorded_path = record.get("kv_store_file")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ValueError(f"{chunk_id}: failure record has no original KV store path")
        if Path(recorded_path).expanduser().resolve() != config.get_kv_store_file().resolve():
            raise ValueError(f"{chunk_id}: failure record refers to a different original KV store")
        content = kv_store[chunk_id]["content"]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if record.get("content_sha256") != digest:
            raise ValueError(f"{chunk_id}: saved chunk content no longer matches its original SHA-256")
        return index, content

    pending: dict[str, tuple[int, str, dict[str, Any]]] = {}
    completed_count = 0
    for chunk_id, record in list(failures.items()):
        try:
            index, content = locate_chunk(chunk_id, record)
            if chunk_id in blocks:
                del failures[chunk_id]
            else:
                pending[chunk_id] = (index, content, record)
                continue
        except Exception as exc:
            record_failure(chunk_id, record, exc)
        completed_count += 1
        if completed_count % 10 == 0:
            checkpoint()

    if pending:
        # Preserve invalidation across a manual repair followed by a later pipeline.
        invalidate_downstream_steps(config)
        try:
            client = create_chat_client(config)
        except Exception as exc:
            for chunk_id, (_, _, record) in pending.items():
                record_failure(chunk_id, record, exc)
        else:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(call_model, client, config.default_model, content, chunk_id, config): chunk_id
                    for chunk_id, (_, content, _) in pending.items()
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing triples"):
                    chunk_id = futures[future]
                    _, _, record = pending[chunk_id]
                    try:
                        block = future.result()
                        validate_snapshot({chunk_id: block}, kv_store)
                        blocks[chunk_id] = block
                        del failures[chunk_id]
                    except Exception as exc:
                        record_failure(chunk_id, record, exc)
                    completed_count += 1
                    if completed_count % 10 == 0:
                        checkpoint()

    checkpoint()
    mark_extraction_state(config, completed=not failures, chunks=len(kv_store))
    if failures:
        raise ExtractionIncompleteError(
            f"{config.dataset_name}: {len(failures)} chunks still need repair; "
            f"see {get_failure_file(config)}"
        )
    return str(output_path)


def main(dataset_name: str) -> str:
    config = get_config(dataset_name)
    print(f"Repairing failed graph extraction for dataset: {dataset_name}")
    try:
        return repair_graph(config)
    except Exception as exc:
        # Use the same redaction as persisted records before printing errors.
        reason = make_failure(config, "", -1, "", exc)["reason"]
        print(f"Graph repair failed: {reason}", file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair one dataset's failed graph-extraction chunks.")
    parser.add_argument("--dataset", required=True, help="Dataset name under the configured data directory")
    args = parser.parse_args()
    try:
        main(args.dataset)
    except Exception:
        raise SystemExit(1) from None
