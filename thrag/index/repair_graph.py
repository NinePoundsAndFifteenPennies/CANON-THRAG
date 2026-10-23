"""Repair failed extraction chunks and blank-sentence chunks."""

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
    save_kv_store,
    validate_snapshot,
)

# Blank-sentence regeneration uses a lower cap so the model keeps enough budget
# for each supporting sentence instead of running into the response limit.
BLANK_SENTENCE_MAX_TRIPLES = 70


def quote_safe(text: str) -> str:
    """Replace ASCII double quotes that break JSON string copying."""

    return text.replace('"', "'")


def has_blank_sentences(block: dict[str, Any]) -> bool:
    """Return True when any triple has an empty support sentence."""

    return any(
        not str(item.get("sentence", "")).strip()
        for item in block.get("triples", [])
        if isinstance(item, dict)
    )


def sanitize_failed_chunk_quotes(
    config: THRAGConfig,
    failures: dict[str, dict[str, Any]],
    blocks: dict[str, dict[str, Any]],
    kv_store: dict[str, dict[str, str]],
) -> None:
    """Rewrite double quotes for failed chunks so the model can copy sentences.

    Only chunks without a saved block are touched, so existing blocks keep a
    byte-identical match with the KV store.
    """

    sanitized: list[str] = []
    for chunk_id, record in failures.items():
        if chunk_id in blocks:
            continue
        entry = kv_store.get(chunk_id)
        if entry is None:
            continue
        content = entry["content"]
        replacement = quote_safe(content)
        if replacement == content:
            continue
        entry["content"] = replacement
        record["content_sha256"] = hashlib.sha256(replacement.encode("utf-8")).hexdigest()
        sanitized.append(chunk_id)
    if sanitized:
        print(
            f"Sanitized double quotes in {len(sanitized)} failed chunks before repair: "
            f"{', '.join(sanitized)}",
            flush=True,
        )
        save_kv_store(config.get_kv_store_file(), kv_store)
        save_failures(config, failures)


def repair_blank_sentences(
    config: THRAGConfig,
    blocks: dict[str, dict[str, Any]],
    kv_store: dict[str, dict[str, str]],
    chunk_ids: list[str],
    output_path: Path,
) -> list[str]:
    """Regenerate chunks whose extracted triples have no support sentence.

    This pass never touches the failure log and never blocks later stages: a
    chunk that still fails validation keeps its previous block.
    """

    targets = [
        chunk_id for chunk_id in chunk_ids
        if chunk_id in blocks and has_blank_sentences(blocks[chunk_id])
    ]
    if not targets:
        return []

    before = {chunk_id: sum(
        1 for item in blocks[chunk_id]["triples"]
        if isinstance(item, dict) and not str(item.get("sentence", "")).strip()
    ) for chunk_id in targets}
    print(
        f"{len(targets)} chunks contain blank sentence fields; regenerating with at most "
        f"{BLANK_SENTENCE_MAX_TRIPLES} triples per chunk.",
        flush=True,
    )
    invalidate_downstream_steps(config)
    try:
        client = create_chat_client(config)
    except Exception as exc:
        print(f"Blank-sentence repair skipped: {type(exc).__name__}: {exc}", flush=True)
        return targets

    residual: list[str] = []
    changed = 0
    try:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(
                    call_model, client, config.default_model, quote_safe(kv_store[chunk_id]["content"]),
                    chunk_id, config, BLANK_SENTENCE_MAX_TRIPLES,
                ): chunk_id
                for chunk_id in targets
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing blank sentences"):
                chunk_id = futures[future]
                try:
                    block = future.result()
                    # The stored content must stay identical to the KV store.
                    block["content"] = kv_store[chunk_id]["content"]
                    validate_snapshot({chunk_id: block}, kv_store)
                except Exception as exc:
                    print(
                        f"{chunk_id}: blank-sentence repair failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    residual.append(chunk_id)
                    continue
                previous = blocks[chunk_id]
                blocks[chunk_id] = block
                changed += 1
                # The graph file is large; checkpoint periodically instead of per chunk.
                if changed % 10 == 0:
                    save_blocks(output_path, [blocks[key] for key in chunk_ids if key in blocks])
                remaining = sum(
                    1 for item in block["triples"] if not str(item.get("sentence", "")).strip()
                )
                print(
                    f"{chunk_id}: blank sentences {before[chunk_id]} -> {remaining}, "
                    f"triples {len(previous['triples'])} -> {len(block['triples'])}",
                    flush=True,
                )
                if remaining:
                    residual.append(chunk_id)
    finally:
        client.close()
        if changed and changed % 10 != 0:
            save_blocks(output_path, [blocks[key] for key in chunk_ids if key in blocks])
    return residual



def repair_graph(config: THRAGConfig) -> str:
    """Repair failed chunks, then regenerate chunks with blank sentences."""

    output_path = config.get_graph_json_file()
    failures = load_failures(config)
    kv_store = load_kv_store(config)
    chunk_ids = list(kv_store)
    blocks = load_existing_blocks(output_path)
    if set(blocks) - set(kv_store):
        raise ValueError("Graph JSON contains chunks absent from the original KV store")

    blank_targets = [
        chunk_id for chunk_id in chunk_ids
        if chunk_id in blocks and has_blank_sentences(blocks[chunk_id])
    ]
    if not failures and not blank_targets:
        print(
            f"{config.dataset_name}: no failed chunks to repair and no blank sentences in "
            f"{len(blocks)}/{len(chunk_ids)} saved blocks; nothing to do.",
            flush=True,
        )
        return str(output_path)

    # Failure bookkeeping stays independent of the blank-sentence pass.
    if failures:
        # Invalidate a previously completed stage before inspecting any repair input.
        mark_extraction_state(config, completed=False, chunks=0)
        positions = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}

        # Copying a source sentence that contains ASCII double quotes makes the
        # model degenerate, so rewrite them for failed chunks before retrying.
        sanitize_failed_chunk_quotes(config, failures, blocks, kv_store)

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

    # Independent pass: never writes the failure log and never blocks later stages.
    repair_blank_sentences(config, blocks, kv_store, chunk_ids, output_path)

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
