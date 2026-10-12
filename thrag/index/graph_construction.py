"""Graph construction for TH-RAG.

This step chunks a dataset's contexts.txt file, stores the chunk text in a KV store,
and extracts topic-aware triples for each chunk with Qwen served by vLLM.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tempfile import NamedTemporaryFile
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from config import THRAGConfig, get_config
from model_services import create_chat_client, get_chat_tokenizer
from prompt.extract_graph import EXTRACTION_PROMPT, MAX_TRIPLES, SYSTEM_PROMPT, TRIPLES_SCHEMA


def chunk_text(text: str, max_tokens: int, overlap: int, tokenizer_path: str) -> list[str]:
    """Split text into overlapping token windows."""

    tokenizer = get_chat_tokenizer(tokenizer_path)
    tokens = tokenizer.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = max(0, end - overlap)

    return chunks


class ExtractionIncompleteError(RuntimeError):
    """Successful chunks are saved, but this dataset still needs repair."""


class ExtractionResponseError(ValueError):
    def __init__(
        self,
        reason: str,
        error_type: str,
        finish_reason: str | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.error_type = error_type
        self.finish_reason = finish_reason
        self.response_text = response_text


def validate_triples_payload(payload: Any) -> list[dict[str, Any]]:
    """Validate every field of the extraction schema without dropping entries."""

    def require_keys(value: Any, keys: set[str], location: str) -> None:
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"{location}: expected exactly the fields {sorted(keys)}")

    def require_text(value: Any, location: str) -> None:
        if not isinstance(value, str) or len(value) < 1:
            raise ValueError(f"{location}: expected a nonempty string")

    require_keys(payload, {"triples"}, "response")
    triples = payload["triples"]
    if not isinstance(triples, list):
        raise ValueError("triples: expected an array")
    if len(triples) > MAX_TRIPLES:
        raise ValueError(f"triples: got {len(triples)} items; maximum is {MAX_TRIPLES}")
    for index, item in enumerate(triples):
        location = f"triples[{index}]"
        require_keys(item, {"triple", "sentence", "subject", "object"}, location)
        triple = item["triple"]
        if not isinstance(triple, list) or len(triple) != 3:
            raise ValueError(f"{location}.triple: expected exactly three strings")
        for part_index, part in enumerate(triple):
            require_text(part, f"{location}.triple[{part_index}]")
        require_text(item["sentence"], f"{location}.sentence")
        for role in ("subject", "object"):
            category = item[role]
            require_keys(category, {"subtopic", "main_topic"}, f"{location}.{role}")
            for key in ("subtopic", "main_topic"):
                require_text(category[key], f"{location}.{role}.{key}")
    return triples


def parse_triples_response(response_text: str) -> list[dict[str, Any]]:
    """Accept only the complete JSON object required by the request schema."""

    return validate_triples_payload(json.loads(response_text))


def validate_block(block: Any) -> None:
    if not isinstance(block, dict) or set(block) != {"chunk_id", "content", "triples"}:
        raise ValueError("Graph block must contain exactly chunk_id, content and triples")
    if not isinstance(block["chunk_id"], str) or not block["chunk_id"]:
        raise ValueError("Graph block chunk_id must be a nonempty string")
    if not isinstance(block["content"], str):
        raise ValueError(f"{block['chunk_id']}: content must be a string")
    validate_triples_payload({"triples": block["triples"]})



def build_kv_store(chunks: list[str]) -> dict[str, dict[str, str]]:
    """Create the chunk lookup structure used during answer generation."""

    return {
        f"chunk-{index:05d}": {"content": chunk}
        for index, chunk in enumerate(chunks)
    }



def load_existing_blocks(output_path: Path) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Graph JSON must be a list of chunk blocks: {output_path}")
    existing: dict[str, dict[str, Any]] = {}
    for block in payload:
        if not isinstance(block, dict) or not isinstance(block.get("chunk_id"), str) or not block["chunk_id"]:
            raise ValueError(f"Graph JSON contains a block without a valid chunk_id: {output_path}")
        chunk_id = block["chunk_id"]
        if chunk_id in existing:
            raise ValueError(f"Graph JSON contains duplicate chunk_id {chunk_id}: {output_path}")
        existing[chunk_id] = block
    return existing


def load_kv_store(config: THRAGConfig) -> dict[str, dict[str, str]]:
    path = config.get_kv_store_file()
    with path.open("r", encoding="utf-8") as handle:
        kv_store = json.load(handle)
    if not isinstance(kv_store, dict):
        raise ValueError(f"KV store must be an object: {path}")
    for chunk_id, value in kv_store.items():
        if (
            not chunk_id
            or not isinstance(value, dict)
            or set(value) != {"content"}
            or not isinstance(value["content"], str)
        ):
            raise ValueError(f"Invalid KV entry for {chunk_id!r}: {path}")
    return kv_store


def validate_snapshot(blocks: dict[str, dict[str, Any]], kv_store: dict[str, dict[str, str]]) -> None:
    for chunk_id, block in blocks.items():
        validate_block(block)
        if block["chunk_id"] != chunk_id or chunk_id not in kv_store:
            raise ValueError(f"Graph chunk {chunk_id} does not match the original KV store")
        if block["content"] != kv_store[chunk_id]["content"]:
            raise ValueError(f"Graph chunk {chunk_id} content differs from the original KV store")


def graph_checkpoint_is_complete(config: THRAGConfig) -> bool:
    """Check extraction completeness from KV and graph artifacts only."""

    if not config.get_kv_store_file().exists() or not config.get_graph_json_file().exists():
        return False
    kv_store = load_kv_store(config)
    blocks = load_existing_blocks(config.get_graph_json_file())
    if set(blocks) != set(kv_store):
        return False
    try:
        validate_snapshot(blocks, kv_store)
    except ValueError:
        return False
    return True


def call_model(
    client: OpenAI,
    model_name: str,
    chunk_text_value: str,
    chunk_id: str,
    config: THRAGConfig | None = None,
) -> dict[str, Any]:
    prompt = EXTRACTION_PROMPT.replace("{{document}}", chunk_text_value.strip())
    config = config or get_config()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=config.max_tokens_response,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "chunk_triples", "strict": True, "schema": TRIPLES_SCHEMA},
        },
        extra_body=config.llm_extra_body(),
    )
    if not response.choices:
        raise ExtractionResponseError(f"{chunk_id}: API response contained no choices", "MissingChoices")
    choice = response.choices[0]
    finish_reason = choice.finish_reason
    content = choice.message.content
    if finish_reason == "length":
        raise ExtractionResponseError(
            f"{chunk_id}: triple extraction response was truncated at "
            f"MAX_TOKENS_RESPONSE={config.max_tokens_response}",
            "TruncatedResponse", finish_reason, content,
        )
    if finish_reason not in {None, "stop"}:
        raise ExtractionResponseError(
            f"{chunk_id}: triple extraction stopped with finish_reason={finish_reason!r}",
            "UnexpectedFinishReason", finish_reason, content,
        )
    if not content or not content.strip():
        raise ExtractionResponseError(
            f"{chunk_id}: triple extraction returned empty content", "EmptyResponse", finish_reason, content,
        )
    try:
        triples = parse_triples_response(content)
    except (json.JSONDecodeError, ValueError) as exc:
        error_type = "JSONDecodeError" if isinstance(exc, json.JSONDecodeError) else "SchemaValidationError"
        raise ExtractionResponseError(str(exc), error_type, finish_reason, content) from exc
    return {
        "chunk_id": chunk_id,
        "content": chunk_text_value,
        "triples": triples,
    }



def _write_json_atomic(output_path: Path, payload: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_blocks(output_path: Path, blocks: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for block in blocks:
        validate_block(block)
        chunk_id = block["chunk_id"]
        if chunk_id in seen:
            raise ValueError(f"Duplicate graph chunk_id: {chunk_id}")
        seen.add(chunk_id)
    _write_json_atomic(output_path, blocks)



def save_kv_store(output_path: Path, kv_store: dict[str, dict[str, str]]) -> None:
    _write_json_atomic(output_path, kv_store)


def get_failure_file(config: THRAGConfig) -> Path:
    return config.results_dir / "fail" / f"{config.dataset_name}.json"


def load_failures(config: THRAGConfig) -> dict[str, dict[str, Any]]:
    path = get_failure_file(config)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Failure log must be an array: {path}")
    failures: dict[str, dict[str, Any]] = {}
    for record in payload:
        if (
            not isinstance(record, dict)
            or record.get("dataset") != config.dataset_name
            or not isinstance(record.get("chunk_id"), str)
            or not record["chunk_id"]
        ):
            raise ValueError(f"Invalid dataset or chunk identity in failure log: {path}")
        chunk_id = record["chunk_id"]
        if chunk_id in failures:
            raise ValueError(f"Duplicate failed chunk_id {chunk_id}: {path}")
        failures[chunk_id] = record
    return failures


def has_pending_failures(config: THRAGConfig) -> bool:
    return bool(load_failures(config))


def save_failures(config: THRAGConfig, failures: dict[str, dict[str, Any]]) -> None:
    path = get_failure_file(config)
    if failures:
        def record_order(record: dict[str, Any]) -> tuple[int, int | str]:
            index = record.get("chunk_index")
            if type(index) is int and index >= 0:
                return (0, index)
            return (1, record["chunk_id"])

        records = sorted(failures.values(), key=record_order)
        _write_json_atomic(path, records)
    else:
        path.unlink(missing_ok=True)


def make_failure(
    config: THRAGConfig, chunk_id: str, index: int, content: str, exc: Exception,
) -> dict[str, Any]:
    def redact(text: str) -> str:
        key = config.llm_api_key
        return text.replace(key, "[REDACTED]") if key else text

    record: dict[str, Any] = {
        "dataset": config.dataset_name,
        "chunk_id": chunk_id,
        "chunk_index": index,
        "source_file": str(config.get_contexts_file().resolve()),
        "kv_store_file": str(config.get_kv_store_file().resolve()),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "error_type": getattr(exc, "error_type", type(exc).__name__),
        "reason": redact(str(exc)),
    }
    if isinstance(exc, ExtractionResponseError):
        record["finish_reason"] = exc.finish_reason
        if exc.response_text:
            # Diagnostic excerpts only; parsing always uses the complete response.
            response_text = redact(exc.response_text)
            record["response_excerpt"] = (
                response_text if len(response_text) <= 1200
                else response_text[:800] + "\n... [excerpt] ...\n" + response_text[-400:]
            )
    return record


def mark_extraction_state(config: THRAGConfig, completed: bool, chunks: int) -> None:
    metadata = {
        "input_file": str(config.get_contexts_file()),
        "output_file": str(config.get_graph_json_file()),
        "kv_store_file": str(config.get_kv_store_file()),
        "chunks": chunks,
    }
    if completed:
        config.mark_step_completed("graph_construction", **metadata)
    else:
        state = config.load_pipeline_state() or {}
        state.setdefault(config.dataset_name, {})["graph_construction"] = {"completed": False, **metadata}
        config.save_pipeline_state(state)


def invalidate_downstream_steps(config: THRAGConfig) -> None:
    """Ensure derived artifacts are rebuilt after the graph JSON changes."""

    state = config.load_pipeline_state() or {}
    dataset_state = state.setdefault(config.dataset_name, {})
    for step_name in (
        "json_to_gexf",
        "edge_embedding",
        "answer_generation_short",
        "answer_generation_long",
        "evaluation_f1",
    ):
        dataset_state.setdefault(step_name, {})["completed"] = False
    config.save_pipeline_state(state)



def run_graph_construction(config: THRAGConfig, force_rebuild: bool = False) -> str:
    if not config.llm_api_key:
        raise ValueError("LLM_API_KEY must be configured before running graph construction.")

    input_path = config.get_contexts_file()
    output_path = config.get_graph_json_file()
    kv_store_path = config.get_kv_store_file()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    resume_checkpoint = not force_rebuild and kv_store_path.exists()
    if resume_checkpoint:
        kv_store = load_kv_store(config)
        existing_blocks = load_existing_blocks(output_path)
        if set(existing_blocks) - set(kv_store):
            raise ValueError("Graph JSON has chunks outside the saved KV store; remove stale artifacts for a full rebuild")
    else:
        full_text = input_path.read_text(encoding="utf-8")
        chunks = chunk_text(full_text, config.max_tokens, config.overlap, config.chat_tokenizer_path)
        kv_store = build_kv_store(chunks)
        save_kv_store(kv_store_path, kv_store)
        existing_blocks = {}

    chunk_ids = list(kv_store.keys())
    chunks = [kv_store[chunk_id]["content"] for chunk_id in chunk_ids]
    if set(existing_blocks) - set(kv_store):
        raise ValueError("Graph JSON has chunks outside the saved KV store; use --force for a full rebuild")

    ordered_blocks: list[dict[str, Any] | None] = [None] * len(chunks)
    pending_indices: list[int] = []

    for index, chunk_id in enumerate(chunk_ids):
        existing_block = existing_blocks.get(chunk_id)
        if existing_block is not None:
            try:
                validate_snapshot({chunk_id: existing_block}, kv_store)
            except ValueError:
                pass
            else:
                ordered_blocks[index] = existing_block
                continue
        pending_indices.append(index)

    # Resume uses the existing fail state unchanged. Fresh runs initialize it once.
    if resume_checkpoint:
        failures = load_failures(config)
    else:
        failures = {
            chunk_ids[index]: make_failure(
                config,
                chunk_ids[index],
                index,
                chunks[index],
                ExtractionResponseError("Chunk is awaiting first-round extraction", "PendingExtraction"),
            )
            for index in pending_indices
        }

    saved_count = len(chunks) - len(pending_indices)
    if resume_checkpoint:
        print(
            f"Resuming graph extraction: {saved_count}/{len(chunks)} chunks already saved; "
            f"{len(pending_indices)} chunks pending."
        )

    if pending_indices:
        invalidate_downstream_steps(config)
    mark_extraction_state(config, completed=False, chunks=len(chunks))
    if not resume_checkpoint:
        save_failures(config, failures)
    save_blocks(output_path, [block for block in ordered_blocks if block is not None])

    if pending_indices:
        try:
            client = create_chat_client(config)
        except Exception as exc:
            for index in pending_indices:
                chunk_id = chunk_ids[index]
                failures[chunk_id] = make_failure(config, chunk_id, index, chunks[index], exc)
        else:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(call_model, client, config.default_model, chunks[index], chunk_ids[index], config): index
                    for index in pending_indices
                }
                for completed_count, future in enumerate(
                    tqdm(
                        as_completed(futures),
                        total=len(chunks),
                        initial=saved_count,
                        desc="Extracting triples",
                    ),
                    start=1,
                ):
                    index = futures[future]
                    chunk_id = chunk_ids[index]
                    try:
                        block = future.result()
                        validate_snapshot({chunk_id: block}, kv_store)
                    except Exception as exc:
                        failures[chunk_id] = make_failure(config, chunk_id, index, chunks[index], exc)
                        print(f"Failed {chunk_id}: {failures[chunk_id]['error_type']}: {failures[chunk_id]['reason']}")
                    else:
                        ordered_blocks[index] = block
                        failures.pop(chunk_id, None)

                    if completed_count % 10 == 0 or completed_count == len(futures):
                        save_blocks(output_path, [block for block in ordered_blocks if block is not None])
                        save_failures(config, failures)

    final_blocks = [block for block in ordered_blocks if block is not None]
    save_blocks(output_path, final_blocks)
    save_failures(config, failures)
    mark_extraction_state(config, completed=not failures, chunks=len(chunks))
    if failures:
        print(f"Saved {len(final_blocks)}/{len(chunks)} chunks; {len(failures)} failures: {get_failure_file(config)}")
    return str(output_path)



def main(dataset_name: str, force_rebuild: bool = False) -> str:
    """CLI-compatible entry point for graph construction."""

    config = get_config(dataset_name)
    print(f"Building graph inputs for dataset: {dataset_name}")
    print(f"Input: {config.get_contexts_file()}")
    print(f"Graph JSON: {config.get_graph_json_file()}")
    print(f"KV store: {config.get_kv_store_file()}")
    return run_graph_construction(config, force_rebuild=force_rebuild)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph-construction artifacts for a dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset>/")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard saved graph progress and recompute the extraction from scratch.",
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset, force_rebuild=args.force)
    if has_pending_failures(get_config(args.dataset)):
        raise SystemExit(1)
