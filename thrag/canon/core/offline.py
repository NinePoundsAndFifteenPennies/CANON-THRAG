"""Batched, resumable offline Qwen requests for branch profiles and sentence prefixes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from itertools import islice
import json
from pathlib import Path
import sqlite3
from typing import Any

from model_services import create_answer_completion, create_chat_client, get_chat_tokenizer


def generate_index_rows(
    runtime: Any, checkpoint: Path, groups: Iterable[tuple[str, list[dict]]], *,
    name: str, instructions: str, properties: dict, parse_item: Callable[[dict], Any],
    tokens_per_row: int, force: bool,
) -> dict[int, Any]:
    """IDs are positions in the sorted branch list or the original sentence table."""
    with closing(sqlite3.connect(checkpoint)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        if force:
            with db:
                db.execute("DELETE FROM results")
        results = {i: json.loads(value) for i, value in db.execute("SELECT id, value FROM results")}
        pending = [(context, [row for row in rows if row["id"] not in results]) for context, rows in groups]
        total = len(results) + sum(len(rows) for _, rows in pending)
        if len(results) == total:
            return results
        print(f"{name}: {len(results)}/{total} rows already prepared", flush=True)

        tokenizer = get_chat_tokenizer(runtime.chat_tokenizer_path)
        template = {"enable_thinking": False}

        def fit(context: str, rows: list[dict]) -> Iterator[tuple]:
            payload = {"rows": [{**row, "id": i} for i, row in enumerate(rows, 1)]}
            if context:
                payload["source_chunk"] = context
            messages = [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ]
            output = 64 + len(rows) * tokens_per_row
            size = tokenizer.count_messages(messages, chat_template_kwargs=template)
            if size + output + 128 <= runtime.max_context_length:
                yield messages, output, [row["id"] for row in rows]
            elif len(rows) > 1:
                middle = len(rows) // 2
                yield from fit(context, rows[:middle])
                yield from fit(context, rows[middle:])
            else:
                raise ValueError(f"{name} row {rows[0]['id']} exceeds MAX_CONTEXT_LENGTH")

        batches = (
            batch for context, rows in pending for start in range(0, len(rows), 32)
            for batch in fit(context, rows[start:start + 32])
        )
        with create_chat_client(runtime) as raw_client:
            client = raw_client.with_options(max_retries=min(2, max(0, runtime.max_retries - 1)))

            def request(messages: list[dict], output: int, ids: list[int]) -> dict[int, Any]:
                # A required property per local ID gives every input exactly one output.
                item = {"type": "object", "properties": properties,
                        "required": list(properties), "additionalProperties": False}
                keys = [str(i) for i in range(1, len(ids) + 1)]
                schema = {"type": "object", "properties": dict.fromkeys(keys, item),
                          "required": keys, "additionalProperties": False}
                for attempt in range(2):
                    response = create_answer_completion(
                        client, model=runtime.default_model, messages=messages, temperature=0,
                        max_tokens=output, extra_body={"chat_template_kwargs": template},
                        response_format={"type": "json_schema", "json_schema": {
                            "name": name, "strict": True, "schema": schema,
                        }},
                    )
                    try:
                        if not response.choices or response.choices[0].finish_reason not in {None, "stop"}:
                            raise ValueError("Incomplete model response")
                        rows = json.loads(response.choices[0].message.content or "")
                        if not isinstance(rows, dict) or rows.keys() != set(keys):
                            raise ValueError("Response must cover each supplied ID once")
                        return {i: parse_item(rows[key]) for i, key in zip(ids, keys, strict=True)}
                    except (ValueError, TypeError, KeyError) as exc:
                        if attempt == 1:
                            raise ValueError(f"Invalid {name} response for IDs {ids}: {exc}") from exc

            workers = max(1, runtime.max_workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while wave := list(islice(batches, workers)):
                    futures = [pool.submit(request, *batch) for batch in wave]
                    failed = None
                    for future in as_completed(futures):
                        error = future.exception()
                        if error is not None:
                            failed = error
                            continue
                        batch_results = future.result()
                        with db:
                            db.executemany("INSERT INTO results VALUES (?, ?)", (
                                (i, json.dumps(value, ensure_ascii=False)) for i, value in batch_results.items()
                            ))
                        results.update(batch_results)
                    print(f"{name}: saved {len(results)}/{total}", flush=True)
                    if failed is not None:
                        raise failed
        return results
