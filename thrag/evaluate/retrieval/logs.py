"""Read the ranked chunks saved by TH-RAG and CANON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_rankings(path: Path, queries: list[str], chunk_ids: set[str]) -> dict[int, list[str]]:
    indices_by_query: dict[str, list[int]] = defaultdict(list)
    for index, query in enumerate(queries):
        indices_by_query[query].append(index)

    rankings: dict[int, list[str]] = {}
    seen: set[int] = set()
    canon_log = None
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("query"), str):
                raise ValueError(f"{location}: expected a retrieval record with a query")
            query = row["query"].strip()
            is_canon = "row_index" in row
            if canon_log is not None and canon_log != is_canon:
                raise ValueError(f"{location}: mixed retrieval log formats")
            canon_log = is_canon

            if is_canon:
                index = row["row_index"]
                if type(index) is not int or not 0 <= index < len(queries) or queries[index] != query:
                    raise ValueError(f"{location}: row_index/query does not match qa.json")
                if index in seen:
                    raise ValueError(f"{location}: repeated row_index {index}")
                seen.add(index)
                retrieval = row.get("retrieval")
                if not isinstance(retrieval, dict) or "chunks" not in retrieval:
                    continue
                chunks = retrieval["chunks"]
            else:
                indices = indices_by_query.get(query, [])
                if len(indices) != 1:
                    raise ValueError(f"{location}: query must match exactly one row in qa.json")
                index = indices[0]
                if "chunk_id" in row:
                    chunks = [row["chunk_id"]]
                elif "sentence_chunk_id" in row:
                    # Source IDs for graph sentences do not represent full chunks.
                    chunks = []
                else:
                    raise ValueError(f"{location}: expected chunk_id or sentence_chunk_id")

            if not isinstance(chunks, list):
                raise ValueError(f"{location}: expected a list of chunk IDs")
            for chunk_id in chunks:
                if not isinstance(chunk_id, str) or chunk_id not in chunk_ids:
                    raise ValueError(f"{location}: chunk {chunk_id!r} is absent from the KV store")
            rankings.setdefault(index, []).extend(chunks)

    return {index: list(dict.fromkeys(chunks)) for index, chunks in rankings.items()}
