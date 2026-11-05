"""Explicit per-question route replay for paired evidence-allocation experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from canon.core.artifacts import source_fingerprint
from canon.core.config import CanonConfig
from canon.core.contracts import QueryContext, RoutingResult
from canon.core.io import digest, read_json


def routing_signature(runtime: Any, method: CanonConfig, prepared: Any, runtime_info: dict[str, Any]) -> str:
    root = runtime.project_root
    names = (
        "dataset_name", "default_model", "llm_thinking", "answer_temperature", "max_context_length",
        "topic_choice_min", "topic_choice_max", "subtopic_choice_min", "subtopic_choice_max",
        "max_retries", "retry_backoff", "llm_endpoint_sha256", "chat_tokenizer_fingerprint",
        "e5_model_path", "e5_max_length", "libraries",
    )
    files = [path for path in (root / "canon/routing").glob("*.py") if path.name != "replay.py"] + [
        root / "canon/retriever.py", root / "canon/core/contracts.py", root / "canon/core/evidence.py",
        root / "canon/core/io.py", root / "canon/hierarchy/builder.py",
        root / "index/topic_choice.py", root / "index/subtopic_choice.py", root / "index/label_selection.py",
        root / "prompt/topic_choice.py", root / "prompt/subtopic_choice.py", root / "model_services.py",
    ]
    return digest({
        "schema": 1, "graph": prepared.identity["hierarchy_content"],
        "evidence": prepared.identity["evidence_content"], "settings": method.routing_settings(),
        "runtime": {name: runtime_info.get(name) for name in names}, "code": source_fingerprint(files, root),
    })


class ReplayOnlyRouter:
    def route(self, context: QueryContext) -> RoutingResult:
        raise ValueError("A per-question replay route must be installed before retrieval")


class FrozenRouter:
    def __init__(self, query: str, result: RoutingResult) -> None:
        self.query, self.result = query, result

    def route(self, context: QueryContext) -> RoutingResult:
        if context.query != self.query:
            raise ValueError("Replayed route belongs to a different question")
        return deepcopy(self.result)


class RoutingReplay:
    def __init__(self, source: Path, expected_signature: str) -> None:
        source = source.expanduser().resolve()
        stages = ("generate_short", "retrieve", "generate_long")
        if (source / "manifest.json").is_file():
            self.root = source
            stage = next((name for name in stages if (source / name / "questions").is_dir()), None)
            if stage is None:
                raise ValueError("Replay source contains no per-question checkpoints")
            self.stage = stage
        elif source.name in stages and (source.parent / "manifest.json").is_file():
            self.root, self.stage = source.parent, source.name
        else:
            raise ValueError("--routing-from must identify a CANON run directory or one of its stages")
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest.get("identity", {}).get("routing_signature") != expected_signature:
            raise ValueError("Replay requires identical V2 graph, evidence, routing code/model and settings; generate a fresh route run first")
        self.directory = self.root / self.stage / "questions"
        # Freeze only successful routing data, never generated answers. Binding the run
        # identity to these contents also catches a source run rebuilt under the same ID.
        self._checkpoints: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("*.json")):
            row = read_json(path)
            if row.get("status") != "ok":
                continue
            retrieval = row.get("retrieval") or {}
            diagnostics = retrieval.get("diagnostics", {})
            self._checkpoints[path.stem] = {
                "qid": row.get("qid"), "query": row.get("query"), "row_index": row.get("row_index"),
                "run_fingerprint": row.get("run_fingerprint"), "paths": retrieval.get("paths"),
                "routing": diagnostics.get("routing", {}), "routing_calls": diagnostics.get("routing_calls"),
            }
        self.snapshot_sha256 = digest(self._checkpoints)

    def metadata(self) -> dict[str, Any]:
        return {
            "source_run": str(self.root), "stage": self.stage, "fingerprint": self.manifest["fingerprint"],
            "routes_sha256": self.snapshot_sha256,
        }

    def load(self, question: dict[str, Any]) -> RoutingResult:
        row = self._checkpoints.get(f"{question['row_index']:06d}", {})
        if (row.get("qid") != question["qid"] or row.get("query") != question["query"]
                or row.get("run_fingerprint") != self.manifest["fingerprint"] or row.get("row_index") != question["row_index"]):
            raise ValueError("Replay checkpoint is not a successful matching question from the source run")
        entries = row.get("paths")
        if not isinstance(entries, list):
            raise ValueError("Replay checkpoint has no routing paths")
        paths = [(entry["topic_id"], entry["subtopic_id"]) for entry in entries]
        if len(paths) != len(set(paths)):
            raise ValueError("Replay checkpoint contains duplicate routing paths")
        return RoutingResult(
            paths, {(entry["topic_id"], entry["subtopic_id"]): float(entry["score"])
                    for entry in entries if entry.get("score") is not None},
            {**deepcopy(row["routing"]), "replayed": True, "replay_source": self.metadata(),
             "source_routing_calls": row["routing_calls"]},
        )
