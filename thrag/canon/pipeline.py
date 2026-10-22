"""Server entrypoint for optional enhancements; never runs author comparisons."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
import re
import subprocess
import sys
import threading
from time import perf_counter
from typing import Any

from config import get_config
from model_services import create_chat_client, get_cpu_e5_encoder
from canon.core.artifacts import PreparedArtifacts, artifact_lock, dependency_versions, model_fingerprint, prepare_artifacts, source_fingerprint
from canon.core.config import CanonConfig, parse_bool
from canon.core.io import atomic_json, atomic_jsonl, digest, file_digest, read_json
from canon.core.telemetry import MeteredClient, record_calls
from canon.retriever import CanonRetriever

STEPS = ("prepare", "retrieve", "generate_short", "generate_long")


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise argparse.ArgumentTypeError("Use a dataset/run name containing letters, digits, '.', '_' or '-'; no directory paths")
    return value


def git_identity(root: Path) -> dict[str, Any]:
    def read(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
            return result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            return None
    status = read("status", "--porcelain")
    return {"commit": read("rev-parse", "HEAD"), "working_tree_dirty": bool(status) if status is not None else None}


def runtime_identity(runtime: Any) -> dict[str, Any]:
    # Explicit allowlist: never serialize the runtime object or .env wholesale.
    names = (
        "dataset_name", "default_model", "chat_model", "llm_thinking", "e5_model_path", "e5_batch_size",
        "e5_max_length", "chat_tokenizer_path", "answer_temperature", "answer_max_tokens", "max_workers",
        "topic_choice_min", "topic_choice_max", "subtopic_choice_min", "subtopic_choice_max",
        "max_retries", "retry_backoff", "top_k1", "top_k2", "top_k1_long", "top_k2_long", "max_context_length",
    )
    identity = {name: getattr(runtime, name) for name in names}
    identity["llm_endpoint_sha256"] = digest(runtime.llm_base_url)
    identity["python_version"] = sys.version
    identity["libraries"] = dependency_versions(("numpy", "networkx", "faiss-cpu", "faiss-gpu", "openai", "transformers", "tokenizers"))
    tokenizer = Path(runtime.chat_tokenizer_path)
    identity["chat_tokenizer_fingerprint"] = model_fingerprint(tokenizer) if tokenizer.is_dir() else None
    return identity


def load_questions(runtime: Any, indices: list[int] | None, limit: int | None) -> tuple[list[dict[str, Any]], str]:
    path = runtime.get_questions_file()
    raw = read_json(path)
    if not isinstance(raw, list) or any(not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip() for item in raw):
        raise ValueError("qa.json must contain a list of nonempty query strings")
    selected = list(range(len(raw))) if indices is None else indices
    if len(set(selected)) != len(selected) or any(index < 0 or index >= len(raw) for index in selected):
        raise ValueError("Question indices must be unique, valid zero-based qa.json positions")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("No questions selected")
    metadata_path = runtime.get_dataset_dir() / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else [{} for _ in raw]
    if not isinstance(metadata, list) or len(metadata) != len(raw):
        raise ValueError("metadata.json and qa.json must have the same number of rows")
    qa_hash = file_digest(path)
    questions = []
    for index in selected:
        item, details = raw[index], metadata[index]
        if not isinstance(details, dict) or ("query" in details and details["query"] != item["query"]):
            raise ValueError(f"QA/metadata alignment mismatch at row {index}")
        questions.append({
            "qid": f"{runtime.dataset_name}:{qa_hash[:12]}:{index}", "row_index": index,
            "query": item["query"].strip(),
            "source_id": item.get("id", details.get("id", details.get("source_record_id", details.get("source_index")))),
        })
    return questions, qa_hash


def make_answerer(runtime: Any, prepared: PreparedArtifacts, retriever: CanonRetriever, client: Any, stage: str) -> Any:
    from generate.graph_rag import GraphRAG

    if stage == "generate_short":
        from generate.graph_based_rag_short import SHORT_SYSTEM_PROMPT as system
        from prompt.answer_short import ANSWER_PROMPT as prompt
        k1, k2, temperature = runtime.top_k1, runtime.top_k2, 0.0
        output_tokens = max(64, min(256, runtime.answer_max_tokens))
    else:
        from generate.graph_based_rag_long import LONG_SYSTEM_PROMPT as system
        from prompt.answer import ANSWER_PROMPT as prompt
        k1, k2, temperature = runtime.top_k1_long, runtime.top_k2_long, runtime.answer_temperature
        output_tokens = runtime.answer_max_tokens
    return GraphRAG(
        dataset_name=runtime.dataset_name, answer_prompt=prompt, system_prompt=system,
        default_top_k1=k1, default_top_k2=k2, temperature=temperature, max_output_tokens=output_tokens,
        retriever=retriever, client=client, config=runtime, chunk_map=prepared.store.chunk_map,
    )


def run_questions(
    stage: str, directory: Path, fingerprint: str, questions: list[dict[str, Any]], runtime: Any,
    prepared: PreparedArtifacts, retriever: CanonRetriever, client: Any, force: bool,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = directory / "questions"
    checkpoint_dir.mkdir(exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    pending = []
    for question in questions:
        path = checkpoint_dir / f"{question['row_index']:06d}.json"
        if path.exists() and not force:
            row = read_json(path)
            if row.get("run_fingerprint") != fingerprint or row.get("qid") != question["qid"]:
                raise ValueError(f"Question checkpoint identity mismatch: {path}")
            if row.get("status") == "ok":
                rows[question["qid"]] = row
                continue
        pending.append(question)
    local = threading.local()

    def process(question: dict[str, Any]) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        started = perf_counter()
        row: dict[str, Any] = {**question, "run_fingerprint": fingerprint}
        try:
            if stage == "retrieve":
                retrieval = retriever.retrieve(question["query"])
                answer, context_tokens = None, 0
            else:
                if not hasattr(local, "rag"):
                    local.rag = make_answerer(runtime, prepared, retriever, client, stage)
                local.rag.last_retrieval = {}
                with record_calls(calls, "answer"):
                    answer, _elapsed, context_tokens = local.rag.answer(question["query"])
                retrieval = local.rag.last_retrieval
            row.update(status="ok", result=answer, retrieval=retrieval, meta={
                "total_spent": retrieval["diagnostics"]["seconds"]["retrieval"],
                "context_tokens": context_tokens, "answer_calls": calls,
                "wall_seconds": perf_counter() - started,
            })
        except Exception as exc:
            error_type = getattr(exc, "error_type", type(exc).__name__)
            partial = getattr(exc, "diagnostics", None)
            retrieval = {"diagnostics": partial} if partial is not None else getattr(getattr(local, "rag", None), "last_retrieval", None)
            row.update(status="error", result=f"[Error] {error_type}", error_type=error_type, retrieval=retrieval, meta={
                "answer_calls": calls, "wall_seconds": perf_counter() - started,
            })
            print(f"{stage} question={question['qid']} failed: {error_type}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=max(1, runtime.max_workers)) as executor:
        futures = {executor.submit(process, question): question for question in pending}
        for future in as_completed(futures):
            row = future.result()
            rows[row["qid"]] = row
            atomic_json(checkpoint_dir / f"{row['row_index']:06d}.json", row)
            print(f"{stage}: saved {len(rows)}/{len(questions)}", flush=True)
    ordered = [rows[question["qid"]] for question in questions]
    if stage != "retrieve":
        atomic_json(directory / "answers.json", [
            {key: row.get(key) for key in ("qid", "row_index", "source_id", "query", "result", "status", "meta")}
            for row in ordered
        ])
    atomic_jsonl(directory / "retrievals.jsonl", (
        {"qid": row["qid"], "row_index": row["row_index"], "query": row["query"],
         "status": row["status"], "retrieval": row.get("retrieval"), "error_type": row.get("error_type")}
        for row in ordered
    ))
    failed = sum(row["status"] != "ok" for row in ordered)
    return {"completed": failed == 0, "total": len(ordered), "failed": failed, "directory": str(directory)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = get_config(args.dataset)
    overrides: dict[str, Any] = {}
    task_file, task_name = getattr(args, "task_file", None), getattr(args, "task", None)
    if (task_file is None) != (task_name is None):
        raise ValueError("--task-file and --task must be supplied together")
    if task_file is not None:
        task_list = read_json(task_file)
        if task_list.get("dataset") != args.dataset:
            raise ValueError("Task file belongs to another dataset")
        matches = [task for task in task_list.get("tasks", []) if task.get("name") == task_name]
        if len(matches) != 1:
            raise ValueError("Task name must identify exactly one task in the file")
        allowed = {f"CANON_{field.name.upper()}": field for field in fields(CanonConfig)}
        for key, value in matches[0]["environment"].items():
            if key not in allowed:
                raise ValueError(f"Unsupported task setting: {key}")
            field = allowed[key]
            parser = parse_bool if type(field.default) is bool else type(field.default)
            overrides[field.name] = parser(value)
    overrides.update({field.name: getattr(args, field.name) for field in fields(CanonConfig) if getattr(args, field.name, None) is not None})
    method = CanonConfig.from_env(**overrides)
    encoder = get_cpu_e5_encoder(runtime.e5_model_path, runtime.e5_batch_size, runtime.e5_max_length)
    prepared = prepare_artifacts(runtime, method, encoder, force=args.force)
    print(f"Evidence artifacts: {prepared.evidence_dir}\nHierarchy artifacts: {prepared.hierarchy_dir}", flush=True)
    steps = [step for step in STEPS if step in args.steps and step != "prepare"]
    if not steps:
        return {"prepared": True}
    questions, qa_hash = load_questions(runtime, args.question_indices, args.limit)
    root = runtime.project_root
    source_files = list((root / "canon").rglob("*.py")) + list((root / "prompt").glob("*.py")) + [
        root / "generate/graph_rag.py", root / "generate/graph_based_rag_short.py", root / "generate/graph_based_rag_long.py",
        root / "index/topic_choice.py", root / "index/subtopic_choice.py", root / "index/label_selection.py",
        root / "config.py", root / "model_services.py",
    ]
    identity = {
        "schema": 1, "method": method.to_dict(), "runtime": runtime_identity(runtime),
        "artifacts": prepared.identity, "qa_sha256": qa_hash, "questions": questions,
        "source_sha256": source_fingerprint(source_files, root),
    }
    fingerprint = digest(identity)
    run_id = args.run_id or f"{method.routing}-{fingerprint[:20]}"
    directory = runtime.results_dir / "canon" / "runs" / runtime.dataset_name / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Run ID already belongs to another configuration/input; select a different --run-id")
    else:
        manifest = {"fingerprint": fingerprint, "identity": identity, "git": git_identity(root), "stages": {}}
    raw_client = None
    try:
        if method.routing != "a" or any(step.startswith("generate_") for step in steps):
            raw_client = create_chat_client(runtime)
        client = MeteredClient(raw_client) if raw_client is not None else None
        retriever = CanonRetriever(prepared.store, prepared.hierarchy, encoder, method, runtime, client)
        with artifact_lock(directory):
            atomic_json(manifest_path, manifest)
            for step in steps:
                manifest["stages"][step] = {"completed": False}
                atomic_json(manifest_path, manifest)
                result = run_questions(step, directory / step, fingerprint, questions, runtime, prepared, retriever, client, args.force)
                manifest["stages"][step] = result
                atomic_json(manifest_path, manifest)
        print(f"CANON outputs: {directory}", flush=True)
        return manifest["stages"]
    finally:
        if raw_client is not None:
            raw_client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=safe_name, required=True)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=["prepare", "generate_short"])
    parser.add_argument("--run-id", type=safe_name)
    parser.add_argument("--task-file", type=Path, help="Optional task list written by canon.ablations")
    parser.add_argument("--task", help="Select one task; explicit module CLI switches override its settings")
    parser.add_argument("--force", action="store_true", help="Rebuild only CANON derived artifacts and selected answer stages; never re-extract facts")
    parser.add_argument("--hierarchy", choices=("none", "split", "split_merge"))
    parser.add_argument("--routing", choices=("baseline", "a", "b", "hybrid"))
    for name in ("soft-links", "bridge", "balance"):
        parser.add_argument(f"--{name}", type=parse_bool, metavar="on|off")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--question-indices", type=int, nargs="+")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    if any(isinstance(value, dict) and value.get("failed", 0) for value in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
