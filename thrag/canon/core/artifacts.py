"""Prepared artifacts keyed separately by extraction, embedding, and hierarchy."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from canon.core.config import CanonConfig
from canon.core.evidence import EvidenceStore
from canon.core.io import atomic_json, digest, file_digest, read_json
from canon.hierarchy import HierarchyArtifacts, build_hierarchy

SCHEMA_VERSION = 1
EVIDENCE_FILES = ("sentences.json", "vocabulary.json", "sentences.npy", "vocabulary.npy")
HIERARCHY_FILES = ("hierarchy.json", "canonical_map.json", "hierarchy_labels.npy")


def dependency_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def source_fingerprint(paths: list[Path], root: Path) -> str:
    return digest({path.relative_to(root).as_posix(): file_digest(path) for path in sorted(set(paths))})


def model_fingerprint(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"E5_MODEL_PATH is not a local directory: {path}")
    files = sorted(file for file in path.rglob("*") if file.is_file() and file.suffix in {
        ".json", ".txt", ".model", ".safetensors", ".bin",
    })
    if not files:
        raise ValueError(f"No local embedding model/tokenizer files found at {path}")
    return source_fingerprint(files, path)


@contextmanager
def artifact_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".building"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Artifact build already locked at {lock}; after a crash, verify no writer remains before removing it") from exc
    try:
        with handle:
            handle.write("CANON artifact build in progress\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def valid_cache(directory: Path, inputs: dict[str, Any], names: tuple[str, ...]) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("inputs") != inputs:
        raise ValueError(f"Artifact identity mismatch: {directory}")
    for name in names:
        path = directory / name
        if not path.is_file() or manifest.get("files", {}).get(name) != file_digest(path):
            raise ValueError(f"Incomplete/corrupt artifact: {path}. Rebuild with --force on the server")
    return True


def finish_cache(directory: Path, inputs: dict[str, Any], names: tuple[str, ...], seconds: float) -> None:
    atomic_json(directory / "manifest.json", {
        "inputs": inputs, "files": {name: file_digest(directory / name) for name in names},
        "build_seconds": seconds,
    })


@dataclass
class PreparedArtifacts:
    store: EvidenceStore
    hierarchy: HierarchyArtifacts
    evidence_dir: Path
    hierarchy_dir: Path
    identity: dict[str, Any]


def prepare_artifacts(runtime: Any, method: CanonConfig, encoder: Any, force: bool = False) -> PreparedArtifacts:
    from index.graph_construction import graph_checkpoint_is_complete, has_pending_failures

    if has_pending_failures(runtime) or not graph_checkpoint_is_complete(runtime):
        raise ValueError("Complete graph JSON/KV extraction without pending failures is required; use the original extraction/repair entrypoints first")
    root = runtime.project_root
    graph_path, kv_path = runtime.get_graph_json_file(), runtime.get_kv_store_file()
    embedding = {
        "path": str(Path(runtime.e5_model_path).expanduser().resolve()),
        "fingerprint": model_fingerprint(Path(runtime.e5_model_path).expanduser().resolve()),
        "batch_size": runtime.e5_batch_size, "max_length": runtime.e5_max_length,
        "libraries": dependency_versions(("numpy", "sentence-transformers", "torch", "transformers", "tokenizers")),
    }
    inputs = {
        "schema": SCHEMA_VERSION, "graph_sha256": file_digest(graph_path), "kv_sha256": file_digest(kv_path),
        "embedding": embedding,
        "source": source_fingerprint([
            root / "canon/core/evidence.py", root / "canon/core/contracts.py", root / "canon/core/io.py",
            root / "model_services.py", root / "index/json_to_gexf.py",
        ], root),
    }
    cache_root = runtime.results_dir / "canon" / "artifacts" / runtime.dataset_name
    evidence_dir = cache_root / ("evidence-" + digest(inputs)[:24])
    chunk_map = {key: value["content"] for key, value in read_json(kv_path).items()}
    if force or not valid_cache(evidence_dir, inputs, EVIDENCE_FILES):
        with artifact_lock(evidence_dir):
            started = perf_counter()
            store = EvidenceStore.build(read_json(graph_path), chunk_map, encoder)
            store.save(evidence_dir)
            finish_cache(evidence_dir, inputs, EVIDENCE_FILES, perf_counter() - started)
    else:
        store = EvidenceStore.load(evidence_dir, chunk_map)
    hierarchy_inputs = {
        "schema": SCHEMA_VERSION, "evidence": digest(inputs), "method": method.hierarchy_settings(),
        "source": source_fingerprint(list((root / "canon/hierarchy").glob("*.py")), root),
        "libraries": dependency_versions(("numpy", "networkx", "faiss-cpu", "faiss-gpu")),
    }
    hierarchy_dir = cache_root / ("hierarchy-" + digest(hierarchy_inputs)[:24])
    if force or not valid_cache(hierarchy_dir, hierarchy_inputs, HIERARCHY_FILES):
        with artifact_lock(hierarchy_dir):
            started = perf_counter()
            hierarchy = build_hierarchy(store, method)
            hierarchy.save(hierarchy_dir)
            finish_cache(hierarchy_dir, hierarchy_inputs, HIERARCHY_FILES, perf_counter() - started)
    else:
        hierarchy = HierarchyArtifacts.load(hierarchy_dir, store)
    return PreparedArtifacts(store, hierarchy, evidence_dir, hierarchy_dir, {
        "evidence": inputs, "hierarchy": hierarchy_inputs,
        "evidence_path": str(evidence_dir), "hierarchy_path": str(hierarchy_dir),
    })
