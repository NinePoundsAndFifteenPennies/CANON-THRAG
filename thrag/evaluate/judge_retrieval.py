"""Evaluate saved HotpotQA and MultiHop-RAG chunk retrievals offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from canon.core.io import atomic_json, read_json
from evaluate.retrieval import hotpotqa, multihop
from evaluate.retrieval.evidence import CorpusChunks
from evaluate.retrieval.logs import load_rankings
from evaluate.retrieval.metrics import score_run

GOLD_BUILDERS = {"hotpotqa": hotpotqa.build_gold, "multihop": multihop.build_gold}


def evaluate(
    dataset: str, benchmark: str, dataset_dir: Path, kv_path: Path,
    runs: dict[str, Path], top_k: int,
) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("--top-k must be positive")
    questions = read_json(dataset_dir / "qa.json")
    metadata = read_json(dataset_dir / "metadata.json")
    if not isinstance(questions, list) or not isinstance(metadata, list) or len(questions) != len(metadata):
        raise ValueError("qa.json and metadata.json must contain the same number of rows")
    queries = []
    for index, (question, details) in enumerate(zip(questions, metadata)):
        query = question["query"]
        if not isinstance(query, str) or not query.strip() or query != details.get("query"):
            raise ValueError(f"qa.json and metadata.json have different queries at row {index}")
        queries.append(query.strip())

    context = (dataset_dir / "contexts.txt").read_text(encoding="utf-8")
    kv_store = read_json(kv_path)
    gold = GOLD_BUILDERS[benchmark](metadata, CorpusChunks(context, kv_store))
    aligned = {index: question for index, question in enumerate(gold) if question.aligned}
    chunk_ids = set(kv_store)
    rankings = {name: load_rankings(path, queries, chunk_ids) for name, path in runs.items()}
    common = set(aligned)
    for ranking in rankings.values():
        common.intersection_update(ranking)
    compared = {index: question for index, question in aligned.items() if index in common}

    return {
        "dataset": dataset,
        "top_k": top_k,
        "total": len(questions),
        "with_gold": sum(bool(question.evidence_chunks) for question in gold),
        "gold_aligned": len(aligned),
        "compared": len(compared),
        "runs": {
            name: {
                "log": str(runs[name]),
                "missing": len(aligned.keys() - ranking.keys()),
                **score_run(compared, ranking, top_k),
            }
            for name, ranking in rankings.items()
        },
    }


def run_argument(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Specify a retrieval log as LABEL=PATH")
    return label.strip(), Path(path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Directory name under data/")
    parser.add_argument("--benchmark", choices=GOLD_BUILDERS, help="Defaults to the dataset name prefix")
    parser.add_argument("--run", action="append", type=run_argument, metavar="LABEL=PATH",
                        help="Log path relative to the storage results directory, or absolute. Repeat to compare runs.")
    parser.add_argument("--top-k", type=int, help="Cutoff for full-list metrics, defaults to TOP_K2")
    parser.add_argument("--output", type=Path, help="Override the retrieval evaluation JSON path")
    args = parser.parse_args()
    benchmark = args.benchmark or args.dataset.lower().split("_", 1)[0]
    if benchmark not in GOLD_BUILDERS:
        parser.error("Select --benchmark for this dataset name")

    from config import get_config

    runtime = get_config(args.dataset)
    dataset_dir = runtime.get_dataset_dir()
    kv_path = runtime.get_kv_store_file()
    entries = args.run or [("thrag", runtime.get_chunk_log_file())]
    runs = {name: runtime.results_dir / path for name, path in entries}
    if len(runs) != len(entries):
        parser.error("Use a different label for each --run")
    top_k = runtime.top_k2 if args.top_k is None else args.top_k
    output = args.output or runtime.get_evaluation_file(eval_method="retrieval")
    inputs = [dataset_dir / name for name in ("qa.json", "metadata.json", "contexts.txt")]
    inputs.extend([kv_path, *runs.values()])
    if output.resolve() in {path.resolve() for path in inputs}:
        parser.error("--output must be different from the input files")

    try:
        result = evaluate(args.dataset, benchmark, dataset_dir, kv_path, runs, top_k)
        atomic_json(output, result)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Compared {result['compared']} of {result['total']} questions "
          f"({result['gold_aligned']} with aligned evidence)")
    print(f"Retrieval evaluation written to {output}")


if __name__ == "__main__":
    main()
