"""Macro-averaged chunk retrieval metrics."""

from __future__ import annotations

from math import log2
from statistics import fmean

from .evidence import GoldQuestion

TOP_FIVE_WEIGHTS = tuple(1.0 / log2(rank + 1) for rank in range(1, 6))


def score_question(gold: GoldQuestion, ranking: list[str], top_k: int) -> tuple[float, ...]:
    relevant = set().union(*gold.evidence_chunks)
    retrieved = set(ranking[:top_k])
    hits = len(retrieved & relevant)
    recall = hits / len(relevant)
    f1 = 2 * hits / (len(retrieved) + len(relevant))

    gains = [chunk_id in relevant for chunk_id in ranking[:5]]
    recall_at_five = sum(gains) / len(relevant)
    dcg = sum(weight for weight, hit in zip(TOP_FIVE_WEIGHTS, gains) if hit)
    ideal_dcg = sum(TOP_FIVE_WEIGHTS[:len(relevant)])
    all_evidence = float(all(retrieved & chunks for chunks in gold.evidence_chunks))
    return recall, f1, recall_at_five, dcg / ideal_dcg, all_evidence


def score_run(
    gold: dict[int, GoldQuestion], rankings: dict[int, list[str]], top_k: int,
) -> dict[str, float | None]:
    names = (f"recall_at_{top_k}", f"f1_at_{top_k}", "recall_at_5", "ndcg_at_5",
             f"all_evidence_hit_at_{top_k}")
    if not gold:
        return dict.fromkeys(names)
    scores = [score_question(question, rankings[index], top_k) for index, question in gold.items()]
    return {name: fmean(values) for name, values in zip(names, zip(*scores))}
