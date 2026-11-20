"""Retrieval metrics. `retrieved` = ordered list of source ids; `relevant` = set of gold ids."""


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def summarize(rows: list[tuple[list[str], set[str]]], k: int) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        f"hit@{k}": sum(hit_at_k(r, g, k) for r, g in rows) / n,
        f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in rows) / n,
        "mrr": sum(reciprocal_rank(r, g) for r, g in rows) / n,
    }
