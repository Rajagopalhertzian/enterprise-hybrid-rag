import math

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1/i for i,x in enumerate(retrieved,1) if x in relevant),0.0)

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg=sum((1/math.log2(i+2)) for i,x in enumerate(retrieved[:k]) if x in relevant)
    ideal=sum(1/math.log2(i+2) for i in range(min(len(relevant),k)))
    return dcg/ideal if ideal else 0.0
