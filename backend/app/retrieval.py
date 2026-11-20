from rank_bm25 import BM25Okapi
from qdrant_client import models
from sentence_transformers import CrossEncoder
from .config import settings
from .ingestion import services

_reranker = None

def retrieve(tenant_id: str, query: str, top_k: int | None = None, allowed_document_ids: list[str] | None = None):
    embedder, qdrant = services()
    limit = top_k or settings.top_k_final
    qvec = embedder.encode(query, normalize_embeddings=True).tolist()
    must = [models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))]
    if allowed_document_ids is not None:
        if not allowed_document_ids: return []
        must.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=allowed_document_ids)))
    filt = models.Filter(must=must)
    dense = qdrant.query_points(collection_name=settings.qdrant_collection, query=qvec,
        using="dense", query_filter=filt, limit=settings.top_k_dense, with_payload=True).points
    # Tenant-filtered sparse lexical retrieval over the same tenant's indexed chunks.
    records, offset = [], None
    while True:
        batch, offset = qdrant.scroll(collection_name=settings.qdrant_collection, scroll_filter=filt,
            limit=256, offset=offset, with_payload=True, with_vectors=False)
        records.extend(batch)
        if offset is None: break
    sparse_ranked = []
    if records:
        corpus = [p.payload.get("text", "") for p in records]
        bm25 = BM25Okapi([x.lower().split() for x in corpus])
        scores = bm25.get_scores(query.lower().split())
        sparse_ranked = sorted(zip(records, scores), key=lambda x: x[1], reverse=True)[:settings.top_k_sparse]
    # RRF fusion; IDs are Qdrant point IDs.
    fused = {}
    for rank, p in enumerate(dense, 1): fused[str(p.id)] = fused.get(str(p.id), 0) + 1/(60+rank)
    for rank, (p, _) in enumerate(sparse_ranked, 1): fused[str(p.id)] = fused.get(str(p.id), 0) + 1/(60+rank)
    by_id = {str(p.id): p for p in dense}
    by_id.update({str(p.id): p for p, _ in sparse_ranked})
    candidates = sorted(fused, key=fused.get, reverse=True)[:max(limit*5, 20)]
    docs = [{"id": pid, "score": fused[pid], **by_id[pid].payload} for pid in candidates]
    global _reranker
    if settings.enable_reranker and docs:
        if _reranker is None: _reranker = CrossEncoder(settings.reranker_model)
        pairs = [(query, d["text"]) for d in docs]
        scores = _reranker.predict(pairs).tolist()
        for d, score in zip(docs, scores): d["rerank_score"] = float(score)
        docs.sort(key=lambda d: d["rerank_score"], reverse=True)
    return docs[:limit]
