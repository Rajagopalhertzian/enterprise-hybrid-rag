"""Hybrid retrieval engine: dense (embeddings) + BM25, fused with reciprocal rank fusion."""
import threading
from pathlib import Path

from .config import Settings


class RagEngine:
    def __init__(self, s: Settings):
        from llama_index.core import Settings as LISettings
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self.s = s
        self.persist_dir = Path(s.data_dir) / "index"
        self.upload_dir = Path(s.data_dir) / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        LISettings.embed_model = HuggingFaceEmbedding(model_name=s.embed_model)
        LISettings.llm = None  # we call the LLM ourselves
        self.splitter = SentenceSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
        self.index = None
        self._retriever = None
        self._load()

    # ---------- persistence ----------
    def _load(self):
        from llama_index.core import StorageContext, load_index_from_storage

        if (self.persist_dir / "docstore.json").exists():
            ctx = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self.index = load_index_from_storage(ctx)
            self._rebuild_retriever()

    def _rebuild_retriever(self):
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.retrievers.bm25 import BM25Retriever

        nodes = list(self.index.docstore.docs.values())
        k = self.s.top_k
        dense = self.index.as_retriever(similarity_top_k=k * 2)
        sparse = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k * 2)
        self._retriever = QueryFusionRetriever(
            [dense, sparse],
            similarity_top_k=k,
            num_queries=1,  # no LLM query rewriting
            mode="reciprocal_rerank",
            use_async=False,
        )

    # ---------- ingestion ----------
    def _parse(self, path: Path):
        from llama_index.core import SimpleDirectoryReader

        if self.s.llama_cloud_key and path.suffix.lower() in {".pdf", ".docx", ".pptx", ".png", ".jpg"}:
            from llama_parse import LlamaParse

            parser = LlamaParse(api_key=self.s.llama_cloud_key, result_type="markdown")
            return SimpleDirectoryReader(
                input_files=[str(path)], file_extractor={path.suffix.lower(): parser}
            ).load_data()
        return SimpleDirectoryReader(input_files=[str(path)]).load_data()

    def ingest(self, path: Path) -> int:
        from llama_index.core import StorageContext, VectorStoreIndex

        docs = self._parse(path)
        for d in docs:
            d.metadata["source"] = path.name
        nodes = self.splitter.get_nodes_from_documents(docs)
        with self._lock:
            if self.index is None:
                self.index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults())
            else:
                self.index.insert_nodes(nodes)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
            self._rebuild_retriever()
        return len(nodes)

    # ---------- retrieval ----------
    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        if self._retriever is None:
            return []
        hits = self._retriever.retrieve(query)[: top_k or self.s.top_k]
        return [
            {
                "doc_id": h.node.node_id,
                "source": h.node.metadata.get("source", "unknown"),
                "text": h.node.get_content(),
                "score": float(h.score or 0.0),
            }
            for h in hits
        ]

    def stats(self) -> dict:
        n = len(self.index.docstore.docs) if self.index else 0
        return {"chunks": n, "embed_model": self.s.embed_model}
