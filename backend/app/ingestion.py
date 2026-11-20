import io, re, uuid
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from .config import settings

_embedder = None
_qdrant = None

def services():
    global _embedder, _qdrant
    if _embedder is None: _embedder = SentenceTransformer(settings.embedding_model)
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.qdrant_url)
        if not _qdrant.collection_exists(settings.qdrant_collection):
            _qdrant.create_collection(settings.qdrant_collection,
                vectors_config={"dense": models.VectorParams(size=_embedder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)})
    return _embedder, _qdrant

def extract_text(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if ext == ".docx":
        d = DocxDocument(io.BytesIO(raw))
        return "\n".join(p.text for p in d.paragraphs)
    if ext in {".txt", ".md", ".csv", ".html"}:
        return raw.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words: return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words): break
        start = max(end - overlap, start + 1)
    return chunks

def index_document(tenant_id: str, document_id: str, filename: str, text: str) -> int:
    embedder, qdrant = services()
    chunks = chunk_text(text)
    if not chunks: return 0
    vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(models.PointStruct(id=str(uuid.uuid4()), vector={"dense": vector},
            payload={"tenant_id": tenant_id, "document_id": document_id, "filename": filename,
                     "chunk_id": f"{document_id}:{i}", "chunk_index": i, "text": chunk}))
    qdrant.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)
    return len(points)
