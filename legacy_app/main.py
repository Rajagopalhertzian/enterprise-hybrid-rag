import asyncio
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import Settings
from .prompting import SYSTEM_PROMPT, build_user_prompt
from .ratelimit import SlidingWindowLimiter

ALLOWED_EXT = {".pdf", ".txt", ".md", ".docx", ".html", ".csv"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

settings = Settings()
limiter = SlidingWindowLimiter(settings.rate_limit_per_min)
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .llm import LLMClient
    from .rag import RagEngine

    state["rag"] = await asyncio.to_thread(RagEngine, settings)
    state["llm"] = LLMClient(settings)
    yield


app = FastAPI(title="Hybrid RAG API", version="0.1.0", lifespan=lifespan)


def auth(x_api_key: str = Header(default="")) -> str:
    ok = any(secrets.compare_digest(x_api_key, k) for k in settings.api_keys)
    if not settings.api_keys or not ok:
        raise HTTPException(401, "Invalid or missing API key")
    if not limiter.allow(x_api_key):
        raise HTTPException(429, "Rate limit exceeded")
    return x_api_key


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    doc_id: str
    source: str
    excerpt: str
    score: float


class QueryOut(BaseModel):
    answer: str
    sources: list[Source]
    usage: dict


@app.get("/health")
def health():
    return {"status": "ok", **state["rag"].stats(), "provider": settings.llm_provider}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), _: str = Depends(auth)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXT)}")
    dest = state["rag"].upload_dir / Path(file.filename).name  # strips any path components
    size = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            f.write(chunk)
    n = await asyncio.to_thread(state["rag"].ingest, dest)
    return {"file": dest.name, "chunks_indexed": n}


@app.post("/query", response_model=QueryOut)
async def query(body: QueryIn, _: str = Depends(auth)):
    if len(body.question) > settings.max_query_chars:
        raise HTTPException(400, "Question too long")
    chunks = await asyncio.to_thread(state["rag"].retrieve, body.question, body.top_k)
    if not chunks:
        return QueryOut(answer="No documents have been indexed yet.", sources=[], usage={})
    try:
        answer, usage = await state["llm"].complete(SYSTEM_PROMPT, build_user_prompt(body.question, chunks))
    except Exception as e:
        raise HTTPException(502, f"LLM provider error: {type(e).__name__}")
    return QueryOut(
        answer=answer,
        sources=[
            Source(doc_id=c["doc_id"], source=c["source"], excerpt=c["text"][:300], score=c["score"])
            for c in chunks
        ],
        usage=usage,
    )
