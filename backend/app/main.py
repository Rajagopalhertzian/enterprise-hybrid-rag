import uuid, time, io
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from prometheus_client import Counter, Histogram, make_asgi_app
from langfuse import Langfuse
from .config import settings
from .db import init_db, get_db, Tenant, User, Document, Membership
from .auth import hash_password, verify_password, create_token, current_user
from .storage import ensure_bucket
from .ingestion import extract_text, index_document
from .retrieval import retrieve
from .llm import answer

REQUESTS = Counter("rag_requests_total", "RAG API requests", ["route","status"])
LATENCY = Histogram("rag_request_seconds", "RAG request latency", ["route"])
langfuse = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global langfuse
    init_db()
    try: ensure_bucket()
    except Exception: pass
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        langfuse = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_host)
    yield
    if langfuse: langfuse.flush()

app = FastAPI(title="Enterprise Hybrid RAG v2", version="2.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start=time.perf_counter()
    try:
        response=await call_next(request)
        REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        return response
    finally:
        LATENCY.labels(request.url.path).observe(time.perf_counter()-start)

class RegisterIn(BaseModel):
    tenant_name: str = Field(min_length=2, max_length=120)
    email: str
    password: str = Field(min_length=12)

class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=6, ge=1, le=20)

@app.get("/health")
def health():
    return {"status":"ok","version":"2.0.0"}

@app.post("/auth/register")
def register(body: RegisterIn, db: Session=Depends(get_db)):
    if db.query(User).filter_by(email=body.email.lower()).first(): raise HTTPException(409,"Email already registered")
    tenant_id, user_id = str(uuid.uuid4()), str(uuid.uuid4())
    db.add(Tenant(id=tenant_id,name=body.tenant_name))
    user=User(id=user_id,tenant_id=tenant_id,email=body.email.lower(),password_hash=hash_password(body.password),role="owner")
    db.add(user); db.commit()
    return {"access_token":create_token(user),"token_type":"bearer","tenant_id":tenant_id}

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm=Depends(), db: Session=Depends(get_db)):
    user=db.query(User).filter_by(email=form.username.lower()).first()
    if not user or not verify_password(form.password,user.password_hash): raise HTTPException(401,"Invalid credentials")
    return {"access_token":create_token(user),"token_type":"bearer"}

@app.get("/me")
def me(user: User=Depends(current_user)):
    return {"user_id":user.id,"tenant_id":user.tenant_id,"email":user.email,"role":user.role}

@app.post("/documents")
async def upload(file: UploadFile=File(...), user: User=Depends(current_user), db: Session=Depends(get_db)):
    name=file.filename or "upload"
    if name.rsplit(".",1)[-1].lower() not in {"pdf","docx","txt","md","csv","html"}: raise HTTPException(400,"Unsupported file type")
    raw=await file.read()
    if len(raw)>25*1024*1024: raise HTTPException(413,"Maximum upload size is 25MB")
    doc_id=str(uuid.uuid4()); key=f"{user.tenant_id}/{doc_id}/{name}"
    try:
        s3=ensure_bucket(); s3.put_object(Bucket=settings.s3_bucket,Key=key,Body=raw)
        text=extract_text(name,raw)
        count=index_document(user.tenant_id,doc_id,name,text)
    except Exception as e:
        raise HTTPException(500,f"Ingestion failed: {type(e).__name__}")
    doc=Document(id=doc_id,tenant_id=user.tenant_id,owner_id=user.id,filename=name,status="indexed",object_key=key,chunk_count=count)
    db.add(doc); db.add(Membership(id=str(uuid.uuid4()),tenant_id=user.tenant_id,user_id=user.id,document_id=doc_id,can_read=True)); db.commit()
    return {"document_id":doc_id,"filename":name,"chunks_indexed":count,"status":"indexed"}

@app.get("/documents")
def documents(user: User=Depends(current_user), db: Session=Depends(get_db)):
    q=db.query(Document).filter_by(tenant_id=user.tenant_id)
    if user.role != "owner":
        allowed = db.query(Membership.document_id).filter_by(tenant_id=user.tenant_id, user_id=user.id, can_read=True).subquery()
        q=q.filter(Document.id.in_(allowed))
    rows=q.order_by(Document.created_at.desc()).all()
    return [{"id":d.id,"filename":d.filename,"status":d.status,"chunks":d.chunk_count,"created_at":d.created_at} for d in rows]

@app.delete("/documents/{document_id}")
def delete_document(document_id: str, user: User=Depends(current_user), db: Session=Depends(get_db)):
    doc=db.query(Document).filter_by(id=document_id,tenant_id=user.tenant_id).first()
    if not doc or (doc.owner_id != user.id and user.role != "owner"): raise HTTPException(404,"Document not found")
    from qdrant_client import QdrantClient, models
    q=QdrantClient(url=settings.qdrant_url)
    try:
        # External cleanup is idempotent: if S3 or the DB transaction fails,
        # the document row remains and this endpoint can safely be retried.
        q.delete(collection_name=settings.qdrant_collection, points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="tenant_id",match=models.MatchValue(value=user.tenant_id)),
            models.FieldCondition(key="document_id",match=models.MatchValue(value=document_id))])))
        ensure_bucket().delete_object(Bucket=settings.s3_bucket,Key=doc.object_key)
        db.query(Membership).filter_by(
            tenant_id=user.tenant_id, document_id=document_id
        ).delete(synchronize_session=False)
        db.delete(doc)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(502, "Document cleanup failed; the document record was retained so cleanup can be retried") from exc
    finally:
        q.close()
    return {"deleted":document_id}

@app.post("/documents/{document_id}/share")
def share_document(document_id: str, email: str, user: User=Depends(current_user), db: Session=Depends(get_db)):
    if user.role != "owner":
        raise HTTPException(403, "Only workspace owners can share documents")
    doc=db.query(Document).filter_by(id=document_id, tenant_id=user.tenant_id).first()
    target=db.query(User).filter_by(email=email.lower(), tenant_id=user.tenant_id).first()
    if not doc or not target: raise HTTPException(404, "Document or workspace user not found")
    existing=db.query(Membership).filter_by(tenant_id=user.tenant_id,user_id=target.id,document_id=document_id).first()
    if existing:
        existing.can_read=True
    else:
        db.add(Membership(id=str(uuid.uuid4()),tenant_id=user.tenant_id,user_id=target.id,document_id=document_id,can_read=True))
    db.commit()
    return {"document_id":document_id,"shared_with":target.email,"can_read":True}

@app.post("/query")
async def query(body: QueryIn, user: User=Depends(current_user), db: Session=Depends(get_db)):
    trace=None
    if langfuse:
        trace=langfuse.trace(name="hybrid-rag-query",user_id=user.id,metadata={"tenant_id":user.tenant_id})
    try:
        db_ids = None
        # Owners can query the entire tenant workspace; members can query only their own uploads.
        if user.role == "owner":
            db_ids = [d[0] for d in db.query(Document.id).filter_by(tenant_id=user.tenant_id).all()]
        else:
            db_ids = [d[0] for d in db.query(Membership.document_id).filter_by(
                tenant_id=user.tenant_id, user_id=user.id, can_read=True).all()]
        contexts=retrieve(user.tenant_id,body.question,body.top_k,db_ids)
        if not contexts: return {"answer":"No relevant documents found in this workspace.","citations":[],"usage":{}}
        result, usage=await answer(body.question,contexts)
        citations=[{"citation":i+1,"document_id":c["document_id"],"filename":c["filename"],"chunk_id":c["chunk_id"],"text":c["text"][:500],"rerank_score":c.get("rerank_score")} for i,c in enumerate(contexts)]
        if trace:
            trace.generation(name="answer-generation",input=body.question,output=result,model=settings.groq_model,usage=usage)
        return {"answer":result,"citations":citations,"usage":usage}
    except Exception as e:
        raise HTTPException(502,f"Query failed: {type(e).__name__}")

@app.get("/analytics")
def analytics(user: User=Depends(current_user), db: Session=Depends(get_db)):
    docs=db.query(Document).filter_by(tenant_id=user.tenant_id).count()
    chunks=db.query(Document).filter_by(tenant_id=user.tenant_id).with_entities(Document.chunk_count).all()
    return {"documents":docs,"chunks":sum(x[0] for x in chunks)}
