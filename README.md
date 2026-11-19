# Enterprise Hybrid RAG v2

An end-to-end multi-tenant RAG application evolved from the supplied FastAPI/LlamaIndex hybrid RAG prototype. This version provides a React UI, JWT tenant workspaces, document ingestion, Qdrant dense retrieval, tenant-scoped BM25 + RRF, optional cross-encoder reranking, Groq generation, Ragas evaluation entrypoint, Langfuse tracing hooks, Prometheus metrics, Docker Compose infrastructure, and GitHub Actions.

## Architecture

- React + TypeScript UI: chat, document management, workspace analytics.
- FastAPI: auth, tenant context, uploads, query and document lifecycle APIs.
- PostgreSQL: tenant, user and document metadata.
- Qdrant: dense vector index with tenant/document payload filters.
- BM25: tenant-scoped lexical retrieval over Qdrant payloads; fused with dense results using RRF.
- Cross-encoder: BGE reranker applied after fusion.
- MinIO: S3-compatible original document storage.
- Redis: provisioned for distributed jobs/rate limiting extensions (not yet wired into request limiting).
- Ragas: offline answer/retrieval evaluation script.
- Langfuse: optional query/generation tracing.
- Prometheus: `/metrics` endpoint.
- GitHub Actions: backend checks and frontend build; optional Modal deployment workflow.

## Run locally

1. Copy `.env.example` to `.env`.
2. Set `JWT_SECRET` to a long random value and `GROQ_API_KEY` to your Groq key.
3. Start services:

   ```bash
   docker compose up --build
   ```

4. Open http://localhost:3000. Register a workspace with a password of at least 12 characters.
5. Upload documents and ask questions.
6. API docs: http://localhost:8000/docs. Metrics: http://localhost:8000/metrics. MinIO console: http://localhost:9001.

## API

- `POST /auth/register` JSON `{ "tenant_name": "...", "email": "...", "password": "12+ chars" }`
- `POST /auth/login` form fields `username`, `password`
- `GET /me`
- `POST /documents` multipart `file`
- `GET /documents`
- `DELETE /documents/{document_id}`
- `POST /query` JSON `{ "question": "...", "top_k": 6 }`
- `GET /analytics`
- `GET /health`
- `GET /metrics`

All document/query APIs require `Authorization: Bearer <JWT>`. Tenant ID is derived from the authenticated user, never trusted from request body. Qdrant filters are applied before both dense and lexical retrieval.

## Evaluation

Ragas evaluation expects JSONL rows with `question`, `answer`, `contexts`, `ground_truth`. Copy `eval_dataset.example.jsonl` and expand it with reviewed reference questions, answers and contexts. Run inside a Python environment with the dependencies installed:

```bash
python -m backend.app.evaluate eval_dataset.jsonl
```

The Ragas evaluator may require an LLM/embedding provider configuration depending on installed Ragas version. Treat it as an offline evaluation tool, not a request-path dependency. Retrieval IR metrics live in `backend/app/evaluation.py`.

## Modal

The included Modal entrypoint is a deployment adapter, not a replacement for the Docker Compose reference stack. Configure a Modal secret named `rag-v2-secrets` containing at minimum:
`JWT_SECRET`, `DATABASE_URL`, `QDRANT_URL`, `GROQ_API_KEY`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, and optional Langfuse credentials. These services must be reachable from Modal; localhost Compose hostnames are not reachable from Modal. For a full deployment use managed PostgreSQL, Qdrant Cloud and S3-compatible object storage, or deploy those services with private networking. Then:

```bash
modal deploy modal_app.py
```

The GitHub deployment workflow requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` repository secrets.

## Important production hardening still required

This is a runnable reference implementation, not a security-audited enterprise product. Before public production use:
- Add email verification, password reset, refresh-token rotation and tenant invitations.
- Move ingestion into a durable Redis-backed worker and add idempotent retry/dead-letter handling.
- Implement PostgreSQL-backed document ACLs for shared workspaces; current membership table records uploader access and tenant filtering provides tenant-level isolation.
- Replace the illustrative BM25 full-tenant scan with persisted per-tenant sparse indexes or Qdrant sparse vectors for large corpora.
- Add Redis-backed rate limiting and job queues (Redis is provisioned but not yet wired).
- Add Alembic migrations, backups, retention policies, malware scanning and upload quotas.
- Add Ragas evaluator provider setup and a curated golden dataset; no evaluation scores are fabricated.
- Add Prometheus scraping/remote-write or managed metrics backend for ephemeral Modal workers.
- Run load, isolation and penetration tests before production.

## Development notes

The original `app/` implementation is preserved under `legacy_app/` for comparison. The new implementation is under `backend/app/`.
