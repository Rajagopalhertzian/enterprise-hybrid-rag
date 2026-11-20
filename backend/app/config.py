from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_env: str = "development"
    jwt_secret: str = "replace-this-in-production"
    access_token_minutes: int = 720
    database_url: str = "sqlite:///./rag.db"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_rag"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranker: bool = True
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_final: int = 6
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    llm_provider: str = "groq"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "rag-documents"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

settings = Settings()
