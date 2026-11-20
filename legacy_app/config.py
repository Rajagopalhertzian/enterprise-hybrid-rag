import os
from dataclasses import dataclass, field


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./storage"))
    api_keys: tuple = field(default_factory=lambda: tuple(_split(os.getenv("API_KEYS", ""))))
    rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MIN", "30")))
    max_query_chars: int = field(default_factory=lambda: int(os.getenv("MAX_QUERY_CHARS", "2000")))
    llama_cloud_key: str = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY", ""))
