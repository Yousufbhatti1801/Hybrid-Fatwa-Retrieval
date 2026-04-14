"""Centralized configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ───────────────────────────────────────────────
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chat_model: str = "gpt-4o-mini"

    # ── Embedding retry ──────────────────────────────────────
    embed_max_retries: int = 5
    embed_base_delay: float = 1.0   # seconds; doubles each retry
    embed_max_delay: float = 60.0

    # ── Pinecone ─────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "fatawa-hybrid"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # ── Data ─────────────────────────────────────────────────
    # Default resolves to <workspace_root>/repos/data regardless of CWD.
    # Override at runtime via DATA_ROOT=<path> in your .env file.
    data_root: Path = (
        Path(__file__).resolve().parent  # src/config/
        .parent                          # src/
        .parent                          # repo root
        .parent                          # repos/
        / "data"                         # repos/data/
    )
    data_sources: list[str] = [
        "Banuri-ExtractedData-Output",
        "IslamQA-ExtractedData-Output",
        "fatwaqa-ExtractedData-Output",
        "urdufatwa-ExtractedData-Output",
    ]

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 10
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    bm25_cache_path: Path = Path(__file__).resolve().parent.parent.parent / ".bm25_cache.pkl"

    # ── Pipeline / token budget ────────────────────────────
    # Total context window for gpt-4o-mini is 128k tokens.
    # We reserve ~2k for the system prompt + instructions overhead,
    # ~1k for the answer, leaving ~3k for context by default.
    context_token_budget: int = 3000   # max tokens for retrieved fatawa
    answer_max_tokens: int = 512       # max tokens in the LLM response

    # ── Ingestion ────────────────────────────────────────────
    batch_size: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
