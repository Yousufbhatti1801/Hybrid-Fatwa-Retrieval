from .pinecone_store import (
    init_index,
    upsert_chunks,
    upsert_records,
    index_stats,
    validate_index_coverage,
)

__all__ = [
    "init_index",
    "index_stats",
    # legacy Chunk-based path
    "upsert_chunks",
    # pipeline dict-based path
    "upsert_records",
    # validation
    "validate_index_coverage",
]
