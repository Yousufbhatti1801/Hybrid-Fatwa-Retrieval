from .embedder import generate_embeddings, generate_embeddings_as_dicts, embed_single, embed_texts
from .pipeline import EmbeddingCheckpoint, embed_chunks, load_checkpoint

__all__ = [
    # low-level (Chunk-based, used by legacy ingest path)
    "embed_single",
    "embed_texts",
    "generate_embeddings",
    "generate_embeddings_as_dicts",
    # pipeline (dict-based, checkpointed)
    "EmbeddingCheckpoint",
    "embed_chunks",
    "load_checkpoint",
]
