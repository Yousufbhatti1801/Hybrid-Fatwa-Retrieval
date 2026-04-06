"""One-shot script: load → chunk → embed → upsert the entire corpus."""

import logging

from src.ingestion import load_all
from src.preprocessing.chunker import chunk_document
from src.embedding import generate_embeddings
from src.indexing import upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    # 1 — Load all CSV fatawa
    docs = load_all()
    logger.info("Loaded %d documents", len(docs))

    # 2 — Chunk
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info("Created %d chunks", len(all_chunks))

    # 3 — Embed
    pairs = generate_embeddings(all_chunks)
    logger.info("Generated %d embeddings", len(pairs))

    # 4 — Upsert to Pinecone
    upsert_chunks(pairs)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
