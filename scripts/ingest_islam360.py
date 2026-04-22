#!/usr/bin/env python3
"""Build Islam360 BM25 cache + embed + upsert to Pinecone (1536-dim).

This is the **canonical** Islam360 ingest entry point.  It:
  1. Loads CSV/JSON fatawa from ``settings.islam360_data_dir``
  2. Builds the BM25 corpus (Urdu-aware tokens) and saves the pickle cache
  3. Streams the records through OpenAI ``text-embedding-3-small`` (1536-d)
  4. Streams the resulting vectors into the Islam360 Pinecone index
     in a SINGLE long-lived call (no per-batch index handshake / fetch_existing
     round-trip).  This is ~10x faster than the previous loop.

Prerequisites
-------------
* CSV or JSON files under ``data/islam-360-fatwa-data/`` with Question + Answer columns.
* ``OPENAI_API_KEY`` and ``PINECONE_API_KEY`` in ``.env``.

Usage::

    python scripts/ingest_islam360.py
    python scripts/ingest_islam360.py --batch-size 200 --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Repo root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("ingest_islam360")


def _embedding_record_stream(
    records: list[dict],
    batch_size: int,
):
    """Yield one Pinecone-ready dict per fatwa.

    Embeddings are generated in batches (single API call per batch) and
    the records are yielded one-by-one so the downstream Pinecone upsert
    can stream them with its own batch size — avoiding any per-record
    Pinecone handshake.
    """
    from src.embedding.embedder import embed_texts_islam360
    from src.islam360.documents import build_metadata

    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        # Embed ONLY the question side (see build_embedding_text docstring).
        # Falls back to r["text"] for older records that predate the split.
        texts = [r.get("embedding_text") or r["text"] for r in batch]

        t0 = time.perf_counter()
        vecs = embed_texts_islam360(texts)
        dt = (time.perf_counter() - t0) * 1000
        log.info(
            "Embedded batch %d-%d / %d  (%.0f ms)",
            start,
            start + len(batch),
            total,
            dt,
        )

        for r, emb in zip(batch, vecs):
            meta = build_metadata(
                category=r.get("category", ""),
                scholar=r.get("scholar", ""),
                language=r.get("language", "ur"),
                source_file=r.get("source_file", ""),
                question=r.get("question", ""),
                answer=r.get("answer", ""),
            )
            meta["doc_id"] = r["id"]
            meta["chunk_index"] = 0
            meta["total_chunks"] = 1
            yield {"id": r["id"], "embedding": emb, "metadata": meta}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ingest Islam360 fatwas into Pinecone + BM25"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Embedding batch size (also used as Pinecone upsert batch size, clamped 100-500)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip vectors already in Pinecone (slower; only useful for resume).",
    )
    p.add_argument(
        "--no-bm25",
        action="store_true",
        help="Skip BM25 (re)build — assume the cache already exists.",
    )
    args = p.parse_args()

    from src.config import get_settings
    from src.indexing.pinecone_store import upsert_to_named_index
    from src.islam360.loader import load_islam360_records, records_to_bm25_docs
    from src.retrieval.bm25_index import BM25Corpus

    settings = get_settings()
    records = load_islam360_records()
    if not records:
        log.error(
            "No records found. Place CSV/JSON under %s",
            settings.islam360_data_dir,
        )
        sys.exit(1)

    log.info(
        "Embedding model=%s dim=%d → index=%s",
        settings.islam360_embedding_model,
        settings.islam360_embedding_dimensions,
        settings.islam360_pinecone_index,
    )

    # 1) BM25 (sparse path)
    if args.no_bm25 and Path(settings.islam360_bm25_cache_path).exists():
        log.info("--no-bm25 set; reusing existing %s", settings.islam360_bm25_cache_path)
    else:
        docs = records_to_bm25_docs(records)
        corpus = BM25Corpus.build(docs)
        corpus.save(settings.islam360_bm25_cache_path)
        log.info(
            "BM25 saved → %s (%d docs)",
            settings.islam360_bm25_cache_path,
            len(docs),
        )

    # 2) Embedding + Pinecone upsert (single streaming call)
    log.info(
        "Streaming %d records into Pinecone (skip_existing=%s, batch=%d)…",
        len(records),
        args.skip_existing,
        args.batch_size,
    )
    t0 = time.perf_counter()
    summary = upsert_to_named_index(
        settings.islam360_pinecone_index,
        settings.islam360_embedding_dimensions,
        _embedding_record_stream(records, args.batch_size),
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        show_progress=True,
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "Done in %.1f min. Summary: %s",
        elapsed / 60.0,
        summary,
    )


if __name__ == "__main__":
    main()
