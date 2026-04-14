"""Embedding pipeline with resumable SQLite checkpointing.

Designed to consume the unified dict format produced by
``src.preprocessing.chunker.preprocess_records()``::

    {"id": str, "text": str, "metadata": dict}

Pipeline behaviour
------------------
1. On startup, read the checkpoint database to discover already-embedded IDs.
2. Yield every record that already exists in the checkpoint (fast, no API call).
3. Stream remaining chunks through the OpenAI API in token-budgeted batches.
4. After each successful batch, atomically commit the new vectors to the
   checkpoint so progress is never lost on crash or interruption.
5. Yield each newly-embedded record.

Checkpoint format
-----------------
SQLite database with a single table::

    embeddings(
        id       TEXT PRIMARY KEY,
        vector   BLOB,   -- float32 little-endian packed bytes (struct)
        metadata TEXT    -- JSON string
    )

Storage: ≈ 12 KB per record (3072 × 4 bytes) → ≈ 1.2 GB for 100 k records.

Usage
-----
::

    from src.embedding.pipeline import embed_chunks, load_checkpoint
    from src.preprocessing.chunker import preprocess_records
    from src.ingestion.dynamic_loader import stream_corpus

    chunks = preprocess_records(stream_corpus("data"))

    for record in embed_chunks(chunks, checkpoint_path=Path("embed.db")):
        # record = {"id": ..., "embedding": [...], "metadata": {...}}
        upload_to_pinecone(record)

    # ── Resume after a crash ──────────────────────────────────────────────
    # Re-running embed_chunks with the same checkpoint_path will skip every
    # ID that was already committed and carry on from where it left off.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import get_settings
from src.embedding.embedder import embed_texts

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Conservative ceiling shared with embedder.py (300 k tokens × 4 chars/token)
_MAX_CHARS_PER_BATCH = 300_000 * 4

_SQL_CREATE = """
CREATE TABLE IF NOT EXISTS embeddings (
    id       TEXT PRIMARY KEY,
    vector   BLOB    NOT NULL,
    metadata TEXT    NOT NULL DEFAULT '{}'
)
"""

_SQL_INSERT = (
    "INSERT OR IGNORE INTO embeddings (id, vector, metadata) VALUES (?, ?, ?)"
)


# ── Binary packing helpers ────────────────────────────────────────────────────

def _pack_vector(vec: list[float]) -> bytes:
    """Encode a float-32 vector as raw little-endian bytes."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(data: bytes) -> list[float]:
    """Decode raw little-endian bytes back to a Python list of floats."""
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


# ── Checkpoint class ──────────────────────────────────────────────────────────

class EmbeddingCheckpoint:
    """Thin wrapper around a SQLite database used as an embedding cache.

    Parameters
    ----------
    path:
        Filesystem path for the ``.db`` file.  Created automatically if
        it does not exist.

    Examples
    --------
    ::

        with EmbeddingCheckpoint(Path("embed.db")) as ckpt:
            print(ckpt.count(), "records already embedded")
            for record in ckpt.iter_all():
                ...
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(path), check_same_thread=False
        )
        # ── Performance PRAGMAs ───────────────────────────────────────
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
        self._conn.execute(_SQL_CREATE)
        self._conn.commit()
        logger.debug("Opened checkpoint at %s (%d records)", path, self.count())

    # ── Query helpers ─────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of stored embeddings."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]

    def get_existing_ids(self) -> set[str]:
        """Return the set of all stored IDs (loaded into memory)."""
        rows = self._conn.execute("SELECT id FROM embeddings").fetchall()
        return {row[0] for row in rows}

    def contains(self, doc_id: str) -> bool:
        """Return True if *doc_id* is already in the checkpoint."""
        return (
            self._conn.execute(
                "SELECT 1 FROM embeddings WHERE id = ?", (doc_id,)
            ).fetchone()
            is not None
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def write_batch(self, records: list[dict]) -> None:
        """Persist a batch of embedding records in a single transaction.

        Each record must have ``id``, ``embedding``, and ``metadata`` keys.
        Records whose IDs already exist are silently skipped (INSERT OR IGNORE).
        """
        rows = [
            (
                r["id"],
                _pack_vector(r["embedding"]),
                json.dumps(r["metadata"], ensure_ascii=False),
            )
            for r in records
        ]
        with self._conn:
            self._conn.executemany(_SQL_INSERT, rows)
        logger.debug("Wrote batch of %d records to checkpoint", len(records))

    # ── Read ──────────────────────────────────────────────────────────────

    def iter_all(self) -> Generator[dict, None, None]:
        """Yield every stored record as ``{"id", "embedding", "metadata"}``."""
        for row in self._conn.execute(
            "SELECT id, vector, metadata FROM embeddings"
        ):
            yield {
                "id": row[0],
                "embedding": _unpack_vector(row[1]),
                "metadata": json.loads(row[2]),
            }

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "EmbeddingCheckpoint":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ── Batching helper ───────────────────────────────────────────────────────────

def _iter_batches(
    chunks: Iterable[dict],
    max_items: int,
    skip_ids: set[str],
) -> Generator[list[dict], None, None]:
    """Yield batches of chunks that are not in *skip_ids*.

    Each batch respects both the item-count limit (*max_items*) and a
    conservative character-count ceiling (``_MAX_CHARS_PER_BATCH``).
    """
    buf: list[dict] = []
    buf_chars = 0

    for chunk in chunks:
        if chunk["id"] in skip_ids:
            continue

        chunk_chars = len(chunk.get("text", ""))

        # Flush the current buffer if adding this chunk would breach limits
        if buf and (
            len(buf) >= max_items or buf_chars + chunk_chars > _MAX_CHARS_PER_BATCH
        ):
            yield buf
            buf = []
            buf_chars = 0

        buf.append(chunk)
        buf_chars += chunk_chars

    if buf:
        yield buf


# ── Public API ────────────────────────────────────────────────────────────────

def embed_chunks(
    chunks: Iterable[dict],
    *,
    checkpoint_path: Path = Path("embed_checkpoint.db"),
    batch_size: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    skip_replay: bool = False,
) -> Generator[dict, None, None]:
    """When *skip_replay* is True, pre-existing checkpoint records are NOT
    replayed into memory — they are only used to populate the skip-ID set.
    Stage 6 reads from the checkpoint DB directly so replay is unnecessary
    and wastes large amounts of RAM for big corpora.
    """
    """Embed preprocessed chunks with resumable checkpointing.

    Parameters
    ----------
    chunks:
        Iterable of ``{"id": str, "text": str, "metadata": dict}`` records,
        e.g. the output of :func:`src.preprocessing.chunker.preprocess_records`.
    checkpoint_path:
        Path to the SQLite ``.db`` file used as a progress store.
        Created automatically on first run; subsequent runs skip any IDs
        already present in this file.
    batch_size:
        Maximum number of chunks per OpenAI API call.
        Defaults to ``settings.batch_size`` (100).
    overwrite:
        When ``True``, delete any existing checkpoint and start from scratch.
        Use with caution – this discards all previously computed embeddings.
    show_progress:
        Show a ``tqdm`` progress bar while new chunks are being embedded.

    Yields
    ------
    dict
        ``{"id": str, "embedding": list[float], "metadata": dict}``
        Records from the checkpoint are yielded first, then newly
        computed records in input order.

    Raises
    ------
    RuntimeError
        If the OpenAI API fails after all retry attempts (propagated from
        :func:`src.embedding.embedder.embed_texts`).
    """
    settings = get_settings()
    effective_batch_size = batch_size if batch_size is not None else settings.batch_size

    if overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Deleted checkpoint at %s (overwrite=True)", checkpoint_path)

    with EmbeddingCheckpoint(checkpoint_path) as ckpt:
        # ── Step 1: replay already-computed embeddings ────────────────────
        pre_existing = ckpt.count()
        if pre_existing:
            if skip_replay:
                logger.info(
                    "Skipping replay of %d pre-existing embeddings (skip_replay=True)."
                    "  Stage 6 will read them directly from the checkpoint DB.",
                    pre_existing,
                )
            else:
                logger.info(
                    "Replaying %d pre-existing embeddings from checkpoint.", pre_existing
                )
                yield from ckpt.iter_all()

        # ── Step 2: collect IDs to skip ───────────────────────────────────
        skip_ids: set[str] = ckpt.get_existing_ids()

        # ── Step 3: stream + batch remaining chunks ───────────────────────
        batch_iter = _iter_batches(chunks, effective_batch_size, skip_ids)

        with tqdm(
            desc="Embedding chunks",
            unit="chunk",
            disable=not show_progress,
        ) as pbar:
            for batch in batch_iter:
                texts = [c["text"] for c in batch]

                # embed_texts has its own retry / back-off logic
                vectors = embed_texts(texts)

                records = [
                    {
                        "id": chunk["id"],
                        "embedding": vec,
                        "metadata": chunk.get("metadata", {}),
                    }
                    for chunk, vec in zip(batch, vectors)
                ]

                # Commit the entire batch atomically before yielding
                ckpt.write_batch(records)

                pbar.update(len(records))
                logger.info(
                    "Embedded and saved batch of %d chunks "
                    "(checkpoint total: %d)",
                    len(records),
                    ckpt.count(),
                )

                yield from records


def load_checkpoint(checkpoint_path: Path) -> Generator[dict, None, None]:
    """Iterate all records stored in an existing checkpoint.

    This is a read-only convenience wrapper – it does *not* run any
    embeddings.  Useful for inspecting progress or feeding already-embedded
    records into the Pinecone indexer without re-processing the corpus.

    Parameters
    ----------
    checkpoint_path:
        Path to an existing ``.db`` checkpoint file.

    Yields
    ------
    dict
        ``{"id": str, "embedding": list[float], "metadata": dict}``

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run embed_chunks() first to generate it."
        )

    with EmbeddingCheckpoint(checkpoint_path) as ckpt:
        logger.info(
            "Loading %d records from checkpoint %s", ckpt.count(), checkpoint_path
        )
        yield from ckpt.iter_all()
