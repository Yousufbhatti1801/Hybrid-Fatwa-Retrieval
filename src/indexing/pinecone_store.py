"""Pinecone index management and upsert logic for hybrid (dense + sparse) vectors.

Design notes
------------
* Metric: cosine (best for normalised OpenAI embeddings).
* Each vector carries a metadata payload with question, answer, category,
  source_file and auxiliary fields for filtering.
* Duplicate prevention: IDs are fetched from the index before upserting;
  records already present are skipped entirely.
* Batch size is clamped to [100, 500] vectors, which is the Pinecone-
  recommended range for serverless indexes.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from itertools import islice
from typing import Any, Iterator

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from src.config import get_settings
from src.indexing.sparse import build_sparse_vector
from src.preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)

_BATCH_MIN = 100
_BATCH_MAX = 500

# Pinecone metadata values must be str / int / float / bool / list[str].
# We truncate long text fields to stay within the 40 KB-per-vector limit.
_MAX_META_TEXT_LEN = 4_000


# ── Internal helpers ──────────────────────────────────────────────────

def _get_client() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key)


def _batched(iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _clamp_batch_size(requested: int) -> int:
    return max(_BATCH_MIN, min(requested, _BATCH_MAX))


def _make_metadata(chunk: Chunk, question: str = "", answer: str = "", source_file: str = "") -> dict:
    """Build the full metadata payload stored alongside each vector."""
    return {
        # — core fields requested —
        "question":    question or "",
        "answer":      answer   or "",
        "category":    chunk.category,
        "source_file": source_file or chunk.source,
        # — auxiliary fields for filtering / display —
        "doc_id":      chunk.doc_id,
        "source":      chunk.source,
        "subcategory": chunk.subcategory,
        "fatwa_no":    chunk.fatwa_no,
        "url":         chunk.url,
        "text":        chunk.text,        # stored for display without re-fetch
        "chunk_index": str(chunk.chunk_index),
    }


def _make_metadata_from_dict(record_meta: dict) -> dict:
    """Build a Pinecone-compatible metadata dict from a pipeline record's
    ``metadata`` sub-dict (produced by ``preprocess_record()``).

    Only scalar / list-of-str values are kept; long text fields are truncated
    to ``_MAX_META_TEXT_LEN`` characters to stay within Pinecone's 40 KB limit.
    """
    def _trunc(v: Any) -> Any:
        return v[:_MAX_META_TEXT_LEN] if isinstance(v, str) else v

    def _clean(v: Any) -> Any:
        # Pinecone rejects null metadata values.
        if v is None:
            return ""
        if isinstance(v, (str, int, float, bool)):
            return v
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        return str(v)

    return {
        # ── four required fields ─────────────────────────────────────
        "question":    _clean(_trunc(record_meta.get("question", ""))),
        "answer":      _clean(_trunc(record_meta.get("answer", ""))),
        "category":    _clean(record_meta.get("category", "")),
        "source_file": _clean(record_meta.get("source_file", "")),
        # ── additional fields for filtering / display ─────────────────
        "doc_id":      _clean(record_meta.get("doc_id", "")),
        "folder":      _clean(record_meta.get("folder", "")),
        "date":        _clean(record_meta.get("date", "")),
        "reference":   _clean(record_meta.get("reference", "")),
        "chunk_index": int(record_meta.get("chunk_index", 0)),
        "total_chunks": int(record_meta.get("total_chunks", 1)),
        "length_flag": _clean(record_meta.get("length_flag", "normal")),
        # ── text stored for display without re-fetch ──────────────────
        "text":        _clean(_trunc(record_meta.get("text", ""))),
    }


def _fetch_existing_ids(index: Any, ids: list[str]) -> set[str]:
    """Return the subset of *ids* that already exist in the index.

    Pinecone fetch accepts up to 1000 IDs per call.
    """
    existing: set[str] = set()
    for batch in _batched(ids, 1000):
        result = index.fetch(ids=batch)
        existing.update(result.vectors.keys())
    return existing


# ── Public API ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def init_index() -> Any:
    """Create the serverless Pinecone index if it doesn't exist.

    Uses **cosine** similarity (best for unit-normalised OpenAI embeddings).
    Returns a live index handle.  Result is cached after first call.
    """
    settings = get_settings()
    pc = _get_client()

    existing_names = {idx.name for idx in pc.list_indexes()}
    if settings.pinecone_index_name not in existing_names:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        logger.info(
            "Created Pinecone index '%s' (dim=%d, metric=cosine)",
            settings.pinecone_index_name,
            settings.embedding_dimensions,
        )
    else:
        logger.info("Using existing Pinecone index '%s'", settings.pinecone_index_name)

    return pc.Index(settings.pinecone_index_name)


def index_stats() -> dict:
    """Return live statistics for the configured index.

    Example output::

        {
            "total_vector_count": 87412,
            "dimension":          3072,
            "index_fullness":     0.08,
            "namespaces":         {"":{"vector_count": 87412}},
        }
    """
    index = init_index()
    stats = index.describe_index_stats()
    result = {
        "total_vector_count": stats.total_vector_count,
        "dimension":          stats.dimension,
        "index_fullness":     stats.index_fullness,
        "namespaces":         {
            ns: {"vector_count": v.vector_count}
            for ns, v in (stats.namespaces or {}).items()
        },
    }
    logger.info("Index stats: %s", result)
    return result


def upsert_chunks(
    pairs: list[tuple[Chunk, list[float]]],
    *,
    doc_map: dict[str, dict] | None = None,
    batch_size: int | None = None,
    skip_existing: bool = True,
) -> dict:
    """Upsert (Chunk, dense_vector) pairs into Pinecone.

    Parameters
    ----------
    pairs:
        Output of ``generate_embeddings()`` — list of (Chunk, vector) tuples.
    doc_map:
        Optional mapping ``doc_id → {question, answer, source_file}`` so the
        full Q&A text can be stored in metadata.  When omitted the fields are
        left empty.
    batch_size:
        Vectors per upsert call.  Clamped to [100, 500].
    skip_existing:
        When True (default), fetch existing IDs and skip already-indexed
        vectors to avoid redundant writes and wasted cost.

    Returns
    -------
    dict with keys ``upserted``, ``skipped``, ``total``.
    """
    settings = get_settings()
    batch_size = _clamp_batch_size(batch_size or settings.batch_size)
    doc_map = doc_map or {}
    index = init_index()

    # Build full vector list
    all_vectors: list[dict] = []
    for chunk, dense_vec in pairs:
        doc_meta = doc_map.get(chunk.doc_id, {})
        all_vectors.append(
            {
                "id":            chunk.chunk_id,
                "values":        dense_vec,
                "sparse_values": build_sparse_vector(chunk.text),
                "metadata":      _make_metadata(
                    chunk,
                    question=doc_meta.get("question", ""),
                    answer=doc_meta.get("answer", ""),
                    source_file=doc_meta.get("source_file", ""),
                ),
            }
        )

    # Duplicate prevention
    if skip_existing and all_vectors:
        all_ids = [v["id"] for v in all_vectors]
        existing_ids = _fetch_existing_ids(index, all_ids)
        if existing_ids:
            logger.info("Skipping %d already-indexed vectors", len(existing_ids))
        all_vectors = [v for v in all_vectors if v["id"] not in existing_ids]
        skipped = len(pairs) - len(all_vectors)
    else:
        skipped = 0

    # Upsert in clamped batches
    upserted = 0
    for batch in _batched(all_vectors, batch_size):
        index.upsert(vectors=batch)
        upserted += len(batch)
        logger.info("Upserted %d vectors (running total: %d)", len(batch), upserted)

    summary = {"upserted": upserted, "skipped": skipped, "total": len(pairs)}
    logger.info("Upsert complete: %s", summary)
    return summary


# ── Pipeline-dict API (new path) ──────────────────────────────────────────────

def upsert_records(
    records: Iterable[dict],
    *,
    namespace: str = "",
    batch_size: int | None = None,
    skip_existing: bool = True,
    show_progress: bool = True,
) -> dict:
    """Upsert embedding records from the pipeline into Pinecone.

    Designed to consume the output of
    :func:`src.embedding.pipeline.embed_chunks` directly::

        for record in embed_chunks(chunks, checkpoint_path=Path("embed.db")):
            # records flow straight into this function
            pass

        # — or — stream the whole generator in one call:
        upsert_records(embed_chunks(chunks, checkpoint_path=Path("embed.db")))

    Parameters
    ----------
    records:
        Iterable of ``{"id": str, "embedding": list[float], "metadata": dict}``.
        The ``metadata`` dict is expected to contain at minimum
        ``question``, ``answer``, ``category``, ``source_file`` keys.
    namespace:
        Pinecone namespace.  Defaults to the default (empty) namespace.
    batch_size:
        Vectors per upsert call.  Clamped to [100, 500].
    skip_existing:
        Fetch each batch's IDs from Pinecone and discard any that are already
        present, enabling safe resume after an interrupted run.
    show_progress:
        Show a ``tqdm`` progress bar.

    Returns
    -------
    dict
        ``{"upserted": int, "skipped": int, "total": int}``
    """
    settings = get_settings()
    effective_batch = _clamp_batch_size(batch_size or settings.batch_size)
    index = init_index()

    upserted = 0
    skipped = 0
    total = 0

    # ── Materialise records so we can do a single bulk ID check ───────────
    all_records = list(records)
    total = len(all_records)

    # ── Bulk duplicate prevention (one pass instead of per-batch) ─────────
    existing_ids: set[str] = set()
    if skip_existing and all_records:
        all_ids = [r["id"] for r in all_records]
        existing_ids = _fetch_existing_ids(index, all_ids)
        if existing_ids:
            logger.info(
                "Skipping %d already-indexed IDs (bulk check).", len(existing_ids)
            )
        all_records = [r for r in all_records if r["id"] not in existing_ids]
        skipped = total - len(all_records)

    with tqdm(desc="Indexing to Pinecone", unit="vec", disable=not show_progress) as pbar:
        for batch in _batched(all_records, effective_batch):
            # ── Build Pinecone vector dicts ───────────────────────────────
            vectors = []
            for r in batch:
                meta = r.get("metadata", {})
                # Store the chunk text in metadata for display at query time
                if "text" not in meta:
                    meta = {**meta}  # don't mutate caller's dict

                vectors.append(
                    {
                        "id":            r["id"],
                        "values":        r["embedding"],
                        "sparse_values": build_sparse_vector(
                            meta.get("text") or meta.get("answer") or ""
                        ),
                        "metadata":      _make_metadata_from_dict(meta),
                    }
                )

            index.upsert(vectors=vectors, namespace=namespace)
            upserted += len(vectors)
            pbar.update(len(vectors))
            logger.info(
                "Upserted %d vectors to Pinecone (running total: %d, skipped: %d)",
                len(vectors),
                upserted,
                skipped,
            )

    summary = {"upserted": upserted, "skipped": skipped, "total": total}
    logger.info("upsert_records complete: %s", summary)
    return summary


def validate_index_coverage(
    dataset_ids: Iterable[str] | None = None,
    *,
    dataset_count: int | None = None,
    namespace: str = "",
    sample_check: bool = False,
) -> dict:
    """Compare the live Pinecone index size against an expected dataset size.

    Two modes
    ---------
    **Count-only** (fast, no ID lookup)
        Pass ``dataset_count=N``.  The function fetches index stats and
        computes coverage as ``index_count / N``.

    **ID-level** (thorough)
        Pass ``dataset_ids=<iterable of str>``.  The function checks which IDs
        are *missing* from the index by probing Pinecone in 1000-ID batches.
        Set ``sample_check=True`` to probe only the first 5000 IDs (fast
        sanity-check rather than full audit).

    Parameters
    ----------
    dataset_ids:
        Iterable of all expected chunk IDs (e.g. ``[r["id"] for r in chunks]``).
        Optional when *dataset_count* is supplied.
    dataset_count:
        Total number of expected vectors.  Used when full ID list is not
        available.  Ignored if *dataset_ids* is provided.
    namespace:
        Pinecone namespace to inspect.
    sample_check:
        When ``True`` and *dataset_ids* is provided, only verify the first
        5000 IDs instead of the full set.

    Returns
    -------
    dict
        Always contains::

            {
                "index_count":   int,   # vectors currently in the index
                "dataset_count": int,   # expected total (0 if unknown)
                "coverage_pct":  float, # index_count / dataset_count × 100
                "is_complete":   bool,  # coverage_pct >= 99.9 %
            }

        When *dataset_ids* is provided, also contains::

            {
                "missing_count": int,
                "missing_ids":   list[str],  # empty when sample_check=False and all present
                "checked_ids":   int,        # number of IDs actually verified
            }

    Raises
    ------
    ValueError
        If neither *dataset_ids* nor *dataset_count* is supplied.
    """
    if dataset_ids is None and dataset_count is None:
        raise ValueError(
            "Provide either dataset_ids or dataset_count."
        )

    # ── Live index count ──────────────────────────────────────────────────
    stats = index_stats()
    ns_stats = stats["namespaces"].get(namespace, {})
    index_count: int = (
        ns_stats.get("vector_count", 0) if namespace
        else stats["total_vector_count"]
    )

    # ── ID-level check ────────────────────────────────────────────────────
    if dataset_ids is not None:
        ids_list = list(dataset_ids)
        expected = len(ids_list)

        if sample_check:
            ids_list = ids_list[:5_000]
            logger.info(
                "sample_check=True: verifying %d of %d IDs",
                len(ids_list),
                expected,
            )

        index = init_index()
        missing: list[str] = []

        with tqdm(
            total=len(ids_list),
            desc="Checking coverage",
            unit="id",
        ) as pbar:
            for batch in _batched(ids_list, 1000):
                result = index.fetch(ids=batch, namespace=namespace)
                found = set(result.vectors.keys())
                missing.extend(vid for vid in batch if vid not in found)
                pbar.update(len(batch))

        coverage = (index_count / expected * 100) if expected else 0.0
        result_dict = {
            "index_count":   index_count,
            "dataset_count": expected,
            "coverage_pct":  round(coverage, 2),
            "is_complete":   coverage >= 99.9,
            "missing_count": len(missing),
            "missing_ids":   missing,
            "checked_ids":   len(ids_list),
        }

    else:
        # ── Count-only mode ───────────────────────────────────────────────
        expected = dataset_count  # type: ignore[assignment]
        coverage = (index_count / expected * 100) if expected else 0.0
        result_dict = {
            "index_count":   index_count,
            "dataset_count": expected,
            "coverage_pct":  round(coverage, 2),
            "is_complete":   coverage >= 99.9,
        }

    logger.info(
        "Index coverage: %d / %d vectors (%.1f%%)",
        index_count,
        result_dict["dataset_count"],
        result_dict["coverage_pct"],
    )
    return result_dict
