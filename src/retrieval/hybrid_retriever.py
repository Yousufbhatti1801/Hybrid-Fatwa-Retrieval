"""Hybrid retrieval: dense (Pinecone cosine) + sparse (rank_bm25), merged by
weighted score fusion.

Pipeline
--------
1. Normalise the Urdu query.
2. Dense path  — embed query → Pinecone top-(k * CANDIDATE_MULTIPLIER).
3. Sparse path — tokenise query → BM25 top-(k * CANDIDATE_MULTIPLIER).
4. Normalise each result list to [0, 1] with min-max scaling.
5. Merge by ID: final_score = dense_weight * norm_dense + sparse_weight * norm_bm25.
   Documents found by only one path get 0 for the missing component.
6. Sort by final_score, return top_k.

Weights are fully configurable and default to 0.7 dense / 0.3 sparse.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import get_settings
from src.embedding.embedder import embed_single
from src.indexing.pinecone_store import init_index
from src.preprocessing.urdu_normalizer import normalize_urdu

if TYPE_CHECKING:
    from src.retrieval.bm25_index import BM25Corpus

logger = logging.getLogger(__name__)

# Retrieve this many candidates per path before merging, to give the fusion
# step enough material to re-rank into a good top-k.
# Increased from 3 → 5: with a large corpus, 3×top_k gives the fusion step
# too little headroom.  At 5× the extra Pinecone overhead is <5ms but recall
# improves measurably, especially for minority categories (ZAKAT, HAJJ, etc.).
_CANDIDATE_MULTIPLIER = 3
_MIN_CANDIDATES = 30   # never fetch fewer than this, even for very small top-k

# ── Cached BM25 corpus (loaded once, reused for all queries) ─────────────────
_BM25_CACHE: "BM25Corpus | None" = None


def _get_bm25_corpus() -> "BM25Corpus | None":
    """Return a cached BM25Corpus, loading from disk (or building) on first call.

    Returns *None* if the corpus is not yet available (e.g. still building
    in a background thread) — callers must handle this gracefully.
    """
    global _BM25_CACHE
    if _BM25_CACHE is None:
        try:
            from src.retrieval.bm25_index import BM25Corpus  # lazy import
            _BM25_CACHE = BM25Corpus.load_or_build()
            logger.info("BM25 corpus cached in memory (%d docs)", len(_BM25_CACHE._docs))
        except Exception as exc:
            logger.warning("BM25 corpus not available: %s", exc)
            return None
    return _BM25_CACHE


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A single retrieval result with its fused score and metadata."""

    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


# ── Score helpers ─────────────────────────────────────────────────────────────

def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalise a {id: score} mapping to [0, 1]."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Dense retrieval (Pinecone) ────────────────────────────────────────────────

def _dense_search(
    query_vec: list[float],
    candidates: int,
    category: str | None = None,
) -> dict[str, tuple[float, dict]]:
    """Query Pinecone with a pure dense vector.

    Parameters
    ----------
    category:
        When provided, adds a Pinecone metadata filter so only vectors whose
        ``category`` field equals this value are returned.

    Returns {chunk_id: (raw_score, metadata_dict)}.
    """
    index = init_index()
    kwargs: dict = {
        "vector": query_vec,
        "top_k": candidates,
        "include_metadata": True,
    }
    if category:
        kwargs["filter"] = {"category": {"$eq": category}}

    response = index.query(**kwargs)
    return {
        match.id: (match.score, dict(match.metadata or {}))
        for match in response.matches
    }


# ── Sparse retrieval (BM25) ───────────────────────────────────────────────────

def _sparse_search(
    query: str,
    candidates: int,
    corpus: "BM25Corpus",
    category: str | None = None,
) -> dict[str, tuple[float, dict]]:
    """Query the BM25 corpus.

    Parameters
    ----------
    category:
        When provided, BM25 hits whose ``category`` metadata field does not
        match are dropped before returning.

    Returns {doc_id: (raw_bm25_score, metadata_dict)}.
    """
    hits = corpus.search(query, top_k=candidates)
    results = {
        h["id"]: (h["score"], h["metadata"])
        for h in hits
        if h["score"] > 0
    }
    if category:
        results = {
            cid: (score, meta)
            for cid, (score, meta) in results.items()
            if meta.get("category", "") == category
        }
    return results


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    bm25_corpus: "BM25Corpus | None" = None,
    *,
    category: str | None = None,
    question_boost: float = 0.15,
) -> list[dict]:
    """Run a hybrid query and return ranked results.

    Parameters
    ----------
    query:
        Raw Urdu query string.
    top_k:
        Number of results to return (default: ``settings.top_k``).
    dense_weight:
        Weight for the dense (Pinecone) path, e.g. ``0.7``.
    sparse_weight:
        Weight for the sparse (BM25) path, e.g. ``0.3``.
    bm25_corpus:
        Pre-loaded ``BM25Corpus`` instance.  If *None* the corpus is loaded
        (or built) automatically via ``BM25Corpus.load_or_build()``.
    category:
        Optional category string (e.g. ``"NAMAZ"``, ``"ZAKAT"``) used to
        filter both the Pinecone dense path (metadata filter) and the BM25
        sparse path (post-filter on the ``category`` field).  When *None*,
        no filtering is applied and all categories are searched.
    question_boost:
        Additive weight (default ``0.15``) applied to a third BM25 score
        computed over the **question field only**.  This rewards results
        where the query closely matches a fatwa\'s original question.
        Set to ``0.0`` to disable.

    Returns
    -------
    List of dicts sorted by descending fused score::

        [
            {
                "text":     str,
                "score":    float,
                "metadata": {
                    "question":    str,
                    "answer":      str,
                    "category":    str,
                    "source_file": str,
                    ...                   # all Pinecone metadata fields
                }
            },
            ...
        ]
    """
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.top_k
    dw = dense_weight if dense_weight is not None else settings.dense_weight
    sw = sparse_weight if sparse_weight is not None else settings.sparse_weight
    candidates = max(_MIN_CANDIDATES, top_k * _CANDIDATE_MULTIPLIER)

    # Normalise query once, reuse for both paths
    normalised = normalize_urdu(query)

    # ── 1. Build / load BM25 corpus ───────────────────────────────────────────
    if bm25_corpus is None:
        bm25_corpus = _get_bm25_corpus()  # may be None if not ready

    # ── 2. Embed query ──────────────────────────────────────────────────────
    query_vec = embed_single(normalised)

    # ── 3. Dense + Sparse paths (run in parallel if BM25 available) ────────
    if bm25_corpus is not None:
        with ThreadPoolExecutor(max_workers=2) as pool:
            dense_future: Future = pool.submit(
                _dense_search, query_vec, candidates, category,
            )
            sparse_future: Future = pool.submit(
                _sparse_search, normalised, candidates, bm25_corpus, category,
            )
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()
    else:
        # BM25 not available — dense only
        logger.info("BM25 not loaded yet; using dense-only retrieval")
        dense_hits = _dense_search(query_vec, candidates, category)
        sparse_hits = {}

    # ── 4. Question-field boost ────────────────────────────────────────────────
    # Score each candidate against the question field only; used as an additive
    # bonus so fatwas whose saved question is semantically close to the user
    # query rank higher than those that only match in the answer body.
    if question_boost > 0 and bm25_corpus is not None:
        raw_q_scores = bm25_corpus.score_questions(normalised, candidates)
        # Only boost IDs already present in dense or sparse results;
        # norm_q is a re-ranking signal, not a source of new candidates.
        allowed = set(dense_hits) | set(sparse_hits)
        raw_q_scores = {k: v for k, v in raw_q_scores.items() if k in allowed}
        norm_q = _minmax_normalize(raw_q_scores)
    else:
        norm_q = {}

    # ── 5. Normalise path scores to [0, 1] ────────────────────────────────────
    norm_dense  = _minmax_normalize({cid: s for cid, (s, _) in dense_hits.items()})
    norm_sparse = _minmax_normalize({cid: s for cid, (s, _) in sparse_hits.items()})

    # ── 6. Merge (union of IDs) ───────────────────────────────────────────────
    all_ids = set(dense_hits) | set(sparse_hits) | set(norm_q)
    fused: list[tuple[str, float, dict]] = []

    for cid in all_ids:
        d_score = norm_dense.get(cid, 0.0)
        s_score = norm_sparse.get(cid, 0.0)
        q_score = norm_q.get(cid, 0.0)
        final = dw * d_score + sw * s_score + question_boost * q_score

        # Prefer metadata from the dense path (richer Pinecone payload);
        # fall back to BM25 metadata if the chunk only appeared there.
        if cid in dense_hits:
            meta = dense_hits[cid][1]
        else:
            meta = sparse_hits[cid][1]

        fused.append((cid, final, meta))

    # ── 7. Sort and truncate ──────────────────────────────────────────────────
    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:top_k]

    results = [
        {
            "text":     meta.get("text", ""),
            "score":    round(score, 6),
            "metadata": meta,
        }
        for _, score, meta in top
    ]

    logger.info(
        "Hybrid search | dense=%d sparse=%d q_boost=%d merged=%d returned=%d"
        " | category=%s question_boost=%.2f | query=%.60s",
        len(dense_hits), len(sparse_hits), len(norm_q), len(fused), len(results),
        category or "*", question_boost, query,
    )
    return results


def hybrid_search_as_chunks(
    query: str,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    bm25_corpus: "BM25Corpus | None" = None,
    *,
    category: str | None = None,
    question_boost: float = 0.15,
) -> list[RetrievedChunk]:
    """Same as ``hybrid_search`` but returns ``RetrievedChunk`` dataclass objects.

    Kept for backwards compatibility with the RAG pipeline.
    """
    raw = hybrid_search(
        query, top_k, dense_weight, sparse_weight, bm25_corpus,
        category=category, question_boost=question_boost,
    )
    return [
        RetrievedChunk(
            chunk_id=r["metadata"].get("doc_id", ""),
            score=r["score"],
            text=r["text"],
            metadata=r["metadata"],
        )
        for r in raw
    ]
