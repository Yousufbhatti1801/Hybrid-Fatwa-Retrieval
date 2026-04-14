"""BM25 corpus for sparse retrieval using the bm25s library.

Previously used ``rank_bm25`` which is pure-Python and scales poorly
(~9-15 seconds per query on 170k docs). The ``bm25s`` library is
numpy-vectorized and is 5-10× faster (~1-3s per query, often
sub-second on warm corpora).

Usage
-----
Build from scratch (run once after ingestion)::

    from src.retrieval.bm25_index import BM25Corpus
    corpus = BM25Corpus.build_from_corpus()
    corpus.save()          # persists to settings.bm25_cache_path

Load at query time::

    corpus = BM25Corpus.load()
    hits = corpus.search("وضو کے احکام", top_k=10)

Each hit in *hits* is::

    {"score": float, "id": str, "text": str, "metadata": {...}}
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import TYPE_CHECKING

import bm25s

from src.config import get_settings
from src.preprocessing.urdu_normalizer import normalize_urdu

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Tokenizer ────────────────────────────────────────────────────────────────
# Urdu words are the natural retrieval unit; we keep Arabic-script tokens and
# any Latin/digit tokens, discarding single characters (noise).
_TOKEN_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\w]+"
)


def _tokenize(text: str) -> list[str]:
    """Full tokenize with normalisation (used at query time — small text)."""
    tokens = _TOKEN_RE.findall(normalize_urdu(text))
    return [t for t in tokens if len(t) > 1]


def _tokenize_fast(text: str) -> list[str]:
    """Fast tokenize for bulk corpus building — skip normalize_urdu().

    Uses simple split() + length filter. BM25 ranking is robust to minor
    tokenisation differences; accuracy-critical normalisation only matters
    at query time where we still use the full _tokenize().
    """
    return [w for w in text.split() if len(w) > 1]


# ── BM25Corpus ───────────────────────────────────────────────────────────────

class BM25Corpus:
    """In-memory BM25 index over the fatawa corpus (bm25s-backed).

    Uses the ``bm25s`` library which is numpy-vectorized and ~5-10×
    faster than the old ``rank_bm25`` implementation. The public API
    is preserved so callers (hybrid_retriever) don't need to change.

    Attributes
    ----------
    _docs:           parallel list of raw document dicts (id, text, metadata)
    _bm25:           bm25s.BM25 full-text index
    _bm25_questions: bm25s.BM25 question-only index (for question boost)
    """

    def __init__(
        self,
        docs: list[dict],
        bm25: "bm25s.BM25",
        bm25_questions: "bm25s.BM25 | None" = None,
    ) -> None:
        self._docs = docs
        self._bm25 = bm25
        self._bm25_questions = bm25_questions

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, docs: list[dict]) -> "BM25Corpus":
        """Build a BM25 index from a list of document dicts.

        Each dict must have at least ``id`` and ``text``; optionally
        ``question``, ``answer``, ``category``, ``source_file``.
        """
        logger.info("Building bm25s index over %d documents…", len(docs))

        # ── Full-text index (question + answer) ───────────────────────────
        full_texts = [d.get("text", "") for d in docs]
        full_tokens = bm25s.tokenize(full_texts, stopwords=None, show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(full_tokens, show_progress=False)

        # ── Question-only index ───────────────────────────────────────────
        q_texts = [
            d.get("question", "") or d.get("text", "")[:300]
            for d in docs
        ]
        bm25_questions = None
        if q_texts:
            q_tokens = bm25s.tokenize(q_texts, stopwords=None, show_progress=False)
            bm25_questions = bm25s.BM25()
            bm25_questions.index(q_tokens, show_progress=False)

        logger.info("bm25s indexes built (full-text + question-only).")
        return cls(docs, bm25, bm25_questions)

    @classmethod
    def build_from_corpus(cls) -> "BM25Corpus":
        """Convenience: load all CSVs and build the BM25 index in one call."""
        from src.ingestion.loader import load_all_as_dicts  # lazy import
        docs = load_all_as_dicts()
        return cls.build(docs)

    @classmethod
    def build_from_checkpoint(cls, checkpoint_path: Path | None = None) -> "BM25Corpus":
        """Build from the embedding checkpoint DB (much faster than CSV reload).

        The checkpoint already contains every document's question + answer
        metadata, so we skip the full CSV scan entirely.
        """
        if checkpoint_path is None:
            for candidate in [
                Path("embed_checkpoint.db"),
                Path(".pipeline_cache/embed_checkpoint.db"),
            ]:
                if candidate.exists():
                    checkpoint_path = candidate
                    break

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError("No embedding checkpoint DB found.")

        import json
        import sqlite3

        logger.info("Building BM25 from checkpoint DB: %s", checkpoint_path)
        conn = sqlite3.connect(str(checkpoint_path))
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.execute("SELECT id, metadata FROM embeddings")

        docs: list[dict] = []
        for row_id, meta_json in cursor:
            meta = json.loads(meta_json) if meta_json else {}
            question = meta.get("question", "")
            answer = meta.get("answer", "")
            text = meta.get("text", "") or f"سوال: {question} جواب: {answer}"
            docs.append({
                "id":          row_id,
                "text":        text,
                "question":    question,
                "answer":      answer,
                "category":    meta.get("category", ""),
                "source_file": meta.get("source_file", ""),
            })
        conn.close()
        logger.info("Loaded %d docs from checkpoint for BM25 build.", len(docs))
        return cls.build(docs)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Pickle the corpus to *path* (defaults to ``settings.bm25_cache_path``).

        Only persists docs + BM25 instances.  The tokenized lists are
        redundant (BM25Okapi keeps its own internal state) and are
        excluded to keep the file small.
        """
        settings = get_settings()
        path = path or settings.bm25_cache_path
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "docs":           self._docs,
                    "bm25":           self._bm25,
                    "bm25_questions": self._bm25_questions,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BM25 corpus saved to %s (%d docs)", path, len(self._docs))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "BM25Corpus":
        """Load a pickled corpus; raises ``FileNotFoundError`` if missing."""
        settings = get_settings()
        path = path or settings.bm25_cache_path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"BM25 cache not found at '{path}'. "
                "Run BM25Corpus.build_from_corpus().save() first."
            )
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        # Reject old rank_bm25 pickles — caller (load_or_build) will rebuild.
        bm25_obj = data.get("bm25")
        if not isinstance(bm25_obj, bm25s.BM25):
            raise ModuleNotFoundError(
                "BM25 cache is from the old rank_bm25 format; rebuild required."
            )
        logger.info("BM25 corpus loaded from %s (%d docs)", path, len(data["docs"]))
        return cls(
            data["docs"],
            bm25_obj,
            data.get("bm25_questions"),
        )

    @classmethod
    def load_or_build(cls, path: Path | None = None) -> "BM25Corpus":
        """Load from cache if available, otherwise build and save.

        Build order:
        1. Try loading from pickle cache (instant).
        2. Try building from the embedding checkpoint DB (fast — no CSV scan).
        3. Fall back to building from raw CSVs (slow).
        """
        import pickle as _pickle
        settings = get_settings()
        path = path or settings.bm25_cache_path
        try:
            return cls.load(path)
        except FileNotFoundError:
            logger.info("BM25 cache not found — building…")
        except (EOFError, _pickle.UnpicklingError, KeyError, ModuleNotFoundError) as exc:
            logger.warning("BM25 cache corrupt or unreadable (%s) — rebuilding…", exc)

        # Prefer checkpoint DB (already has all text + metadata, no CSV re-scan)
        try:
            corpus = cls.build_from_checkpoint()
        except FileNotFoundError:
            logger.info("No checkpoint DB found — falling back to CSV scan…")
            corpus = cls.build_from_corpus()

        try:
            corpus.save(path)
        except OSError as exc:
            logger.warning("BM25 cache save skipped: %s", exc)
        return corpus

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return the top-k BM25 hits for *query*.

        Returns
        -------
        List of dicts, each with keys::

            {
                "score":    float,   # raw BM25 score (non-negative)
                "id":       str,
                "text":     str,
                "metadata": {
                    "question":    str,
                    "answer":      str,
                    "category":    str,
                    "source_file": str,
                }
            }

        Results are sorted by descending score.
        """
        if not query or not query.strip():
            return []

        # ── Result cache: avoid the BM25 scan for repeat queries ──
        cache_key = (query, top_k)
        if hasattr(self, "_search_cache") and cache_key in self._search_cache:
            return self._search_cache[cache_key]

        k = min(top_k, len(self._docs))
        if k == 0:
            return []

        try:
            query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
            doc_ids, scores = self._bm25.retrieve(
                query_tokens, k=k, show_progress=False
            )
        except Exception as exc:
            logger.warning("bm25s retrieve failed: %s", exc)
            return []

        results: list[dict] = []
        for idx, score in zip(doc_ids[0], scores[0]):
            if score <= 0:
                break
            doc = self._docs[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "id":    doc.get("id", ""),
                    "text":  doc.get("text", ""),
                    "metadata": {
                        "question":    doc.get("question", ""),
                        "answer":      doc.get("answer", ""),
                        "category":    doc.get("category", ""),
                        "source_file": doc.get("source_file", ""),
                    },
                }
            )

        if not hasattr(self, "_search_cache"):
            self._search_cache = {}
        if len(self._search_cache) >= 2048:
            self._search_cache.pop(next(iter(self._search_cache)))
        self._search_cache[cache_key] = results

        return results

    def score_questions(self, query: str, top_k: int) -> dict[str, float]:
        """Return per-document BM25 scores computed over the **question field only**.

        Used by :func:`src.retrieval.hybrid_retriever.hybrid_search` to add an
        additive boost to results where the query closely matches the fatwa's
        original question.

        Returns
        -------
        dict[str, float]
            ``{doc_id: raw_bm25_score}`` for the top-*top_k* matches.
            Documents with a zero score are excluded.
        """
        if self._bm25_questions is None:
            return {}
        if not query or not query.strip():
            return {}

        cache_key = (query, top_k, "q")
        if hasattr(self, "_search_cache") and cache_key in self._search_cache:
            return self._search_cache[cache_key]

        k = min(top_k, len(self._docs))
        if k == 0:
            return {}

        try:
            query_tokens = bm25s.tokenize(query, stopwords=None, show_progress=False)
            doc_ids, scores = self._bm25_questions.retrieve(
                query_tokens, k=k, show_progress=False
            )
        except Exception as exc:
            logger.warning("bm25s retrieve (questions) failed: %s", exc)
            return {}

        result: dict[str, float] = {}
        for idx, score in zip(doc_ids[0], scores[0]):
            if score <= 0:
                continue
            doc_id = self._docs[int(idx)].get("id", "")
            if doc_id:
                result[doc_id] = float(score)

        if not hasattr(self, "_search_cache"):
            self._search_cache = {}
        if len(self._search_cache) < 2048:
            self._search_cache[cache_key] = result
        return result

    # ── Info ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return f"BM25Corpus(docs={len(self._docs)})"
