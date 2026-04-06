"""BM25 corpus for sparse retrieval using the rank_bm25 library.

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

from rank_bm25 import BM25Okapi

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
    """In-memory BM25 index over the fatawa corpus.

    Attributes
    ----------
    _docs:      parallel list of raw document dicts (id, text, metadata)
    _tokenized: tokenized form of each document's text
    _bm25:      fitted BM25Okapi instance
    """

    def __init__(
        self,
        docs: list[dict],
        tokenized: list[list[str]],
        bm25: BM25Okapi,
        bm25_questions: BM25Okapi | None = None,
        question_tokenized: list[list[str]] | None = None,
    ) -> None:
        self._docs = docs
        self._tokenized = tokenized
        self._bm25 = bm25
        # Second index built over the question field only; used for the
        # question-match boost during hybrid retrieval.
        self._bm25_questions: BM25Okapi | None = bm25_questions
        self._question_tokenized: list[list[str]] | None = question_tokenized

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, docs: list[dict]) -> "BM25Corpus":
        """Build a BM25 index from a list of document dicts.

        Each dict must have at least ``id`` and ``text``; optionally
        ``question``, ``answer``, ``category``, ``source_file``.

        Uses fast whitespace-based tokenisation for bulk building.
        The full normalised tokeniser is only used at query time.
        """
        logger.info("Building BM25 index over %d documents…", len(docs))

        # ── Full-text index (question + answer) ───────────────────────────
        tokenized = [_tokenize_fast(d.get("text", "")) for d in docs]
        bm25 = BM25Okapi(tokenized)

        # ── Question-only index ───────────────────────────────────────────
        q_tokenized: list[list[str]] = [
            _tokenize_fast(d.get("question", "") or d.get("text", "")[:300])
            for d in docs
        ]
        bm25_questions = BM25Okapi(q_tokenized) if q_tokenized else None

        logger.info("BM25 indexes built (full-text + question-only).")
        return cls(docs, tokenized, bm25, bm25_questions, q_tokenized)

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
        logger.info("BM25 corpus loaded from %s (%d docs)", path, len(data["docs"]))
        return cls(
            data["docs"],
            data.get("tokenized", []),
            data["bm25"],
            data.get("bm25_questions"),
            data.get("question_tokenized", []),
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
        query_tokens = _tokenize_fast(normalize_urdu(query))
        if not query_tokens:
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Grab indices of top_k highest scores
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[dict] = []
        for idx, score in indexed:
            if score <= 0:
                break
            doc = self._docs[idx]
            results.append(
                {
                    "score": score,
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
        query_tokens = _tokenize_fast(normalize_urdu(query))
        if not query_tokens:
            return {}
        scores: list[float] = self._bm25_questions.get_scores(query_tokens).tolist()
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return {
            self._docs[idx].get("id", ""): score
            for idx, score in indexed
            if score > 0
        }

    # ── Info ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._docs)

    def __repr__(self) -> str:
        return f"BM25Corpus(docs={len(self._docs)})"
