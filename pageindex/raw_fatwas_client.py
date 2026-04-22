"""Facade for the raw fatwas (inverted index) retrieval mode.

Mirrors the ``PageIndexClient`` pattern so the Flask app can call
``search()`` and ``preload()`` identically.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pageindex.raw_fatwas_index import RawFatwasIndex, _tokenize, _expand_query_terms

logger = logging.getLogger(__name__)

_INDEX = RawFatwasIndex()


class RawFatwasClient:
    """Singleton-style facade. Safe to instantiate at module load time."""

    def preload(self) -> None:
        """Build the inverted index from fatawa_lookup.json."""
        try:
            t0 = time.perf_counter()
            _INDEX.build()
            logger.info("Raw fatwas index preload complete (%.1fs, %d fatwas)",
                        time.perf_counter() - t0, _INDEX.size)
        except FileNotFoundError as exc:
            logger.warning("Raw fatwas preload skipped — lookup not built: %s", exc)
        except Exception:
            logger.exception("Raw fatwas preload failed")

    def search(self, raw_question: str) -> dict:
        """End-to-end search. Returns the same JSON shape as
        ``/api/search_pageindex`` so the frontend can reuse the
        stacked-sections renderer.
        """
        return self._search_impl(raw_question, rerank=True)

    def fast_search(self, raw_question: str, *, top_n_per_school: int = 4) -> dict:
        """Keyword-only variant: skips the per-school LLM re-rank.

        Used by the smart router (``src.routing.router``) as a cheap
        probe — ~200-500 ms end-to-end, no LLM calls.  The inverted-
        index ``score`` field is the only relevance signal in the
        output; consumers should use ``score_raw_fatwas`` in
        ``src.routing.router`` to interpret it.
        """
        return self._search_impl(raw_question, rerank=False, top_n=top_n_per_school)

    def _search_impl(
        self,
        raw_question: str,
        *,
        rerank: bool = True,
        top_n: int = 4,
    ) -> dict:
        if not _INDEX.loaded:
            raise FileNotFoundError(
                "Raw fatwas index not built. "
                "Run `python -m pageindex.convert` first, then restart the app."
            )

        results = _INDEX.search_by_school(
            raw_question, top_n_per_school=top_n, rerank=rerank,
        )

        # Extract keywords for the response (same shape as pageindex)
        tokens = _expand_query_terms(_tokenize(raw_question))

        return {
            "query":        raw_question,
            "search_query": raw_question,
            "extracted": {
                "core_question": raw_question,
                "category_hint": None,
                "keywords":      tokens[:10],
            },
            "results":      results,
            "rerank_used":  rerank,
        }
