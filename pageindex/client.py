"""Thin facade over the PageIndex retrieval primitives.

Used by app.py so it doesn't have to know about the internal layout
of the ``pageindex`` package. Two public methods:

  - ``search(question)`` — full pipeline: extract → 4-school descent
  - ``summarise(fatwa_id)`` — on-demand 2-3 sentence Urdu summary of
    one fatwa, used by the "خلاصہ" button on each result card
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

# Make src.* importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI                                # noqa: E402
from src.config import get_settings                       # noqa: E402

from pageindex.pipeline_pageindex import extract_core_question
from pageindex.query_enrich import merge_pageindex_keywords
from pageindex.search_pageindex import (
    pageindex_search,
    preload as _preload_search,
    _load_lookup,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


_SUMMARY_SYSTEM = (
    "You are an Urdu Islamic fiqh assistant. "
    "Respond ONLY in formal Urdu, in 2-3 short sentences, "
    "summarising the given fatwa faithfully. "
    "Do not add opinions, citations, or preamble."
)


class PageIndexClient:
    """Singleton-style facade. Safe to instantiate at module load time."""

    def preload(self) -> None:
        """Force-load the tree and lookup. Called by app.py warmup thread."""
        try:
            _preload_search()
            logger.info("PageIndex preload complete")
        except FileNotFoundError as exc:
            logger.warning("PageIndex preload skipped — index not built: %s", exc)
        except Exception:
            logger.exception("PageIndex preload failed")

    def search(self, raw_question: str) -> dict:
        """End-to-end search: returns the JSON shape that
        ``/api/search_pageindex`` will hand to the frontend.
        """
        s = get_settings()
        extracted = extract_core_question(raw_question)
        base_terms = (
            extracted.get("search_terms")
            or extracted.get("keywords")
            or []
        )
        norm = (
            (extracted.get("normalized_urdu") or "").strip()
            or (extracted.get("core_question") or "").strip()
        )
        blob = "\n".join(
            p for p in (norm, (raw_question or "").strip()) if p
        )
        merged = merge_pageindex_keywords(
            base_terms,
            blob,
            apply_synonym_expansion=bool(
                s.pageindex_lexical_synonym_expand
            ),
        )
        nav_q = norm or extracted["core_question"]
        results = pageindex_search(
            nav_q,
            category_hint=extracted.get("category_hint"),
            keywords=merged,
            user_raw_query=(raw_question or "").strip(),
        )
        return {
            "query":         raw_question,
            "search_query":  nav_q,
            "extracted":     extracted,
            "keywords_merged": merged,
            "results":       results,
        }

    def summarise(self, fatwa_id: str, *, model: str | None = None) -> dict:
        """One ``gpt-4o-mini`` call to summarise a single fatwa from the
        flat lookup. Returns ``{"summary": str, "fatwa_id": str}``.
        Raises ``KeyError`` if the id isn't in the lookup.
        """
        lookup = _load_lookup()
        rec = lookup.get(fatwa_id)
        if not rec:
            raise KeyError(f"fatwa_id not found: {fatwa_id}")

        body = (
            (rec.get("question_text") or "") + "\n\n"
            + (rec.get("answer_text") or "")
        ).strip()
        if not body:
            return {"fatwa_id": fatwa_id, "summary": ""}

        # Trim very long fatawa to keep cost predictable
        body = body[:6000]

        settings = get_settings()
        try:
            comp = _client().chat.completions.create(
                model=model or settings.chat_model,
                temperature=0,
                max_tokens=180,
                timeout=8,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user",   "content": body},
                ],
            )
            summary = (comp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("summarise LLM call failed: %s", exc)
            summary = ""

        return {"fatwa_id": fatwa_id, "summary": summary}
