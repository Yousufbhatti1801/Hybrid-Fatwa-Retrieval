"""Canonical Islam360 document shape for indexing and BM25."""

from __future__ import annotations

import hashlib
from typing import Any


def stable_id(question: str, answer: str, source_hint: str = "") -> str:
    raw = f"{question.strip()}::{answer.strip()}::{source_hint}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_embedding_text(question: str, answer: str = "") -> str:
    """Text used to generate the dense (OpenAI) embedding.

    **Question only.**  The user-side query is also a question, so embedding
    only the question side gives us truly symmetric Q-to-Q retrieval in the
    vector space.  The answer body is intentionally excluded because:

    * Answers can be long and dilute the topical signal of the question.
    * Answers often mention tangential terms in passing, which would pull
      the doc vector toward the wrong neighborhood for those terms.

    Falls back to the answer ONLY if the question field is empty (so we
    still produce a usable vector for malformed records).
    """
    q = (question or "").strip()
    if q:
        return q
    a = (answer or "").strip()
    return a[:1500]  # last-resort fallback; should be rare


def build_index_text(question: str, answer: str) -> str:
    """Text used for **BM25** indexing — Question + full Answer.

    BM25 is keyword-driven and benefits from indexing the answer body so
    a user query whose terms only appear in the answer can still match.
    The separate question-only BM25 index (built from ``question``) handles
    the topical Q-to-Q signal.
    """
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q and not a:
        return ""
    if not a:
        return q
    if not q:
        return a
    return f"{q}\n{a}"


def build_metadata(
    *,
    category: str = "",
    scholar: str = "",
    language: str = "",
    source_file: str = "",
    question: str = "",
    answer: str = "",
    sect: str = "",
    source: str = "",
    folder: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "category": category or "GENERAL",
        "scholar": scholar or "",
        "language": language or "ur",
        "corpus_source": "islam360",
        "source_file": source_file or "",
        # ``folder`` was previously hard-coded to the top-level directory
        # name (``"islam-360-fatwa-data"``), which discarded the sect
        # information carried by the parent sub-folder (``Banuri-*``,
        # ``urdufatwa-*``, ``IslamQA-*``, ``fatwaqa-*``).  We now store
        # the real parent-folder name so downstream code (and future BM25
        # rebuilds) can filter on it.
        "folder": folder or "islam-360-fatwa-data",
        "source_name": "Islam360",
        "maslak": "",
        # ``sect`` is one of ``deobandi`` / ``barelvi`` / ``ahle_hadith``
        # and ``source`` one of ``banuri`` / ``urdu_fatwa`` /
        # ``ahle_hadith_1`` / ``ahle_hadith_2``.  These drive the strict
        # per-sect retrieval filter — see :mod:`src.islam360.url_index`
        # for the canonical taxonomy.
        "sect": sect,
        "source": source,
        # ``text`` is the BM25 indexed text (Q + A) — the dense embedding
        # input is built separately via ``build_embedding_text(question)``.
        "text": build_index_text(question, answer),
    }
