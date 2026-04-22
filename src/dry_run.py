"""Dry run mode for the Hybrid+Vectorless RAG pipeline.

Patches every external API call (OpenAI Embeddings, OpenAI Chat, Pinecone)
with deterministic in-memory mocks so the entire pipeline can be exercised
without any network access, API keys, or cost.

What is mocked
--------------
* **OpenAI Embeddings** — ``embed_single`` / ``embed_texts`` return
  deterministic unit-length vectors derived from an MD5 hash of the input
  text.  Same input always produces the same vector.
* **Hybrid Search** — ``hybrid_search()`` is replaced end-to-end with a
  simple Urdu keyword matcher over a built-in ``MOCK_CORPUS`` of 8 fatawa.
  The Pinecone dense path and BM25 loading are not invoked at all.
* **Pinecone** — ``init_index()`` returns a lightweight in-memory index.
  Upserts are tracked in a dict; queries return ranked MOCK_CORPUS results.
* **OpenAI Chat** — ``OpenAI()`` returns a mock client whose
  ``chat.completions.create()`` assembles a valid Urdu answer directly from
  the retrieved context string without any HTTP call.

What is NOT mocked (runs exactly as in production)
---------------------------------------------------
* CSV parsing, Urdu normalisation, chunking, context trimming
* Prompt building (system prompt + user turn)
* Guardrail logic (all five guards evaluate the mock answer)
* Output validation
* BM25 tokenisation (used by the keyword relevance scorer)
* Schema analysis and schema mapping stages

This means dry run exercises the complete pipeline *logic* and catches
import errors, type mismatches, missing fields, and schema regressions —
without ever leaving the local machine.

Usage — context manager (recommended)
--------------------------------------
::

    from src.dry_run import DryRunContext

    with DryRunContext():
        from src.pipeline.rag import query
        result = query("نماز کی نیت کا طریقہ کیا ہے؟")
        print(result["answer"])
        print(result["timings"])

Usage — decorator
-----------------
::

    from src.dry_run import DryRunContext

    @DryRunContext()
    def test_my_function():
        from src.pipeline.guardrails import guarded_query
        return guarded_query("سوال یہاں")

Usage — persistent activation (e.g. test harness)
--------------------------------------------------
::

    import src.dry_run as dry_run

    dry_run.activate()
    # ... run tests ...
    dry_run.deactivate()

Usage — CLI quick-test
----------------------
::

    python -m src.dry_run                            # 3 sample queries
    python -m src.dry_run --query "نماز قصر کے احکام"
    python -m src.dry_run --query "سوال" --top-k 3 --validate
    python -m src.dry_run --json                     # machine-readable output
    python -m src.dry_run --stages ingest            # test ingest stages only
    python -m src.dry_run --stages query             # test query stages only
    python -m src.dry_run --stages all               # full pipeline smoke test
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
import struct
import sys
from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level state
# ─────────────────────────────────────────────────────────────────────────────

_DRY_RUN_ACTIVE: bool = False

_SENTINEL = object()   # used to detect missing env vars


def is_active() -> bool:
    """Return ``True`` when dry run mode is currently active."""
    return _DRY_RUN_ACTIVE


# ─────────────────────────────────────────────────────────────────────────────
# Built-in mock corpus  (8 representative Urdu fatawa, 4 categories)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_CORPUS: list[dict] = [
    {
        "id":          "dry_namaz_001",
        "question":    "نماز میں نیت کا کیا حکم ہے؟",
        "answer": (
            "نماز کی نیت فرض ہے۔ نیت دل کا ارادہ ہے اور اس کا محل دل ہے نہ زبان۔ "
            "فقہاءِ احناف کے نزدیک نماز شروع کرتے وقت دل سے یہ ارادہ ضروری ہے کہ "
            "کونسی نماز پڑھ رہا ہوں — فجر، ظہر، عصر، مغرب یا عشاء۔"
        ),
        "category":    "NAMAZ",
        "source_file": "dry_namaz.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-01",
    },
    {
        "id":          "dry_namaz_002",
        "question":    "سفر میں نماز قصر کا کیا طریقہ ہے؟",
        "answer": (
            "مسافر پر قصر واجب ہے — چار رکعت والی فرض نماز (ظہر، عصر، عشاء) کو "
            "دو رکعت پڑھنا ہوگا۔ مسافت کم از کم 78 کلومیٹر (48 میل) ہونی چاہیے۔ "
            "15 دن سے کم قیام کرنے کی نیت ہو تو قصر جاری رہتا ہے۔"
        ),
        "category":    "NAMAZ",
        "source_file": "dry_namaz.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-02",
    },
    {
        "id":          "dry_wudu_001",
        "question":    "وضو کن چیزوں سے ٹوٹتا ہے؟",
        "answer": (
            "وضو ٹوٹنے کی وجوہات: پیشاب یا پاخانہ آنا، ریح خارج ہونا، ناک سے خون بہنا "
            "یا پیپ نکلنا اگر بہہ کر منہ تک پہنچے، نیند جس میں جسم ڈھیلا ہو جائے، "
            "اور بے ہوشی۔ ان میں سے کچھ بھی ہو جائے تو نئے سرے سے وضو ضروری ہے۔"
        ),
        "category":    "WUDU",
        "source_file": "dry_wudu.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-03",
    },
    {
        "id":          "dry_wudu_002",
        "question":    "تیمم کب جائز ہے اور اس کا طریقہ کیا ہے؟",
        "answer": (
            "تیمم اس وقت جائز ہے جب پانی نہ ملے، بیماری کی وجہ سے پانی نقصاندہ ہو، "
            "یا پانی کی مقدار صرف پینے کے لیے کافی ہو۔ "
            "طریقہ: نیت کریں، دونوں ہاتھ پاک مٹی پر مار کر چہرہ مسح کریں، "
            "پھر دوبارہ مٹی پر ہاتھ مار کر دونوں ہاتھ کہنیوں سمیت مسح کریں۔"
        ),
        "category":    "WUDU",
        "source_file": "dry_wudu.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-04",
    },
    {
        "id":          "dry_zakat_001",
        "question":    "زکوٰۃ کا نصاب کیا ہے اور کتنی شرح سے ادا کی جائے؟",
        "answer": (
            "سونے کا نصاب ساڑھے سات تولہ (87.48 گرام) اور چاندی کا نصاب ساڑھے باون تولہ "
            "(612.36 گرام) ہے۔ جب ان میں سے کسی کا مالک ایک سال تک رہے تو ڈھائی فیصد "
            "(2.5%) کے حساب سے زکوٰۃ ادا کرنا فرض ہے۔ نقد رقم اور تجارتی سامان پر بھی "
            "چاندی کے نصاب کے برابر ہونے پر زکوٰۃ واجب ہے۔"
        ),
        "category":    "ZAKAT",
        "source_file": "dry_zakat.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-05",
    },
    {
        "id":          "dry_zakat_002",
        "question":    "زکوٰۃ کے مصارف کیا ہیں؟",
        "answer": (
            "قرآن کریم (سورۃ التوبہ: 60) میں زکوٰۃ کے آٹھ مصارف بیان ہوئے: فقراء، مساکین، "
            "عاملینِ زکوٰۃ، مؤلفۃ القلوب، گردن آزاد کرانا، مقروض، فی سبیل اللہ، اور مسافر۔ "
            "اپنے والدین، دادا دادی، نانا نانی، اولاد، شوہر یا بیوی کو زکوٰۃ دینا جائز نہیں۔"
        ),
        "category":    "ZAKAT",
        "source_file": "dry_zakat.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-06",
    },
    {
        "id":          "dry_fast_001",
        "question":    "رمضان المبارک کا روزہ کس پر فرض ہے؟",
        "answer": (
            "رمضان کا روزہ ہر مسلمان بالغ عاقل پر فرض ہے جو مقیم ہو اور صحت مند ہو۔ "
            "مسافر، مریض، حاملہ یا دودھ پلانے والی خاتون اور حائضہ کو رخصت ہے؛ "
            "یہ بعد میں قضا کریں۔ ضعیف بوڑھا جو روزہ نہ رکھ سکے فدیہ دے سکتا ہے۔"
        ),
        "category":    "FAST",
        "source_file": "dry_fast.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-07",
    },
    {
        "id":          "dry_fast_002",
        "question":    "روزے کی حالت میں کون سی چیزیں روزہ توڑ دیتی ہیں؟",
        "answer": (
            "روزہ توڑنے والی چیزیں: جان بوجھ کر کھانا پینا، جماع کرنا، قصداً قے کرنا، "
            "انجکشن کے ذریعے غذائیت پہنچانا (احتیاطاً)، اور حیض یا نفاس کا آنا۔ "
            "بھول کر کھانا پینا روزہ نہیں توڑتا۔ "
            "جان بوجھ کر توڑنے پر صرف جماع کی صورت میں کفارہ (60 روزے یا 60 فقراء کو کھانا) واجب ہے۔"
        ),
        "category":    "FAST",
        "source_file": "dry_fast.csv",
        "folder":      "DRY_RUN",
        "fatwa_no":    "DR-08",
    },
]

# Build the standard text field for each mock doc
for _r in MOCK_CORPUS:
    _r["text"] = f"سوال: {_r['question']} جواب: {_r['answer']}"

_MOCK_IDS = [r["id"] for r in MOCK_CORPUS]

# ─────────────────────────────────────────────────────────────────────────────
# Mock embeddings  (deterministic, no OpenAI call)
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=4096)
def _make_mock_embedding(text: str, dim: int = 3072) -> list[float]:
    """Return a deterministic unit-length vector derived from shake_256(text).

    Uses hashlib.shake_256 (an XOF — extendable output function) to produce
    exactly ``dim × 4`` bytes in a single C-level call, then unpacks them as
    float32 values with struct.unpack.  This is ~50× faster than the previous
    Python LCG implementation and can be cached across repeated texts.

    Same input always produces the same vector.
    """
    raw  = hashlib.shake_256(text.encode("utf-8")).digest(dim * 4)
    vec  = list(struct.unpack_from(f"<{dim}f", raw))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _mock_embed_single(text: str) -> list[float]:
    """Mock for ``src.embedding.embedder.embed_single``."""
    try:
        from src.config import get_settings   # noqa: PLC0415
        dim = get_settings().embedding_dimensions
    except Exception:
        dim = 3072
    return _make_mock_embedding(text, dim)


def _mock_embed_texts(texts: list[str]) -> list[list[float]]:
    """Mock for ``src.embedding.embedder.embed_texts``."""
    return [_mock_embed_single(t) for t in texts]


def _mock_call_with_retry(_client: Any, texts: list[str]) -> list[list[float]]:
    """Mock for ``src.embedding.embedder._call_with_retry``."""
    return _mock_embed_texts(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Mock hybrid search  (keyword relevance over MOCK_CORPUS)
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077Fا-یA-Za-z\d]{2,}")

# Common Urdu function words that otherwise dominate token-overlap scoring.
_URDU_STOPWORDS = {
    "ہے", "ہیں", "تھا", "تھی", "تھے", "کا", "کی", "کے", "کو", "میں", "پر",
    "سے", "اور", "یا", "یہ", "وہ", "کیا", "کیوں", "کب", "کس", "کن", "کہ",
    "تو", "ہی", "بھی", "نہیں", "اگر", "لیے", "لئے", "ہیں؟", "ہے؟",
}

_SCORE_BASE   = 0.70   # base score for results with any overlap
_SCORE_NO_HIT = 0.25   # base score for results with zero overlap
_SCORE_DECAY  = 0.04   # per-rank decay


def _tokenise_for_rank(text: str) -> list[str]:
    """Tokenise text for ranking, filtering common Urdu stop words."""
    toks = [t.strip().lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t and t not in _URDU_STOPWORDS and len(t) >= 2]


def _keyword_score(query: str, doc_text: str, doc_question: str = "") -> float:
    q_tokens = set(_tokenise_for_rank(query))
    d_tokens = set(_tokenise_for_rank(doc_text))
    qq_tokens = set(_tokenise_for_rank(doc_question))
    if not q_tokens:
        return _SCORE_NO_HIT
    overlap_text = q_tokens & d_tokens
    overlap_q    = q_tokens & qq_tokens

    if not overlap_text and not overlap_q:
        return _SCORE_NO_HIT

    # Heavily reward matches in the stored fatwa question field.
    score = _SCORE_BASE
    score += 0.45 * len(overlap_q) / len(q_tokens)
    score += 0.20 * len(overlap_text) / len(q_tokens)

    # Extra boost for distinctive (longer) terms found in the question title.
    score += 0.04 * sum(1 for t in q_tokens if len(t) >= 4 and t in qq_tokens)

    return min(score, 1.5)


def _mock_hybrid_search(
    query: str,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    bm25_corpus: Any = None,
    *,
    category: str | None = None,
    maslak: str | None = None,   # accepted for parity with the live signature
    question_boost: float = 0.15,
) -> list[dict]:
    """Mock for ``src.retrieval.hybrid_retriever.hybrid_search``.

    Scores MOCK_CORPUS by Urdu keyword overlap; applies optional category filter.
    Returns the same dict format as the live retriever.
    """
    try:
        from src.config import get_settings   # noqa: PLC0415
        k = top_k or get_settings().top_k
    except Exception:
        k = top_k or 5

    pool = MOCK_CORPUS
    if category:
        filtered = [r for r in pool if r["category"].upper() == category.upper()]
        pool = filtered if filtered else MOCK_CORPUS   # fallback to all on miss

    scored: list[tuple[float, dict]] = [
        (_keyword_score(query, r["text"], r.get("question", "")), r) for r in pool
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[dict] = []
    for i, (raw_score, doc) in enumerate(scored[:k]):
        score = max(0.0, raw_score - i * _SCORE_DECAY)
        results.append({
            "text":  doc["text"],
            "score": round(score, 4),
            "metadata": {
                "question":    doc["question"],
                "answer":      doc["answer"],
                "category":    doc["category"],
                "source_file": doc["source_file"],
                "doc_id":      doc["id"],
                "fatwa_no":    doc.get("fatwa_no", ""),
                "folder":      doc.get("folder", "DRY_RUN"),
                "text":        doc["text"][:500],
            },
        })
    logger.debug(
        "[dry_run] mock_hybrid_search: query=%.50s  category=%s  returned=%d",
        query, category or "*", len(results),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Mock Pinecone index  (in-memory, used for upsert/indexing operations)
# ─────────────────────────────────────────────────────────────────────────────

class _MockPineconeIndex:
    """Minimal Pinecone index interface for dry-run ingest/query stages."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}   # id → {"values", "sparse_values", "metadata"}

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float] | None = None,
        top_k: int = 5,
        include_metadata: bool = True,
        filter: dict | None = None,   # noqa: A002
        **_kwargs: Any,
    ) -> Any:
        pool = list(self._store.values()) if self._store else [
            {"id": r["id"], "score": 0.55 - i * 0.03, "metadata": {
                "question":    r["question"],
                "answer":      r["answer"],
                "category":    r["category"],
                "source_file": r["source_file"],
                "text":        r["text"][:400],
            }}
            for i, r in enumerate(MOCK_CORPUS)
        ]

        # Apply category filter if present
        if filter:
            cat_eq = (filter.get("category") or {}).get("$eq")
            if cat_eq:
                pool = [
                    v for v in pool
                    if (v.get("metadata") or {}).get("category", "").upper() == cat_eq.upper()
                ]

        # Return as mock match objects, capped at top_k
        matches = []
        for i, item in enumerate(pool[:top_k]):
            m = MagicMock()
            m.id       = item.get("id", f"mock_{i}")
            m.score    = float(item.get("score", max(0.0, 0.7 - i * 0.05)))
            m.metadata = item.get("metadata", {})
            matches.append(m)

        response = MagicMock()
        response.matches = matches
        return response

    # ── Upsert ─────────────────────────────────────────────────────────────

    def upsert(self, vectors: list[dict] | None = None, **_kwargs: Any) -> Any:
        for v in (vectors or []):
            self._store[v["id"]] = v
        resp = MagicMock()
        resp.upserted_count = len(vectors or [])
        logger.debug("[dry_run] mock_index.upsert: %d vectors", len(vectors or []))
        return resp

    # ── Fetch ──────────────────────────────────────────────────────────────

    def fetch(self, ids: list[str] | None = None, **_kwargs: Any) -> Any:
        resp = MagicMock()
        resp.vectors = {vid: self._store[vid] for vid in (ids or []) if vid in self._store}
        return resp

    # ── Stats ──────────────────────────────────────────────────────────────

    def describe_index_stats(self) -> Any:
        stats = MagicMock()
        stats.total_vector_count = len(self._store) or len(MOCK_CORPUS)
        stats.dimension          = 3072
        stats.index_fullness     = 0.0
        stats.namespaces         = {}
        return stats

    # ── Delete ─────────────────────────────────────────────────────────────

    def delete(self, ids: list[str] | None = None, **_kwargs: Any) -> None:
        for vid in (ids or []):
            self._store.pop(vid, None)
        logger.debug("[dry_run] mock_index.delete: %d ids", len(ids or []))


_GLOBAL_MOCK_INDEX = _MockPineconeIndex()


def _mock_init_index() -> _MockPineconeIndex:
    """Mock for ``src.indexing.pinecone_store.init_index``."""
    logger.debug("[dry_run] init_index() → _MockPineconeIndex")
    return _GLOBAL_MOCK_INDEX


# ─────────────────────────────────────────────────────────────────────────────
# Mock OpenAI LLM client
# ─────────────────────────────────────────────────────────────────────────────

# Match the جواب / متن block emitted by prompt_builder._FATWA_BLOCK
_QUESTION_RE = re.compile(
    r"سوال \(User Question\):\n[═\s]*\n(.+?)\n\n[═\s]*\nسیاق و سباق",
    re.DOTALL,
)

_CONTEXT_BLOCK_RE = re.compile(
    r"سوال \(Original Question\):\n(?P<oq>.+?)\n\n"
    r"جواب / متن \(Retrieved Text\):\n(?P<ans>.+?)(?=\n\n┌|\Z)",
    re.DOTALL,
)


def _extract_first_answer(messages: list[dict]) -> str:
    """Pick the best matching retrieved snippet for the user question."""
    user_content = next(
        (m.get("content", "") for m in (messages or []) if m.get("role") == "user"),
        "",
    )

    q_match = _QUESTION_RE.search(user_content)
    user_q = (q_match.group(1).strip() if q_match else "")

    blocks = []
    for m in _CONTEXT_BLOCK_RE.finditer(user_content):
        oq = m.group("oq").strip()
        ans = m.group("ans").strip()
        blocks.append((oq, ans))

    if blocks:
        best_oq, best_ans = max(
            blocks,
            key=lambda b: _keyword_score(user_q, b[1], b[0]),
        )

        snippet = best_ans
        # Hard-cap to keep the mock answer realistic
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "…"
        return f"منتخب فتویٰ: {best_oq}\n{snippet}"

    # Fallback for older prompt shapes: take first retrieved text block if present.
    legacy_match = re.search(
        r"جواب / متن \(Retrieved Text\):\n(.+?)(?=\n[┌╔─═]|\Z)",
        user_content,
        re.DOTALL,
    )
    if legacy_match:
        snippet = legacy_match.group(1).strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "…"
        return f"فتویٰ 1 کی روشنی میں:\n{snippet}"
    return MOCK_CORPUS[0]["answer"]


class _MockStreamChunk:
    """Single token chunk object matching the OpenAI streaming API shape."""
    def __init__(self, token: str) -> None:
        delta         = MagicMock()
        delta.content = token
        choice        = MagicMock()
        choice.delta  = delta
        self.choices  = [choice]


class _MockCompletion:
    """Non-streaming completion matching ``completion.choices[0].message.content``."""
    def __init__(self, text: str) -> None:
        message         = MagicMock()
        message.content = text
        choice          = MagicMock()
        choice.message  = message
        self.choices    = [choice]


class _MockCompletions:
    """Mock ``client.chat.completions``."""

    def create(
        self,
        model:       str | None = None,
        messages:    list[dict] | None = None,
        temperature: float       = 0,
        max_tokens:  int | None  = None,
        stream:      bool        = False,
        **_kwargs:   Any,
    ) -> Any:
        answer = _extract_first_answer(messages or [])
        logger.debug("[dry_run] mock_llm.create: stream=%s  answer_len=%d", stream, len(answer))
        if stream:
            # Yield token-by-token (word level for speed)
            tokens = answer.split()
            return iter(
                [_MockStreamChunk(t + " ") for t in tokens]
                + [_MockStreamChunk("")]
            )
        return _MockCompletion(answer)


class _MockChat:
    def __init__(self) -> None:
        self.completions = _MockCompletions()


class MockOpenAI:
    """Drop-in replacement for ``openai.OpenAI``.

    Accepts the same keyword arguments so it can be swapped in without
    any caller changes.
    """
    def __init__(self, api_key: str | None = None, **_kwargs: Any) -> None:
        self.chat = _MockChat()

    # Some code paths call ``OpenAI(api_key=...).embeddings.create(...)``
    # directly; expose a stub so those don't crash.
    @property
    def embeddings(self) -> Any:
        stub = MagicMock()
        stub.create.return_value = MagicMock(
            data=[MagicMock(embedding=_mock_embed_single("stub")) for _ in range(1)]
        )
        return stub


# ─────────────────────────────────────────────────────────────────────────────
# PageIndex (vectorless mode) — deterministic mocks for the 4-school search
# ─────────────────────────────────────────────────────────────────────────────

_PI_SCHOOLS = ("Banuri", "fatwaqa", "IslamQA", "urdufatwa")
_PI_LABELS = {
    "Banuri":    "Banuri Town — Deobandi",
    "fatwaqa":   "FatwaQA — Ahle Hadees",
    "IslamQA":   "IslamQA — Ahle Hadees",
    "urdufatwa": "UrduFatwa — Barelvi",
}


def _mock_pi_extract_core_question(raw_query: str, **_kw: Any) -> dict:
    """Deterministic stand-in for ``pipeline_pageindex.extract_core_question``."""
    text = (raw_query or "").strip() or "test question"
    keys = [w for w in text.split() if len(w) > 2][:5]
    s_terms = (keys + keys) if keys else ["test", "نماز"]
    return {
        "core_question":     text,
        "normalized_urdu":  text,
        "category_hint":     "OTHER",
        "keywords":          keys,
        "search_terms":      s_terms,
        "schools_to_search": list(_PI_SCHOOLS),
    }


_PI_MASLAK_MOCK = {
    "Banuri":    "Deobandi",
    "fatwaqa":   "Ahle Hadees",
    "IslamQA":   "Ahle Hadees",
    "urdufatwa": "Barelvi",
}


def _mock_pi_search(
    core_question: str,
    *,
    category_hint: str | None = None,
    keywords: list[str] | None = None,
    schools: list[str] | None = None,
    top_n: int = 4,
    **_kw: Any,
) -> list[dict]:
    """Deterministic stand-in for ``search_pageindex.pageindex_search``.

    Returns one fabricated card per school (each with a ``fatawa`` list
    of ``top_n`` mocked entries) so the dry-run UI flow can be exercised
    without any LLM call or built tree/lookup.
    """
    schools = schools or list(_PI_SCHOOLS)
    out: list[dict] = []
    for sid in schools:
        fatawa = []
        for i in range(1, max(1, top_n) + 1):
            fatawa.append({
                "fatwa_id":      f"dry__{sid}__OTHER__dry_run__DRY-RUN-{sid}-{i:03d}_0",
                "fatwa_no":      f"DRY-RUN-{sid}-{i:03d}",
                "category":      category_hint or "OTHER",
                "subtopic":      "dry_run",
                "query_text":    f"{core_question} (variant {i})",
                "question_text": f"[DRY-RUN {sid} #{i}] {core_question}",
                "answer_text":   (
                    f"یہ ایک dry-run جواب ہے ({sid} #{i})۔ "
                    "اصل LLM یا tree استعمال نہیں ہوا۔"
                ),
                "url":           "",
                "relevance_pct": max(20, 90 - (i - 1) * 15),
            })
        primary = fatawa[0]
        out.append({
            "school_id":     sid,
            "school_label":  _PI_LABELS.get(sid, sid),
            "maslak":        _PI_MASLAK_MOCK.get(sid, ""),
            "fatwa_id":      primary["fatwa_id"],
            "fatwa_no":      primary["fatwa_no"],
            "category":      primary["category"],
            "subtopic":      primary["subtopic"],
            "query_text":    primary["query_text"],
            "question_text": primary["question_text"],
            "answer_text":   primary["answer_text"],
            "url":           primary["url"],
            "relevance_pct": primary["relevance_pct"],
            "fatawa":        fatawa,
            "navigation":    {"reason": "dry-run mock", "llm_reranked": True},
        })
    return out


def _mock_pi_summarise(self: Any, fatwa_id: str, **_kw: Any) -> dict:
    """Stand-in for ``PageIndexClient.summarise``."""
    return {
        "fatwa_id": fatwa_id,
        "summary":  "[DRY-RUN] یہ ایک نمونہ خلاصہ ہے، کوئی LLM کال نہیں ہوئی۔",
    }


def _mock_pi_preload(self: Any) -> None:
    """No-op preload for PageIndexClient."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Patch table
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is (dotted.target.path, replacement).
# order does not matter — ExitStack applies them all simultaneously.

_PATCH_TABLE: list[tuple[str, Any]] = [
    # ── OpenAI Embeddings ─────────────────────────────────────────────────
    ("src.embedding.embedder.embed_single",          _mock_embed_single),
    ("src.embedding.embedder.embed_texts",           _mock_embed_texts),
    ("src.embedding.embedder._call_with_retry",      _mock_call_with_retry),
    # Re-exports from the embedder module (bound at import time in other modules)
    ("src.retrieval.hybrid_retriever.embed_single",  _mock_embed_single),
    ("src.embedding.pipeline.embed_texts",           _mock_embed_texts),
    # ── Hybrid search (replaces both dense + sparse paths end-to-end) ─────
    ("src.retrieval.hybrid_retriever.hybrid_search", _mock_hybrid_search),
    ("src.pipeline.rag.hybrid_search",               _mock_hybrid_search),
    # ── Pinecone index (for ingest / upsert operations) ───────────────────
    ("src.indexing.pinecone_store.init_index",       _mock_init_index),
    ("src.retrieval.hybrid_retriever.init_index",    _mock_init_index),
    # ── OpenAI Chat ───────────────────────────────────────────────────────
    # rag.py imports OpenAI at module level; guardrails imports it locally
    ("src.pipeline.rag.OpenAI",                      MockOpenAI),
    ("openai.OpenAI",                                MockOpenAI),
    # ── PageIndex (vectorless mode) ───────────────────────────────────────
    # Patches at the import sites used by app.py and pageindex.client.
    # Each patch is silently skipped if its target module isn't installed.
    ("pageindex.pipeline_pageindex.extract_core_question", _mock_pi_extract_core_question),
    ("pageindex.client.extract_core_question",             _mock_pi_extract_core_question),
    ("pageindex.search_pageindex.pageindex_search",        _mock_pi_search),
    ("pageindex.client.pageindex_search",                  _mock_pi_search),
    ("pageindex.client.PageIndexClient.summarise",         _mock_pi_summarise),
    ("pageindex.client.PageIndexClient.preload",           _mock_pi_preload),
]


# ─────────────────────────────────────────────────────────────────────────────
# DryRunContext  — context manager + decorator
# ─────────────────────────────────────────────────────────────────────────────

class DryRunContext:
    """Activate dry run mode for the duration of a ``with`` block.

    Can also be used as a decorator::

        @DryRunContext()
        def test_something():
            ...

    Entering the context manager:
    1. Injects placeholder ``OPENAI_API_KEY`` / ``PINECONE_API_KEY`` env vars
       if they are absent (needed for pydantic-settings validation).
    2. Clears the ``@lru_cache`` on ``get_settings()`` so Settings are
       reloaded with the placeholder keys.
    3. Applies all 11 mock patches via ``ExitStack``.

    Exiting restores the original env vars and clears the settings cache
    again so subsequent real calls load the actual keys from ``.env``.
    """

    def __init__(self) -> None:
        self._stack: ExitStack | None = None
        self._prev_openai:   Any = _SENTINEL
        self._prev_pinecone: Any = _SENTINEL

    # ── Context manager protocol ──────────────────────────────────────────

    def __enter__(self) -> "DryRunContext":
        global _DRY_RUN_ACTIVE  # noqa: PLW0603

        logger.info("[dry_run] Activating dry run mode (%d patches)", len(_PATCH_TABLE))

        # 1. Guarantee placeholder API keys exist so Settings can load
        self._prev_openai   = os.environ.get("OPENAI_API_KEY",  _SENTINEL)
        self._prev_pinecone = os.environ.get("PINECONE_API_KEY", _SENTINEL)
        if self._prev_openai is _SENTINEL:
            os.environ["OPENAI_API_KEY"]  = "dry-run-placeholder"
        if self._prev_pinecone is _SENTINEL:
            os.environ["PINECONE_API_KEY"] = "dry-run-placeholder"

        # 2. Clear cached Settings so they reload with the placeholders
        try:
            from src.config import get_settings   # noqa: PLC0415
            get_settings.cache_clear()
        except Exception:
            pass

        # 3. Apply all patches
        self._stack = ExitStack()
        self._stack.__enter__()
        for target, replacement in _PATCH_TABLE:
            try:
                self._stack.enter_context(patch(target, replacement))
            except (ModuleNotFoundError, AttributeError) as exc:
                # Gracefully skip patches whose target module isn't installed
                logger.debug("[dry_run] Skipping patch '%s': %s", target, exc)

        _DRY_RUN_ACTIVE = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _DRY_RUN_ACTIVE  # noqa: PLW0603

        _DRY_RUN_ACTIVE = False

        if self._stack:
            self._stack.__exit__(exc_type, exc_val, exc_tb)

        # Restore env vars
        if self._prev_openai is _SENTINEL:
            os.environ.pop("OPENAI_API_KEY", None)
        elif self._prev_openai is not _SENTINEL:
            os.environ["OPENAI_API_KEY"] = self._prev_openai  # type: ignore[assignment]

        if self._prev_pinecone is _SENTINEL:
            os.environ.pop("PINECONE_API_KEY", None)
        elif self._prev_pinecone is not _SENTINEL:
            os.environ["PINECONE_API_KEY"] = self._prev_pinecone  # type: ignore[assignment]

        # Clear settings cache so subsequent live calls reload from .env
        try:
            from src.config import get_settings   # noqa: PLC0415
            get_settings.cache_clear()
        except Exception:
            pass

        logger.info("[dry_run] Deactivated.")

    # ── Decorator form ────────────────────────────────────────────────────

    def __call__(self, func: Any) -> Any:
        """Use ``DryRunContext()`` as a function decorator."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with DryRunContext():
                return func(*args, **kwargs)
        return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def dry_run() -> DryRunContext:
    """Return a new :class:`DryRunContext` (same as calling ``DryRunContext()``)."""
    return DryRunContext()


def activate() -> None:
    """Permanently activate dry run mode for the current process.

    Useful in test scripts where you want dry run for the entire lifetime::

        import src.dry_run as dry_run
        dry_run.activate()
        # everything from here on is mocked
    """
    global _DRY_RUN_ACTIVE   # noqa: PLW0603
    if _DRY_RUN_ACTIVE:
        logger.debug("[dry_run] Already active — activate() is a no-op.")
        return
    # Store the context so deactivate() can undo it
    _PERSISTENT_CTX = DryRunContext()
    _PERSISTENT_CTX.__enter__()
    globals()["_PERSISTENT_CTX"] = _PERSISTENT_CTX


def deactivate() -> None:
    """Deactivate dry run mode previously started by :func:`activate`."""
    ctx = globals().get("_PERSISTENT_CTX")
    if ctx is not None:
        ctx.__exit__(None, None, None)
        globals().pop("_PERSISTENT_CTX", None)
    else:
        logger.warning("[dry_run] deactivate() called but no activate() ctx found.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_QUERIES = [
    "نماز کی نیت کا طریقہ کیا ہے؟",
    "زکوٰۃ کا نصاب کتنا ہے؟",
    "پائتھن پروگرامنگ کیسے سیکھیں؟",   # out-of-domain — guardrail test
]


def _run_query_stage(questions: list[str], top_k: int, validate: bool, as_json: bool) -> list[dict]:
    """Run questions through guarded_query inside a DryRunContext."""
    from src.pipeline.guardrails import GuardrailConfig, guarded_query   # noqa: PLC0415

    cfg = GuardrailConfig(
        min_context_score=0.05,
        min_top_score=0.05,
        min_overlap_ratio=0.03,
        min_urdu_ratio=0.20,
    )
    results = []
    for q in questions:
        logger.info("Query: %s", q)
        try:
            gr = guarded_query(q, config=cfg, top_k=top_k)
            entry: dict = {
                "question":        q,
                "answer":          gr.answer,
                "guardrail_hits":  gr.guardrail_hits,
                "passed_preflight": gr.passed_preflight,
                "num_chunks":      gr.num_chunks,
                "timings":         gr.timings,
                "retrieved": [
                    {
                        "rank":      i + 1,
                        "score":     s.get("score", s.get("score", 0.0)) if isinstance(s, dict) else 0.0,
                        "category":  (s.get("category", "—") if isinstance(s, dict) else getattr(s, "category", "—")),
                        "question":  (s.get("question", "")[:80] if isinstance(s, dict) else ""),
                    }
                    for i, s in enumerate(gr.sources[:5])
                ],
                "error": "",
            }
            if validate:
                from src.pipeline.output_validator import validate as ov_validate  # noqa: PLC0415
                report = ov_validate(gr.answer, gr.sources)
                entry["validation"] = {
                    "valid":  report["valid"],
                    "issues": [i["code"] for i in report["issues"]],
                    "scores": report["scores"],
                }
        except Exception as exc:
            logger.exception("Query failed: %s", q)
            entry = {"question": q, "answer": "", "error": str(exc)}
        results.append(entry)
    return results


def _run_ingest_stage(as_json: bool) -> dict:
    """Run stages 1–4 of the orchestrator in dry-run mode."""
    from orchestrator import orchestrate   # noqa: PLC0415
    import tempfile, pathlib   # noqa: PLC0415, E401

    work_dir = pathlib.Path(tempfile.mkdtemp(prefix="dry_run_"))
    logger.info("Ingest dry-run work_dir: %s", work_dir)
    results = orchestrate(
        data_root=pathlib.Path("data"),
        work_dir=work_dir,
        batch_size=10,
        requested_stages={1, 2, 3, 4},
        force=False,
        fail_fast=False,
        dry_run=True,
    )
    return {"stages_run": [r.stage for r in results], "statuses": {r.stage: r.status for r in results}}


def _print_query_result(entry: dict, idx: int, as_json: bool) -> None:
    if as_json:
        return
    W = 76
    print(f"\n{'─' * W}")
    print(f"  [{idx}] {entry['question']}")
    print(f"{'─' * W}")
    print(f"  Preflight passed : {entry.get('passed_preflight', '—')}")
    print(f"  Guardrail hits   : {entry.get('guardrail_hits', []) or 'none'}")
    print(f"  Chunks retrieved : {entry.get('num_chunks', '—')}")

    print(f"\n  Retrieved (top {len(entry.get('retrieved', []))}):")
    for r in entry.get("retrieved", []):
        print(f"    [{r['rank']}] score={r['score']:.4f}  cat={r['category']}  Q: {r['question'][:60]}")

    print(f"\n  Answer:")
    for line in (entry.get("answer") or "(none)").split("\n"):
        print(f"    {line}")

    if "validation" in entry:
        v = entry["validation"]
        status = "VALID" if v["valid"] else "INVALID"
        print(f"\n  Validation: {status}  issues={v['issues']}")
        s = v["scores"]
        print(f"  Scores: grounding={s['groundedness']:.2f}  urdu={s['urdu_ratio']:.2f}  "
              f"halluc_risk={s['hallucination_risk']:.2f}")

    if entry.get("error"):
        print(f"\n  ERROR: {entry['error']}")


def _cli() -> None:
    import argparse   # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Dry run mode — validates pipeline logic without API calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", "-q",
        metavar="TEXT",
        action="append",
        dest="queries",
        help="Urdu query to run (repeatable). Defaults to 3 built-in sample queries.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int, default=5,
        metavar="N",
        help="Number of results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--stages",
        choices=["query", "ingest", "all"],
        default="query",
        help="Which pipeline stages to dry-run (default: query).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run output validator on each answer.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON instead of formatted text.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not args.verbose:
        for _lib in ("httpx", "httpcore", "urllib3", "pinecone"):
            logging.getLogger(_lib).setLevel(logging.WARNING)

    questions = args.queries or _SAMPLE_QUERIES
    report: dict = {"mode": "dry_run", "stages": args.stages}

    with DryRunContext():
        if not args.as_json:
            print("\n" + "═" * 76)
            print("  DRY RUN MODE  —  no OpenAI or Pinecone calls")
            print(f"  Patches active: {len(_PATCH_TABLE)}  |  MOCK_CORPUS: {len(MOCK_CORPUS)} docs")
            print("═" * 76)

        if args.stages in ("query", "all"):
            if not args.as_json:
                print(f"\n  Running {len(questions)} query(ies) through guarded_query()…")
            query_results = _run_query_stage(questions, args.top_k, args.validate, args.as_json)
            report["queries"] = query_results
            if not args.as_json:
                for i, entry in enumerate(query_results, 1):
                    _print_query_result(entry, i, args.as_json)

        if args.stages in ("ingest", "all"):
            if not args.as_json:
                print("\n  Running ingest stages 1–4 (orchestrator dry-run mode)…")
            try:
                ingest_result = _run_ingest_stage(args.as_json)
                report["ingest"] = ingest_result
                if not args.as_json:
                    print(f"  Ingest stages: {ingest_result}")
            except Exception as exc:
                logger.warning("Ingest dry-run failed: %s", exc)
                report["ingest"] = {"error": str(exc)}

    # ── Summary ───────────────────────────────────────────────────────────
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print()
        print("═" * 76)
        q_results = report.get("queries", [])
        errors    = [r for r in q_results if r.get("error")]
        print(f"  Results: {len(q_results)} queries  |  {len(errors)} errors")
        if errors:
            for e in errors:
                print(f"    ✗ {e['question'][:60]}: {e['error']}")
        else:
            print("  ✓ All queries completed without exceptions.")
        print("═" * 76)
        print()

    sys.exit(1 if any(r.get("error") for r in report.get("queries", [])) else 0)


if __name__ == "__main__":
    _cli()
