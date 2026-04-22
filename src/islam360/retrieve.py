"""Islam360 retrieval pipeline.

Two entry points
----------------

``Islam360Retriever.retrieve_fast()`` — DEFAULT (settings.islam360_use_fast_path)
    1.  Light canonicalise (Urdu normalise; quick LLM Roman→Urdu translate
        only if the query has no Arabic-script chars at all)
    2.  BM25 full-text + BM25 question-field on the canonicalised query
    3.  Combined score = full + boost * question  (linear, no rerank)
    4.  Floor check: top-1 must score ≥ ``islam360_fast_min_bm25_score``
    5.  Synthesize answer from top-k

    No LLM query-rewrite (it drifts the query). No dense vectors. No LLM
    re-rank. No strict 0.55 gate. This mirrors the simple BM25s setup that
    worked in <2s with high precision on Urdu queries.

``Islam360Retriever.retrieve()`` — heavy pipeline (kept for ablation):
    rewrite → embed → Pinecone dense → BM25 ×2 → RRF → dedupe → LLM rerank
    → strict gate → synthesize.  Slower; layered LLM steps tend to drift
    valid Urdu hits out of the top-5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import concurrent.futures as _cf
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.config import get_settings
from src.embedding.embedder import embed_query_islam360
from src.indexing.pinecone_store import ensure_serverless_index
from src.islam360.query_rewrite import rewrite_query
from src.islam360.url_index import (
    ALL_SECTS,
    SECT_AHLE_HADITH,
    SECT_BARELVI,
    SECT_DEOBANDI,
    SECT_TO_SOURCES,
    SOURCE_TO_SECT,
    get_sect_for_id,
    get_source_for_id,
)
from src.preprocessing.urdu_normalizer import normalize_urdu
from src.retrieval.bm25_index import BM25Corpus, _tokenize as _bm25_tokenize

logger = logging.getLogger(__name__)

NO_RELEVANT_FATWA = "No relevant fatwa found."

# Detect Arabic-script characters (Urdu / Arabic / Persian). If a query has
# at least one such char we skip the LLM canonicalise and pass through to
# BM25 directly with just Urdu normalisation.
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

_bm360_lock = __import__("threading").Lock()
_bm360_corpus: BM25Corpus | None = None


# ── BM25 corpus singleton ────────────────────────────────────────────────────

def get_islam360_bm25() -> BM25Corpus:
    global _bm360_corpus
    if _bm360_corpus is not None:
        return _bm360_corpus
    with _bm360_lock:
        if _bm360_corpus is None:
            from pathlib import Path

            from src.islam360.loader import load_islam360_records, records_to_bm25_docs

            settings = get_settings()
            p = Path(settings.islam360_bm25_cache_path)
            try:
                _bm360_corpus = BM25Corpus.load(p)
            except FileNotFoundError:
                docs = records_to_bm25_docs(load_islam360_records())
                if not docs:
                    raise RuntimeError(
                        f"No records under {settings.islam360_data_dir}. "
                        "Add CSV/JSON fatwas and run: python scripts/ingest_islam360.py"
                    )
                _bm360_corpus = BM25Corpus.build(docs)
                try:
                    _bm360_corpus.save(p)
                except OSError as e:
                    logger.warning("Could not save Islam360 BM25 cache: %s", e)
        return _bm360_corpus


# ── Pinecone dense ───────────────────────────────────────────────────────────

def _pinecone_filter(language: str | None) -> dict[str, Any] | None:
    """Permissive filter — corpus + (optional) language only.

    We deliberately omit any category filter: Islam360 CSVs use raw text
    labels (e.g. ``"نماز"``, ``"عبادات"``) that don't match the LLM's
    short uppercase intent codes (``"NAMAZ"``).  Filtering on those zeroed
    out the dense candidate pool for most queries.
    """
    parts: list[dict[str, Any]] = [{"corpus_source": {"$eq": "islam360"}}]
    if language and language not in ("mixed", "unknown", ""):
        parts.append({"language": {"$eq": language}})
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}


def _dense_islam360(
    query_vec: list[float],
    top_k: int,
    flt: dict[str, Any] | None,
) -> dict[str, tuple[float, dict[str, Any]]]:
    settings = get_settings()
    idx = ensure_serverless_index(
        settings.islam360_pinecone_index,
        settings.islam360_embedding_dimensions,
    )
    kwargs: dict[str, Any] = {
        "vector": query_vec,
        "top_k": top_k,
        "include_metadata": True,
    }
    if flt:
        kwargs["filter"] = flt
    resp = idx.query(**kwargs)
    return {
        m.id: (float(m.score or 0), dict(m.metadata or {}))
        for m in resp.matches
    }


# ── Sparse (BM25 full-text + question-field boost) ───────────────────────────

def _sparse_islam360(
    keyword_query: str,
    top_k: int,
    corpus: BM25Corpus,
    *,
    language: str | None,
) -> dict[str, tuple[float, dict[str, Any]]]:
    """BM25 full-text hits filtered to islam360 corpus."""
    hits = corpus.search(keyword_query, top_k=top_k * 3)
    out: dict[str, tuple[float, dict[str, Any]]] = {}
    for h in hits:
        if h["score"] <= 0:
            continue
        meta = h.get("metadata") or {}
        if meta.get("corpus_source", "") != "islam360":
            continue
        if language and language not in ("mixed", "unknown", ""):
            mlang = meta.get("language", "")
            if mlang and mlang != language:
                continue
        out[h["id"]] = (float(h["score"]), meta)
        if len(out) >= top_k:
            break
    return out


def _question_boost_ids(
    keyword_query: str,
    top_k: int,
    corpus: BM25Corpus,
) -> dict[str, float]:
    """BM25 over question field only — filtered to islam360 corpus."""
    raw = corpus.score_questions(keyword_query, top_k=top_k * 4)
    if not raw:
        return {}
    # Filter to islam360 docs by walking the underlying doc list.
    keep: dict[str, float] = {}
    id_to_corpus = {
        d.get("id"): d.get("corpus_source", "")
        for d in getattr(corpus, "_docs", [])
    }
    for cid, score in raw.items():
        if id_to_corpus.get(cid) == "islam360":
            keep[cid] = score
        if len(keep) >= top_k:
            break
    return keep


# ── Fusion (Reciprocal Rank Fusion) ──────────────────────────────────────────

def _rrf(rankings: list[list[str]], *, k: int = 60) -> dict[str, float]:
    """Standard RRF: score(d) = Σ 1 / (k + rank_i(d))."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return out


def _fuse(
    dense: dict[str, tuple[float, dict]],
    sparse: dict[str, tuple[float, dict]],
    q_boost: dict[str, float],
) -> list[tuple[str, float, dict]]:
    """RRF over dense + sparse + question-only ranks; metadata pulled from
    whichever source supplied it (dense preferred for richer payload)."""
    dense_rank = sorted(dense.keys(), key=lambda c: -dense[c][0])
    sparse_rank = sorted(sparse.keys(), key=lambda c: -sparse[c][0])
    qboost_rank = sorted(q_boost.keys(), key=lambda c: -q_boost[c])

    fused = _rrf([dense_rank, sparse_rank, qboost_rank])

    out: list[tuple[str, float, dict]] = []
    for cid, score in fused.items():
        meta = (
            dense[cid][1] if cid in dense
            else sparse[cid][1] if cid in sparse
            else {}
        )
        out.append((cid, score, meta))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ── Rare-anchor must-match filter ────────────────────────────────────────────
#
# Failure mode this guards against: a query like "bandar ko hath lagane se
# kapre napak ho jate hai?" (→ بندر کو ہاتھ لگانے سے کپڑے ناپاک ہوتے ہیں) has
# one subject-specific anchor (بندر) and five structural/filler tokens
# (ہاتھ, لگانے, کپڑے, ناپاک, ہوتے). BM25 is additive, so a fatwa about
# DOGS that shares 5 of those 6 tokens outscores any sparse-match on بندر
# and the synth LLM dutifully writes a "monkey" answer from dog rulings.
#
# Rule: every chosen candidate must contain at least ONE query token that
# is NOT already shared by the majority of the candidate pool. If none do,
# the query is asking about a topic the corpus likely does not cover, and
# we return "no relevant fatwa" instead of confabulating.

# Urdu/Arabic grammatical fillers that carry no topical signal. Kept
# small on purpose — we only strip the obvious ones. Over-stripping
# would remove legitimate anchors.
_URDU_FILLERS: frozenset[str] = frozenset({
    "کا", "کی", "کے", "کو", "سے", "پر", "میں", "تک", "یا", "اور",
    "ہے", "ہیں", "ہو", "ہوا", "ہوئی", "ہوئے", "ہوتا", "ہوتی", "ہوتے",
    "ہوجاتا", "ہوجاتی", "ہوجاتے", "ہوجائے", "ہوگا", "ہوگی", "ہوں",
    "جاتا", "جاتی", "جاتے", "جائے", "جایے", "گیا", "گئی", "گئے",
    "یہ", "وہ", "اس", "اسی", "ان", "انہیں", "انہی", "ہم", "تم", "میں",
    "کب", "کس", "کیا", "کیوں", "نہیں", "اگر", "تو", "بھی", "تھا", "تھی", "تھے",
    # Common interrogative/command verb forms.  These are high-frequency
    # Urdu verbs that appear in nearly every fatwa question ("what should
    # one recite/do/say/give?") but are NOT topic-distinguishing.  Keeping
    # them out of the anchor candidate pool prevents the bug where a
    # query like "جادو ہو جائے تو کیا پڑھے" promotes ``پڑھے`` to an
    # anchor (because it's rare in a pool saturated with magic fatwas)
    # and then filters out every magic fatwa whose question doesn't
    # happen to use the word ``پڑھے`` — leaving only tangential fatwas
    # that do.  Stripping these from the anchor candidate set ONLY
    # (BM25 ranking still sees them) fixes that failure mode.
    "اوپر", "نیچے", "ساتھ", "بعد", "پہلے", "دوران", "طرح", "طرف",
    "پڑھے", "پڑھا", "پڑھی", "پڑھنا", "پڑھیں", "پڑھتا", "پڑھتی",
    "پڑھتے", "پڑھنے", "پڑھو",
    "کرے", "کریں", "کرنا", "کرتا", "کرتی", "کرتے", "کرنے", "کرو", "کروں",
    "کہے", "کہیں", "کہنا", "کہتا", "کہتی", "کہتے", "کہنے", "کہا", "کہی",
    "دے", "دیں", "دینا", "دیتا", "دیتی", "دیتے", "دینے", "دیا", "دی",
    "لے", "لیں", "لینا", "لیتا", "لیتی", "لیتے", "لینے", "لیا", "لی", "لو",
    "ہونا", "ہونے", "ہوجانا", "ہوجائیں", "ہوجاؤ",
    "آئے", "آنا", "آتا", "آتی", "آتے", "آئیں", "آیا", "آئی",
    "کتنا", "کتنی", "کتنے", "کونسا", "کونسی", "کون", "کہاں", "کیسے", "کیسا", "کیسی",
    "مجھے", "آپ", "آپکا", "آپکی", "آپکے", "میرا", "میری", "میرے",
    "was", "were", "the", "and", "for", "with", "that", "this",
})


def _extract_anchor_tokens(
    query: str,
    pool: list[dict[str, Any]],
    *,
    pool_size: int = 30,
    max_share: float = 0.30,
) -> tuple[list[str], dict[str, int]]:
    """Return query tokens that appear in fewer than ``max_share`` of the pool.

    These are the topic-distinguishing tokens — typically the subject
    noun of the question (``بندر``, ``شیعہ``, ``کتا``).  Tokens that
    are either fillers (``کا``, ``ہے``) or shared by most pool entries
    (verbs/adjectives like ``لگانے``, ``ناپاک`` that form the structural
    template of the question) are NOT anchors.

    The default threshold is 30 %% — empirically this cleanly separates
    subject nouns (typically 5–15 %% of the pool) from structural verbs
    (35–90 %%).  Raising it causes structural words to leak through;
    lowering it misses legitimate niche subjects.

    Returns
    -------
    (anchors, df_map)
        ``anchors`` — list of rare query tokens; ``df_map`` — every query
        token's pool document frequency (for logging).
    """
    if not pool:
        return [], {}

    pool_sample = pool[: max(1, min(pool_size, len(pool)))]
    q_tokens = [t for t in _bm25_tokenize(query) if t not in _URDU_FILLERS and len(t) >= 2]
    if not q_tokens:
        return [], {}

    # Pre-tokenize each pool entry once (reuses BM25-equivalent tokenizer
    # so normalisation is consistent with what the index saw).
    pool_token_sets: list[set[str]] = []
    for c in pool_sample:
        m = c.get("metadata") or {}
        full = (c.get("text") or f"{m.get('question','')}\n{m.get('answer','')}")
        pool_token_sets.append(set(_bm25_tokenize(full)))

    sz = len(pool_sample)
    df_map: dict[str, int] = {}
    for t in q_tokens:
        df_map[t] = sum(1 for s in pool_token_sets if t in s)

    threshold = sz * max_share
    # df=0 tokens are NOT anchors — they match nothing in the pool,
    # so keeping them would force every candidate to fail the filter.
    # This happens in practice when the fiqh-synonym expander
    # (``_expand_fiqh_synonyms``) appends a spelling variant that is
    # absent from the current sect's corpus (e.g. ``زکات`` added to
    # a pool whose fatwas uniformly use ``زکوٰۃ``).  Those tokens
    # should be treated as "query-expansion noise" rather than as
    # a distinguishing requirement.
    anchors = [t for t, d in df_map.items() if 0 < d < threshold]
    return anchors, df_map


def _filter_by_rare_anchors(
    chosen: list[dict[str, Any]],
    anchors: list[str],
    *,
    min_hits: int = 1,
    mode: str = "question",
) -> list[dict[str, Any]]:
    """Keep candidates that contain ≥ ``min_hits`` distinct anchor tokens.

    ``mode="question"`` (default) checks only the QUESTION field — a
    sharp topical signal that avoids false positives from long answer
    bodies that *incidentally* mention the anchor once.

    ``mode="body"`` checks the full (question + answer) text — useful
    as a graceful fallback when Q-only returns empty, typically because
    of Urdu morphological variation (e.g. query uses ``ہنسنا`` but the
    fatwa's question uses ``ہنسی``/``قہقہہ`` while the body expands
    to the full word).  We fall back to this rather than return
    no_match so the LLM reranker gets a chance to judge relevance.

    Each candidate gets two annotations for downstream scoring:

        ``anchor_hits``      — anchors present in the question
        ``anchor_hits_body`` — anchors present in the full body
    """
    if not anchors:
        return chosen
    anchor_set = set(anchors)
    kept: list[dict[str, Any]] = []
    for c in chosen:
        m = c.get("metadata") or {}
        q_text = str(m.get("question", "") or "")
        full = c.get("text") or f"{q_text}\n{m.get('answer','')}"
        q_tokens = set(_bm25_tokenize(q_text))
        full_tokens = set(_bm25_tokenize(full))
        q_hit = len(q_tokens & anchor_set)
        body_hit = len(full_tokens & anchor_set)
        c["anchor_hits"] = q_hit
        c["anchor_hits_body"] = body_hit
        if mode == "question":
            passes = q_hit >= min_hits
        else:  # "body"
            passes = body_hit >= min_hits
        if passes:
            kept.append(c)
    return kept


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop near-duplicate fatwas (same Q + A)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for it in items:
        m = it.get("metadata") or {}
        key = hashlib.sha1(
            (str(m.get("question", ""))[:200] + "|" + str(m.get("answer", ""))[:400])
            .encode("utf-8", "ignore")
        ).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        kept.append(it)
    return kept


# ── LLM Re-ranker (Q-to-Q centric) ───────────────────────────────────────────

RERANK_SYSTEM = """You score how well each fatwa CANDIDATE answers the SAME
fiqh issue as the user's QUESTION.

Scoring rubric (0.0–1.0):
- 0.90–1.00: Same masala. The candidate's QUESTION is essentially the user's question.
- 0.65–0.89: Same fiqh sub-topic and the answer addresses what the user asked.
- 0.35–0.64: Related fiqh area but a different specific ruling.
- 0.00–0.34: Tangential or off-topic.

Rules:
- Compare QUESTION-to-QUESTION first.  Only use the answer to confirm topic.
- Penalize candidates whose question is about a different masala even if the
  answer happens to mention the user's term in passing.
- Be strict: if you would not show this fatwa to the user, score < 0.35.

Return ONLY JSON: {"scores": [<float>, ...]} — one per candidate, in order.
"""


def _rerank_llm(
    user_question: str,
    items: list[dict[str, Any]],
) -> list[float]:
    if not items:
        return []
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, max_retries=1, timeout=45.0)
    blocks = []
    for i, it in enumerate(items, 1):
        m = it.get("metadata") or {}
        q = str(m.get("question", "") or "")[:600]
        a = str(m.get("answer", "") or "")[:500]
        blocks.append(f"[{i}] CANDIDATE QUESTION:\n{q}\nCANDIDATE ANSWER (excerpt):\n{a}")
    user = (
        f"USER QUESTION:\n{user_question}\n\nCANDIDATES:\n"
        + "\n\n".join(blocks)
    )
    try:
        comp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RERANK_SYSTEM},
                {"role": "user", "content": user[:14000]},
            ],
        )
        data = json.loads((comp.choices[0].message.content or "{}"))
        scores = data.get("scores")
        if isinstance(scores, list):
            out = [max(0.0, min(1.0, float(x))) for x in scores[: len(items)]]
            while len(out) < len(items):
                out.append(0.0)
            return out
    except Exception as exc:
        logger.warning("LLM re-rank failed: %s — using fusion order", exc)
    # Fallback: gentle decay so we still rank, but flag uncertainty.
    return [max(0.0, 0.55 - i * 0.03) for i in range(len(items))]


# ── BM25-first fast path ─────────────────────────────────────────────────────

# Quick deterministic Roman-Urdu → Urdu map for the most common Islamic
# terms.  Used as a first pass before any (optional) LLM translation, so
# that queries like "namaz ka treeqa" are repaired without an API round-
# trip for the most-frequent vocabulary.  Keep this list focused: only
# unambiguous religious/legal terms.
_ROMAN_URDU_MAP: dict[str, str] = {
    # worship
    "namaz": "نماز", "namaaz": "نماز", "salah": "نماز", "salat": "نماز",
    "wudu": "وضو", "wuzu": "وضو", "ablution": "وضو",
    "ghusl": "غسل", "tayammum": "تیمم",
    "roza": "روزہ", "rozah": "روزہ", "sawm": "روزہ", "fasting": "روزہ",
    "azan": "اذان", "adhan": "اذان", "iqamah": "اقامہ",
    "masjid": "مسجد", "mosque": "مسجد",
    # hajj / umrah
    "hajj": "حج", "haj": "حج", "umrah": "عمرہ", "ihram": "احرام",
    # finance
    "zakat": "زکوۃ", "zakaat": "زکوۃ", "zakah": "زکوۃ",
    "riba": "سود", "sood": "سود", "interest": "سود",
    "bay": "بیع", "bai": "بیع", "trade": "تجارت",
    # marriage / family
    "nikah": "نکاح", "nikkah": "نکاح", "marriage": "نکاح",
    "talaq": "طلاق", "divorce": "طلاق", "khula": "خلع",
    "mehr": "مہر", "iddat": "عدت", "iddah": "عدت",
    "wirasat": "وراثت", "inheritance": "وراثت", "meeras": "میراث",
    # food
    "halal": "حلال", "haram": "حرام", "makrooh": "مکروہ", "mubah": "مباح",
    # rulings
    "hukm": "حکم", "hukum": "حکم", "ruling": "حکم",
    "fatwa": "فتویٰ", "fatawa": "فتاویٰ",
    "masla": "مسئلہ", "masail": "مسائل",
    "treeqa": "طریقہ", "tareeqa": "طریقہ", "tariqa": "طریقہ", "method": "طریقہ",
    # connectives (only the unambiguous ones)
    "ka": "کا", "ki": "کی", "ke": "کے",
    "kya": "کیا", "kia": "کیا",
    "hai": "ہے", "hain": "ہیں",
}

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _has_arabic_script(text: str) -> bool:
    return bool(_ARABIC_SCRIPT_RE.search(text or ""))


def _quick_roman_to_urdu(text: str) -> str:
    """Token-level deterministic Roman-Urdu → Urdu substitution.

    Only replaces tokens that are in :data:`_ROMAN_URDU_MAP`; everything
    else is left untouched.  This handles the bulk of common queries
    without needing an LLM call.  If the query has no covered tokens at
    all the function returns the original text unchanged.
    """
    if not text:
        return text
    parts = _WORD_RE.split(text)
    matches = _WORD_RE.findall(text)
    out: list[str] = []
    for i, m in enumerate(matches):
        out.append(parts[i])
        out.append(_ROMAN_URDU_MAP.get(m.lower(), m))
    if len(parts) > len(matches):
        out.append(parts[-1])
    return "".join(out)


_CANONICALISE_SYS = """You convert an Islamic-jurisprudence question into a
TERSE Urdu keyword phrase suitable for a BM25 keyword search over a fatwa
corpus.

OUTPUT RULES — these are hard constraints:
1. Output ONLY the keyword phrase. No explanation, no punctuation, no
   quotes, no English.
2. Keep the core Islamic topic words (3–8 Urdu words is ideal): the
   SUBJECT + the specific ACTION/ASPECT being asked about + any
   essential ENTITY (recipient, object, relationship).
3. PRESERVE these even if the user buried them in narrative:
   • WHO the fatwa concerns — a brother, wife, neighbour, non-Muslim,
     scholar, orphan, etc.  ``"can I give zakat to my brother?"`` MUST
     keep ``بھائی``; ``"money given to a non-Muslim"`` MUST keep
     ``غیرمسلم``.
   • WHAT object — land, gold, salary, insurance, etc.
   • Which SPECIFIC ruling aspect — "is it allowed", "is it obligatory",
     "how many", "when", "on what amount".
4. DROP every filler word: salutation ("السلام علیکم"), polite phrases
   ("مہربانی کر کے", "جواب دیں", "بتا دیں", "براہ کرم"), self-reference
   ("میں نے پوچھنا تھا", "مجھے بتائیں"), framing ("کے بارے میں", "سوال
   یہ ہے کہ", "یہ معلوم کرنا ہے") and narrative backstory that doesn't
   change the fiqh question (ages, timing, "late", "old", "small", a
   relative's habits).  These are NOT content — they poison retrieval.
5. Examples:
   • "how to pray" / "namaz ka tareeqa"  →  نماز کا طریقہ
   • "ruling on interest / riba ka hukum" →  سود کا حکم
   • "tell me about divorce / talaq ke baray me bata do" →  طلاق کے احکام
   • "is home loan with interest haram?" →  سودی قرضہ کا حکم
   • "who is zakat obligatory on?" →  زکوٰۃ کس پر فرض ہے
   • "Salam, I wanted to ask — my brother has some land and gets
     wheat, he's getting old and his kids are studying, can I give him
     zakat?" →  بھائی کو زکات دینا (keep the RECIPIENT, drop the
     narrative)
   • "I live in England and receive child benefit, is it permissible
     for a Muslim to take it?" →  غیرمسلم حکومت سے الاؤنس لینا
6. If the input is already Urdu, output it compressed to its core
   keywords (still applying rules 3 and 4).
7. NEVER invent an answer, ruling, or opinion. You only rewrite the
   query."""


def _llm_translate_to_urdu(text: str) -> str:
    """Turn a possibly-mixed query into a terse Urdu keyword phrase.

    Only invoked when the deterministic map left Latin content words
    behind (see :func:`_canonicalise_for_bm25`).  The output is designed
    for BM25 — it is intentionally shorter than a natural translation so
    filler n-grams like ``کے بارے میں بتا دیں`` don't dominate scoring.
    """
    settings = get_settings()
    try:
        client = OpenAI(api_key=settings.openai_api_key, max_retries=1, timeout=15.0)
        comp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=60,
            messages=[
                {"role": "system", "content": _CANONICALISE_SYS},
                {"role": "user", "content": text[:500]},
            ],
        )
        out = (comp.choices[0].message.content or "").strip()
        # Strip stray quotes / trailing punctuation the model sometimes
        # adds despite the instructions.
        out = out.strip("\"'،؟?.! \t\r\n")
        return out or text
    except Exception as exc:
        logger.warning("Roman→Urdu LLM translate failed: %s — using deterministic map only", exc)
        return text


_LATIN_TOKEN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
_LATIN_CONTENT_RE = re.compile(r"\b[a-zA-Z]{3,}\b")

# Above this many content tokens, we treat a pure-Urdu query as a
# verbose narrative and invoke the LLM compressor.  Calibrated from
# the corpus: legitimate fiqh questions average 3–8 content tokens;
# anything above ~12 is virtually always narrative + backstory that
# injects noise into BM25.  Set slightly higher to avoid compressing
# tight, specific queries (e.g. a detailed 10-token inheritance case).
_VERBOSE_TOKEN_THRESHOLD = 13


def _count_content_tokens(text: str) -> int:
    """Count non-filler Urdu tokens in *text*.

    Mirrors what BM25 would see as retrieval signal — stopwords and
    short/low-content tokens are excluded so the count reflects the
    query's actual informational density, not its raw length.
    """
    if not text:
        return 0
    tokens = _bm25_tokenize(text)
    return sum(1 for t in tokens if t not in _URDU_FILLERS and len(t) >= 2)


def _is_verbose_urdu_query(text: str) -> bool:
    """True if *text* is a long Urdu narrative that needs compression.

    Two signals jointly indicate narrative noise:
      • content-token count above ``_VERBOSE_TOKEN_THRESHOLD``;
      • the text is primarily Arabic-script (the Roman-Urdu path
        already routes through the LLM whenever needed, so we only
        gate pure-Urdu narratives here).
    """
    if not text or not _has_arabic_script(text):
        return False
    return _count_content_tokens(text) > _VERBOSE_TOKEN_THRESHOLD


def _has_residual_content(text: str) -> bool:
    """True if any Latin-script word ≥3 chars is left after mapping.

    Short tokens (``ka``, ``me``, ``ki``, …) are allowed because they are
    grammatical fillers that carry little retrieval signal. Content
    words that the deterministic map missed (``tareqqa``, ``masjidain``,
    …) are red flags — they are almost always typos/variants of Islamic
    terms and dropping them silently leaves BM25 with too-generic
    queries (e.g. "نماز کا ?" matches every prayer-related fatwa).
    """
    return bool(_LATIN_CONTENT_RE.search(text or ""))


def _strip_short_latin(text: str) -> str:
    """Remove only short Latin filler tokens (≤2 chars) after mapping.

    By the time this is called the query has already been through the
    map + (if needed) the LLM translator, so anything Latin left is
    filler noise, not content.
    """
    cleaned = _LATIN_TOKEN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _canonicalise_for_bm25(query: str) -> str:
    """Prepare *query* for BM25 lookup.

    Strategy (ordered by cost):

    1. Deterministic roman→Urdu map for common Islamic terms and
       grammatical particles (free, handles the 80 %% case).
    2. LLM-based compression is invoked when EITHER of:
         (a) Latin *content words* (≥3 chars) still remain after the
             map — the input is incompletely translated; or
         (b) the query is a **verbose Urdu narrative** (>
             ``_VERBOSE_TOKEN_THRESHOLD`` content tokens).  Long
             personal questions like ``"السلام علیکم میں نے پوچھنا
             تھا ... میرا بھائی ... کیا زکات دے سکتی ہوں"`` carry
             heavy narrative noise (salutation, backstory, pronouns)
             that poisons BM25 ranking unless compressed to the core
             fiqh question.
       Either way the LLM uses the ``_CANONICALISE_SYS`` prompt which
       is specifically designed to drop fillers and keep only the 3-7
       core Islamic keywords.
    3. Strip any residual short Latin filler tokens (``"ka"`` etc.)
       so BM25 only sees Urdu tokens.
    4. Expand fiqh synonyms (see ``_expand_fiqh_synonyms``).
    5. Normalise Urdu characters.
    """
    q = (query or "").strip()
    if not q:
        return ""

    # 1) cheap deterministic map
    mapped = _quick_roman_to_urdu(q)

    # 2) Decide whether to invoke the LLM compressor.
    settings = get_settings()
    has_residual_latin = _has_residual_content(mapped)
    is_verbose_urdu = _is_verbose_urdu_query(mapped)
    use_llm = (
        settings.islam360_fast_canonicalise_roman
        and (has_residual_latin or is_verbose_urdu)
    )
    if use_llm:
        translated = _llm_translate_to_urdu(q)
        # Keep the LLM output only if it actually produced Urdu AND
        # actually shortened the query (a pass-through isn't useful
        # and might even add LLM-introduced noise).  For the
        # "verbose narrative" trigger we only accept outputs that
        # dropped at least a third of the content tokens, otherwise
        # fall back to the original.
        if translated and _has_arabic_script(translated):
            if is_verbose_urdu and not has_residual_latin:
                before = _count_content_tokens(mapped)
                after = _count_content_tokens(translated)
                if after <= max(3, int(before * 0.66)):
                    logger.info(
                        "[islam360-fast] compressed verbose Urdu: %d → %d tokens",
                        before, after,
                    )
                    mapped = translated
                else:
                    logger.info(
                        "[islam360-fast] LLM compression did not shrink (%d→%d), "
                        "keeping original mapped query",
                        before, after,
                    )
            else:
                mapped = translated
        else:
            logger.info(
                "[islam360-fast] LLM translate did not produce Urdu — falling "
                "back to deterministic map: %r → %r", q, mapped,
            )

    # 3) strip any leftover Latin filler tokens once we have Urdu content.
    if _has_arabic_script(mapped):
        mapped = _strip_short_latin(mapped)

    # 4) expand fiqh synonyms — BM25 is purely lexical, so a query using
    #    casual Urdu ("ہنسی") won't match a fatwa whose title uses the
    #    technical term ("قہقہہ") even though they refer to the same
    #    masala.  We append known synonyms inline so BM25 matches either
    #    variant.  See ``_expand_fiqh_synonyms``.
    mapped = _expand_fiqh_synonyms(mapped)

    # 4b) LLM-powered dynamic expansion for the long tail of fiqh
    #     terms (خلع, عدت, فطرانہ, ایصالِ ثواب, …) that the
    #     hand-curated dictionary can't enumerate.  Cached per unique
    #     canonical query so three parallel sect retrievals share one
    #     round-trip.  Runs AFTER the deterministic dictionary so we
    #     don't waste an LLM call on queries that are already fully
    #     covered — but more importantly, it sees the deterministic
    #     additions in its "present tokens" set and won't duplicate
    #     them.  Graceful no-op on timeout/error.
    mapped = _llm_expand_query(mapped)

    # 5) normalise and return.
    return normalize_urdu(mapped)


def _rerank_query_for(raw_query: str, canon_query: str) -> str:
    """Return the best form of the query to pass to the LLM reranker.

    For short/normal queries we want the reranker to see exactly what
    the user typed — preserves typos, idiom, sect hints, etc.

    For verbose narratives (where the canonicaliser compressed the
    query), we pass the compressed form so the reranker judges
    candidates against the CORE fiqh question, not the narrative.
    The raw query's backstory ("my brother has land, kids are in
    school, ...") makes the reranker reject candidates that address
    the core masala but don't mention every detail.

    We also strip appended fiqh synonyms (see
    ``_expand_fiqh_synonyms``) — those help BM25 but are noise for
    the LLM, which handles synonymy natively.
    """
    if not raw_query:
        return canon_query or ""
    if not canon_query:
        return raw_query
    # Pick the compressed form iff it's significantly shorter than
    # the raw input (i.e. the compressor actually fired).
    raw_content = _count_content_tokens(raw_query) if _has_arabic_script(raw_query) else 999
    canon_content = _count_content_tokens(canon_query)
    if raw_content > _VERBOSE_TOKEN_THRESHOLD and canon_content < raw_content * 0.66:
        # Strip trailing synonym tokens added by _expand_fiqh_synonyms.
        stripped = _strip_trailing_synonyms(canon_query)
        return stripped or canon_query
    return raw_query


def _strip_trailing_synonyms(canon: str) -> str:
    """Remove appended synonym tokens from a canonicalised query.

    Two expansion stages append synonyms to the END of the BM25
    query: ``_expand_fiqh_synonyms`` (hand-curated dictionary) and
    ``_llm_expand_query`` (dynamic LLM proposal).  For the reranker
    these alternates add no value and can crowd / mislead the
    prompt; return just the "content" prefix up to where the
    expansions began.
    """
    if not canon:
        return canon
    tokens = canon.split()
    cut = len(tokens)

    # 1) Strip LLM-added trailing tokens first — we know them
    #    exactly from the tracker.
    llm_extras = _LLM_EXPAND_TRAILING.get(canon, ())
    if llm_extras:
        llm_set = set(llm_extras)
        while cut > 0 and tokens[cut - 1] in llm_set:
            cut -= 1

    # 2) Then strip hand-curated fiqh-dictionary trailing tokens —
    #    tokens that are synonyms whose partner already appears
    #    earlier in the query.
    while cut > 0:
        t = tokens[cut - 1]
        syns = _FIQH_SYNONYM_INDEX.get(t)
        if not syns:
            break
        if any(s in tokens[: cut - 1] for s in syns):
            cut -= 1
            continue
        break

    return " ".join(tokens[:cut]).strip() or canon


# ── Fiqh-term synonym dictionary ────────────────────────────────────────────
#
# Common Islamic jurisprudence terms that have multiple Urdu / Arabic
# renderings.  BM25 is purely lexical, so without this expansion a query
# like "namaz me hansi" (canonicalised → "نماز میں ہنسی") would fail to
# match a fatwa titled "قہقہہ سے وضو توڑنا" even though both discuss the
# same masala — loud laughter during prayer breaking wudu/namaz.
#
# The map is intentionally conservative: only pairs/groups where the
# terms truly denote the same fiqh concept.  Each group lists synonyms;
# if *any* synonym is present in the canonicalised query, the *others*
# are appended so BM25 can match fatwas using any of the variants.
#
# Adding a new group?  Keep groups small and unambiguous — don't lump
# e.g. "murder" and "assault" together even though they're related.

_FIQH_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    # ════════════════════════════════════════════════════════════════
    # CORE VOCAB-GAP FIXES (proven to move retrieval metrics)
    # ════════════════════════════════════════════════════════════════

    # Laughter in prayer / wudu context.  قہقہہ = loud laughter
    # (the technical fiqh term that affects wudu), ہنسی / ہنسنا
    # = generic laugh/verb. Different corpora use different forms
    # for the SAME masala — textbook case for synonym expansion.
    ("قہقہہ", "قہقہے", "ہنسی", "ہنسنا", "ہنسنے", "ہنسا"),
    # Interest / usury.  Urdu corpora use سود while Arabic-leaning
    # ones use ربا (same concept).
    ("سود", "ربا", "ربوا"),
    # Zakat — multiple attested spellings BM25 treats as distinct.
    ("زکات", "زکوٰۃ", "زکاۃ", "زکوۃ"),
    # Wudu — rare variant spelling with hamza.
    ("وضو", "وضوء"),
    # Magic — Arabic/Urdu pair (proven fix for the "کسی کے اوپر
    # جادو" bug where ahle_hadith corpus uses سحر almost exclusively).
    ("جادو", "سحر"),

    # ════════════════════════════════════════════════════════════════
    # IBADAT — ‘worship’ domain
    # ════════════════════════════════════════════════════════════════
    #
    # Note: we deliberately do NOT expand نماز ↔ صلاۃ because that
    # pulls niche Arabic-titled fatwas (صلاۃ الخسوف etc.) to the top
    # and displaces general namaz fatwas.  Same restraint for ذکر,
    # درود, and قرآن — all appear in too many queries to safely
    # expand.  Only domain-specific terms are included below.

    # Fasting — Urdu/Arabic pair, textbook vocab gap.
    ("روزہ", "روزے", "صوم", "صیام"),
    # Ramadan — orthographic variants.
    ("رمضان", "رمضان المبارک", "رمضان شریف"),
    # Fitr zakat — three attested phrasings for ONE concept.
    # Site-specific: banuri writes "صدقہ فطر"; fatwaqa uses "زکاۃ
    # الفطر"; many users type "فطرانہ".  Without expansion these
    # three pools never meet.
    ("فطرانہ", "صدقہِ فطر", "صدقہ فطر", "زکاۃ الفطر", "زکوٰۃ الفطر"),
    # Sacrificial slaughter on Eid al-Adha.  Arabic term is
    # اضحیہ, Urdu is قربانی — pure vocab gap.
    ("قربانی", "اضحیہ", "عید الاضحیٰ", "عیدالاضحیٰ"),
    # Newborn aqiqah — spelling variants.
    ("عقیقہ", "عقیقت"),
    # Funeral prayer — phrase variants.  "نمازِ جنازہ" and
    # "صلاۃ الجنازہ" refer to the same ritual.
    ("جنازہ", "نماز جنازہ", "نمازِ جنازہ", "صلاۃ الجنازہ"),
    # Missed-prayer/missed-fast make-up.  قضا / قضاء / قضائے
    # are morphological variants of the same fiqh concept.
    ("قضا", "قضاء", "قضائے"),
    # I'tikaf.
    ("اعتکاف", "اعتکافات"),
    # Friday prayer — orthographic variants of same word.
    ("جمعہ", "جمعة", "جمعۃ"),
    # Umrah.
    ("عمرہ", "عمرة"),
    # Supplication — dua / duas with Arabic-variant spelling.
    ("دعا", "دعاء", "دعائیں"),

    # ════════════════════════════════════════════════════════════════
    # TAHARAT — ‘ritual purity’ domain
    # ════════════════════════════════════════════════════════════════
    #
    # ناپاکی / نجاست NOT expanded: نجاست has a precise technical
    # meaning ("najis substance"); ناپاکی is broader everyday usage.
    # Not true synonyms in fiqh contexts.

    # Full bath — جنابت is the state requiring ghusl.
    ("غسل", "غسلِ جنابت", "غسل جنابت", "جنابت"),
    # Dry ablution — shadda variant.
    ("تیمم", "تیمّم"),
    # Menstruation — "حیض" is the masdar; "حیضہ" / "حائضہ" are
    # derived forms used interchangeably in fatwa titles.
    ("حیض", "حیضہ", "حائضہ"),
    # Post-natal bleeding.
    ("نفاس", "نفاسی"),
    # Irregular bleeding.
    ("استحاضہ", "استحاضة"),

    # ════════════════════════════════════════════════════════════════
    # MUNAKAHAT — ‘marriage/divorce’ domain
    # ════════════════════════════════════════════════════════════════

    # Judicial divorce initiated by the wife.  خلع / مخالعہ /
    # فسخِ نکاح are three procedurally distinct routes often
    # conflated in user queries — keep as ONE expansion group
    # because casual users don't distinguish them.
    ("خلع", "مخالعہ", "فسخِ نکاح", "فسخ نکاح"),
    # Waiting period after divorce / widowhood.
    ("عدت", "عدّت", "عدة"),
    # Wife-as-mother oath — ظہار in Urdu, ظہارت as a rarer form.
    ("ظہار", "ظہارت"),
    # Oath of abstinence (husband's vow).
    ("ایلاء", "ایلا"),
    # Wife maintenance.
    ("نفقہ", "نفقة", "نفقات"),
    # Child custody.
    ("حضانت", "حضانة"),
    # Wedding feast.
    ("ولیمہ", "ولیمة"),
    # Dower — singular/plural and Arabic variants.
    ("مہر", "مہور"),

    # ════════════════════════════════════════════════════════════════
    # MU‘AMALAT — ‘financial transactions’ domain
    # ════════════════════════════════════════════════════════════════
    #
    # بیع / ایجارہ / مرابحہ / کفالت NOT expanded to their generic
    # Urdu counterparts (خرید، کرایہ، ضمانت) because those everyday
    # words are far broader than the fiqh contract types.

    # Silent-partnership contract.
    ("مضاربت", "مضاربہ"),
    # Loan.
    ("قرض", "قرضہ"),
    # Mortgage / pledge.  گروی is the Urdu equivalent of رہن.
    ("رہن", "گروی"),
    # Agency.
    ("وکالت", "توکیل"),

    # ════════════════════════════════════════════════════════════════
    # MIRATH — ‘inheritance’ domain
    # ════════════════════════════════════════════════════════════════

    # Inheritance.  فرائض = science of inheritance share allocation.
    ("وراثت", "میراث", "فرائض"),
    # Heirs — plural variants.
    ("وارث", "ورثاء", "ورثہ"),
    # Estate.
    ("ترکہ", "ترکة"),

    # ════════════════════════════════════════════════════════════════
    # UQUBAT — ‘penal / retributive’ domain
    # ════════════════════════════════════════════════════════════════

    # Equal retribution.
    ("قصاص", "قصاصاً"),
    # Blood-money.
    ("دیت", "دیات"),

    # ════════════════════════════════════════════════════════════════
    # SPIRITUAL / UNSEEN domain
    # ════════════════════════════════════════════════════════════════

    # Prophetic protective-recitation practice.
    ("رقیہ", "رُقیہ", "دمِ شرعی", "دم شرعی"),

    # ════════════════════════════════════════════════════════════════
    # OATHS / VOWS / EXPIATION
    # ════════════════════════════════════════════════════════════════

    ("قسم", "حلف"),
    ("نذر", "منت"),
    ("کفارہ", "کفارات"),

    # ════════════════════════════════════════════════════════════════
    # GROUPS DELIBERATELY NOT INCLUDED (failure cases from testing)
    # ════════════════════════════════════════════════════════════════
    #
    #   • نماز / صلاۃ / صلوٰۃ — pulls niche Arabic-titled fatwas
    #     (صلاۃ الخسوف, صلاۃ التسبیح) and displaces general namaz
    #     queries.
    #   • فرض / واجب / سنت / مستحب — these are DISTINCT obligation
    #     levels in fiqh, not synonyms.  Conflating them is a
    #     correctness error, not just a noise issue.
    #   • حدث / ناپاکی — حدث is a specific ritual nullifier;
    #     ناپاکی is general everyday impurity.  Not interchangeable.
    #   • حکم / احکام / مسئلہ — too generic.  Every fiqh query
    #     mentions these so expansion drowns other signal.
    #   • حلال / جائز / حرام / ناجائز — rulings, not topics;
    #     expansion bleeds across unrelated masalas.
    #   • نکاح / شادی — شادی is colloquial for "wedding (event)",
    #     نکاح is the legal contract; not reliable synonyms.
    #   • اجارہ / کرایہ — کرایہ is everyday word for "rent money";
    #     expansion injects far too much noise.
    #   • ذکر / تسبیح / دعا / درود / قرآن — too broad; appear in
    #     most queries; expansion has no upside.
    #   • نظر / نظرِ بد — نظر is a high-frequency everyday word
    #     ("view", "sight"); false-positive rate too high.
)

# Pre-compute: term → set of other synonyms in its group.  Lookup is
# O(tokens_in_query × 1) at query time.
_FIQH_SYNONYM_INDEX: dict[str, tuple[str, ...]] = {}
for _group in _FIQH_SYNONYM_GROUPS:
    for _t in _group:
        _FIQH_SYNONYM_INDEX[_t] = tuple(x for x in _group if x != _t)


def _expand_fiqh_synonyms(text: str) -> str:
    """Append known synonyms for any fiqh term present in *text*.

    BM25 is purely lexical — a fatwa titled ``قہقہہ سے وضو ٹوٹنا``
    will not match a query tokenised as ``ہنسی`` no matter how high its
    term-frequency.  This function closes that gap by appending the
    sibling terms from each triggered synonym group.  Only triggered
    groups are expanded (we never inject tokens out of thin air).

    Original order is preserved — new synonyms go at the end so
    user-intent terms still dominate the BM25 scoring through higher
    query-term-frequency weights.
    """
    if not text:
        return text
    tokens = text.split()
    present = {t for t in tokens if t in _FIQH_SYNONYM_INDEX}
    if not present:
        return text
    extras: list[str] = []
    seen = set(tokens)
    for t in present:
        for syn in _FIQH_SYNONYM_INDEX[t]:
            if syn not in seen:
                extras.append(syn)
                seen.add(syn)
    if not extras:
        return text
    logger.info(
        "[islam360-fast] expanded fiqh synonyms: %s → +%s",
        sorted(present), extras,
    )
    return text + " " + " ".join(extras)


# ── LLM-powered dynamic query expansion ────────────────────────────────────
#
# The hand-curated ``_FIQH_SYNONYM_GROUPS`` above covers only the half-dozen
# most common vocab gaps (قہقہہ/ہنسی, سود/ربا, زکات/زکوٰۃ, وضو/وضوء).
# The long tail of fiqh terminology — خلع, عدة, ظہار, فطرانہ, ایصالِ
# ثواب, ولیمہ, متعہ, حج بدل, قضا نماز — is impractical to curate by
# hand.  So we ask the LLM to propose up to ``islam360_llm_expand_max_tokens``
# additional Urdu tokens that are TRUE synonyms / technical variants of
# terms already in the query, then append those to BM25.
#
# Design constraints:
#   • Strict prompt (synonyms only, never broader/narrower concepts)
#     because the downstream rerank gate is strong but not infinite —
#     if expansion drifts into a different masala the user sees
#     irrelevant results.
#   • Hard timeout (``islam360_llm_expand_timeout_s``) with empty
#     fallback so a slow API never blocks retrieval.
#   • In-process LRU cache (``_LLM_EXPAND_CACHE_SIZE``) so the three
#     parallel sect calls share a single LLM invocation per unique
#     query.  Flask warm-up empties it; otherwise it lives per-process.

_LLM_EXPAND_SYS = """You are a FIQH (Islamic jurisprudence) QUERY EXPANDER for
a hybrid Urdu/Arabic fatwa search engine.  Your job is to make the
BM25 keyword search succeed across four fatwa corpora that each use
slightly different vocabulary:

  • Banuri (Deobandi)     — mixed Urdu / classical Arabic terms
  • UrduFatwa (Barelvi)   — mostly Urdu, ي-dotted spellings
  • IslamQA (Ahle-Hadith) — Arabic-leaning, many Arabic-only titles
  • FatwaQA (Ahle-Hadith) — Urdu with Arabic technical vocabulary

THE PROBLEM YOU SOLVE: A user types a question using ONE vocabulary;
the answer they need lives in a corpus using a DIFFERENT vocabulary
for the same masala (e.g. user says "جادو", fatwa says "سحر"; user
says "ڈراؤنا خواب", fatwa says "حُلمٌ سیئ" or "کابوس"; user says
"فطرانہ", fatwa says "زکاۃ الفطر").  You close that gap.

╔══════════════════════════════════════════════════════════════════╗
║  TASK PROTOCOL                                                    ║
╚══════════════════════════════════════════════════════════════════╝

STEP 1 — IDENTIFY 2-5 KEY FIQH CONCEPTS IN THE QUERY.
  Ignore particles / question words / pronouns.  Focus on the
  CONTENT nouns and verbs that a fatwa would answer.  Most queries
  have 2-4 key concepts.

STEP 2 — FOR EACH CONCEPT, LIST REALISTIC ALTERNATIVE PHRASINGS.
  For each concept propose 1-4 alternative ways a scholar or
  different fatwa-site might write that SAME concept.  Types of
  alternatives that are VALID:
    a) Arabic ↔ Urdu equivalent  (روزہ ↔ صوم،  جادو ↔ سحر،
       خوابِ بد ↔ کابوس ↔ ڈراؤنا خواب،  فطرانہ ↔ زکاۃ الفطر).
    b) Colloquial ↔ technical fiqh term  (ہنسی ↔ قہقہہ،
       سود ↔ ربا،  بچاؤ ↔ حفاظت ↔ پناہ).
    c) Orthographic variants of same word  (زکات / زکوٰۃ / زکاۃ،
       عدت / عدّت / عدة،  وضو / وضوء).
    d) Morphological variants of a root already present
       (حیض ↔ حیضہ ↔ حائضہ،  قرض ↔ قرضہ،  خلع ↔ مخالعہ).
    e) Standard procedurally-adjacent fiqh term that shares
       the SAME masala  (خلع ↔ فسخِ نکاح،  تدابیر ↔ اذکار
       ↔ وظائف when the query is clearly about protective
       recitations,  دعائے خوف ↔ مسنون دعا ↔ ماثور دعا).

STEP 3 — WRITE THE OUTPUT AS STRICT JSON.

  Output MUST be a single JSON object:

    {
      "concepts": [
        {"term": "<concept from query>", "alternates": ["<alt1>", "<alt2>", ...]},
        {"term": "<concept from query>", "alternates": ["<alt1>", ...]},
        ...
      ]
    }

  Rules on the JSON:
    • 2-5 concept entries typical; up to 8 allowed.
    • Each `alternates` list has 0-4 items.  Empty list is OK
      if that concept has no reliable alternative.
    • Every alternate is a SHORT Urdu/Arabic phrase (1-3 words).
    • No English, no transliteration, no Latin-script tokens.
    • No explanation outside the JSON.

╔══════════════════════════════════════════════════════════════════╗
║  FIQH DOMAIN REFERENCE (stay INSIDE the query's domain)           ║
╚══════════════════════════════════════════════════════════════════╝

  ● IBADAT — نماز، روزہ، زکات، حج، عمرہ، اعتکاف، قربانی، عقیقہ،
      تراویح، تہجد، جمعہ، صدقہ، فطرانہ، جنازہ، قضا، اذان، اقامت
  ● TAHARAT — وضو، غسل، تیمم، جنابت، حیض، نفاس، استحاضہ، نجاست
  ● MUNAKAHAT — نکاح، مہر، ولیمہ، طلاق، خلع، فسخ، عدت، ظہار،
      ایلاء، نفقہ، حضانت، رضاعت، ولی
  ● MU'AMALAT — بیع، خرید و فروخت، اجارہ، سود، ربا، قرض، رہن،
      مضاربت، مشارکت، مرابحہ، وکالت، ضمانت، کفالت، وقف
  ● MIRATH / FARA'ID — وراثت، میراث، ترکہ، وارث، حصہ، عصبہ
  ● UQUBAT — حدود، قصاص، دیت، تعزیر، زنا، قذف، چوری، شراب
  ● DHABH / AT'IMAH — ذبح، ذبیحہ، حلال، حرام، مردار
  ● AQIDAH — ایمان، توحید، شرک، بدعت، کفر، نفاق
  ● RUQYAH / UNSEEN — جادو، سحر، جن، رقیہ، دمِ شرعی، تعویذ،
      خواب، کابوس، خوف، وسوسہ، دعائے خوف، ماثور دعا، وظائف
  ● OATHS & VOWS — قسم، حلف، نذر، منت، کفارہ

╔══════════════════════════════════════════════════════════════════╗
║  HARD PROHIBITIONS (these break the search — never do them)       ║
╚══════════════════════════════════════════════════════════════════╝

1. NO ANSWER-DOMAIN TERMS.  Don't list content that would appear
   ONLY in the answer.  For "what to recite when magic happens" do
   NOT output قرآن، آیات، معوذتین، منزل — those are what the
   answer cites, not synonyms of query words.  Your task is
   query-vocabulary expansion, NOT answer prediction.

2. NO ENTITY INVENTION.  If the query says "کسی کے اوپر"
   (on someone), that is a generic pronoun — do NOT expand to
   عزیزہ، بیٹا، بیوی، ماں، or any specific person.

3. NO CROSS-DOMAIN DRIFT.  If the query is MU'AMALAT (finance), do
   NOT emit IBADAT or TAHARAT terms.  All concepts you emit must
   share the same fiqh domain as the query.

4. NO INTERROGATIVE EXPANSION.  کیا، کیسے، کب، کیوں، کہاں،
   کون — these are question markers; they have no fiqh synonyms.
   Don't list them as concepts.

5. NO OBLIGATION-LEVEL SWAPS.  فرض / واجب / سنت / مستحب / مکروہ /
   حرام are DISTINCT fiqh categories.  Never treat them as
   synonyms of one another.

6. NO GENERIC RULING WORDS.  حکم، جواز، حلال، حرام — these
   appear in every fatwa and their expansion adds pure noise.

7. DEFAULT BEHAVIOUR: BE HELPFUL, NOT SILENT.  Most real queries
   have AT LEAST one expandable concept.  Only emit an empty
   `concepts` array when the query is genuinely just particles /
   question words.

╔══════════════════════════════════════════════════════════════════╗
║  WORKED EXAMPLES (input → output JSON)                            ║
╚══════════════════════════════════════════════════════════════════╝

── IBADAT ────────────────────────────────────────────────────────

  INPUT:  نماز میں ہنسی کا حکم
  OUTPUT: {"concepts":[
    {"term":"ہنسی","alternates":["قہقہہ","قہقہے","ہنسنا"]}
  ]}

  INPUT:  فطرانہ کی مقدار کتنی ہے
  OUTPUT: {"concepts":[
    {"term":"فطرانہ","alternates":["صدقہ فطر","زکاۃ الفطر","زکوٰۃ الفطر"]},
    {"term":"مقدار","alternates":["کتنا","اندازہ"]}
  ]}

  INPUT:  قربانی کے مسائل
  OUTPUT: {"concepts":[
    {"term":"قربانی","alternates":["اضحیہ","عید الاضحیٰ","ذبح"]}
  ]}

── TAHARAT ───────────────────────────────────────────────────────

  INPUT:  حیض کی حالت میں قرآن پڑھنا
  OUTPUT: {"concepts":[
    {"term":"حیض","alternates":["حیضہ","حائضہ","ماہواری"]}
  ]}

── MUNAKAHAT ─────────────────────────────────────────────────────

  INPUT:  خلع لینا جائز ہے
  OUTPUT: {"concepts":[
    {"term":"خلع","alternates":["مخالعہ","فسخِ نکاح","عدت"]}
  ]}

── MU'AMALAT ─────────────────────────────────────────────────────

  INPUT:  بینک سے قرض لینا
  OUTPUT: {"concepts":[
    {"term":"بینک","alternates":["بنک","مالیاتی ادارہ"]},
    {"term":"قرض","alternates":["قرضہ","ادھار"]}
  ]}
  (NOT سود — that's a distinct ruling, not a synonym of قرض.)

── RUQYAH / UNSEEN (multi-concept — THIS is where expansion matters most) ─

  INPUT:  اگر کسی کے اوپر جادو ہو جائے تو کیا پڑھے
  OUTPUT: {"concepts":[
    {"term":"جادو","alternates":["سحر","ساحر"]}
  ]}
  (Do NOT invent عزیزہ/بیٹا — "کسی" is generic.  Do NOT output
   قرآن/آیات — those are answer-domain.)

  INPUT:  دعائے خوف یا خوابِ بد سے بچاؤ کی مخصوص تدابیر کیا ہیں؟
  OUTPUT: {"concepts":[
    {"term":"دعائے خوف","alternates":["خوف کی دعا","ماثور دعا","مسنون دعا"]},
    {"term":"خوابِ بد","alternates":["ڈراؤنا خواب","برا خواب","کابوس","حلم سیئ"]},
    {"term":"بچاؤ","alternates":["حفاظت","پناہ","بچنا"]},
    {"term":"تدابیر","alternates":["اعمال","اذکار","وظائف"]}
  ]}
  (Four concepts, all in the RUQYAH/اذکار domain, every
   alternate a realistic fatwa-site phrasing.  This is the
   model expansion for multi-concept queries.)

  INPUT:  رقیہ کا طریقہ
  OUTPUT: {"concepts":[
    {"term":"رقیہ","alternates":["دمِ شرعی","دم"]},
    {"term":"طریقہ","alternates":["طریقہ کار"]}
  ]}

── COUNTER-EXAMPLES (do NOT produce these) ───────────────────────

  INPUT:  حائضہ کے لیے کیا پڑھنا
  BAD:    {"concepts":[{"term":"پڑھنا","alternates":["قرآن","ذکر"]}]}
  GOOD:   {"concepts":[{"term":"حائضہ","alternates":["حیض","حیضہ","ماہواری"]}]}
  (قرآن/ذکر are answer-domain; don't emit them.)

  INPUT:  بیمار کے لیے کون سی دعا
  BAD:    {"concepts":[{"term":"دعا","alternates":["شفا","درود","منزل"]}]}
  GOOD:   {"concepts":[{"term":"بیمار","alternates":["مریض","مرض","بیماری"]}]}
  (شفا/درود/منزل are specific duas — answer content, not query
   synonyms.)
"""


def _postprocess_expansion(
    alternates: list[str], present_tokens: set[str]
) -> list[str]:
    """Sanitize the flat list of alternate phrases into a clean BM25 token list.

    Accepts the `alternates` values harvested from the LLM's structured
    JSON output (a list of short Urdu/Arabic phrases).  Each phrase is
    tokenized through the SAME tokenizer BM25 uses downstream — that
    way any normalisation BM25 applies (e.g. hamza folding, ي/ی
    unification) also applies to the expansion, and the new tokens
    integrate cleanly into the ranking.

    Drops: empty/too-short tokens, tokens already in the query,
    Latin-script tokens, grammatical fillers, and duplicates.
    """
    if not alternates:
        return []
    toks: list[str] = []
    seen: set[str] = set()
    for phrase in alternates:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        for t in _bm25_tokenize(phrase.strip()):
            if not t or len(t) < 2:
                continue
            if any("a" <= ch.lower() <= "z" for ch in t):
                continue
            if t in present_tokens or t in seen:
                continue
            if t in _URDU_FILLERS:
                continue
            toks.append(t)
            seen.add(t)
    return toks


def _parse_expansion_json(raw: str) -> list[str]:
    """Extract the flat list of alternate phrases from the LLM's JSON response.

    Expected shape (see ``_LLM_EXPAND_SYS``)::

        {"concepts": [
            {"term": "<concept>", "alternates": ["<a1>", "<a2>", ...]},
            ...
        ]}

    Returns the concatenation of every ``alternates`` list across all
    concepts.  Falls back gracefully if the LLM produces malformed
    JSON (e.g. trailing prose, code-fenced blocks) by extracting the
    first ``{ ... }`` substring we can parse.  Returns an empty list
    when the response is unparseable — NEVER raises.
    """
    if not raw:
        return []
    raw = raw.strip()

    # Strip code-fence wrappers if the LLM ignored the JSON-only rule.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: locate the first balanced JSON object substring.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        else:
            return []

    if not isinstance(parsed, dict):
        return []
    concepts = parsed.get("concepts")
    if not isinstance(concepts, list):
        return []

    alts: list[str] = []
    for entry in concepts:
        if not isinstance(entry, dict):
            continue
        lst = entry.get("alternates")
        if not isinstance(lst, list):
            continue
        for item in lst:
            if isinstance(item, str) and item.strip():
                alts.append(item.strip())
    return alts


def _llm_expand_query_uncached(
    canon_query: str,
    *,
    max_tokens: int,
    timeout_s: float,
    model: str,
) -> tuple[str, ...]:
    """One LLM round-trip to propose SAME-MASALA alternate phrasings.

    Uses structured JSON output (per ``_LLM_EXPAND_SYS``) so the LLM
    is forced to (a) identify 2-5 key fiqh concepts in the query and
    (b) list realistic alternate phrasings for each.  This is much
    more reliable than asking for a flat token list — plain-text
    prompts tend to default to empty output for long multi-concept
    queries.

    Returns a tuple of BM25-ready tokens (hashable for the outer LRU
    cache).  Empty tuple on any failure — callers treat that as "no
    expansion, proceed with the deterministic query".
    """
    settings = get_settings()
    use_model = model or settings.chat_model
    present = set(canon_query.split())

    def _call() -> tuple[tuple[str, ...], str]:
        client = OpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=timeout_s)
        comp = client.chat.completions.create(
            model=use_model,
            temperature=0,
            # Budget generously: a 4-concept expansion with 3 alts
            # each plus JSON scaffolding is ~200 tokens.
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_EXPAND_SYS},
                {"role": "user", "content": canon_query[:500]},
            ],
        )
        raw = (comp.choices[0].message.content or "").strip()
        alts = _parse_expansion_json(raw)
        toks = _postprocess_expansion(alts, present)
        return tuple(toks[:max_tokens]), raw

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call)
            toks, raw = fut.result(timeout=timeout_s + 1.0)
    except _cf.TimeoutError:
        logger.warning("[islam360-fast] LLM query-expand hard-timeout (%.1fs)", timeout_s)
        return ()
    except Exception as exc:
        logger.warning("[islam360-fast] LLM query-expand failed: %s", exc)
        return ()

    # Log the raw LLM response at DEBUG and the final token list at
    # INFO — lets us audit expansion quality without drowning the log
    # in full JSON for every query.  An empty `toks` with non-empty
    # `raw` means the LLM returned something but _parse_expansion_json
    # or _postprocess_expansion rejected it — that's a signal worth
    # surfacing.
    if toks:
        logger.info("[islam360-fast] LLM expansion: %s → +%s", canon_query, list(toks))
    else:
        if raw:
            logger.info(
                "[islam360-fast] LLM expansion empty (raw=%r) for query=%r",
                raw[:300], canon_query,
            )
        else:
            logger.info("[islam360-fast] LLM expansion empty (no response) for query=%r", canon_query)
    logger.debug("[islam360-fast] LLM expansion raw=%r", raw)
    return toks


# Public cached wrapper — the cache key is just the canonical query
# string, which is exactly what we want (three parallel sect calls
# with the same query share a single LLM round-trip).
@lru_cache(maxsize=256)
def _llm_expand_query_cached(canon_query: str, max_tokens: int, timeout_s: float, model: str) -> tuple[str, ...]:
    return _llm_expand_query_uncached(
        canon_query, max_tokens=max_tokens, timeout_s=timeout_s, model=model,
    )


# Tracks which trailing tokens came from the LLM expander so
# ``_strip_trailing_synonyms`` can remove them when building the
# reranker input.  Keyed by the POST-normalise canonical query the
# BM25 stage receives.  Bounded — we evict oldest on overflow.
_LLM_EXPAND_TRAILING: dict[str, tuple[str, ...]] = {}
_LLM_EXPAND_TRAILING_MAX = 512


def _remember_llm_expansion(final_query: str, extras: tuple[str, ...]) -> None:
    if not extras:
        return
    if len(_LLM_EXPAND_TRAILING) >= _LLM_EXPAND_TRAILING_MAX:
        # FIFO eviction — Python 3.7+ dicts preserve insertion order.
        try:
            oldest = next(iter(_LLM_EXPAND_TRAILING))
            _LLM_EXPAND_TRAILING.pop(oldest, None)
        except StopIteration:
            pass
    _LLM_EXPAND_TRAILING[final_query] = extras


def _llm_expand_query(canon_query: str) -> str:
    """Append LLM-suggested synonyms to *canon_query* for BM25.

    Idempotent with respect to the hand-curated dictionary — the
    LLM is asked to add tokens that are NOT already present, so
    running ``_expand_fiqh_synonyms`` first and this function second
    yields the union without duplicates.

    Failures are silent (the caller just gets back the original
    query unchanged) — this stage is pure recall-enhancement, never
    a correctness gate.
    """
    s = get_settings()
    if not s.islam360_llm_expand_enabled or not canon_query:
        return canon_query
    # Only invoke on Urdu-script input — the compression step has
    # already run, so anything still in Latin is filler/noise that
    # won't benefit from fiqh synonym hunting.
    if not _has_arabic_script(canon_query):
        return canon_query
    try:
        extras = _llm_expand_query_cached(
            canon_query,
            int(s.islam360_llm_expand_max_tokens),
            float(s.islam360_llm_expand_timeout_s),
            str(s.islam360_llm_expand_model or ""),
        )
    except Exception as exc:  # defensive — lru_cache wrapper shouldn't raise
        logger.warning("[islam360-fast] LLM expand wrapper error: %s", exc)
        return canon_query
    if not extras:
        return canon_query
    logger.info(
        "[islam360-fast] LLM expanded query: %r → +%s",
        canon_query, list(extras),
    )
    expanded = canon_query + " " + " ".join(extras)
    # Record the added tokens keyed by the POST-normalise form that
    # the rest of the pipeline will see.  normalize_urdu is applied
    # on both sides so the key matches when the stripper looks up.
    try:
        _remember_llm_expansion(normalize_urdu(expanded), extras)
    except Exception:
        pass
    return expanded


# ── Sect detection from user query ───────────────────────────────────────────
#
# A query may explicitly name a sect ("deobandi fatwa do", "barelvi mufti
# kya kehte hain", "ahle hadees ka moaqqif"). When that happens we scope
# retrieval to that sect's source(s) only — otherwise we do a 3-way split
# (see :meth:`Islam360Retriever.retrieve_all_sects`).
#
# Keywords are matched case-insensitively as whole words so that
# incidental substrings ("bandar" ≠ "barelvi") don't trigger false
# positives. Roman-Urdu AND Urdu-script variants are covered.

_SECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    SECT_DEOBANDI: (
        "deobandi", "deobandis", "deoband", "deobandee",
        "banuri", "binori", "binoori",
        "دیوبندی", "دیوبند", "بنوری",
    ),
    SECT_BARELVI: (
        "barelvi", "barelwi", "bareilvi", "bareilwi", "brelvi", "brelwi",
        "urdufatwa", "urdu fatwa", "urdu-fatwa",
        "بریلوی", "بریلی", "بَریلوی", "رضوی",
    ),
    SECT_AHLE_HADITH: (
        "ahle hadees", "ahl e hadees", "ahl-e-hadees",
        "ahle hadith", "ahl e hadith", "ahl-e-hadith",
        "ahlehadees", "ahlehadith",
        "islamqa", "islam qa",
        "fatwaqa", "fatwa qa",
        "salafi", "salafiyya",
        "اہل حدیث", "اہلِ حدیث", "سلفی",
    ),
}


_SECT_WORD_BOUNDARY_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def detect_sect_in_query(query: str) -> str | None:
    """Return the canonical sect code if *query* explicitly names a sect.

    Matching is **word-boundary** so that substrings inside an
    unrelated word don't trigger a false positive.  Returns ``None``
    when the query is sect-neutral (the default case — ~99 %% of
    real-world queries).
    """
    q = (query or "").lower()
    if not q:
        return None
    # Build a tokenized view once so Roman tokens and Urdu tokens
    # can both be matched cheaply.
    tokens = set(_SECT_WORD_BOUNDARY_RE.findall(q))
    for sect, keywords in _SECT_KEYWORDS.items():
        for kw in keywords:
            low = kw.lower()
            # Multi-word keywords must appear as a substring; single-word
            # ones must be a whole token (avoids false positives where
            # e.g. "banuri" is embedded in a URL fragment inside the query).
            if " " in low or "-" in low:
                if low in q:
                    return sect
            elif low in tokens:
                return sect
    return None


# ── Per-sect post-filtering ──────────────────────────────────────────────────


def _sect_of_candidate(c: dict[str, Any]) -> str:
    """Return the sect code for a candidate, preferring in-metadata labels.

    Order of precedence:
        1. ``metadata['sect']`` (populated by future BM25 rebuilds).
        2. ``metadata['source']`` → ``SOURCE_TO_SECT`` (same populated path).
        3. Sidecar lookup by chunk id (``get_sect_for_id``).
    """
    meta = c.get("metadata") or {}
    sect = meta.get("sect")
    if sect:
        return sect
    src = meta.get("source")
    if src and src in SOURCE_TO_SECT:
        return SOURCE_TO_SECT[src]
    return get_sect_for_id(c.get("id", "")) or ""


def _source_of_candidate(c: dict[str, Any]) -> str:
    meta = c.get("metadata") or {}
    src = meta.get("source")
    if src:
        return src
    return get_source_for_id(c.get("id", "")) or ""


def _filter_by_sect(
    candidates: list[dict[str, Any]],
    sect: str,
) -> list[dict[str, Any]]:
    """Return only candidates whose derived sect exactly equals *sect*.

    This is the hard filter that prevents "all results are Banuri"
    contamination — the allow-list is authoritative, and anything that
    resolves to a different sect or to blank is dropped.
    """
    allowed = SECT_TO_SOURCES.get(sect, frozenset())
    out: list[dict[str, Any]] = []
    for c in candidates:
        src = _source_of_candidate(c)
        if src in allowed:
            out.append(c)
    return out


# ── Public retriever ─────────────────────────────────────────────────────────

class Islam360Retriever:
    """End-to-end Islam360 retrieval.

    By default ``retrieve()`` dispatches to :meth:`retrieve_fast` (BM25-first,
    no LLM rewrite/rerank) which mirrors the original working setup. Set
    ``settings.islam360_use_fast_path = False`` (or call
    :meth:`retrieve_heavy` directly) to use the layered hybrid pipeline.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    # ── Default dispatcher ───────────────────────────────────────────────
    def retrieve(self, user_query: str, *, top_k: int = 5, **kwargs: Any) -> dict[str, Any]:
        """Top-level retrieval entry point.

        Routing rules:
            1. If the user query explicitly names a sect
               ("deobandi fatwa kya hai", "barelvi mufti ...") we run a
               **sect-scoped** single retrieval restricted to that sect's
               sources — the output shape is identical to a regular
               fast-path retrieval.
            2. Otherwise we run the standard single retrieval across all
               corpora (no sect filter).  The 3-way per-sect split is
               exposed via :meth:`retrieve_all_sects` and is driven by
               the Flask ``/api/query-all-schools`` endpoint.
        """
        sect = detect_sect_in_query(user_query)
        if self.settings.islam360_use_fast_path:
            return self.retrieve_fast(user_query, top_k=top_k, sect=sect)
        return self.retrieve_heavy(user_query, **kwargs)

    # ── Per-sect dispatch helpers ────────────────────────────────────────

    def retrieve_by_sect(
        self,
        user_query: str,
        sect: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve strictly within a single sect's sources.

        Enforces a hard allow-list filter derived from
        :data:`SECT_TO_SOURCES` — candidates whose sidecar-resolved
        source is not on the list are dropped before scoring is even
        considered final.  If nothing survives, returns a structured
        no-match payload so the caller can render "No relevant fatwa
        found in <sect>" without confabulating a cross-sect result.
        """
        if sect not in SECT_TO_SOURCES:
            raise ValueError(f"Unknown sect: {sect!r}")
        if self.settings.islam360_use_fast_path:
            return self.retrieve_fast(user_query, top_k=top_k, sect=sect)
        return self.retrieve_heavy(user_query, sect=sect)

    def retrieve_all_sects(
        self,
        user_query: str,
        *,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Run THREE independent per-sect retrievals and return them separately.

        This implements Phase 4 / Phase 7 of the sect-aware spec: each
        sect is retrieved with its own strict source filter, results
        are **never merged into one ranked list**, and the payload
        exposes a ``by_sect`` map keyed by sect code so the UI can
        render three columns side-by-side.

        Answer synthesis is performed per-sect as well — we never show
        a Banuri fatwa under the Barelvi answer, etc.
        """
        out: dict[str, dict[str, Any]] = {}
        for sect in ALL_SECTS:
            try:
                out[sect] = self.retrieve_by_sect(
                    user_query, sect, top_k=top_k,
                )
            except Exception as exc:
                logger.exception("retrieve_by_sect(%s) failed: %s", sect, exc)
                out[sect] = {
                    "answer": NO_RELEVANT_FATWA,
                    "sources": [],
                    "log": {"error": f"retrieval_failed: {exc}"},
                    "no_match": True,
                }

        # Contamination guard — catch bugs early.  Every returned source
        # inside a sect payload must resolve to that same sect; if not,
        # the metadata fell out of sync (usually a stale sidecar) and we
        # strip the offending entry so the user never sees cross-sect
        # leakage.
        leaks: dict[str, list[str]] = {}
        for sect, payload in out.items():
            expected = SECT_TO_SOURCES.get(sect, frozenset())
            clean: list[dict[str, Any]] = []
            bad: list[str] = []
            for s in payload.get("sources") or []:
                src = _source_of_candidate(s)
                if src in expected:
                    clean.append(s)
                else:
                    bad.append(f"{s.get('id','?')}<{src or 'unknown'}>")
            if bad:
                leaks[sect] = bad
                payload["sources"] = clean
                if not clean:
                    payload["no_match"] = True
                    payload["answer"] = NO_RELEVANT_FATWA
        if leaks:
            logger.error(
                "[islam360] CROSS-SECT LEAK DETECTED — stripped: %s", leaks,
            )

        return {
            "by_sect": out,
            "detected_sect": detect_sect_in_query(user_query),
        }

    # ── Fast path: BM25-first, no LLM in the retrieval loop ───────────────
    def retrieve_fast(
        self,
        user_query: str,
        *,
        top_k: int = 5,
        sect: str | None = None,
    ) -> dict[str, Any]:
        """BM25-first retrieval, optionally scoped to a single sect.

        When ``sect`` is provided the BM25 candidate pool is widened
        (sect-filtered post-hoc) so we still have enough survivors per
        sect even when one source dominates the corpus in volume.
        """
        s = self.settings
        canon = _canonicalise_for_bm25(user_query)
        if not canon:
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": {"error": "empty_query", "sect": sect},
                "no_match": True,
                "sect": sect,
            }

        corpus = get_islam360_bm25()

        # Widen the pool considerably when a sect filter is active — one
        # source can easily out-number another 5×, so a narrow pool
        # would leave the minority sect with too few candidates after
        # filtering.  No sect filter → normal pool.
        pool = max(top_k * 6, 30)
        if sect is not None:
            pool = max(top_k * 40, 400)
        full_hits = corpus.search(canon, top_k=pool)
        q_scores = corpus.score_questions(canon, top_k=pool)

        # Filter to islam360 corpus only (the cache may also hold legacy docs).
        full_hits = [
            h for h in full_hits
            if (h.get("metadata") or {}).get("corpus_source", "") == "islam360"
        ]

        # Index full hits by id, capture metadata + raw full-text score.
        merged: dict[str, dict[str, Any]] = {}
        for h in full_hits:
            cid = h["id"]
            meta = dict(h.get("metadata") or {})
            merged[cid] = {
                "id": cid,
                "full_score": float(h.get("score", 0.0)),
                "q_score": 0.0,
                "metadata": meta,
                "text": h.get("text", "") or f"{meta.get('question','')}\n{meta.get('answer','')}",
            }

        # Layer in question-field scores; they may surface ids NOT in full_hits
        # (rare, but possible if the question matches strongly while the answer
        # noise dominated). For those we synthesise a metadata stub from the
        # underlying corpus doc list.
        id_to_doc = {d.get("id"): d for d in getattr(corpus, "_docs", [])}
        for cid, qsc in q_scores.items():
            doc = id_to_doc.get(cid)
            if doc is None or doc.get("corpus_source", "") != "islam360":
                continue
            entry = merged.get(cid)
            if entry is None:
                merged[cid] = {
                    "id": cid,
                    "full_score": 0.0,
                    "q_score": float(qsc),
                    "metadata": {
                        "question": doc.get("question", ""),
                        "answer": doc.get("answer", ""),
                        "category": doc.get("category", ""),
                        "source_file": doc.get("source_file", ""),
                        "folder": doc.get("folder", ""),
                        "source_name": doc.get("source_name", ""),
                        "maslak": doc.get("maslak", ""),
                        "scholar": doc.get("scholar", ""),
                        "language": doc.get("language", ""),
                        "corpus_source": doc.get("corpus_source", ""),
                    },
                    "text": doc.get("text", ""),
                }
            else:
                entry["q_score"] = float(qsc)

        # Combined score: full-text + boosted question-field match.
        boost = float(s.islam360_fast_question_boost)
        for entry in merged.values():
            entry["score"] = entry["full_score"] + boost * entry["q_score"]
            entry["final_score"] = entry["score"]  # alias for UI

        ranked = sorted(merged.values(), key=lambda e: e["score"], reverse=True)
        ranked = _dedupe(ranked)

        # ── Sect filter (HARD allow-list) ───────────────────────────────
        # If the caller asked for a specific sect, drop every candidate
        # whose source is not on that sect's allow-list.  This happens
        # BEFORE the rare-anchor gate so the rare-anchor pool analysis
        # operates on the sect-scoped candidate set (otherwise an
        # anchor that's rare corpus-wide but common within the sect
        # would wrongly eliminate all sect results).
        sect_pre_filter_ids: list[str] = []
        count_after_sect: int | None = None
        if sect is not None:
            sect_pre_filter_ids = [c["id"] for c in ranked[:top_k]]
            ranked = _filter_by_sect(ranked, sect)
            count_after_sect = len(ranked)

        # ── Rare-anchor must-match gate ─────────────────────────────────
        # Drop candidates that share the query's structural tokens but
        # miss its topic-distinguishing token (e.g. dog fatwas when the
        # user asked about a monkey). If *none* of the top-k candidates
        # contain any rare anchor, we know the corpus doesn't actually
        # answer this question.
        # If the sect filter removed every candidate, there's no point
        # running the rare-anchor analysis — the corpus just doesn't
        # have a fatwa in this sect for this query.
        if sect is not None and not ranked:
            log_payload = {
                "mode": "bm25_first",
                "canonical_query": canon,
                "reason": "sect_filter_empty",
                "sect": sect,
                "allowed_sources": sorted(SECT_TO_SOURCES.get(sect, set())),
                "sect_pre_filter_ids": sect_pre_filter_ids,
            }
            logger.info(
                "[islam360-fast] %s",
                json.dumps(log_payload, ensure_ascii=False)[:2000],
            )
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": log_payload,
                "no_match": True,
                "sect": sect,
            }

        # Anchor extraction is concept-aware: the user's original
        # rare tokens AND the LLM/dictionary synonym expansions BOTH
        # contribute to the anchor set, and a candidate passes if it
        # contains ANY one of them (``min_hits=1`` below).
        #
        # Historical context: we used to strip expansions before anchor
        # extraction, because the old plain-text LLM expander was
        # prone to hallucinating tokens that became rare-anchor
        # ``must-match`` requirements and filtered out every genuinely
        # relevant fatwa (the ``کسی کے اوپر جادو`` → ``عزیزہ``
        # regression).  That's no longer the failure mode — the new
        # JSON-structured concept expander (see ``_LLM_EXPAND_SYS``
        # and ``_parse_expansion_json``) grounds every alternate in
        # an explicit concept from the query, dramatically lowering
        # hallucination risk.
        #
        # More importantly, stripping expansions here was itself a
        # bigger bug in the opposite direction: it rejected fatwas
        # whose title used SYNONYM vocabulary instead of the user's
        # exact wording.  E.g. a user asking about ``خوابِ بد`` whose
        # query is expanded to include ``ڈراؤنا خواب / کابوس / برا
        # خواب`` would STILL have anchors restricted to
        # ``{خوف, بد, بچاو}`` and reject the top-scoring barelvi
        # fatwa ``(342) ڈراونے خواب`` (score 19.7) simply because
        # its title uses the Urdu colloquial form instead of the
        # Arabic-leaning ``بد``.  Including expansions as anchor
        # candidates fixes that class of miss while preserving
        # precision — the reranker's 0.75 threshold is still the
        # final gate.
        anchors, df_map = _extract_anchor_tokens(canon, ranked)
        pre_filter_ids = [c["id"] for c in ranked[:top_k]]
        count_after_anchor: int | None = None
        anchor_mode_used: str | None = None
        if anchors:
            # First pass — strict: anchor must appear in the candidate's
            # question.  This catches topical false positives where the
            # query's distinguishing token is only mentioned in passing
            # in the answer body.
            anchored = _filter_by_rare_anchors(
                ranked, anchors,
                min_hits=s.islam360_fast_min_anchor_hits,
                mode="question",
            )
            if anchored:
                ranked = anchored
                count_after_anchor = len(ranked)
                anchor_mode_used = "question"
            else:
                # Fallback — Q-only came up empty.  Most common cause is
                # Urdu morphological variation: the query uses one form
                # of a verb/noun and the fatwa's question uses another
                # (ہنسنا vs ہنسی vs قہقہہ).  Retry against the full
                # body before declaring no-match; the LLM reranker will
                # filter out any off-topic survivors.
                relaxed = _filter_by_rare_anchors(
                    ranked, anchors,
                    min_hits=s.islam360_fast_min_anchor_hits,
                    mode="body",
                )
                if relaxed:
                    ranked = relaxed
                    count_after_anchor = len(ranked)
                    anchor_mode_used = "body_fallback"
                    logger.info(
                        "[islam360-fast] anchor Q-only empty, fell back to body "
                        "match (sect=%s, kept=%d)",
                        sect, len(ranked),
                    )
                else:
                    # Not in question AND not in body — the corpus
                    # genuinely doesn't have a fatwa about this masala.
                    log_payload = {
                        "mode": "bm25_first",
                        "canonical_query": canon,
                        "reason": "rare_anchor_miss",
                        "sect": sect,
                        "anchors": anchors,
                        "anchor_df_in_pool": df_map,
                        "pre_filter_top_ids": pre_filter_ids,
                        "pre_filter_top_questions": [
                            (c.get("metadata") or {}).get("question", "")[:100]
                            for c in merged.values() if c["id"] in pre_filter_ids
                        ],
                    }
                    logger.info(
                        "[islam360-fast] %s",
                        json.dumps(log_payload, ensure_ascii=False)[:2000],
                    )
                    return {
                        "answer": NO_RELEVANT_FATWA,
                        "sources": [],
                        "log": log_payload,
                        "no_match": True,
                        "sect": sect,
                    }

        # ── Score-based relevance floor (per-candidate) ─────────────────
        #
        # Previously we only gated the top-1 on an absolute score floor,
        # which meant the remaining top_k-1 slots could be padded with
        # weak filler matches whose scores were far below the leader.
        # Now every candidate must clear BOTH:
        #
        #   (a) absolute floor — `islam360_fast_min_bm25_score` (drops
        #       obvious garbage that BM25 gave a tiny residual score).
        #   (b) relative floor — ≥ `islam360_fast_min_score_ratio` of the
        #       #1 candidate's score (drops anything significantly weaker
        #       than the best hit; if only 1-2 candidates are genuinely
        #       strong, top_k shrinks rather than padding with noise).
        #
        # Exception: always keep at least the #1 candidate if it clears
        # (a) — otherwise a long-tail query with one weak-but-valid hit
        # would hit no_match unnecessarily.
        abs_floor = float(s.islam360_fast_min_bm25_score)
        rel_ratio = float(s.islam360_fast_min_score_ratio)
        top_score = ranked[0]["score"] if ranked else 0.0
        rel_floor = top_score * rel_ratio
        floor = max(abs_floor, rel_floor)

        # Top-1 fast-fail: if even the best hit doesn't clear the absolute
        # floor, there's nothing worth showing.
        if not ranked or top_score < abs_floor:
            log_payload = {
                "mode": "bm25_first",
                "canonical_query": canon,
                "sect": sect,
                "reason": "top_score_below_floor",
                "top_score": round(top_score, 3),
                "abs_floor": abs_floor,
                "pre_filter_top_ids": pre_filter_ids,
            }
            logger.info(
                "[islam360-fast] %s",
                json.dumps(log_payload, ensure_ascii=False)[:2000],
            )
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": log_payload,
                "no_match": True,
                "sect": sect,
            }

        # Apply the combined floor to all other candidates.  #1 is always
        # kept (it passed the absolute floor above); the rest must also
        # clear the relative floor.
        strong: list[dict[str, Any]] = [ranked[0]]
        for c in ranked[1:]:
            if c["score"] >= floor:
                strong.append(c)

        # ── Anchor-coverage floor ─────────────────────────────────────
        #
        # When the query has multiple rare-anchor tokens, the best hits
        # usually cover 2-3 of them while weaker "padding" hits cover
        # only 1.  Forcing every candidate to cover at least
        # `ceil(best_coverage * ratio)` anchors drops the padders while
        # preserving genuinely topical results.
        #
        # No-op when only one anchor was identified (the rare-anchor
        # filter already required it to be present).
        best_anchors = max((c.get("anchor_hits", 0) for c in strong), default=0)
        anchor_floor_hits = 0
        if anchors and best_anchors >= 2:
            cov_ratio = float(s.islam360_fast_min_anchor_coverage_ratio)
            anchor_floor_hits = max(1, math.ceil(best_anchors * cov_ratio))
            strong = [
                c for c in strong
                if c.get("anchor_hits", 0) >= anchor_floor_hits
            ]

        # ── Secondary sort: richer topical coverage beats raw BM25 ────
        #
        # Within the surviving pool, sort by (anchor_hits DESC, score
        # DESC).  BM25 already rewards multi-anchor hits, but long
        # answer bodies can inflate single-anchor scores past
        # multi-anchor ones — this re-sort corrects that.
        strong.sort(
            key=lambda c: (c.get("anchor_hits", 0), c["score"]),
            reverse=True,
        )

        # ── LLM question-to-question reranker (strict precision gate) ──
        #
        # Up to this point the candidate set survived structural gates
        # (sect allow-list, anchor presence, score floor).  That's
        # enough to drop *obvious* junk, but it still lets through
        # fatwas whose QUESTION is only tangentially related to the
        # user's query.  We now ask the LLM to score each candidate's
        # question (not the full answer) against the user's query and
        # drop anything below `islam360_rerank_threshold`.
        #
        # Quality-over-quantity contract: it's acceptable to return 0
        # results if nothing clears the threshold.  The existing
        # `NO_RELEVANT_FATWA` no-match path handles that cleanly.
        rerank_applied = False
        rerank_threshold = float(s.islam360_rerank_threshold)
        rerank_rejected: list[dict[str, Any]] = []  # logged per-candidate

        if s.islam360_rerank_enabled and strong:
            # Cap the LLM input size — no point scoring 50 candidates
            # when only 5 can be returned.  We trust the pre-rerank
            # ordering (BM25 × anchor) to surface the true contenders.
            max_cand = max(top_k, int(s.islam360_rerank_max_candidates))
            rerank_pool = strong[:max_cand]

            # For verbose narratives, pass the COMPRESSED fiqh question
            # to the reranker.  The raw user_query carries personal
            # backstory ("my brother has land, kids are studying, late
            # inheritance, …") that pushes the LLM to reject candidates
            # which don't mention every specific detail — even when the
            # candidate directly addresses the core masala ("can I
            # give zakat to a relative").  The canonicalised ``canon``
            # already has synonyms appended which is noisy, so we strip
            # those back off before passing to the reranker.
            rerank_query = _rerank_query_for(user_query, canon)

            scores = _llm_rerank_by_question(
                rerank_query, rerank_pool, sect=sect,
            )
            if scores:
                rerank_applied = True
                kept: list[dict[str, Any]] = []
                for c in rerank_pool:
                    entry = scores.get(c["id"])
                    if entry is None:
                        # LLM didn't score this one → treat as uncertain
                        # and REJECT (strict rule).
                        rerank_rejected.append({
                            "id": c["id"],
                            "score": None,
                            "reason": "no_llm_score_returned",
                            "Q": (c.get("metadata") or {}).get("question", "")[:80],
                        })
                        continue
                    c["rerank_score"]  = entry["score"]
                    c["rerank_reason"] = entry["reason"]
                    if entry["score"] >= rerank_threshold:
                        kept.append(c)
                    else:
                        rerank_rejected.append({
                            "id": c["id"],
                            "score": round(entry["score"], 3),
                            "reason": entry["reason"],
                            "Q": (c.get("metadata") or {}).get("question", "")[:80],
                        })

                # Re-sort kept candidates by rerank score (primary) then
                # BM25 score (tiebreak).  This is what actually decides
                # the visible ranking.
                kept.sort(
                    key=lambda c: (c.get("rerank_score", 0.0), c["score"]),
                    reverse=True,
                )
                strong = kept
            # else: rerank unavailable (timeout / bad JSON) → see
            # policy below.  `rerank_applied` stays False so the log
            # makes this transparent.

        # Rerank failure policy under body-fallback anchor mode.
        #
        # When we fell back to body-match anchor filtering, the pool is
        # noisier (anchor might appear only in a passing mention in
        # the answer body).  The LLM reranker is what normally cleans
        # that noise up.  If the reranker failed (timeout / bad JSON)
        # publishing the pre-rerank ordering would show demonstrably
        # irrelevant fatwas to the user (confirmed in testing — see
        # the "namaz me has diya" case).  Honour the no-filler
        # contract and return no_match instead.
        if (
            s.islam360_rerank_enabled
            and not rerank_applied
            and anchor_mode_used == "body_fallback"
            and s.islam360_strict_when_body_fallback
        ):
            log_payload = {
                "mode": "bm25_first",
                "canonical_query": canon,
                "sect": sect,
                "reason": "rerank_unavailable_in_body_fallback_mode",
                "anchor_mode": anchor_mode_used,
                "pool_size": len(strong),
                "top_score": round(top_score, 3),
            }
            logger.info(
                "[islam360-fast] %s",
                json.dumps(log_payload, ensure_ascii=False)[:2000],
            )
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": log_payload,
                "no_match": True,
                "sect": sect,
            }

        # Strict no-match: if the rerank gate emptied the pool, bail
        # out with the same contract as the existing gates.
        if s.islam360_rerank_enabled and rerank_applied and not strong:
            log_payload = {
                "mode": "bm25_first",
                "canonical_query": canon,
                "sect": sect,
                "reason": "rerank_threshold_miss",
                "rerank_threshold": rerank_threshold,
                "rerank_rejected": rerank_rejected[:15],
                "pre_filter_top_ids": pre_filter_ids,
            }
            logger.info(
                "[islam360-fast] %s",
                json.dumps(log_payload, ensure_ascii=False)[:3000],
            )
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": log_payload,
                "no_match": True,
                "sect": sect,
            }

        chosen = strong[:top_k]

        # Enrich each chosen source with its original URL / fatwa number
        # from the sidecar lookup (built from the raw CSVs — see
        # :mod:`src.islam360.url_index`).  Missing URLs just stay blank.
        try:
            from src.islam360.url_index import enrich_metadata

            for c in chosen:
                c["metadata"] = enrich_metadata(c["id"], c.get("metadata") or {})
        except Exception as exc:  # pragma: no cover — never fatal
            logger.warning("URL enrichment skipped: %s", exc)

        # Final contamination check — every chosen source must belong
        # to the requested sect. Any violation is a hard error: it
        # means the sidecar lookup and the BM25 cache disagree about
        # which source a doc belongs to. We strip the violator and
        # log a big red flag so the issue is visible in logs.
        if sect is not None:
            expected = SECT_TO_SOURCES.get(sect, frozenset())
            leaked: list[str] = []
            safe: list[dict[str, Any]] = []
            for c in chosen:
                src = _source_of_candidate(c)
                if src in expected:
                    safe.append(c)
                else:
                    leaked.append(f"{c['id']}<{src or 'unknown'}>")
            if leaked:
                logger.error(
                    "[islam360-fast] SECT LEAK in %s retrieval — dropped %s",
                    sect, leaked,
                )
            chosen = safe

        log_payload = {
            "mode": "bm25_first",
            "canonical_query": canon,
            "sect": sect,
            "had_arabic_script_input": _has_arabic_script(user_query),
            "full_hits": len(full_hits),
            "q_hits": len(q_scores),
            "merged": len(merged),
            "after_sect_filter": count_after_sect,
            "after_anchor_filter": count_after_anchor,
            "anchor_mode": anchor_mode_used,
            "pool_size": len(ranked),
            "abs_floor": abs_floor,
            "rel_ratio": rel_ratio,
            "effective_floor": round(floor, 3),
            "top_score": round(top_score, 3),
            "best_anchor_hits": best_anchors,
            "anchor_floor_hits": anchor_floor_hits,
            "rerank_applied": rerank_applied,
            "rerank_threshold": rerank_threshold,
            "rerank_rejected": rerank_rejected[:15],
            "requested_top_k": top_k,
            "returned": len(chosen),   # may be < top_k on weak queries
            "rare_anchors": anchors,
            "anchor_df_in_pool": df_map,
            "candidates_preview": [
                {
                    "id": c["id"],
                    "full": round(c["full_score"], 3),
                    "q": round(c["q_score"], 3),
                    "score": round(c["score"], 3),
                    "anchor_hits": c.get("anchor_hits", 0),
                    "rerank": round(c.get("rerank_score", 0.0), 3)
                               if c.get("rerank_score") is not None else None,
                    "src": _source_of_candidate(c),
                    "sect": _sect_of_candidate(c),
                    "Q": (c.get("metadata") or {}).get("question", "")[:80],
                }
                for c in ranked[:8]
            ],
            "final_ids": [c["id"] for c in chosen],
            "final_scores": [round(c["score"], 3) for c in chosen],
            "final_rerank": [
                round(c.get("rerank_score", 0.0), 3)
                if c.get("rerank_score") is not None else None
                for c in chosen
            ],
            "final_sources": [_source_of_candidate(c) for c in chosen],
        }
        logger.info("[islam360-fast] %s", json.dumps(log_payload, ensure_ascii=False)[:4000])

        if not chosen:
            # Should be unreachable (top-1 gate above catches the empty
            # case), but defend anyway.
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "log": log_payload,
                "no_match": True,
                "sect": sect,
            }

        answer = _synthesize_answer(user_query, chosen)
        return {
            "answer": answer,
            "sources": chosen,
            "log": log_payload,
            "no_match": False,
            "sect": sect,
        }

    # ── Heavy path (kept for ablation / future tuning) ────────────────────
    def retrieve_heavy(
        self,
        user_query: str,
        *,
        use_llm_rewrite: bool = True,
        sect: str | None = None,
    ) -> dict[str, Any]:
        rw = rewrite_query(user_query, use_llm=use_llm_rewrite)

        sem_q = normalize_urdu(rw.semantic_query)
        kw_q = normalize_urdu(rw.keyword_query or rw.semantic_query)

        vec = embed_query_islam360(sem_q)
        if len(vec) != self.settings.islam360_embedding_dimensions:
            logger.error(
                "Embedding dim mismatch: got %d expected %d",
                len(vec),
                self.settings.islam360_embedding_dimensions,
            )

        lang = rw.language if rw.language not in ("mixed", "unknown") else None
        flt = _pinecone_filter(language=lang)

        n_cand = self.settings.islam360_retrieval_candidates

        corpus = get_islam360_bm25()
        dense = _dense_islam360(vec, max(n_cand, 50), flt)
        sparse = _sparse_islam360(kw_q, max(n_cand, 50), corpus, language=lang)
        qboost = _question_boost_ids(kw_q, max(n_cand, 50), corpus)

        fused = _fuse(dense, sparse, qboost)[: max(n_cand, 30)]

        if not fused:
            log_payload = {
                "semantic_query": sem_q,
                "keyword_query": kw_q,
                "embedding_dims": len(vec),
                "dense_hits": len(dense),
                "sparse_hits": len(sparse),
                "qboost_hits": len(qboost),
                "error": "no_candidates",
            }
            logger.warning("[islam360] no fused candidates: %s", log_payload)
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "rewritten_query": rw,
                "log": log_payload,
                "no_match": True,
            }

        candidates: list[dict[str, Any]] = []
        for cid, fscore, meta in fused:
            candidates.append({
                "id": cid,
                "fusion_score": round(fscore, 6),
                "metadata": meta,
                "text": meta.get("text", "")
                or f"{meta.get('question', '')}\n{meta.get('answer', '')}",
            })

        candidates = _dedupe(candidates)

        # Hard sect allow-list (heavy path mirrors fast path semantics)
        if sect is not None:
            candidates = _filter_by_sect(candidates, sect)
            if not candidates:
                log_payload = {
                    "semantic_query": sem_q,
                    "keyword_query": kw_q,
                    "sect": sect,
                    "reason": "sect_filter_empty",
                }
                logger.info("[islam360] %s", log_payload)
                return {
                    "answer": NO_RELEVANT_FATWA,
                    "sources": [],
                    "rewritten_query": rw,
                    "log": log_payload,
                    "no_match": True,
                    "sect": sect,
                }

        rscores = _rerank_llm(user_query, candidates)
        ranked: list[dict[str, Any]] = []
        for it, rs in zip(candidates, rscores):
            it2 = dict(it)
            it2["rerank_score"] = round(rs, 4)
            # Final score weighted toward the LLM relevance judgment.
            it2["final_score"] = round(0.30 * it["fusion_score"] + 0.70 * rs, 6)
            ranked.append(it2)
        ranked.sort(key=lambda x: x["final_score"], reverse=True)

        final_k = self.settings.islam360_rerank_top_k
        min_sc = self.settings.islam360_min_relevance_score
        chosen = [x for x in ranked[:final_k] if x["rerank_score"] >= min_sc]

        # Enrich with URL / fatwa-number from the sidecar lookup so the UI
        # can render a "cross-verify" link. Safe no-op if lookup empty.
        try:
            from src.islam360.url_index import enrich_metadata

            for c in chosen:
                c["metadata"] = enrich_metadata(c["id"], c.get("metadata") or {})
        except Exception as exc:  # pragma: no cover
            logger.warning("URL enrichment skipped: %s", exc)

        log_payload = {
            "semantic_query": sem_q,
            "keyword_query": kw_q,
            "intent": rw.intent_category,
            "language_filter": lang,
            "embedding_dims": len(vec),
            "dense_hits": len(dense),
            "sparse_hits": len(sparse),
            "qboost_hits": len(qboost),
            "fused_unique": len(fused),
            "after_dedupe": len(candidates),
            "candidates_preview": [
                {
                    "id": c["id"],
                    "fusion": c["fusion_score"],
                    "rerank": c.get("rerank_score"),
                    "final": c.get("final_score"),
                    "q": (c.get("metadata") or {}).get("question", "")[:80],
                }
                for c in ranked[:8]
            ],
            "min_relevance_floor": min_sc,
            "final_ids": [c["id"] for c in chosen],
        }
        logger.info("[islam360] %s", json.dumps(log_payload, ensure_ascii=False)[:4000])

        if not chosen:
            return {
                "answer": NO_RELEVANT_FATWA,
                "sources": [],
                "rewritten_query": rw,
                "log": log_payload,
                "no_match": True,
            }

        answer = _synthesize_answer(user_query, chosen)
        return {
            "answer": answer,
            "sources": chosen,
            "rewritten_query": rw,
            "log": log_payload,
            "no_match": False,
        }


# ── Synthesis ────────────────────────────────────────────────────────────────

# Hard wall-clock timeout for the LLM synthesis step. The OpenAI SDK's
# internal timeout has proven unreliable on this machine (some calls hang
# indefinitely on SSL / httpx retry loops). We wrap the call in a thread
# and kill it after this many seconds, falling back to a deterministic
# summary of the top fatwa so the user always gets a useful response.
_SYNTH_HARD_TIMEOUT_S = 35.0


# ════════════════════════════════════════════════════════════════════════
# LLM question-to-question reranker (strict precision gate)
# ════════════════════════════════════════════════════════════════════════
#
# Relevance contract enforced here:
#
#   A fatwa is RELEVANT only if the fatwa's QUESTION directly asks the
#   same fiqh issue as the user's query.  Partial keyword overlap, topic
#   adjacency, or "somewhat related" — all treated as IRRELEVANT.
#
# The actual prompt + scoring logic lives in
# ``src.retrieval.openai_reranker`` so every retrieval path in the
# codebase shares ONE battle-tested reranker.  The wrapper below just
# preserves the historic call-site shape for this module.


def _llm_rerank_by_question(
    user_query: str,
    candidates: list[dict[str, Any]],
    *,
    sect: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Ask the LLM to score each candidate's QUESTION against *user_query*.

    Thin wrapper around :func:`src.retrieval.openai_reranker.score_candidates`
    that preserves the historic return shape this module's callers
    expect (``{id: {"score": float, "reason": str}}``).  The actual
    scoring logic, prompt, timeout handling, and graceful-fallback
    policy all live in the externalized ``openai_reranker`` module so
    every retrieval path in the codebase (islam360, legacy hybrid,
    pageindex, smart router) shares ONE battle-tested reranker.

    On any failure (timeout, bad JSON, network, missing key) returns
    an empty mapping — callers must treat that as "rerank unavailable"
    and keep the pre-rerank ordering.  Never raises.
    """
    # Lazy-import to avoid a hard circular dep with src.retrieval at
    # module load time (src.retrieval pulls in hybrid_retriever which
    # in turn imports pinecone; we don't want those side-effects paid
    # by every islam360 caller).
    from src.retrieval.openai_reranker import score_candidates

    scored = score_candidates(user_query, candidates, sect=sect)
    return {
        cid: {"score": entry.score, "reason": entry.reason}
        for cid, entry in scored.items()
    }


def _fallback_answer(user_query: str, chunks: list[dict[str, Any]]) -> str:
    """Deterministic fallback when the LLM call fails or times out.

    Returns the top-1 fatwa's question + answer verbatim (truncated),
    prefixed with a one-line note. No model calls, no latency.
    """
    if not chunks:
        return NO_RELEVANT_FATWA
    m = chunks[0].get("metadata") or {}
    q = (m.get("question") or "").strip()
    a = (m.get("answer") or "").strip()
    if not (q or a):
        return NO_RELEVANT_FATWA
    header = "(Direct fatwa excerpt — LLM synthesis unavailable)"
    body = ""
    if q:
        body += f"سوال: {q}\n\n"
    if a:
        body += f"جواب:\n{a[:2500]}"
    return f"{header}\n\n{body}".strip()


def _synthesize_answer(user_query: str, chunks: list[dict[str, Any]]) -> str:
    import concurrent.futures

    settings = get_settings()
    parts = []
    for i, ch in enumerate(chunks, 1):
        m = ch.get("metadata") or {}
        parts.append(
            f"[{i}] سوال: {m.get('question', '')}\nجواب: {m.get('answer', '')[:3500]}"
        )
    sys_msg = (
        "You are a careful Islamic fiqh assistant. Answer ONLY using the "
        "provided fatwa excerpts. Write in the same language as the user's "
        "question (Urdu if they wrote Urdu). If the excerpts do not contain "
        "enough detail to answer, say so briefly — do NOT invent rulings."
    )
    user_msg = (
        f"User question:\n{user_query}\n\nFatwa excerpts:\n"
        + "\n\n".join(parts)
    )

    def _call_llm() -> str:
        client = OpenAI(
            api_key=settings.openai_api_key,
            max_retries=0,
            timeout=_SYNTH_HARD_TIMEOUT_S - 5,
        )
        comp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=700,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg[:14000]},
            ],
        )
        return (comp.choices[0].message.content or "").strip()

    logger.info("[islam360-fast] synthesize: chunks=%d timeout=%.1fs", len(chunks), _SYNTH_HARD_TIMEOUT_S)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call_llm)
            result = fut.result(timeout=_SYNTH_HARD_TIMEOUT_S)
            if result:
                return result
            logger.warning("Synthesis returned empty — using fallback")
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Synthesis hard-timeout after %.1fs — using fallback", _SYNTH_HARD_TIMEOUT_S
        )
    except Exception as exc:
        logger.warning("Synthesis failed: %s — using fallback", exc)
    return _fallback_answer(user_query, chunks)
