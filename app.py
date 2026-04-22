"""
app.py — Flask web frontend for the Islamic Fatawa RAG System.

Usage
-----
    # Dry-run (no API keys needed — uses mock backend):
    python app.py --dry-run

    # Production (requires .env with OPENAI_API_KEY + PINECONE_API_KEY):
    python app.py

    # Custom port:
    python app.py --port 8080

    # With guardrails always on:
    python app.py --guardrails
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

# ── Force UTF-8 I/O on Windows ───────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONUTF8", "1")

# ── Arg parse (before any src import) ────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Islamic Fatawa RAG — Web UI")
    p.add_argument("--dry-run",    action="store_true", help="Use mock backend (no API calls)")
    p.add_argument("--guardrails", action="store_true", default=True,
                   help="Always apply safety guardrails (default: on)")
    p.add_argument("--no-guardrails", dest="guardrails", action="store_false",
                   help="Disable guardrails")
    p.add_argument("--port",  type=int, default=5000, help="Server port [5000]")
    p.add_argument("--host",  type=str, default="127.0.0.1", help="Bind address [127.0.0.1]")
    p.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve [5]")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"], help="Log verbosity")
    return p.parse_args()


_args = _parse()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, _args.log_level),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
for _lib in ("httpx", "httpcore", "openai", "pinecone", "urllib3", "werkzeug"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger("app")

# ── Optional-dep stubs for dry-run ────────────────────────────────────────────
if _args.dry_run:
    from unittest.mock import MagicMock
    for _pkg, _subs in [
        ("pinecone", ["pinecone.data"]),
        ("rank_bm25", []),
        ("tqdm", ["tqdm.auto"]),
    ]:
        if importlib.util.find_spec(_pkg) is None:
            sys.modules.setdefault(_pkg, MagicMock())
            for _s in _subs:
                sys.modules.setdefault(_s, MagicMock())

# ── Load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ── Src imports ───────────────────────────────────────────────────────────────
from src.config import get_settings

if _args.dry_run:
    from src.dry_run import DryRunContext
    _dry_ctx = DryRunContext()
    _dry_ctx.__enter__()
    log.info("Dry-run mode active — no external API calls.")

# Import pipeline after dry-run patches are in place
from src.pipeline.guardrails import GuardrailConfig, guarded_query
from src.pipeline.rag import query as rag_query
from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL
from src.pipeline.output_validator import validate as ov_validate
from src.preprocessing.urdu_normalizer import normalize_urdu

# ── Islam360 retriever (new corpus) ───────────────────────────────────────────
# In this build the active Pinecone index is `islam360-fatwa-1536` and the
# legacy 4-school index (`fatawa-hybrid`) is not present in the project.
# We route the main /api/query endpoint through the Islam360 hybrid pipeline
# (dense + BM25 + RRF + LLM rerank).  The retriever is created lazily so
# that --dry-run still works without API keys.
_islam360_retriever = None

def _get_islam360_retriever():
    global _islam360_retriever
    if _islam360_retriever is None:
        from src.islam360.retrieve import Islam360Retriever
        _islam360_retriever = Islam360Retriever()
        log.info("Islam360 retriever initialised (index=islam360-fatwa-1536)")
    return _islam360_retriever


def _islam360_query_for_flask(
    question: str,
    *,
    top_k: int,
    maslak: str | None = None,  # accepted for API compat; not used
    category: str | None = None,  # accepted for API compat; not used
):
    """Bridge the Islam360 retriever to the shape Flask expects.

    Returns ``(answer, sources, blocked, guard_hits, num_chunks)`` so the
    rest of /api/query can stay unchanged.

    Sources are normalised so each item exposes ``metadata`` (with
    ``question``/``answer``/etc.) and ``score`` — matching the legacy
    pipeline's source format used by the JSON serialiser below.
    """
    retr = _get_islam360_retriever()
    # NOTE: ``retrieve()`` dispatches to retrieve_fast() by default
    # (BM25-first, no LLM rewrite/rerank) — the simple, fast pipeline that
    # matches the original working setup. It also auto-detects a sect in
    # the query (``"deobandi ka fatwa"``, ``"barelvi kehte hain"``) and
    # scopes the retrieval to that sect's source allow-list.
    # ``maslak`` (from the UI maslak dropdown) is the explicit override.
    from src.islam360.url_index import SECT_TO_SOURCES

    # Map legacy maslak UI values onto the canonical sect codes so an
    # explicit selection wins over auto-detection.
    MASLAK_TO_SECT: dict[str, str] = {
        "deobandi":     "deobandi",
        "barelvi":      "barelvi",
        "ahle_hadith":  "ahle_hadith",
        "ahle_hadees":  "ahle_hadith",
        "ahle hadees":  "ahle_hadith",
        "ahle hadith":  "ahle_hadith",
    }
    sect_override: str | None = None
    if maslak:
        sect_override = MASLAK_TO_SECT.get(maslak.strip().lower())
    if sect_override is not None and sect_override in SECT_TO_SOURCES:
        out = retr.retrieve_by_sect(question, sect_override, top_k=top_k)
    else:
        out = retr.retrieve(question, top_k=top_k)

    raw_sources = out.get("sources") or []
    sources: list[dict] = []
    for s in raw_sources:
        meta = dict(s.get("metadata") or {})
        # Make sure required display fields exist for the UI card.
        meta.setdefault("source_name", meta.get("scholar") or "Islam360")
        meta.setdefault("source_file", meta.get("source_file", ""))
        meta.setdefault("category", meta.get("category", "—"))
        meta.setdefault("maslak", meta.get("maslak", ""))
        sources.append({
            "id":       s.get("id"),
            "score":    float(s.get("final_score", s.get("rerank_score", 0.0))),
            "metadata": meta,
        })

    answer = out.get("answer") or NO_ANSWER_SENTINEL
    no_match = bool(out.get("no_match"))
    return answer, sources, no_match, [], len(sources)

# PageIndex (vectorless) retrieval — second mode toggleable from the UI
from pageindex.client import PageIndexClient
_pi_client = PageIndexClient()

# Raw Fatwas (inverted index) retrieval — third mode
from pageindex.raw_fatwas_client import RawFatwasClient
_raw_client = RawFatwasClient()

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CATEGORIES = [
    "ALL", "NAMAZ", "WUDU", "ZAKAT", "FAST", "HAJJ",
    "NIKKAH", "DIVORCE", "INHERITANCE", "FOOD", "JIHAD",
    "TAUHEED", "FORGIVING", "ADHAN", "OTHER",
]

_CAT_ICONS = {
    "ALL": "🕌", "NAMAZ": "🕋", "WUDU": "💧", "ZAKAT": "💰", "FAST": "🌙",
    "HAJJ": "🕌", "NIKKAH": "💍", "DIVORCE": "📜", "INHERITANCE": "⚖️",
    "FOOD": "🍽️", "JIHAD": "🌿", "TAUHEED": "☪️", "FORGIVING": "🤝",
    "ADHAN": "📢", "OTHER": "📖",
}

_CAT_UR = {
    "ALL": "تمام", "NAMAZ": "نماز", "WUDU": "وضو", "ZAKAT": "زکوٰۃ",
    "FAST": "روزہ", "HAJJ": "حج", "NIKKAH": "نکاح", "DIVORCE": "طلاق",
    "INHERITANCE": "وراثت", "FOOD": "کھانا", "JIHAD": "جہاد",
    "TAUHEED": "توحید", "FORGIVING": "معافی", "ADHAN": "اذان", "OTHER": "دیگر",
}

@app.template_filter('cat_icon')
def cat_icon(cat): return _CAT_ICONS.get(cat, "📖")

@app.template_filter('cat_display')
def cat_display(cat): return _CAT_UR.get(cat, cat)

# Expose filters as globals so they work inside Jinja2 calls in the template
app.jinja_env.globals['cat_icon']    = cat_icon
app.jinja_env.globals['cat_display'] = cat_display

_guard_cfg = GuardrailConfig(
    min_context_score=0.01,
    min_top_score=0.01,
    min_overlap_ratio=0.05,  # Very lenient: allow ~5% token overlap (paraphrases OK)
    min_urdu_ratio=0.10,
)


@lru_cache(maxsize=1)
def _get_expander_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


# ── Performance caches ───────────────────────────────────────────────────
# 1. Cache query preprocessing results — maps raw user input → (urdu_question, retrieval_query)
_EXPANSION_CACHE: dict[str, tuple[str, str]] = {}
_EXPANSION_CACHE_MAX = 2048

# 2. Cache full query results (identical questions get instant responses)
_RESULT_CACHE: dict[str, tuple[float, dict]] = {}  # key → (timestamp, response)
_RESULT_CACHE_MAX = 500
_RESULT_CACHE_TTL = 3600  # 1 hour


def _get_result_cache(key: str) -> dict | None:
    """Return cached result if fresh, else None."""
    entry = _RESULT_CACHE.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time.perf_counter() - ts > _RESULT_CACHE_TTL:
        del _RESULT_CACHE[key]
        return None
    return result


def _set_result_cache(key: str, result: dict) -> None:
    """Cache a successful result."""
    if len(_RESULT_CACHE) >= _RESULT_CACHE_MAX:
        # Evict oldest entry
        oldest = min(_RESULT_CACHE, key=lambda k: _RESULT_CACHE[k][0])
        del _RESULT_CACHE[oldest]
    _RESULT_CACHE[key] = (time.perf_counter(), result)


def _normalize_query_for_retrieval(question: str) -> str:
    q = normalize_urdu((question or "").strip())
    repl = {
        "ازو": "عضو",
        "اعضو": "عضو",
        "ازو خاص": "عضو خاص",
        "اعضو خاص": "عضو خاص",
    }
    for a, b in repl.items():
        q = q.replace(a, b)
    return " ".join(q.split())


def _is_intimacy_query(question: str) -> bool:
    q = _normalize_query_for_retrieval(question).lower()
    keys = [
        "بیوی", "شوہر", "عضو", "عضو خاص", "شرمگاہ", "منہ", "مباشرت",
        "جماع", "دبر", "قبل", "بوس", "oral",
    ]
    return any(k in q for k in keys)


def _rule_based_expansion(question: str) -> str:
    q = _normalize_query_for_retrieval(question)
    if _is_intimacy_query(q):
        return (
            f"{q} میاں بیوی مباشرت عضو تناسل شرمگاہ منہ میں لینا "
            "جماع کے آداب حلال یا حرام فقہ"
        )
    return q


def _is_formal_urdu(q: str) -> bool:
    """Fast heuristic: is this query already formal Urdu?

    Used to skip the LLM translation step on the hot path. A query counts
    as formal Urdu when >60% of non-space characters are in the Arabic
    script block and there are fewer than 3 Latin letters.
    """
    arabic = sum(
        1 for c in q
        if ("\u0600" <= c <= "\u06FF")
        or ("\u0750" <= c <= "\u077F")
        or ("\uFB50" <= c <= "\uFDFF")
        or ("\uFE70" <= c <= "\uFEFF")
    )
    latin = sum(1 for c in q if c.isascii() and c.isalpha())
    total = sum(1 for c in q if not c.isspace())
    if total == 0:
        return False
    return (arabic / total) > 0.60 and latin < 3


def _llm_translate_and_expand(question: str, already_urdu: bool) -> tuple[str, str]:
    """One LLM call: detect language, translate to Urdu, normalize slang, expand.

    Handles English, Roman Urdu, and informal/slang Urdu. Returns
    ``(urdu_question, retrieval_query)``.

    To keep BM25's lexical anchor intact across languages, the retrieval
    query is built deterministically as ``urdu_question + " " + synonyms``.
    The LLM only provides the list of synonyms — it never rewrites the
    question for retrieval. This guarantees that formal-Urdu and
    Roman-Urdu versions of the same question produce identical retrieval
    anchors.
    """
    try:
        settings = get_settings()
        client = _get_expander_client()
        if already_urdu:
            instruction = (
                "The user query is already in Urdu. Set urdu_question to the "
                "query verbatim (normalise only obvious typos/slang). "
                "Do NOT paraphrase or rewrite it."
            )
        else:
            instruction = (
                "The user query may be in English, Roman Urdu, or mixed. "
                "Translate it to concise formal Urdu (Arabic script). "
                "Preserve the exact intent — do not over-generalise."
            )
        system_msg = (
            "You are an Islamic-fiqh search assistant. "
            f"{instruction}\n\n"
            "Then produce 3–6 high-value Urdu/Arabic fiqh synonyms or "
            "near-synonyms that a fatwa corpus is likely to contain. "
            "Prefer specific fiqh terminology over generic words. "
            "Do NOT answer the question. Do NOT add explanations.\n\n"
            'Return ONLY a JSON object with these fields:\n'
            '  "urdu_question": string  (formal Urdu, 5-15 words)\n'
            '  "synonyms": array of 3-6 short strings (fiqh terms/phrases)'
        )
        user_msg = f"User query: {question[:500]}"
        comp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=220,
            response_format={"type": "json_object"},
        )
        raw = (comp.choices[0].message.content or "").strip()
        urdu_q = ""
        synonyms: list[str] = []
        try:
            data = json.loads(raw)
            urdu_q = " ".join(str(data.get("urdu_question", "")).split())
            syn_raw = data.get("synonyms") or []
            if isinstance(syn_raw, list):
                synonyms = [" ".join(str(s).split()) for s in syn_raw if str(s).strip()]
            elif isinstance(syn_raw, str):
                synonyms = [s.strip() for s in syn_raw.split("،") if s.strip()]
        except (json.JSONDecodeError, TypeError):
            log.warning("Preprocessor JSON parse failed; using raw text. raw=%.120s", raw)
            urdu_q = " ".join(raw.split())
        if not urdu_q:
            urdu_q = _normalize_query_for_retrieval(question)
        # Deterministic retrieval-query construction: verbatim question + synonyms.
        # This is the key invariant — same urdu_question → same retrieval_query,
        # regardless of input language.
        if synonyms:
            retrieval_q = f"{urdu_q} ({'، '.join(synonyms[:6])})"
        else:
            retrieval_q = urdu_q
        log.info(
            "Multilingual preprocess | urdu_mode=%s | original=%.80s | urdu=%.80s | syns=%d",
            already_urdu, question, urdu_q, len(synonyms),
        )
        return urdu_q, retrieval_q
    except Exception as exc:
        log.warning("Multilingual preprocessor failed; falling back: %s", exc)
        normalised = _normalize_query_for_retrieval(question)
        return normalised, _rule_based_expansion(normalised)


def _preprocess_query(question: str) -> tuple[str, str]:
    """Return ``(urdu_question, retrieval_query)`` for any input language.

    Uses a single unified LLM call for all languages so that formal Urdu
    and Roman Urdu variants of the same question produce the same
    retrieval query (and therefore the same fatwas). The ``already_urdu``
    hint tells the LLM to preserve the question verbatim instead of
    paraphrasing.
    """
    raw = (question or "").strip()
    if not raw or _args.dry_run:
        return raw, raw

    if raw in _EXPANSION_CACHE:
        return _EXPANSION_CACHE[raw]

    # Intimacy queries use a deterministic rule-based expansion instead of
    # the LLM — the rule set is hand-tuned to catch euphemisms the LLM may
    # refuse to expand.
    normalised = _normalize_query_for_retrieval(raw)
    if _is_intimacy_query(normalised):
        result = (normalised, _rule_based_expansion(normalised))
    else:
        already_urdu = _is_formal_urdu(raw)
        # When the input is already formal Urdu, feed the normalised
        # version to the LLM so typos/ZWJ noise don't leak into retrieval.
        llm_input = normalised if already_urdu else raw
        result = _llm_translate_and_expand(llm_input, already_urdu=already_urdu)

    if len(_EXPANSION_CACHE) < _EXPANSION_CACHE_MAX:
        _EXPANSION_CACHE[raw] = result
    return result


def _recover_answer_from_sources(question: str, sources: list[dict], maslak: str | None = None) -> str:
    """If model returned sentinel despite retrieved sources, synthesize an extractive answer.

    This keeps output grounded by using only provided chunks.
    """
    if _args.dry_run or not sources:
        return NO_ANSWER_SENTINEL
    try:
        settings = get_settings()
        client = _get_expander_client()

        blocks: list[str] = []
        for i, s in enumerate(sources[:3], 1):
            meta = s.get("metadata", s) if isinstance(s, dict) else {}
            q = str(meta.get("question", "")).strip()
            a = str(meta.get("answer", meta.get("text", ""))).strip()
            src = str(meta.get("source_name", meta.get("source_file", "نامعلوم"))).strip()
            blocks.append(
                f"[{i}] ماخذ: {src}\nسوال: {q}\nمتن: {a[:2200]}"
            )

        sect_line = f"مسلک: {maslak}\n" if maslak else ""
        user_prompt = (
            "درج ذیل فتاویٰ کے متن سے صرف استخراجی (extractive) جواب دیں۔\n"
            "قواعد:\n"
            "1) صرف نیچے دیے گئے متن سے جواب دیں۔\n"
            "2) اگر جزوی جواب ہو تو جزوی مگر واضح جواب دیں، Sentinel نہ دیں۔\n"
            "3) مختصر 4-8 نکات میں جواب دیں۔\n"
            "4) آخر میں [1] [2] جیسے حوالہ دیں۔\n\n"
            f"{sect_line}"
            f"سوال: {question}\n\n"
            "متن:\n" + "\n\n".join(blocks)
        )

        comp = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": "You are a strict extractive Urdu assistant. Use provided text only."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=min(700, settings.answer_max_tokens),
        )
        ans = (comp.choices[0].message.content or "").strip()
        return ans or NO_ANSWER_SENTINEL
    except Exception as exc:
        log.warning("Answer recovery failed: %s", exc)
        return NO_ANSWER_SENTINEL


def _query_terms(question: str) -> list[str]:
    import re
    txt = _normalize_query_for_retrieval(question).lower()
    toks = re.findall(r"[\u0600-\u06FFa-zA-Z0-9]{3,}", txt)
    stop = {
        "کیا", "کی", "کا", "کے", "ہے", "ہیں", "ہو", "ہوتا", "ہوتی", "ہوتے",
        "میں", "پر", "سے", "اور", "کب", "کس", "کیوں", "نہیں", "اگر",
    }
    return [t for t in toks if t not in stop]


def _count_relevant_sources(sources: list[dict], terms: list[str]) -> int:
    if not sources or not terms:
        return 0
    n = 0
    for s in sources[:3]:
        meta = s.get("metadata", s) if isinstance(s, dict) else {}
        hay = " ".join([
            str(meta.get("question", "")),
            str(meta.get("answer", ""))[:800],
            str(meta.get("source_file", "")),
        ]).lower()
        if any(t in hay for t in terms):
            n += 1
    return n

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        categories=CATEGORIES,
        dry_run=_args.dry_run,
        guardrails_on=_args.guardrails,
    )


@app.route("/api/query", methods=["POST"])
def api_query():
    data     = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    category = (data.get("category") or "").strip().upper()
    maslak   = (data.get("maslak") or "").strip()
    top_k    = int(data.get("top_k") or _args.top_k)
    use_grd  = bool(data.get("guardrails", _args.guardrails))
    validate  = bool(data.get("validate", True))
    use_query_expansion = bool(data.get("expand_query", True))

    if not question:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    cat_filter    = None  # NOTE: category filtering disabled—metadata not properly indexed
    maslak_filter = maslak if maslak and maslak != "ALL" else None

    # ── Full-query result cache check ──────────────────────────────────
    # Identical questions (same maslak + top_k + guardrails flags) return
    # the cached response instantly. TTL-based, 1 hour.
    norm_q = normalize_urdu((question or "").strip().lower())
    cache_key = f"{norm_q}|{maslak_filter or ''}|{top_k}|{int(use_grd)}|{int(validate)}"
    cached = _get_result_cache(cache_key)
    if cached is not None:
        # Clone so mutation on cached response doesn't leak
        resp = dict(cached)
        resp["from_cache"] = True
        return jsonify(resp)

    if use_query_expansion:
        urdu_q, retrieval_q = _preprocess_query(question)
    else:
        urdu_q, retrieval_q = question, question
    effective_top_k = max(top_k, 10) if _is_intimacy_query(urdu_q) else top_k

    t0 = time.perf_counter()
    try:
        # NOTE: We pass the RAW user question (not the legacy-preprocessed
        # ``urdu_q``) to the Islam360 retriever.  The legacy
        # ``_preprocess_query`` was tuned for a different corpus and its
        # synonym expansion drifts the query away from Islam360's wording.
        # Islam360 has its own purpose-built LLM rewriter that handles
        # Urdu, English, and roman-Urdu natively — feeding it the raw
        # question gives dramatically better retrieval than feeding it
        # twice-translated text.
        answer, sources, blocked, guard_hits, num_chunks = (
            _islam360_query_for_flask(
                question,
                top_k=effective_top_k,
                category=cat_filter,
                maslak=maslak_filter,
            )
        )

        if (answer or "").strip() == NO_ANSWER_SENTINEL and sources:
            recovered = _recover_answer_from_sources(urdu_q, sources, maslak_filter)
            if recovered and recovered.strip() != NO_ANSWER_SENTINEL:
                answer = recovered
                blocked = False
                guard_hits = [h for h in (guard_hits or []) if h != "HallucinationGuard"]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Output validation
        val_report = None
        if validate and not blocked:
            try:
                val_report = ov_validate(answer, sources)
            except Exception:
                pass

        # Normalise sources for JSON — dedupe by content first, then build
        # user-facing cards. Guards against any retrieval path that still
        # returns the same fatwa under multiple composite IDs.
        import re as _re
        _seen_content: set[str] = set()

        def _dedupe_key(meta: dict) -> str:
            """Aggressive content key — strips whitespace, punctuation,
            and diacritics so minor formatting variations collapse."""
            q = (meta.get("question") or meta.get("question_text") or "").strip()
            # Take first 120 chars of question only (answer fields can
            # have subtle formatting differences that break exact match)
            q = q[:120]
            # Collapse any whitespace, zero-width chars, and punctuation
            q = _re.sub(r"[\s\u200c\u200d\u200e\u200f\u061c\u064b-\u065f،۔؟\?،;,!\.\(\)\[\]'\"]+", "", q)
            return q.lower()

        clean_sources = []
        for s in sources:
            if len(clean_sources) >= 5:
                break
            meta = s.get("metadata", s) if isinstance(s, dict) else {}

            _key = _dedupe_key(meta)
            if _key and _key in _seen_content:
                continue
            if _key:
                _seen_content.add(_key)

            folder = meta.get("folder", "") or meta.get("source", "")
            reference_url = (
                meta.get("reference")
                or meta.get("url")
                or (meta.get("date") if str(meta.get("date", "")).startswith("http") else "")
            )
            source_name = (
                meta.get("source_name")
                or folder.replace("-ExtractedData-Output", "").strip()
                or meta.get("source_file", "—")
            )
            clean_sources.append({
                "category":    meta.get("category", "—"),
                "source_name": source_name,
                "maslak":      meta.get("maslak", ""),
                "sect":        meta.get("sect", ""),
                "source":      meta.get("source", ""),
                "source_file": meta.get("source_file", meta.get("source", "—")),
                "reference":   reference_url,
                "question":    meta.get("question", ""),
                # Full fatwa answer text (truncated at 1500 chars for transport).
                # The UI renders this verbatim so users can see exactly which
                # fatwa drove the answer, not just its question title.
                "answer":      str(meta.get("answer", meta.get("text", "")))[:1500],
                "fatwa_no":    meta.get("fatwa_no", ""),
                "score":       round(s.get("score", 0.0), 4) if isinstance(s, dict) else 0.0,
                "anchor_hits": s.get("anchor_hits", 0) if isinstance(s, dict) else 0,
                "rerank_score": (
                    round(s.get("rerank_score", 0.0), 3)
                    if isinstance(s, dict) and s.get("rerank_score") is not None
                    else None
                ),
                "rerank_reason": (
                    s.get("rerank_reason", "") if isinstance(s, dict) else ""
                ),
            })

        payload = {
            "answer":       answer,
            "sources":      clean_sources,
            "blocked":      blocked,
            "guard_hits":   guard_hits,
            "num_chunks":   num_chunks,
            "elapsed_ms":   elapsed_ms,
            "validation":   _fmt_validation(val_report),
            "original_question": question,
            "urdu_question":     urdu_q,
            "retrieval_query":   retrieval_q,
            "effective_top_k": effective_top_k,
            "dry_run":      _args.dry_run,
            "from_cache":   False,
        }
        # Cache the successful result (skip if blocked/empty)
        if answer and not blocked:
            _set_result_cache(cache_key, payload)
        return jsonify(payload)

    except Exception as exc:
        log.exception("Query failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


def _fmt_validation(report) -> dict | None:
    if not report:
        return None
    scores = report.get("scores", {})
    issues = [i.get("code", "") for i in report.get("issues", []) if i.get("severity") != "ok"]
    return {
        "valid":      report.get("valid", True),
        "grounding":  round(scores.get("groundedness", 0) * 100, 1),
        "urdu":       round(scores.get("urdu_ratio", 0) * 100, 1),
        "halluc":     round(scores.get("hallucination_risk", 0) * 100, 1),
        "issues":     issues,
    }


@app.route("/api/query-all-schools", methods=["POST"])
def api_query_all_schools():
    """Query each sect's source(s) independently and return three separate results.

    Behaviour (per the sect-aware spec):

    * Deobandi → retrieves ONLY from the Banuri source
    * Barelvi  → retrieves ONLY from the UrduFatwa source
    * Ahle Hadees → retrieves from IslamQA + FatwaQA only

    Each sect is retrieved with its own strict source allow-list filter,
    scored independently, and answered independently by the LLM using
    **only** that sect's fatwas as context.  Results are NEVER merged
    into a single ranked list — cross-sect contamination is considered
    a hard error and logged accordingly.
    """
    import concurrent.futures

    from src.islam360.retrieve import detect_sect_in_query
    from src.islam360.url_index import (
        ALL_SECTS,
        SECT_AHLE_HADITH,
        SECT_BARELVI,
        SECT_DEOBANDI,
    )

    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    category = (data.get("category") or "").strip().upper()
    top_k = int(data.get("top_k") or _args.top_k)
    use_query_expansion = bool(data.get("expand_query", True))

    if not question:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    if use_query_expansion:
        urdu_q, retrieval_q = _preprocess_query(question)
    else:
        urdu_q, retrieval_q = question, question
    effective_top_k = max(top_k, 10) if _is_intimacy_query(urdu_q) else top_k

    # Human-readable labels for each sect card in the UI.
    SECT_LABELS: dict[str, str] = {
        SECT_DEOBANDI:    "Deobandi",
        SECT_BARELVI:     "Barelvi",
        SECT_AHLE_HADITH: "Ahle Hadees",
    }
    SECT_SOURCE_DESC: dict[str, str] = {
        SECT_DEOBANDI:    "Banuri Institute",
        SECT_BARELVI:     "UrduFatwa",
        SECT_AHLE_HADITH: "IslamQA + FatwaQA",
    }

    retr = _get_islam360_retriever()

    t0 = time.perf_counter()

    # Run the three retrievals in parallel — they're independent (each
    # operates on a disjoint source allow-list) so there's no risk of
    # cross-talk, and parallelism makes the 3× LLM-synthesis cost roughly
    # 1× wall-clock.
    def _run_sect(sect_code: str) -> dict:
        return retr.retrieve_by_sect(question, sect_code, top_k=effective_top_k)

    per_sect: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_run_sect, s): s for s in ALL_SECTS}
        for fut in concurrent.futures.as_completed(futs):
            sect = futs[fut]
            try:
                per_sect[sect] = fut.result()
            except Exception as exc:
                log.exception("Sect %s retrieval failed: %s", sect, exc)
                per_sect[sect] = {
                    "answer": "(retrieval failed)",
                    "sources": [],
                    "no_match": True,
                    "log": {"error": str(exc)},
                }

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    def _format_sources(raw_sources: list[dict]) -> list[dict]:
        out: list[dict] = []
        for s in (raw_sources or [])[:5]:
            meta = s.get("metadata", s) if isinstance(s, dict) else {}
            folder = meta.get("folder", "") or meta.get("source", "")
            reference_url = (
                meta.get("reference")
                or meta.get("url")
                or (meta.get("date") if str(meta.get("date", "")).startswith("http") else "")
            )
            source_name = (
                meta.get("source_name")
                or folder.replace("-ExtractedData-Output", "").strip()
                or meta.get("source_file", "Islam360")
            )
            out.append({
                "category":    meta.get("category", "—"),
                "source_name": source_name,
                "maslak":      meta.get("maslak", ""),
                "sect":        meta.get("sect", ""),
                "source":      meta.get("source", ""),
                "source_file": meta.get("source_file", meta.get("source", "—")),
                "reference":   reference_url,
                "question":    meta.get("question", ""),
                "answer":      str(meta.get("answer", meta.get("text", "")))[:1500],
                "fatwa_no":    meta.get("fatwa_no", ""),
                "score":       round(s.get("score", s.get("final_score", 0.0)), 4)
                                    if isinstance(s, dict) else 0.0,
                "anchor_hits": s.get("anchor_hits", 0) if isinstance(s, dict) else 0,
                "rerank_score": (
                    round(s.get("rerank_score", 0.0), 3)
                    if isinstance(s, dict) and s.get("rerank_score") is not None
                    else None
                ),
                "rerank_reason": (
                    s.get("rerank_reason", "") if isinstance(s, dict) else ""
                ),
            })
        return out

    # Preserve the canonical order: Deobandi → Barelvi → Ahle Hadees.
    results = []
    for sect in ALL_SECTS:
        payload = per_sect.get(sect, {})
        sources = _format_sources(payload.get("sources"))
        results.append({
            "maslak":       SECT_LABELS[sect],
            "sect":         sect,
            "source_label": SECT_SOURCE_DESC[sect],
            "answer":       payload.get("answer") or "(no answer)",
            "no_match":     bool(payload.get("no_match")),
            "sources":      sources,
            "num_chunks":   len(sources),
            "elapsed_ms":   elapsed_ms if sect == ALL_SECTS[0] else 0,
        })

    return jsonify({
        "query":             question,
        "original_question": question,
        "urdu_question":     urdu_q,
        "retrieval_query":   retrieval_q,
        "effective_top_k":   effective_top_k,
        "detected_sect":     detect_sect_in_query(question),
        "category":          category if category and category != "ALL" else None,
        "results":           results,
        "dry_run":           _args.dry_run,
    })


# ── PageIndex (vectorless) routes ────────────────────────────────────────────

@app.route("/api/search_pageindex", methods=["POST"])
def search_pageindex():
    """Vectorless retrieval via LLM tree navigation over the PageIndex tree.

    Returns 4 results (one per madhab). See ``pageindex/`` for the
    full pipeline. Wholly independent of /api/query — both modes can
    coexist.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw_query = (data.get("question") or "").strip()
    if not raw_query:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    t0 = time.perf_counter()
    try:
        payload = _pi_client.search(raw_query)
    except FileNotFoundError as exc:
        log.warning("PageIndex not built yet: %s", exc)
        return jsonify({
            "error": "PageIndex not built. Run "
                     "`python -m pageindex.convert && "
                     "python -m pageindex.ingest_pageindex` first.",
            "detail": str(exc),
        }), 503
    except Exception as exc:
        log.exception("PageIndex search failed")
        return jsonify({"error": str(exc)}), 500

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    payload["elapsed_ms"] = elapsed_ms
    payload["dry_run"]    = _args.dry_run
    payload["mode"]       = "pageindex"
    return jsonify(payload)


@app.route("/api/summarise", methods=["POST"])
def summarise_fatwa():
    """On-demand 2-3 sentence Urdu summary of a single fatwa.

    Triggered by the "خلاصہ" button on each PageIndex result card.
    """
    data = request.get_json(force=True, silent=True) or {}
    fatwa_id = (data.get("fatwa_id") or "").strip()
    if not fatwa_id:
        return jsonify({"error": "fatwa_id missing"}), 400
    try:
        result = _pi_client.summarise(fatwa_id)
    except KeyError:
        return jsonify({"error": "fatwa_id not found"}), 404
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        log.exception("summarise failed")
        return jsonify({"error": str(exc)}), 500
    return jsonify(result)


@app.route("/api/search_raw_fatwas", methods=["POST"])
def search_raw_fatwas():
    """Raw fatwas retrieval via inverted index + keyword scoring.

    No LLM calls, no embeddings — pure keyword matching. Sub-second.
    """
    data = request.get_json(force=True, silent=True) or {}
    raw_query = (data.get("question") or "").strip()
    if not raw_query:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    t0 = time.perf_counter()
    try:
        payload = _raw_client.search(raw_query)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        log.exception("Raw fatwas search failed")
        return jsonify({"error": str(exc)}), 500

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    payload["elapsed_ms"] = elapsed_ms
    payload["dry_run"]    = _args.dry_run
    payload["mode"]       = "pageindex-raw-fatwas"
    return jsonify(payload)


# ── Smart router (/api/query-smart) ──────────────────────────────────────────
#
# Escalating three-stage retrieval:
#
#   1. raw-fatwas fast probe   (rerank=False, ~200-500 ms, keyword only)
#   2. Islam360 retrieve_fast  (BM25 + strict LLM rerank, ~5-8 s)
#   3. PageIndex tree descent  (LLM category navigation, ~3-5 s)
#
# The first stage that clears its confidence bar wins; the rest are
# skipped.  See ``src/routing/router.py`` for the scoring functions
# and escalation rules, and ``src/config/settings.py`` for thresholds.

@app.route("/api/query-smart", methods=["POST"])
def api_query_smart():
    """Confidence-routed retrieval across raw-fatwas → Islam360 → PageIndex.

    Request body mirrors ``/api/query-all-schools``:

    .. code-block:: json

       { "question": "...", "top_k": 5, "force_path": "raw_fatwas|islam360|pageindex|null" }

    ``force_path`` is an escape-hatch for debugging / ablation — when
    set, the router runs ONLY that stage regardless of confidence.

    Response is the same three-sect shape as ``/api/query-all-schools``
    PLUS a ``routing`` sub-object with the full escalation trace.
    """
    from src.routing import route_query

    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    top_k = int(data.get("top_k") or _args.top_k)
    force_path = data.get("force_path")
    if force_path not in (None, "", "raw_fatwas", "islam360", "pageindex"):
        return jsonify({"error": f"invalid force_path={force_path!r}"}), 400
    if not question:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    t0 = time.perf_counter()
    try:
        result = route_query(
            question,
            top_k=top_k,
            retriever=_get_islam360_retriever(),
            pi_client=_pi_client,
            raw_client=_raw_client,
            force_path=force_path or None,
        )
    except Exception as exc:
        log.exception("smart-router failed")
        return jsonify({"error": str(exc)}), 500

    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    result["dry_run"] = _args.dry_run
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if _warmup_ready else "warming_up",
        "dry_run": _args.dry_run,
        "ready": _warmup_ready,
    })


# ── Warm-up: pre-load expensive objects in a background thread ───────────────

import threading

_warmup_ready = False


def _warmup():
    """Pre-load BM25 corpus, Pinecone index handle, and OpenAI clients
    so the first query doesn't pay the cold-start penalty."""
    global _warmup_ready
    t0 = time.perf_counter()
    log.info("Warming up caches (background)…")

    if _args.dry_run:
        log.info("  (dry-run — skipping BM25 / Pinecone warm-up)")
    else:
        # BM25 corpus (loaded from pickle or built once)
        try:
            from src.retrieval.hybrid_retriever import _get_bm25_corpus
            _get_bm25_corpus()
            log.info("  ✓ BM25 corpus cached")
        except Exception as exc:
            log.warning("  ✗ BM25 warm-up skipped: %s", exc)

        # Pinecone index handle + real warmup query (kills HTTPS handshake)
        try:
            from src.indexing.pinecone_store import init_index
            idx = init_index()
            # Real query to warm the HTTPS connection pool
            try:
                idx.query(vector=[0.0] * 1536, top_k=1, include_metadata=False)
                log.info("  ✓ Pinecone index cached + warmed (real query)")
            except Exception:
                log.info("  ✓ Pinecone index handle cached (warmup query failed — non-fatal)")
        except Exception as exc:
            log.warning("  ✗ Pinecone warm-up skipped: %s", exc)

    # OpenAI clients (embedding + chat)
    # We make REAL dummy calls here to trigger TLS handshake,
    # DNS resolution, and connection pool setup. Otherwise the first
    # real query pays a 10-15 second cold-connection penalty.
    try:
        from src.embedding.embedder import embed_single
        embed_single("warmup")  # triggers full HTTPS handshake
        log.info("  ✓ OpenAI embedding client warmed (real call)")
    except Exception as exc:
        log.warning("  ✗ OpenAI embedding warm-up skipped: %s", exc)

    try:
        from src.pipeline.rag import _get_chat_client
        _get_chat_client()
        log.info("  ✓ OpenAI chat client cached")
    except Exception as exc:
        log.warning("  ✗ OpenAI chat warm-up skipped: %s", exc)

    # Islam360 URL sidecar lookup (built once from raw CSVs, cached to disk)
    if not _args.dry_run:
        try:
            from src.islam360.url_index import get_url_lookup
            _url_map = get_url_lookup()
            log.info("  ✓ Islam360 URL lookup ready (%d ids)", len(_url_map))
        except Exception as exc:
            log.warning("  ✗ Islam360 URL lookup skipped: %s", exc)

    # PageIndex tree + flat lookup (skipped silently if not yet built)
    try:
        _pi_client.preload()
        log.info("  ✓ PageIndex tree + lookup loaded")
    except Exception as exc:
        log.warning("  ✗ PageIndex warm-up skipped: %s", exc)

    # Raw Fatwas inverted index
    try:
        _raw_client.preload()
        log.info("  ✓ Raw fatwas inverted index built")
    except Exception as exc:
        log.warning("  ✗ Raw fatwas warm-up skipped: %s", exc)

    _warmup_ready = True
    log.info("Warm-up done in %.1fs", time.perf_counter() - t0)


# Only start warm-up in the actual server process, not the reloader
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not _args.debug:
    threading.Thread(target=_warmup, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Islamic Fatawa RAG Web UI on http://%s:%d", _args.host, _args.port)
    app.run(host=_args.host, port=_args.port, debug=_args.debug)
