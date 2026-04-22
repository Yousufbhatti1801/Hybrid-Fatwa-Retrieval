"""Mandatory fiqh query rewrite for retrieval.

Primary output contract (LLM stage):
    - normalized_query
    - enhanced_query
    - domain
    - keywords

The returned ``RewrittenQuery`` keeps legacy fields used by retrieval
(``semantic_query``, ``keyword_query``, ``intent_category``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

_DOMAINS = {
    "عبادات",
    "معاملات",
    "نکاح و طلاق",
    "معاشرت / ازدواجی تعلقات",
    "اخلاقیات",
    "دیگر",
}

_DOMAIN_TO_INTENT = {
    "عبادات": "WORSHIP",
    "معاملات": "FINANCE",
    "نکاح و طلاق": "MARRIAGE",
    "معاشرت / ازدواجی تعلقات": "MARRIAGE",
    "اخلاقیات": "SOCIAL",
    "دیگر": "GENERAL",
}

_TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]{2,}")
_STOP = {
    "کیا", "کی", "کا", "کے", "ہے", "ہیں", "ہو", "میں", "پر", "سے", "اور",
    "is", "of", "in", "to", "or", "the", "and", "a", "it",
}

# Rule-based expansions (English / Roman Urdu → fiqh keywords)
_TERM_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\binterest\b|\briba\b|\bsood\b|\bسود\b", re.I), "riba سود"),
    (re.compile(r"\bprayer\b|\bsalah\b|\bnamaz\b|\bنماز\b", re.I), "namaz salah نماز"),
    (re.compile(r"\bcharity\b|\bzakat\b|\bzakah\b|\bزکوٰۃ\b", re.I), "zakat zakah زکوٰۃ"),
    (re.compile(r"\bfasting\b|\broza\b|\bصوم\b", re.I), "roza صوم روزہ"),
    (re.compile(r"\bwudu\b|\bablution\b|\bوضو\b", re.I), "wudu wazu وضو"),
    (re.compile(r"\bdivorce\b|\btalaq\b|\bطلاق\b", re.I), "talaq talaaq طلاق"),
    (re.compile(r"\bmarriage\b|\bnikah\b|\bنکاح\b", re.I), "nikah nikah نکاح"),
    (re.compile(r"\bhajj\b|\bhaj\b|\bحج\b", re.I), "hajj حج"),
]

SYSTEM = """You are an expert Islamic jurisprudence (Fiqh) query processor.

Your job is to transform user queries into accurate, retrieval-ready queries.

Follow these rules:
1) Detect true intent even if wording is informal, misspelled, indirect, or sensitive.
2) Normalize into clear formal Urdu, removing slang/ambiguity.
3) Classify domain as exactly one of:
     - عبادات
     - معاملات
     - نکاح و طلاق
     - معاشرت / ازدواجی تعلقات
     - اخلاقیات
     - دیگر
4) IMPORTANT: physical relations between الزوجین belong to
     "معاشرت / ازدواجی تعلقات" unless divorce is explicitly asked.
5) Do not force unrelated terms (for example, do not inject طلاق if absent).
6) Do not answer the question.

Return ONLY strict JSON in this exact shape:
{
    "normalized_query": "...",
    "enhanced_query": "...",
    "domain": "...",
    "keywords": ["...", "...", "..."]
}
"""


@dataclass
class RewrittenQuery:
    optimized_query: str       # legacy alias = semantic_query
    semantic_query: str        # for the dense embedding (clean)
    keyword_query: str         # for BM25 (synonym-expanded)
    intent_category: str
    inferred_category_code: str
    language: str
    is_specific: bool
    synonyms: list[str]
    raw: dict


def _rule_expand(q: str) -> str:
    out = q
    for pat, ins in _TERM_MAP:
        if pat.search(out):
            out = f"{out} {ins}"
    return " ".join(out.split())


def _infer_domain(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ("ازدواجی", "مباشرت", "ہمبستری", "oral", "intimacy", "زوجین", "شوہر", "بیوی", "الزوجین")):
        if any(k in t for k in ("طلاق", "خلع", "divorce", "talaq")):
            return "نکاح و طلاق"
        return "معاشرت / ازدواجی تعلقات"
    if any(k in t for k in ("طلاق", "خلع", "عدت", "نکاح", "divorce", "talaq", "nikah", "khula")):
        return "نکاح و طلاق"
    if any(k in t for k in ("نماز", "روزہ", "زکو", "زکات", "وضو", "غسل", "حج", "عمرہ", "اذان", "fast", "zakat", "wudu", "hajj", "salah", "namaz")):
        return "عبادات"
    if any(k in t for k in ("سود", "ربا", "بینک", "قرض", "کاروبار", "تجارت", "finance", "business", "loan", "interest")):
        return "معاملات"
    if any(k in t for k in ("غیبت", "جھوٹ", "حسد", "تکبر", "اخلاق", "slander", "backbiting", "lying")):
        return "اخلاقیات"
    return "دیگر"


def _extract_keywords(text: str, *, cap: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in _TOKEN_RE.findall(text or ""):
        tk = t.strip()
        if not tk or tk in _STOP or tk in seen:
            continue
        seen.add(tk)
        out.append(tk)
        if len(out) >= cap:
            break
    return out


def rewrite_query(user_query: str, *, use_llm: bool = True) -> RewrittenQuery:
    """Rewrite user query for Islam360 retrieval."""
    q = (user_query or "").strip()
    base = _rule_expand(q)

    if not use_llm:
        domain = _infer_domain(q or base)
        norm = q or base
        enh = norm
        kws = _extract_keywords(norm, cap=8)
        keyword = _rule_expand(enh)
        if kws:
            keyword = (keyword + " " + " ".join(kws)).strip()
        return RewrittenQuery(
            optimized_query=enh,
            semantic_query=enh,
            keyword_query=keyword,
            intent_category=_DOMAIN_TO_INTENT.get(domain, "GENERAL"),
            inferred_category_code="",
            language="ur" if re.search(r"[\u0600-\u06FF]", q) else "en",
            is_specific=len(q.split()) <= 8,
            synonyms=kws,
            raw={
                "normalized_query": norm,
                "enhanced_query": enh,
                "domain": domain,
                "keywords": kws,
            },
        )

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key, max_retries=1, timeout=20.0)
    try:
        comp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=settings.islam360_query_rewrite_temperature,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"User query:\n{base[:2000]}"},
            ],
        )
        raw_txt = (comp.choices[0].message.content or "{}").strip()
        data = json.loads(raw_txt)
    except Exception as exc:
        logger.warning("LLM query rewrite failed: %s — using rule-only expansion", exc)
        return RewrittenQuery(
            optimized_query=q or base,
            semantic_query=q or base,
            keyword_query=base,
            intent_category="GENERAL",
            inferred_category_code="",
            language="ur" if re.search(r"[\u0600-\u06FF]", q) else "en",
            is_specific=len(q.split()) <= 8,
            synonyms=[],
            raw={"error": str(exc)},
        )

    norm = str(
        data.get("normalized_query")
        or data.get("semantic_query")
        or data.get("optimized_query")
        or q
        or base
    ).strip()
    enhanced = str(
        data.get("enhanced_query")
        or data.get("semantic_query")
        or data.get("optimized_query")
        or norm
        or q
        or base
    ).strip()

    domain = str(data.get("domain") or "").strip()
    if domain not in _DOMAINS:
        domain = _infer_domain(enhanced or norm)

    raw_keywords = data.get("keywords")
    if not isinstance(raw_keywords, list):
        raw_keywords = data.get("synonyms") if isinstance(data.get("synonyms"), list) else []
    keywords = [str(s).strip() for s in raw_keywords if str(s).strip()][:8]
    if not keywords:
        keywords = _extract_keywords(f"{norm} {enhanced}", cap=8)

    keyword = str(data.get("keyword_query", "")).strip()
    if not keyword:
        keyword = _rule_expand(enhanced)
    if keywords:
        keyword = (keyword + " " + " ".join(keywords)).strip()

    return RewrittenQuery(
        optimized_query=enhanced,
        semantic_query=enhanced,
        keyword_query=keyword,
        intent_category=str(data.get("intent_category") or _DOMAIN_TO_INTENT.get(domain, "GENERAL")),
        inferred_category_code="",  # deliberately unused — see retrieve.py docstring
        language=str(data.get("language") or ("ur" if re.search(r"[\u0600-\u06FF]", enhanced) else "en"))[:8],
        is_specific=bool(data.get("is_specific", len((enhanced or q).split()) <= 8)),
        synonyms=keywords,
        raw=data,
    )
