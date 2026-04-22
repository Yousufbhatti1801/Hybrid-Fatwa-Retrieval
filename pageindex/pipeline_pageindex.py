"""Single LLM call: normalize and classify the user's fiqh query.

Primary schema (requested by product):

    - ``normalized_query`` : concise formal Urdu query
    - ``enhanced_query``   : retrieval-ready fiqh phrasing
    - ``domain``           : one of the six Urdu fiqh domains
    - ``keywords``         : 3-8 core retrieval keywords

Backward compatibility keys are still returned for existing callers:
``core_question``, ``normalized_urdu``, ``category_hint``, ``search_terms``.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

# Make src.* importable when running standalone
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI                              # noqa: E402
from src.config import get_settings                    # noqa: E402

logger = logging.getLogger(__name__)

# Same category vocabulary as app.py CATEGORIES (line 112).
_CATEGORIES = [
    "NAMAZ", "WUDU", "ZAKAT", "FAST", "HAJJ", "NIKKAH", "DIVORCE",
    "INHERITANCE", "FOOD", "JIHAD", "TAUHEED", "FORGIVING", "ADHAN",
    "OTHER",
]

_DOMAINS = [
    "عبادات",
    "معاملات",
    "نکاح و طلاق",
    "معاشرت / ازدواجی تعلقات",
    "اخلاقیات",
    "دیگر",
]

ALL_SCHOOLS = ["Banuri", "fatwaqa", "IslamQA", "urdufatwa"]


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


# ──────────────────────────────────────────────────────────────────────────
# Heuristic fallback (used on any LLM failure)
# ──────────────────────────────────────────────────────────────────────────

# 2+ chars: many fiqh terms are 2–3 characters; 3+ was dropping real tokens.
_TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]{2,}")
_STOP = {
    "کیا", "کی", "کا", "کے", "ہے", "ہیں", "ہو", "ہوتا", "ہوتی", "ہوتے",
    "میں", "پر", "سے", "اور", "کب", "کس", "کیوں", "نہیں", "اگر",
    "تو", "بھی", "نہ", "وہ", "یے", "ہوں", "گا", "گے", "تھا",
    "is", "of", "in", "to", "or", "the", "and", "a", "it",
}
_STOP2 = frozenset(
    {"ہے", "میں", "سے", "کا", "کی", "کے", "نہ", "وہ", "یہ", "تو", "کو", "پر", "is", "of", "in", "or", "to", "a", "it"},
)


def _token_terms(text: str, *, cap: int = 20) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in _TOKEN_RE.findall(text or ""):
        tl = t.strip()
        if not tl:
            continue
        if tl in _STOP:
            continue
        if len(tl) == 2 and tl in _STOP2:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(tl)
        if len(out) >= cap:
            break
    return out


def _detect_domain(text: str) -> str:
    t = (text or "").lower()

    if any(k in t for k in ("ازدواجی", "مباشرت", "ہمبستری", "oral", "intimacy", "شوہر", "بیوی", "الزوجین", "زوجین")):
        if any(k in t for k in ("طلاق", "خلع", "divorce", "talaq")):
            return "نکاح و طلاق"
        return "معاشرت / ازدواجی تعلقات"
    if any(k in t for k in ("طلاق", "خلع", "عدت", "نکاح", "طلاق", "divorce", "talaq", "nikah", "khula")):
        return "نکاح و طلاق"
    if any(k in t for k in ("نماز", "روزہ", "زکو", "زکات", "وضو", "غسل", "حج", "عمرہ", "اذان", "سجدہ", "fast", "zakat", "wudu", "hajj", "salah", "namaz")):
        return "عبادات"
    if any(k in t for k in ("سود", "ربا", "بینک", "قرض", "کاروبار", "تجارت", "finance", "business", "loan", "interest")):
        return "معاملات"
    if any(k in t for k in ("غیبت", "جھوٹ", "حسد", "تکبر", "اخلاق", "slander", "backbiting", "lying")):
        return "اخلاقیات"
    return "دیگر"


def _infer_category_hint(enhanced_query: str, domain: str, keywords: list[str]) -> str | None:
    blob = f"{enhanced_query} {' '.join(keywords)}".lower()
    checks = [
        ("ZAKAT", ("زکو", "زکات", "زکوۃ", "zakat", "zakah", "فطرانہ")),
        ("FAST", ("روزہ", "صوم", "fast", "roza", "sawm")),
        ("NAMAZ", ("نماز", "صلا", "salah", "salat", "namaz")),
        ("WUDU", ("وضو", "غسل", "تیمم", "طہارت", "wudu", "ablution")),
        ("HAJJ", ("حج", "عمرہ", "hajj", "umrah")),
        ("DIVORCE", ("طلاق", "خلع", "عدت", "talaq", "divorce", "khula")),
        ("NIKKAH", ("نکاح", "ازدواجی", "زوجین", "شوہر", "بیوی", "nikah", "marriage")),
        ("INHERITANCE", ("وراثت", "میراث", "ترکہ", "inheritance")),
        ("FOOD", ("حلال", "حرام", "کھانا", "food", "halal")),
        ("JIHAD", ("جہاد", "jihad")),
        ("TAUHEED", ("توحید", "شرک", "tauheed", "tawheed")),
        ("ADHAN", ("اذان", "اقامت", "adhan", "azan")),
    ]
    for cat, terms in checks:
        if any(term in blob for term in terms):
            return cat
    if domain in {"نکاح و طلاق", "معاشرت / ازدواجی تعلقات"}:
        return "NIKKAH"
    return None


def _heuristic_extract(raw_query: str) -> dict:
    raw = (raw_query or "").strip()
    toks = _token_terms(raw, cap=16)
    domain = _detect_domain(raw)
    normalized = raw
    enhanced = raw
    category_hint = _infer_category_hint(enhanced, domain, toks)
    return {
        "normalized_query": normalized,
        "enhanced_query":   enhanced,
        "domain":           domain,
        "core_question":    enhanced,
        "normalized_urdu":  normalized,
        "category_hint":    category_hint,
        "keywords":          toks[:8],
        "search_terms":      toks[:16],
        "schools_to_search": list(ALL_SCHOOLS),
    }


# ──────────────────────────────────────────────────────────────────────────
# LLM extraction
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert Islamic jurisprudence (Fiqh) query processor. "
    "Always return STRICT JSON only. Do not answer the question. "
    "Detect true intent even if phrasing is informal, indirect, misspelled, or sensitive. "
    "Use respectful and proper fiqh terminology where helpful. "
    "For physical relations between الزوجین, classify as معاشرت / ازدواجی تعلقات "
    "unless divorce is explicitly asked. Never force طلاق terms if absent."
)

_USER_TEMPLATE = (
    "صارف کا سوال (Roman Urdu / Urdu / mixed ہو سکتا ہے):\n"
    "{q}\n\n"
    "Tasks:\n"
    "1) normalized_query: واضح، سادہ، معیاری اردو میں ایک سطر۔\n"
    "2) enhanced_query: retrieval-ready فقہی سوال، بنیادی موضوع برقرار رکھیں۔\n"
    "3) domain: exactly one of these values: {domains}.\n"
    "4) keywords: 3 to 8 precise retrieval keywords.\n\n"
    "Rules:\n"
    "- غیر ضروری الفاظ، slang، ambiguity ہٹا دیں۔\n"
    "- sensitive موضوعات میں مؤدبانہ فقہی اصطلاحات استعمال کریں۔\n"
    "- ازدواجی جسمانی تعلقات = معاشرت / ازدواجی تعلقات (جب تک طلاق explicit نہ ہو)۔\n"
    "- unrelated terms شامل نہ کریں۔\n\n"
    "Return strict JSON with EXACTLY this shape:\n"
    "{{\n"
    '  "normalized_query": "...",\n'
    '  "enhanced_query": "...",\n'
    '  "domain": "...",\n'
    '  "keywords": ["...", "...", "..."]\n'
    "}}"
)


def extract_core_question(raw_query: str, *, model: str | None = None) -> dict:
    """Run one ``gpt-4o-mini`` call to normalize + classify a fiqh query.

    Primary schema follows the strict 4-key contract:
    ``normalized_query``, ``enhanced_query``, ``domain``, ``keywords``.
    Backward compatibility aliases are also returned.
    """
    raw = (raw_query or "").strip()
    if not raw:
        return _heuristic_extract("")

    try:
        settings = get_settings()
        client = _client()
        comp = client.chat.completions.create(
            model=model or settings.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
            timeout=14,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    q=raw, domains=", ".join(_DOMAINS))},
            ],
        )
        text = (comp.choices[0].message.content or "").strip()
        parsed = json.loads(text)

        # Defensive coercion
        norm = str(
            parsed.get("normalized_query")
            or parsed.get("normalized_urdu")
            or ""
        ).strip()
        core = str(
            parsed.get("enhanced_query")
            or parsed.get("core_question")
            or norm
            or raw
        ).strip() or raw
        if not norm:
            norm = core

        domain = str(parsed.get("domain") or "").strip()
        if domain not in _DOMAINS:
            domain = _detect_domain(core or norm)

        keywords = parsed.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()][:8]
        if not keywords:
            keywords = _token_terms(f"{norm} {core}", cap=8)

        s_terms = parsed.get("search_terms") or []
        if not isinstance(s_terms, list):
            s_terms = []
        s_terms = [str(k).strip() for k in s_terms if str(k).strip()][:20]
        if not s_terms:
            s_terms = _token_terms(f"{norm} {core} {' '.join(keywords)}", cap=20)

        cat = _infer_category_hint(core, domain, keywords)

        return {
            "normalized_query": norm,
            "enhanced_query":   core,
            "domain":           domain,
            "core_question":     core,
            "normalized_urdu":   norm,
            "category_hint":     cat,
            "keywords":          keywords,
            "search_terms":      s_terms if s_terms else keywords,
            "schools_to_search": list(ALL_SCHOOLS),
        }
    except Exception as exc:
        logger.warning("extract_core_question LLM failed (%s); falling back",
                       exc)
        return _heuristic_extract(raw)
