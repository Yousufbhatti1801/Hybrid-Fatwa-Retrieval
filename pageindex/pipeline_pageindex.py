"""Single LLM call: clean / classify the user's raw query.

Used by ``/api/search_pageindex`` to turn a raw question into:

  - ``core_question``  : a precise Urdu fiqh question
  - ``category_hint``  : one of the existing CATEGORIES (or None)
  - ``keywords``       : 3-8 useful retrieval keywords (Urdu/Arabic)

Falls back to a deterministic heuristic if the LLM call fails.
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

ALL_SCHOOLS = ["Banuri", "fatwaqa", "IslamQA", "urdufatwa"]


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


# ──────────────────────────────────────────────────────────────────────────
# Heuristic fallback (used on any LLM failure)
# ──────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]{3,}")
_STOP = {
    "کیا", "کی", "کا", "کے", "ہے", "ہیں", "ہو", "ہوتا", "ہوتی", "ہوتے",
    "میں", "پر", "سے", "اور", "کب", "کس", "کیوں", "نہیں", "اگر",
}


def _heuristic_extract(raw_query: str) -> dict:
    toks = [t for t in _TOKEN_RE.findall(raw_query or "") if t not in _STOP]
    return {
        "core_question":     (raw_query or "").strip(),
        "category_hint":     None,
        "keywords":          toks[:8],
        "schools_to_search": list(ALL_SCHOOLS),
    }


# ──────────────────────────────────────────────────────────────────────────
# LLM extraction
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a precise Urdu Islamic fiqh query analyser. "
    "Always return STRICT JSON. Never include any commentary."
)

_USER_TEMPLATE = (
    "صارف کا خام سوال (User's raw question):\n{q}\n\n"
    "آپ کا کام:\n"
    "1) سوال کو ایک واضح، مختصر فقہی سوال میں صاف کریں (Urdu).\n"
    "2) متعلقہ زمرہ (category_hint) چنیں — صرف اس فہرست سے: "
    "{cats}. اگر کوئی واضح نہ ہو تو null دیں۔\n"
    "3) 3 سے 8 کلیدی الفاظ (Urdu/Arabic) دیں۔\n\n"
    "Return strict JSON with this exact shape:\n"
    "{{\n"
    '  "core_question": "<one-line clean Urdu fiqh question>",\n'
    '  "category_hint": "<one of the listed categories, or null>",\n'
    '  "keywords":      ["…", "…"]\n'
    "}}\n"
    "Do not write anything outside the JSON."
)


def extract_core_question(raw_query: str, *, model: str | None = None) -> dict:
    """Run one ``gpt-4o-mini`` call to clean the query.

    Returns a dict with keys: ``core_question``, ``category_hint``,
    ``keywords``, ``schools_to_search``. Always returns a valid dict —
    on any error, falls back to a deterministic heuristic.
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
            max_tokens=200,
            timeout=10,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(
                    q=raw, cats=", ".join(_CATEGORIES))},
            ],
        )
        text = (comp.choices[0].message.content or "").strip()
        parsed = json.loads(text)

        # Defensive coercion
        core = (parsed.get("core_question") or raw).strip() or raw
        cat = parsed.get("category_hint")
        if isinstance(cat, str):
            cat = cat.strip().upper()
            if cat not in _CATEGORIES:
                cat = None
        else:
            cat = None
        keywords = parsed.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k).strip() for k in keywords if str(k).strip()][:8]

        return {
            "core_question":     core,
            "category_hint":     cat,
            "keywords":          keywords,
            "schools_to_search": list(ALL_SCHOOLS),
        }
    except Exception as exc:
        logger.warning("extract_core_question LLM failed (%s); falling back",
                       exc)
        return _heuristic_extract(raw)
