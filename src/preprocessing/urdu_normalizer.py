"""Urdu-specific text normalisation utilities."""

from __future__ import annotations

import re
import unicodedata

# ── Character normalisation map ───────────────────────────────────────────────
# Maps Arabic characters to their canonical Urdu equivalents
_CHAR_MAP: dict[str, str] = {
    "ك": "ک",    # Arabic kaf        → Urdu kaf
    "ي": "ی",    # Arabic yeh        → Urdu yeh
    "ئ": "ی",    # yeh with hamza    → Urdu yeh  (very common in Urdu)
    "ى": "ی",    # alef maqsurah     → Urdu yeh
    "ة": "ہ",    # taa marbuta       → Urdu heh
    "ؤ": "و",    # waw with hamza    → waw
    "ۀ": "ہ",    # heh with yeh      → heh
    "ۂ": "ہ",    # heh goal          → heh
    "ٱ": "ا",    # alef wasla        → alef
    "أ": "ا",    # alef with hamza above → alef
    "إ": "ا",    # alef with hamza below → alef
    "آ": "ا",    # alef with madda   → alef
    "ٻ": "ب",    # Sindhi dotless beh variant → beh
    "ٮ": "ب",    # dotless beh       → beh
    "\u200c": "",  # ZWNJ — remove
    "\u200d": "",  # ZWJ  — remove
    "\uFEFF": "",  # BOM  — remove
}

_CHAR_RE = re.compile("|".join(re.escape(k) for k in _CHAR_MAP))

# Diacritics / tashkeel (harakat) — usually not needed for retrieval
_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670"
    r"\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Collapse whitespace including zero-width spaces and directional marks
_WHITESPACE_RE = re.compile(
    r"[\s\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069]+"
)

# ── Punctuation standardisation ───────────────────────────────────────────────
# Arabic punctuation → standard equivalents
_PUNCT_MAP: dict[str, str] = {
    "،": "،",   # keep Urdu comma as-is (canonical)
    "؟": "؟",   # keep Urdu question mark as-is
    "؛": "؛",   # keep Urdu semicolon as-is
    "٪": "%",
    "٫": ".",
    "٬": ",",
}
_PUNCT_RE = re.compile("|".join(re.escape(k) for k in _PUNCT_MAP))

# Collapse repeated punctuation (e.g. !!!! → !)
_REPEAT_PUNCT_RE = re.compile(r"([!؟?.,،؛;:]{2,})")

# Remove stray non-Urdu/Arabic/Latin characters (keep digits too)
_JUNK_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
    r"a-zA-Z0-9\s!؟?.,،؛;:\-\(\)\[\]\"'\u060C\u061B\u061F]"
)


def normalize_urdu(text: str, *, strip_diacritics: bool = True) -> str:
    """Normalise Urdu text for consistent indexing and retrieval.

    Steps:
    1.  Unicode NFC normalisation
    2.  Character mapping  (Arabic variants → canonical Urdu)
    3.  Punctuation standardisation
    4.  Junk character removal
    5.  Optional diacritic (tashkeel) removal
    6.  Collapse repeated punctuation
    7.  Whitespace collapse
    8.  Strip leading/trailing whitespace
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = _CHAR_RE.sub(lambda m: _CHAR_MAP[m.group()], text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group()], text)
    text = _JUNK_RE.sub(" ", text)

    if strip_diacritics:
        text = _DIACRITICS_RE.sub("", text)

    text = _REPEAT_PUNCT_RE.sub(lambda m: m.group()[0], text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()
