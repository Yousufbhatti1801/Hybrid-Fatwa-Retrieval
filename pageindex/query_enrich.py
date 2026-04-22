"""Query expansion for PageIndex: merge LLM search terms with fiqh synonym map.

Reuses the same hand-tuned term expansions as ``raw_fatwas_index`` for
recall; the LLM re-ranker (Step B) filters false positives.
"""

from __future__ import annotations

from pageindex.raw_fatwas_index import _expand_query_terms, _tokenize

# Cap merged list — full list is used for *hints*; scoring uses a subset.
_MAX_TERMS: int = 32


def merge_pageindex_keywords(
    base_keywords: list[str] | None,
    normalized_urdu: str = "",
    *,
    apply_synonym_expansion: bool = False,
) -> list[str]:
    """Merge user/LLM keywords and tokenize the Urdu line.

    ``apply_synonym_expansion`` (namaz/salah → اردو) helps recall in some
    paths but can pollute *lexical* candidate scoring; default OFF.
    """
    seen: dict[str, None] = {}
    out: list[str] = []

    def _add(t: str) -> None:
        t = (t or "").strip().lower()
        if len(t) < 2 or t in seen:
            return
        seen[t] = None
        out.append(t)

    for k in base_keywords or []:
        if isinstance(k, str):
            _add(k)
            for piece in k.replace("،", " ").split():
                _add(piece)

    blob = " ".join(
        s for s in (normalized_urdu,) if s
    )
    for t in _tokenize(blob):
        _add(t)

    if not out:
        return []

    if apply_synonym_expansion:
        try:
            for t in _expand_query_terms(_tokenize(" ".join(out))):
                if t:
                    _add(str(t))
        except Exception:
            pass

    return out[:_MAX_TERMS]
