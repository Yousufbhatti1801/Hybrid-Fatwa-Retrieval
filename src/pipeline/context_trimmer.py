"""Token-budget enforcement for retrieved context.

Urdu text does not have a fast tiktoken tokenizer, so we use a conservative
character-based approximation: 1 token ≈ 4 characters (matches OpenAI's
rule-of-thumb for non-Latin scripts, erring on the safe side).

The trimmer:
1. Accepts the ranked list from ``hybrid_search()``.
2. Greedily includes results in score order until adding the next chunk
   would exceed the budget.
3. Truncates the *last* chunk that would overflow rather than dropping it
   entirely, so the budget is used as fully as possible.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4          # conservative estimate for Urdu/Arabic


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def trim_to_budget(
    results: list[dict],
    token_budget: int,
) -> list[dict]:
    """Return a copy of *results* trimmed to fit within *token_budget* tokens.

    Parameters
    ----------
    results:
        Ranked list of dicts from ``hybrid_search()``.  Each has a ``text``
        key with the chunk content.
    token_budget:
        Maximum total tokens permitted across all included chunks.

    Returns
    -------
    A (possibly shorter, possibly tail-truncated) list of result dicts that
    fits within the budget.  Metadata and scores are preserved unchanged.
    Score ordering is maintained.
    """
    trimmed: list[dict] = []
    used = 0

    for r in results:
        text = r.get("text", "")
        chunk_tokens = _estimate_tokens(text)

        if used + chunk_tokens <= token_budget:
            trimmed.append(r)
            used += chunk_tokens
        else:
            remaining = token_budget - used
            if remaining > 20:            # worth including a partial chunk
                max_chars = remaining * _CHARS_PER_TOKEN
                truncated = dict(r)
                truncated["text"] = text[:max_chars].rsplit(" ", 1)[0]  # word boundary
                trimmed.append(truncated)
            break

    return trimmed
