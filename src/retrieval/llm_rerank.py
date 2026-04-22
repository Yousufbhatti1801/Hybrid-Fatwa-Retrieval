"""LLM-based re-ranking of hybrid-retrieved fatwa chunks for topical relevance."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM = """You score how well each fatwa candidate matches the USER QUESTION for the same fiqh topic.
Score 0.0–1.0:
- 0.85–1.0: Same issue or directly answers the user (e.g. same hukm topic).
- 0.5–0.84: Related fiqh area but not the best match.
- 0.0–0.49: Different topic, tangential, or wrong masala.

Return ONLY JSON: {"scores": [float, ...]} — one score per candidate in order, same length as input."""


def rerank_fatwa_candidates(
    user_question: str,
    candidates: list[dict[str, Any]],
    *,
    final_k: int,
    min_relevance: float | None = None,
) -> list[dict[str, Any]]:
    """Re-order candidates by LLM relevance; drop low scores.

    ``candidates`` are dicts from ``hybrid_search`` (keys: text, score, metadata).
    """
    if not candidates:
        return []

    settings = get_settings()
    floor = min_relevance if min_relevance is not None else settings.retrieval_rerank_min_score

    if not settings.retrieval_rerank_enabled or len(candidates) == 1:
        return candidates[:final_k]

    blocks: list[str] = []
    for i, c in enumerate(candidates, 1):
        m = c.get("metadata") or {}
        q = str(m.get("question", "") or "")[:600]
        preview = str(c.get("text", "") or m.get("answer", ""))[:500]
        blocks.append(f"[{i}] Fatwa Q: {q}\n    Preview: {preview}")

    user_block = (
        f"USER QUESTION:\n{user_question[:1500]}\n\nCANDIDATES:\n"
        + "\n\n".join(blocks)
    )

    client = OpenAI(api_key=settings.openai_api_key, max_retries=1, timeout=45.0)
    try:
        comp = client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_block[:14000]},
            ],
        )
        data = json.loads((comp.choices[0].message.content or "{}"))
        scores = data.get("scores")
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError("bad scores shape")
        rel = [max(0.0, min(1.0, float(x))) for x in scores]
    except Exception as exc:
        logger.warning("LLM re-rank failed (%s) — using hybrid order", exc)
        return candidates[:final_k]

    ranked: list[dict[str, Any]] = []
    for c, r in zip(candidates, rel):
        item = dict(c)
        item["rerank_relevance"] = round(r, 4)
        hybrid = float(c.get("score", 0.0))
        item["score"] = round(0.45 * hybrid + 0.55 * r, 6)
        ranked.append(item)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    kept = [x for x in ranked if x.get("rerank_relevance", 0) >= floor]
    if not kept:
        logger.info("Re-rank dropped all candidates (floor=%.2f) — keeping top hybrid only", floor)
        return candidates[:final_k]

    return kept[:final_k]
