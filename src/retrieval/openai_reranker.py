"""External OpenAI LLM reranker — a purpose-built precision filter for retrieved fatwas.

This is the canonical reranker for the whole pipeline.  It is deliberately
externalized into its own module so any retrieval path — Islam360 fast
path, legacy hybrid, PageIndex, future experiments — can call the SAME
well-tuned reranker with one line::

    from src.retrieval.openai_reranker import rerank_candidates

    kept = rerank_candidates(
        user_query="...",
        candidates=[{"id": ..., "metadata": {"question": ..., "answer": ...}}, ...],
        threshold=0.75,
        top_k=5,
    )

What this reranker does
-----------------------

The **RELEVANCE CONTRACT** enforced here is strict: a candidate fatwa is
deemed relevant only if the fatwa's **question** directly asks the same
fiqh issue as the user's query.  Partial keyword overlap, topic
adjacency, and "somewhat related" all score BELOW the 0.75 pass-bar and
are dropped.  This is the precision gate that prevents the UI from
showing tangentially related fatwas to the user.

It uses a single listwise scoring call (one OpenAI chat completion
across ALL candidates) rather than pairwise calls.  Listwise is faster
(1× the cost of pairwise × N), reduces drift between scores, and lets
the LLM apply relative judgements which empirically correlate better
with the user's "is this actually the same masala?" intuition.

Each score comes with a short `reason` string explaining the verdict,
which is logged and surfaced for debugging / observability.

Failure policy
--------------

The reranker **never raises**.  On any failure (network error, bad
JSON, hard timeout, missing API key) the returned score map is empty
and the caller is expected to fall back to the pre-rerank ordering.
This is explicit in every retrieval path that calls it.

Hard wall-clock timeouts are enforced via a ThreadPoolExecutor — the
OpenAI SDK's internal timeout has proven unreliable on some platforms
(hangs in SSL retry loops).  We kill the call after
``settings.islam360_rerank_timeout_s`` seconds regardless.
"""

from __future__ import annotations

import concurrent.futures as _cf
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)


# ── Prompt ─────────────────────────────────────────────────────────────
#
# The system prompt defines a 7-bucket rubric (1.00 / 0.90 / 0.80 /
# 0.75 / 0.60 / 0.40 / 0.00) with explicit "MIN PASS" and "REJECT"
# labels so the LLM has no ambiguity about the threshold semantics.
#
# Prior iteration lessons that are baked into this prompt:
#
#   1. Rate the QUESTION+answer_preview jointly, not the question
#      alone.  Urdu fatwa titles are often 2–5 words and need the
#      answer snippet to disambiguate scope.
#
#   2. Reject topic-adjacency explicitly.  Without this rule the LLM
#      happily rated "monkey fatwa" as a 0.6–0.7 match for a "dog"
#      question because they're both najasat/animal topics.
#
#   3. If unsure between two buckets, pick the LOWER one.  The
#      default is to REJECT marginally-related fatwas rather than
#      publish them — quality over quantity.
#
#   4. `reason` must be ≤12 words of English.  Longer reasons
#      produced unreliable JSON and wasted tokens; 12 words is
#      enough for a humans-debuggable log line.

_RERANK_SYS = (
    "You are a precise Islamic-fiqh relevance rater for a fatwa search "
    "engine. Rate how well each candidate fatwa addresses the USER "
    "QUERY.  A candidate is given as a short QUESTION title plus an "
    "ANSWER snippet — USE BOTH when judging.  Urdu fatwa titles are "
    "often terse (2–5 words), so the answer snippet is often what "
    "reveals whether the fatwa actually addresses the user's concern.\n\n"
    "SCORE RUBRIC (0.00 – 1.00):\n"
    "  1.00  — addresses the user's EXACT masala & sub-question\n"
    "  0.90  — SAME masala, slightly different phrasing or scope\n"
    "  0.80  — SAME fiqh topic & same concern, different sub-angle "
    "(e.g. user asks 'does X break namaz?', fatwa answers 'X in namaz "
    "is makrooh / breaks wudu' — both are the same masala)\n"
    "  0.75  — clearly the same fiqh issue, minor scope difference (MIN PASS)\n"
    "  0.60  — adjacent topic but different fiqh concern (REJECT)\n"
    "  0.40  — only keyword overlap, different topic (REJECT)\n"
    "  0.00  — unrelated (REJECT)\n\n"
    "INTERPRETATION GUIDE:\n"
    "  • Same masala = same subject + same fiqh concern.  If the user "
    "asks about laughter breaking namaz and the fatwa discusses laughter "
    "during namaz and its effect on namaz/wudu, that IS the same masala "
    "(rate 0.85–1.00), even if the fatwa title is just 'laughter in "
    "namaz' without spelling out 'does it break'.\n"
    "  • Reject topic-adjacency: a fatwa about monkeys is NOT a match "
    "for a dog question; a fatwa about Botox injection is NOT a match "
    "for breaking-fast.\n"
    "  • If unsure between two buckets, pick the LOWER one.\n\n"
    "OUTPUT FORMAT — return STRICT JSON only, no prose:\n"
    '  {"scores":[{"id":"<id>","score":<0-1 float>,"reason":"<<=12 words>"}, ...]}\n'
    "  • One entry per candidate, preserving input order.\n"
    "  • `reason` must be a terse English phrase explaining the score."
)


# ── Dataclasses ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankScore:
    """A single candidate's rerank result.

    Attributes
    ----------
    score
        Float in ``[0.0, 1.0]``.  Values at or above the caller's
        threshold pass the precision gate.
    reason
        Terse English phrase (≤12 words) explaining the verdict.
        Logged for observability; never shown to the end user.
    """

    score: float
    reason: str


# ── Low-level scoring API ─────────────────────────────────────────────


def score_candidates(
    user_query: str,
    candidates: list[dict[str, Any]],
    *,
    model: str | None = None,
    timeout_s: float | None = None,
    sect: str | None = None,
) -> dict[str, RerankScore]:
    """Ask OpenAI to score each candidate's relevance to *user_query*.

    Parameters
    ----------
    user_query
        The user's question (already canonicalized / compressed if the
        caller has a cheaper form).  Truncated to 500 chars internally.
    candidates
        A list of candidate dicts.  Each must have ``id`` and
        ``metadata`` keys; ``metadata`` should contain ``question`` and
        ``answer`` fields.  Any candidate without an ``id`` is skipped.
    model
        OpenAI chat model to use.  ``None`` → reads
        ``settings.islam360_rerank_model`` (falling back to
        ``settings.chat_model`` when that's empty).
    timeout_s
        Hard wall-clock timeout.  ``None`` → reads
        ``settings.islam360_rerank_timeout_s``.
    sect
        Optional sect code (``deobandi`` / ``barelvi`` / ``ahle_hadith``)
        — used only in log lines for debugging.

    Returns
    -------
    ``dict[candidate_id, RerankScore]``
        Empty dict on ANY failure — the caller MUST treat that as
        "rerank unavailable" and keep the pre-rerank ordering.  Never
        raises.

    Notes
    -----
    Runs a single listwise LLM call for ALL candidates.  Sends both the
    question title and a 350-char answer preview per candidate so the
    LLM can judge scope even when titles are terse (common in Urdu).
    """
    if not candidates:
        return {}

    settings = get_settings()
    model = model or settings.islam360_rerank_model or settings.chat_model
    timeout_s = (
        float(timeout_s)
        if timeout_s is not None
        else float(settings.islam360_rerank_timeout_s)
    )

    lines: list[str] = []
    for c in candidates:
        cid = c.get("id")
        if not isinstance(cid, str):
            continue
        m = c.get("metadata") or {}
        q = str(m.get("question", "") or "").strip().replace("\n", " ")
        a = str(m.get("answer", "") or "").strip().replace("\n", " ")
        a = re.sub(r"\s+", " ", a)
        lines.append(
            f'- id: "{cid}"\n'
            f'  question: "{q[:250]}"\n'
            f'  answer_preview: "{a[:350]}"'
        )
    if not lines:
        return {}

    user_msg = (
        f"USER QUERY:\n{user_query[:500]}\n\n"
        f"CANDIDATES ({len(lines)}):\n" + "\n".join(lines)
    )

    def _call() -> str:
        client = OpenAI(
            api_key=settings.openai_api_key,
            max_retries=0,
            timeout=max(5.0, timeout_s - 2.0),
        )
        comp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=60 + 50 * len(candidates),
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RERANK_SYS},
                {"role": "user", "content": user_msg[:12000]},
            ],
        )
        return (comp.choices[0].message.content or "").strip()

    raw = ""
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_call)
            raw = fut.result(timeout=timeout_s)
    except _cf.TimeoutError:
        logger.warning(
            "[openai-reranker] hard-timeout after %.1fs (sect=%s, n=%d) — "
            "falling back to pre-rerank ordering",
            timeout_s, sect, len(candidates),
        )
        return {}
    except Exception as exc:
        logger.warning(
            "[openai-reranker] LLM call failed (sect=%s, n=%d): %s — "
            "falling back to pre-rerank ordering",
            sect, len(candidates), exc,
        )
        return {}

    try:
        parsed = json.loads(raw)
        entries = parsed.get("scores") if isinstance(parsed, dict) else None
        if not isinstance(entries, list):
            raise ValueError("missing 'scores' array")
        out: dict[str, RerankScore] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("id")
            score = e.get("score")
            if not isinstance(cid, str):
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            out[cid] = RerankScore(
                score=max(0.0, min(1.0, score_f)),
                reason=str(e.get("reason", ""))[:140],
            )
        return out
    except Exception as exc:
        logger.warning(
            "[openai-reranker] bad JSON from LLM (sect=%s): %s — raw=%r",
            sect, exc, raw[:200],
        )
        return {}


# ── High-level: rerank + filter + sort ────────────────────────────────


def rerank_candidates(
    user_query: str,
    candidates: list[dict[str, Any]],
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    model: str | None = None,
    timeout_s: float | None = None,
    sect: str | None = None,
    attach_scores: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Rerank *candidates* and return ``(kept, rejected, applied)``.

    This is the high-level convenience wrapper most callers want.  It
    runs :func:`score_candidates`, annotates each candidate with its
    score/reason (when ``attach_scores=True``), then splits the pool
    into kept (≥ threshold, sorted desc) and rejected (< threshold) and
    returns both plus an ``applied`` flag.

    Parameters
    ----------
    threshold
        Minimum score to pass the precision gate.  ``None`` → reads
        ``settings.islam360_rerank_threshold`` (default 0.75).
    top_k
        Cap on the size of the ``kept`` list.  ``None`` → no cap
        (everything that clears the threshold is kept).
    attach_scores
        When True, each kept candidate gets two new keys added:
        ``rerank_score`` (float) and ``rerank_reason`` (str).  Rejected
        candidates ALSO get them so the caller can log the full trace.

    Returns
    -------
    (kept, rejected, applied)
        ``kept`` is the precision-filtered, score-sorted candidate
        list (at most ``top_k`` items).  ``rejected`` is every
        candidate that scored below the threshold OR wasn't scored at
        all (the latter are treated as uncertain → rejected, to keep
        the precision contract strict).  ``applied`` is True only when
        the LLM actually returned scores; when False, ``kept`` is
        empty and the caller should fall back to pre-rerank ordering.
    """
    if not candidates:
        return [], [], False

    settings = get_settings()
    thr = float(threshold) if threshold is not None else float(settings.islam360_rerank_threshold)

    scores = score_candidates(
        user_query,
        candidates,
        model=model,
        timeout_s=timeout_s,
        sect=sect,
    )
    if not scores:
        return [], [], False

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for c in candidates:
        cid = c.get("id")
        entry = scores.get(cid) if isinstance(cid, str) else None
        item = dict(c) if attach_scores else c
        if entry is None:
            if attach_scores:
                item["rerank_score"] = None
                item["rerank_reason"] = "no_llm_score_returned"
            rejected.append(item)
            continue
        if attach_scores:
            item["rerank_score"] = entry.score
            item["rerank_reason"] = entry.reason
        if entry.score >= thr:
            kept.append(item)
        else:
            rejected.append(item)

    # Primary sort by rerank score desc, tiebreak by original BM25 score.
    kept.sort(
        key=lambda c: (
            float(c.get("rerank_score") or 0.0),
            float(c.get("score") or 0.0),
        ),
        reverse=True,
    )
    if top_k is not None and top_k > 0:
        kept = kept[:top_k]

    return kept, rejected, True


# ── Public API ────────────────────────────────────────────────────────

__all__ = [
    "RerankScore",
    "score_candidates",
    "rerank_candidates",
]
