"""Guardrails for the Islamic Fatawa RAG pipeline.

Five independent guards are applied in sequence both *before* the LLM call
(pre-flight) and *after* (post-generation):

Pre-flight (applied to retrieved context)
-----------------------------------------
1. ContextGuard      — reject if no chunks were retrieved, or if every chunk
                       scores below ``min_context_score``.
2. ConfidenceGuard   — reject if the *best* retrieval score is below
                       ``min_top_score``.

Post-generation (applied to the LLM answer)
--------------------------------------------
3. HallucinationGuard — light overlap check: reject if the answer shares
                        almost no n-gram tokens with the retrieved context
                        (proxy for a model that ignored the context entirely).
4. LanguageGuard      — ensure output is predominantly Urdu script; strip
                        or warn on excessive Latin/foreign text.
5. LengthGuard        — reject answers that are shorter than a minimum word
                        count (empty/non-answers) or longer than a maximum
                        (rambling / context bleed).

All thresholds are configurable at construction time so they can be tuned
per deployment without touching any other module.

Usage
-----
::

    from src.pipeline.guardrails import guarded_query

    result = guarded_query("نماز کے احکام کیا ہیں؟")
    print(result["answer"])          # always safe to access
    print(result["guardrail_hits"])  # list of triggered guard names

    # Fine-tune thresholds:
    from src.pipeline.guardrails import GuardrailConfig, guarded_query
    cfg = GuardrailConfig(min_top_score=0.30, max_answer_words=600)
    result = guarded_query(question, config=cfg)

Standalone guard usage
----------------------
::

    from src.pipeline.guardrails import ContextGuard, GuardrailConfig
    guard = ContextGuard(GuardrailConfig())
    verdict = guard.check(retrieved_results)
    if not verdict.passed:
        print(verdict.reason)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL

logger = logging.getLogger(__name__)

# ── Arabic/Urdu script detector ───────────────────────────────────────────────

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _urdu_ratio(text: str) -> float:
    """Fraction of non-space characters that are Arabic-script."""
    stripped = text.replace(" ", "")
    if not stripped:
        return 0.0
    return len(_ARABIC_RE.findall(stripped)) / len(stripped)


def _word_count(text: str) -> int:
    return len(text.split())


def _token_set(text: str) -> set[str]:
    """Rough token set for overlap calculation (Urdu words + Latin words)."""
    return set(re.findall(r"[\u0600-\u06FF\w]{2,}", text))


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class GuardrailConfig:
    """Centralised thresholds for all five guards.

    Attributes
    ----------
    min_context_score:
        Average retrieval score across *all* returned chunks below which the
        context is considered too weak to answer.  [default: 0.10]
    min_top_score:
        The single highest retrieval score must exceed this value; otherwise
        even the best match is not confident enough.  [default: 0.20]
    min_context_chunks:
        Minimum number of non-empty chunks required to attempt an answer.
        [default: 1]
    min_overlap_ratio:
        Hallucination proxy: minimum fraction of answer tokens that must also
        appear in the retrieved context.  A value of 0 disables the check.
        [default: 0.10]
    min_answer_words:
        Answers shorter than this are treated as non-answers / refusals and
        kept as-is (not rejected — a legitimate "مناسب جواب دستیاب نہیں" is
        short by design).  The guard only *logs* a warning here.  [default: 5]
    max_answer_words:
        Answers longer than this word count are truncated at the last Urdu
        sentence boundary.  [default: 800]
    min_urdu_ratio:
        Minimum fraction of Arabic-script characters in the answer.  Answers
        with a lower ratio are flagged; the raw answer is still returned with
        a warning rather than being silently rejected.  [default: 0.40]
    sentinel:
        The exact Urdu refusal string used when a pre-flight guard triggers.
        Defaults to ``NO_ANSWER_SENTINEL`` from ``prompt_builder``.
    """

    min_context_score:   float = 0.10
    min_top_score:       float = 0.20
    min_context_chunks:  int   = 1
    min_overlap_ratio:   float = 0.10
    min_answer_words:    int   = 5
    max_answer_words:    int   = 800
    min_urdu_ratio:      float = 0.40
    # Query pre-flight: reject queries that are too short or contain no
    # Urdu script at all (catches blank input and purely-Latin off-topic queries).
    min_query_urdu_ratio: float = 0.25   # 0.0 = disable
    min_query_words:      int   = 1      # 0 = disable
    sentinel:            str   = field(default_factory=lambda: NO_ANSWER_SENTINEL)


# ── Verdict ───────────────────────────────────────────────────────────────────

@dataclass
class GuardVerdict:
    """Result returned by every guard's ``check()`` method."""

    guard: str        # name of the guard
    passed: bool
    reason: str = ""  # human-readable explanation when passed=False
    detail: dict = field(default_factory=dict)


# ── Guard 0 — Query validation (pre-flight, before retrieval) ─────────────────

class QueryGuard:
    """Guard 0 — Validates the query itself before any retrieval is attempted.

    Blocks:
    * Empty or whitespace-only queries.
    * Queries that are entirely Latin / non-Urdu script when
      ``config.min_query_urdu_ratio > 0``.

    Runs *before* the hybrid retriever is called so no Pinecone / OpenAI
    tokens are spent on clearly invalid inputs.
    """

    name = "QueryGuard"

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, question: str) -> GuardVerdict:
        stripped = question.strip()

        if not stripped:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason="Empty query — nothing to answer.",
                detail={"query_len": 0},
            )

        words = stripped.split()
        if self._cfg.min_query_words > 0 and len(words) < self._cfg.min_query_words:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason=(
                    f"Query too short ({len(words)} word(s)); "
                    f"minimum is {self._cfg.min_query_words}."
                ),
                detail={"word_count": len(words)},
            )

        if self._cfg.min_query_urdu_ratio > 0:
            ratio = _urdu_ratio(stripped)
            if ratio < self._cfg.min_query_urdu_ratio:
                return GuardVerdict(
                    guard=self.name,
                    passed=False,
                    reason=(
                        f"Query Urdu ratio {ratio:.1%} is below threshold "
                        f"{self._cfg.min_query_urdu_ratio:.1%}. "
                        "Query does not appear to be in Urdu."
                    ),
                    detail={"urdu_ratio": round(ratio, 3)},
                )

        return GuardVerdict(
            guard=self.name,
            passed=True,
            detail={"word_count": len(words), "urdu_ratio": round(_urdu_ratio(stripped), 3)},
        )


# ── Individual guards ─────────────────────────────────────────────────────────

class ContextGuard:
    """Guard 1 — Context presence and average score check.

    Rejects if:
    * Fewer than ``config.min_context_chunks`` non-empty chunks were retrieved.
    * The mean retrieval score across all chunks is below ``config.min_context_score``.
    """

    name = "ContextGuard"

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, results: list[dict]) -> GuardVerdict:
        non_empty = [r for r in results if r.get("text", "").strip()]
        n = len(non_empty)

        if n < self._cfg.min_context_chunks:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason=f"Only {n} non-empty chunk(s) retrieved "
                       f"(minimum: {self._cfg.min_context_chunks}).",
                detail={"chunks": n},
            )

        scores = [float(r.get("score", 0.0)) for r in non_empty]
        avg_score = sum(scores) / len(scores)

        if avg_score < self._cfg.min_context_score:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason=f"Average retrieval score {avg_score:.3f} is below "
                       f"threshold {self._cfg.min_context_score}.",
                detail={"avg_score": round(avg_score, 4), "chunks": n},
            )

        return GuardVerdict(
            guard=self.name,
            passed=True,
            detail={"avg_score": round(avg_score, 4), "chunks": n},
        )


class ConfidenceGuard:
    """Guard 2 — Top-score confidence check.

    Rejects if the single highest retrieval score is below
    ``config.min_top_score``, meaning even the best candidate is too weak.
    """

    name = "ConfidenceGuard"

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, results: list[dict]) -> GuardVerdict:
        if not results:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason="No results returned by retriever.",
                detail={"top_score": 0.0},
            )

        top_score = max(float(r.get("score", 0.0)) for r in results)

        if top_score < self._cfg.min_top_score:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason=f"Best retrieval score {top_score:.3f} is below "
                       f"confidence threshold {self._cfg.min_top_score}.",
                detail={"top_score": round(top_score, 4)},
            )

        return GuardVerdict(
            guard=self.name,
            passed=True,
            detail={"top_score": round(top_score, 4)},
        )


class HallucinationGuard:
    """Guard 3 — Answer-context overlap (hallucination proxy).

    Computes the fraction of unique answer tokens that also appear in the
    concatenated retrieved context.  A very low overlap suggests the model
    answered from parametric knowledge rather than the supplied context.

    Note: this is a *heuristic* — it cannot detect subtle paraphrasing but
    catches the most egregious cases (e.g. model ignoring the context
    entirely and answering from training data).
    """

    name = "HallucinationGuard"

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, answer: str, results: list[dict]) -> GuardVerdict:
        # Skip if ratio check is disabled or answer is the sentinel
        if self._cfg.min_overlap_ratio <= 0:
            return GuardVerdict(guard=self.name, passed=True,
                                detail={"skipped": True})
        if answer.strip() == self._cfg.sentinel.strip():
            return GuardVerdict(guard=self.name, passed=True,
                                detail={"skipped": "sentinel answer"})

        context_text = " ".join(
            r.get("text", "") + " " + r.get("metadata", {}).get("question", "")
            for r in results
        )
        answer_tokens = _token_set(answer)
        context_tokens = _token_set(context_text)

        if not answer_tokens:
            return GuardVerdict(guard=self.name, passed=True,
                                detail={"skipped": "empty answer"})

        overlap = answer_tokens & context_tokens
        ratio = len(overlap) / len(answer_tokens)

        if ratio < self._cfg.min_overlap_ratio:
            return GuardVerdict(
                guard=self.name,
                passed=False,
                reason=f"Answer–context token overlap {ratio:.1%} is below "
                       f"threshold {self._cfg.min_overlap_ratio:.1%}. "
                       "Possible out-of-context generation.",
                detail={
                    "overlap_ratio":   round(ratio, 4),
                    "answer_tokens":   len(answer_tokens),
                    "overlap_tokens":  len(overlap),
                },
            )

        return GuardVerdict(
            guard=self.name,
            passed=True,
            detail={"overlap_ratio": round(ratio, 4)},
        )


class LanguageGuard:
    """Guard 4 — Urdu output validation.

    Checks that the answer is predominantly Arabic-script.  When the ratio
    falls below ``config.min_urdu_ratio`` the guard flags it as a WARNING
    but does not hard-reject — mixed answers can still be partially useful.
    Instead, leading or trailing Latin paragraphs are stripped.

    Returns the (possibly cleaned) answer alongside the verdict.
    """

    name = "LanguageGuard"

    # Matches paragraphs/lines composed entirely of ASCII / Latin characters
    # (common when the model produces an English preamble or disclaimer)
    _LATIN_PARA_RE = re.compile(
        r"(?m)^[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\n]{30,}$"
    )

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, answer: str) -> tuple[GuardVerdict, str]:
        """Return ``(GuardVerdict, cleaned_answer)``."""
        if answer.strip() == self._cfg.sentinel.strip():
            return (
                GuardVerdict(guard=self.name, passed=True,
                             detail={"skipped": "sentinel answer"}),
                answer,
            )

        ratio = _urdu_ratio(answer)
        cleaned = answer

        # Strip obviously Latin-only lines / paragraphs
        cleaned = self._LATIN_PARA_RE.sub("", cleaned).strip()

        # Recalculate ratio on cleaned text
        cleaned_ratio = _urdu_ratio(cleaned)

        if cleaned_ratio < self._cfg.min_urdu_ratio:
            verdict = GuardVerdict(
                guard=self.name,
                passed=False,
                reason=f"Answer Urdu ratio {cleaned_ratio:.1%} is below "
                       f"threshold {self._cfg.min_urdu_ratio:.1%}. "
                       "Output may not be in Urdu.",
                detail={
                    "original_ratio": round(ratio, 3),
                    "cleaned_ratio":  round(cleaned_ratio, 3),
                },
            )
        else:
            verdict = GuardVerdict(
                guard=self.name,
                passed=True,
                detail={
                    "original_ratio": round(ratio, 3),
                    "cleaned_ratio":  round(cleaned_ratio, 3),
                },
            )

        return verdict, cleaned if cleaned else answer


class LengthGuard:
    """Guard 5 — Answer length validation.

    * Too short (< ``config.min_answer_words``): warns but does not reject —
      a legitimate refusal sentinel is intentionally short.
    * Too long (> ``config.max_answer_words``): truncates at the last Urdu
      sentence boundary (۔) within the allowed length and appends "…".
    """

    name = "LengthGuard"

    # Urdu full stop, question mark, exclamation mark
    _SENTENCE_END_RE = re.compile(r"[۔؟!.?]")

    def __init__(self, config: GuardrailConfig) -> None:
        self._cfg = config

    def check(self, answer: str) -> tuple[GuardVerdict, str]:
        """Return ``(GuardVerdict, possibly_truncated_answer)``."""
        wc = _word_count(answer)
        cleaned = answer

        if wc < self._cfg.min_answer_words:
            return (
                GuardVerdict(
                    guard=self.name,
                    passed=True,     # don't reject — could be a valid sentinel
                    reason=f"Short answer ({wc} words < {self._cfg.min_answer_words}).",
                    detail={"word_count": wc, "action": "warn_only"},
                ),
                cleaned,
            )

        if wc > self._cfg.max_answer_words:
            cleaned = self._truncate(answer)
            truncated_wc = _word_count(cleaned)
            return (
                GuardVerdict(
                    guard=self.name,
                    passed=True,     # truncated — still usable
                    reason=f"Answer truncated from {wc} → {truncated_wc} words "
                           f"(max: {self._cfg.max_answer_words}).",
                    detail={
                        "original_words": wc,
                        "truncated_words": truncated_wc,
                        "action": "truncated",
                    },
                ),
                cleaned,
            )

        return (
            GuardVerdict(
                guard=self.name,
                passed=True,
                detail={"word_count": wc},
            ),
            cleaned,
        )

    def _truncate(self, text: str) -> str:
        """Cut at last sentence boundary within the word budget."""
        words = text.split()
        budget_text = " ".join(words[: self._cfg.max_answer_words])
        # Find the last sentence-ending punctuation and cut there
        matches = list(self._SENTENCE_END_RE.finditer(budget_text))
        if matches:
            cut = matches[-1].end()
            return budget_text[:cut].strip()
        # No sentence boundary found — hard cut with ellipsis
        return budget_text.rstrip() + " …"


# ── Guardrail pipeline ────────────────────────────────────────────────────────

@dataclass
class GuardedResult:
    """Full result returned by :func:`guarded_query`.

    Always safe to access — ``answer`` is the final (possibly cleaned or
    replaced-by-sentinel) string.

    Attributes
    ----------
    answer:
        Final answer string after all guards have been applied.
    sources:
        Retrieved chunk metadata list (same as ``query()``).
    num_chunks:
        Number of chunks passed to the LLM (after trimming).
    passed:
        ``True`` if all pre-flight guards passed and the LLM was called.
        ``False`` if a pre-flight guard rejected the query (no LLM call).
    guardrail_hits:
        Names of guards that triggered (either rejected or modified the output).
    verdicts:
        Full :class:`GuardVerdict` objects for every guard, in order.
    prompt:
        The exact user-turn sent to the LLM (empty string when pre-flight
        guard rejected).
    timings:
        Per-step timing dict from the underlying ``query()`` call, plus
        ``guardrail_ms`` for guardrail overhead.
    raw_answer:
        The LLM answer *before* post-generation guards modified it.
    """

    answer: str
    sources: list[dict] = field(default_factory=list)
    num_chunks: int = 0
    passed: bool = True
    guardrail_hits: list[str] = field(default_factory=list)
    verdicts: list[GuardVerdict] = field(default_factory=list)
    prompt: str = ""
    timings: dict = field(default_factory=dict)
    raw_answer: str = ""


def run_guardrails(
    question: str,
    retrieved: list[dict],
    answer: str,
    config: GuardrailConfig,
) -> tuple[str, list[GuardVerdict], list[str]]:
    """Apply all guards to an already-retrieved + already-generated result.

    This function is exposed so callers that manage their own retrieval /
    generation loop can still use the guardrail logic.

    Parameters
    ----------
    question:
        Original user question (used only for logging).
    retrieved:
        List of retrieval result dicts (``{"text", "score", "metadata"}``).
    answer:
        Raw LLM answer string.
    config:
        Guardrail thresholds.

    Returns
    -------
    (final_answer, verdicts_list, hit_names_list)
    """
    verdicts: list[GuardVerdict] = []
    hits: list[str] = []

    # Guard 0 — query validation (before retrieval)
    query_v = QueryGuard(config).check(question)
    verdicts.append(query_v)
    if not query_v.passed:
        hits.append(query_v.guard)
        logger.warning("QueryGuard: %s | query=%.80s", query_v.reason, question)
        return config.sentinel, verdicts, hits

    # Pre-flight guards (on retrieved context — no LLM call yet boundary)
    context_v = ContextGuard(config).check(retrieved)
    verdicts.append(context_v)
    if not context_v.passed:
        hits.append(context_v.guard)

    conf_v = ConfidenceGuard(config).check(retrieved)
    verdicts.append(conf_v)
    if not conf_v.passed:
        hits.append(conf_v.guard)

    # Post-generation guards
    hall_v = HallucinationGuard(config).check(answer, retrieved)
    verdicts.append(hall_v)
    if not hall_v.passed:
        hits.append(hall_v.guard)
        logger.warning(
            "HallucinationGuard: %s | query=%.60s", hall_v.reason, question
        )
        answer = config.sentinel

    lang_v, answer = LanguageGuard(config).check(answer)
    verdicts.append(lang_v)
    if not lang_v.passed:
        hits.append(lang_v.guard)
        logger.warning(
            "LanguageGuard: %s | query=%.60s", lang_v.reason, question
        )

    len_v, answer = LengthGuard(config).check(answer)
    verdicts.append(len_v)
    if len_v.reason:   # triggered (warn or truncate)
        hits.append(len_v.guard)
        logger.info("LengthGuard: %s", len_v.reason)

    return answer, verdicts, hits


def guarded_query(
    question: str,
    *,
    config: GuardrailConfig | None = None,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    context_token_budget: int | None = None,
    category: str | None = None,
    bm25_corpus: Any | None = None,
) -> GuardedResult:
    """Run the full RAG pipeline wrapped with all five guardrails.

    This is the **recommended entry point** for production use.  It calls
    :func:`src.pipeline.rag.query` internally and wraps the result with
    safety checks.

    Parameters
    ----------
    question:
        Raw Urdu user question.
    config:
        :class:`GuardrailConfig` instance.  Defaults to ``GuardrailConfig()``
        (all default thresholds).
    top_k / dense_weight / sparse_weight / context_token_budget:
        Forwarded to the underlying ``query()`` call.
    bm25_corpus:
        Pre-loaded ``BM25Corpus`` instance forwarded to
        :func:`src.retrieval.hybrid_retriever.hybrid_search`.  Useful for
        testing with a mini corpus or when the corpus is already loaded in
        memory.  When *None* the retriever loads or builds it automatically.
    category:
        Optional category filter forwarded to hybrid retrieval.

    Returns
    -------
    :class:`GuardedResult`

    Examples
    --------
    ::

        from src.pipeline.guardrails import guarded_query, GuardrailConfig

        result = guarded_query("نماز قصر کے احکام")
        if result.passed:
            print(result.answer)
        else:
            print("Rejected by:", result.guardrail_hits)

        # Strict mode:
        strict = GuardrailConfig(min_top_score=0.40, min_overlap_ratio=0.20)
        result = guarded_query("سوال", config=strict)
    """
    import time
    cfg = config or GuardrailConfig()
    gr_t0 = time.perf_counter()
    verdicts: list[GuardVerdict] = []
    hits: list[str] = []

    # ── Step 1: Retrieval only (without LLM call yet) ─────────────────────
    # We run the retrieval step independently so we can apply pre-flight
    # guards before spending tokens on the LLM.
    from src.config import get_settings              # noqa: PLC0415
    from src.pipeline.context_trimmer import trim_to_budget  # noqa: PLC0415
    from src.pipeline.prompt_builder import build_messages   # noqa: PLC0415
    from src.preprocessing.urdu_normalizer import normalize_urdu  # noqa: PLC0415
    from src.retrieval.hybrid_retriever import hybrid_search  # noqa: PLC0415

    settings = get_settings()
    budget = context_token_budget or settings.context_token_budget

    # ── Step 0: Query guard — reject before touching the retriever ──────────
    query_v = QueryGuard(cfg).check(question)
    verdicts.append(query_v)
    if not query_v.passed:
        hits.append(query_v.guard)
        logger.warning("QueryGuard: %s | query=%.80s", query_v.reason, question)
        gr_elapsed = (time.perf_counter() - gr_t0) * 1000
        return GuardedResult(
            answer=cfg.sentinel,
            sources=[],
            num_chunks=0,
            passed=False,
            guardrail_hits=hits,
            verdicts=verdicts,
            prompt="",
            timings={"guardrail_ms": round(gr_elapsed, 1)},
            raw_answer="",
        )

    normalised = normalize_urdu(question)
    try:
        retrieved = hybrid_search(
            normalised,
            top_k=top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            category=category,
            bm25_corpus=bm25_corpus,
        )
    except Exception as _retrieval_exc:  # noqa: BLE001
        logger.error(
            "Retrieval failed — returning sentinel | query=%.80s | error=%s",
            question, _retrieval_exc,
        )
        return GuardedResult(
            answer=NO_ANSWER_SENTINEL,
            sources=[],
            num_chunks=0,
            passed=False,
            guardrail_hits=["RetrievalError"],
            verdicts=[],
            prompt="",
            timings={"retrieval_error": str(_retrieval_exc)},
            raw_answer="",
        )

    # ── Step 2: Pre-flight guards ─────────────────────────────────────────
    ctx_v = ContextGuard(cfg).check(retrieved)
    verdicts.append(ctx_v)
    if not ctx_v.passed:
        hits.append(ctx_v.guard)

    conf_v = ConfidenceGuard(cfg).check(retrieved)
    verdicts.append(conf_v)
    if not conf_v.passed:
        hits.append(conf_v.guard)

    if hits:
        # Pre-flight failed — return sentinel immediately, no LLM call
        logger.warning(
            "Pre-flight guardrail(s) triggered: %s | query=%.80s", hits, question
        )
        gr_elapsed = (time.perf_counter() - gr_t0) * 1000
        return GuardedResult(
            answer=cfg.sentinel,
            sources=[r["metadata"] for r in retrieved],
            num_chunks=len(retrieved),
            passed=False,
            guardrail_hits=hits,
            verdicts=verdicts,
            prompt="",
            timings={"guardrail_ms": round(gr_elapsed, 1)},
            raw_answer="",
        )

    # ── Step 3: LLM call ──────────────────────────────────────────────────
    trimmed = trim_to_budget(retrieved, budget)
    messages = build_messages(question, trimmed)
    user_message = messages[-1]["content"]

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=settings.openai_api_key)
    llm_t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0,
        max_tokens=settings.answer_max_tokens,
    )
    llm_ms = (time.perf_counter() - llm_t0) * 1000
    raw_answer: str = completion.choices[0].message.content or cfg.sentinel

    # ── Step 4: Post-generation guards ────────────────────────────────────
    hall_v = HallucinationGuard(cfg).check(raw_answer, trimmed)
    verdicts.append(hall_v)
    answer = raw_answer
    if not hall_v.passed:
        hits.append(hall_v.guard)
        logger.warning("HallucinationGuard triggered: %s", hall_v.reason)
        answer = cfg.sentinel

    lang_v, answer = LanguageGuard(cfg).check(answer)
    verdicts.append(lang_v)
    if not lang_v.passed:
        hits.append(lang_v.guard)
        logger.warning("LanguageGuard triggered: %s", lang_v.reason)

    len_v, answer = LengthGuard(cfg).check(answer)
    verdicts.append(len_v)
    if len_v.reason:
        hits.append(len_v.guard)
        logger.info("LengthGuard: %s", len_v.reason)

    gr_elapsed = (time.perf_counter() - gr_t0) * 1000

    if hits:
        logger.info(
            "Guardrail hits: %s | passed_preflight=%s | query=%.80s",
            hits, True, question,
        )

    return GuardedResult(
        answer=answer,
        sources=[r["metadata"] for r in trimmed],
        num_chunks=len(trimmed),
        passed=True,
        guardrail_hits=hits,
        verdicts=verdicts,
        prompt=user_message,
        timings={"llm_ms": round(llm_ms, 1), "guardrail_ms": round(gr_elapsed, 1)},
        raw_answer=raw_answer,
    )
