"""Post-generation output validator for the Islamic Fatawa RAG pipeline.

Distinct from :mod:`src.pipeline.guardrails`
--------------------------------------------
* **Guardrails** are *real-time binary gates* — they block generation when a
  signal is detected before or immediately after the LLM call.
* **This validator** is a *post-hoc auditor* — it inspects a completed
  answer and returns a structured report suitable for logging, monitoring,
  human review, CI quality gates, or batch re-ranking.  It never raises
  exceptions and never blocks anything.

Five checks are applied to every answer
----------------------------------------
1. **Groundedness**    — Measures token overlap between the answer and the
   retrieved context.  Low overlap flags a possible hallucination.
2. **Citation**        — Detects whether the answer references source fatawa
   using the expected in-text citation patterns ("فتویٰ 1", source filenames,
   fatwa numbers stored in metadata).
3. **Urdu quality**    — Checks Arabic-script ratio, detects mojibake,
   flags excessive Latin text, and catches the no-answer sentinel.
4. **Hallucination probe** — Detects specific-fact tokens in the answer
   (numerals, years, proper names) that do not appear anywhere in the
   retrieved context — a lightweight fabrication signal.
5. **Answer strength** — Flags answers that are empty, too short, are just
   the sentinel string, or are suspiciously long (context bleed).

Return schema
-------------
Every call to :func:`validate` returns::

    {
        "valid":   bool,           # False if any "error"-severity issue exists
        "issues":  [               # may be empty
            {
                "code":     str,   # e.g. "WEAK_ANSWER", "MISSING_CITATION"
                "severity": str,   # "error" | "warning" | "info"
                "message":  str,   # human-readable description
                "detail":   dict,  # quantitative evidence
            },
            ...
        ],
        "scores": {                # always populated
            "groundedness":    float,  # [0, 1] token-overlap proxy
            "citation_score":  float,  # [0, 1] fraction of sources cited
            "urdu_ratio":      float,  # [0, 1] Arabic-script character ratio
            "answer_words":    int,
            "context_words":   int,
            "hallucination_risk": float,  # [0, 1]  0 = clean, 1 = high risk
        },
        "answer_preview": str,     # first 200 chars of the answer
    }

Usage — single answer
---------------------
::

    from src.pipeline.output_validator import validate

    rag_result = query("نماز کی نیت کا طریقہ کیا ہے؟")
    report = validate(rag_result["answer"], rag_result["sources"])
    if not report["valid"]:
        for issue in report["issues"]:
            print(issue["code"], issue["message"])

Usage — validate a full rag.query() result dict
-----------------------------------------------
::

    from src.pipeline.output_validator import validate_rag_result

    rag_result = query("سوال یہاں")
    report = validate_rag_result(rag_result)
    print(report["valid"], report["issues"])

Usage — batch
-------------
::

    from src.pipeline.output_validator import validate_batch

    rows = [{"answer": ..., "sources": [...]} for ... in results]
    reports = validate_batch(rows)

CLI
---
::

    echo '{"answer": "جواب", "sources": [{"text": "..."}]}' | python -m src.pipeline.output_validator
    python -m src.pipeline.output_validator --help
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Regex / script helpers
# ─────────────────────────────────────────────────────────────────────────────

# Arabic-script Unicode ranges (covers Urdu fully)
_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Standalone Latin ("ASCII word") tokens — signals English/code insertion
_LATIN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")

# In-text citation patterns the model is instructed to use:
#   "فتویٰ 1"  "فتوی 2"  "(فتویٰ 3)"  "فتوی نمبر ۴"  "فتویٰ نمبر: 12345"
_FATWA_REF_RE = re.compile(
    r"فتو[یٰ]\s*(?:نمبر\s*:?\s*)?[\d۰-۹]+"
)

# Numeric tokens: Western digits + Urdu/Arabic-Indic digits + year-like strings
_NUMERIC_RE = re.compile(r"\b[\d۰-۹]{1,6}\b")

# Eastern Arabic / Urdu digit → Western ASCII digit translation table
_EASTERN_TO_WESTERN = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _normalize_digits(text: str) -> str:
    """Normalise Eastern Arabic/Urdu digits to Western ASCII digits."""
    return text.translate(_EASTERN_TO_WESTERN)

# Mojibake heuristic: Arabic text that contains mid-word Latin substitutions
# typically comes from a broken encoding conversion.
_MOJIBAKE_RE = re.compile(
    r"[\u0600-\u06FF]{1,3}[A-Za-z]{2,}[\u0600-\u06FF]{1,3}"
)


def _urdu_ratio(text: str) -> float:
    s = text.replace(" ", "")
    return len(_ARABIC_RE.findall(s)) / len(s) if s else 0.0


def _token_set(text: str) -> set[str]:
    """Multi-script token set (Urdu words + Latin words, length >= 2)."""
    return set(re.findall(r"[\u0600-\u06FF\w]{2,}", text))


def _bigram_set(text: str) -> set[tuple[str, str]]:
    """Ordered bigrams over the token set for richer overlap matching."""
    tokens = list(_token_set(text))
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidatorConfig:
    """Thresholds for all five validation checks.

    All defaults are tuned for production fatawa responses.
    Loosen them for testing with a mini corpus.

    Attributes
    ----------
    min_groundedness:
        Minimum token-overlap ratio between answer and context.
        Below this value a HALLUCINATION issue is raised.  [default: 0.08]
    min_citation_score:
        Fraction of retrieved sources that should be cited.  A value of 0
        means at least *one* citation of any kind is expected.  [default: 0.0
        — requires at least one citation pattern, not a per-source ratio]
    require_citation:
        When True, a MISSING_CITATION warning is issued if no in-text fatwa
        reference patterns are detected.  [default: True]
    min_urdu_ratio:
        Minimum Arabic-script character ratio in the answer.  [default: 0.35]
    max_latin_words:
        Maximum number of standalone Latin words allowed before the
        MIXED_LANGUAGE issue is raised.  [default: 8]
    min_answer_words:
        Answers shorter than this word count are flagged as WEAK_ANSWER.
        The sentinel string is treated by its own check.  [default: 6]
    max_answer_words:
        Answers longer than this word count are flagged as OVERLY_LONG.
        [default: 1000]
    max_hallucination_risk:
        Maximum allowed hallucination-risk score (fabricated numerals /
        year-like tokens not present in context).  [default: 0.4]
    mojibake_threshold:
        Maximum number of mojibake pattern matches before flagging.
        [default: 2]
    sentinel:
        The canonical no-answer string.  [default: NO_ANSWER_SENTINEL]
    """

    min_groundedness:      float = 0.08
    require_citation:      bool  = True
    min_urdu_ratio:        float = 0.35
    max_latin_words:       int   = 8
    min_answer_words:      int   = 6
    max_answer_words:      int   = 1000
    max_hallucination_risk: float = 0.40
    mojibake_threshold:    int   = 2
    sentinel:              str   = field(default_factory=lambda: NO_ANSWER_SENTINEL)


# ─────────────────────────────────────────────────────────────────────────────
# Issue codes
# ─────────────────────────────────────────────────────────────────────────────

class Code:
    """Canonical issue code strings."""

    # errors — make valid=False
    HALLUCINATION      = "HALLUCINATION"
    EMPTY_ANSWER       = "EMPTY_ANSWER"
    WRONG_LANGUAGE     = "WRONG_LANGUAGE"

    # warnings — valid stays True unless combined with an error
    WEAK_ANSWER        = "WEAK_ANSWER"
    SENTINEL_RETURNED  = "SENTINEL_RETURNED"
    MISSING_CITATION   = "MISSING_CITATION"
    MIXED_LANGUAGE     = "MIXED_LANGUAGE"
    MOJIBAKE           = "MOJIBAKE"
    OVERLY_LONG        = "OVERLY_LONG"
    HIGH_HALLUCINATION_RISK = "HIGH_HALLUCINATION_RISK"
    NO_CONTEXT         = "NO_CONTEXT"

    # info
    LOW_GROUNDEDNESS   = "LOW_GROUNDEDNESS"
    PARTIAL_CITATION   = "PARTIAL_CITATION"


_ERROR_CODES = {Code.HALLUCINATION, Code.EMPTY_ANSWER, Code.WRONG_LANGUAGE}


# ─────────────────────────────────────────────────────────────────────────────
# Issue dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    code:     str
    severity: str          # "error" | "warning" | "info"
    message:  str
    detail:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code":     self.code,
            "severity": self.severity,
            "message":  self.message,
            "detail":   self.detail,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks    (each returns list[ValidationIssue])
# ─────────────────────────────────────────────────────────────────────────────

def _check_groundedness(
    answer: str,
    context_text: str,
    cfg: ValidatorConfig,
) -> tuple[list[ValidationIssue], float]:
    """Check 1 — Token-overlap groundedness.

    Returns
    -------
    (issues, groundedness_score)
    """
    issues: list[ValidationIssue] = []
    answer_tokens  = _token_set(answer)
    context_tokens = _token_set(context_text)

    if not answer_tokens:
        return issues, 0.0

    if not context_tokens:
        issues.append(ValidationIssue(
            code=Code.NO_CONTEXT,
            severity="warning",
            message="No retrieved context was provided. Grounding cannot be assessed.",
            detail={"answer_tokens": len(answer_tokens)},
        ))
        return issues, 0.0

    overlap = answer_tokens & context_tokens
    score   = len(overlap) / len(answer_tokens)

    if score < cfg.min_groundedness:
        issues.append(ValidationIssue(
            code=Code.HALLUCINATION,
            severity="error",
            message=(
                f"Answer tokens overlap with context is only {score:.1%} "
                f"(minimum: {cfg.min_groundedness:.1%}). "
                "The answer may not be grounded in retrieved fatawa."
            ),
            detail={
                "overlap_ratio":   round(score, 4),
                "answer_tokens":   len(answer_tokens),
                "overlap_tokens":  len(overlap),
                "context_tokens":  len(context_tokens),
            },
        ))
    elif score < cfg.min_groundedness * 2:
        issues.append(ValidationIssue(
            code=Code.LOW_GROUNDEDNESS,
            severity="info",
            message=(
                f"Answer–context overlap is moderate ({score:.1%}). "
                "Review whether the answer is sufficiently grounded."
            ),
            detail={"overlap_ratio": round(score, 4)},
        ))

    return issues, round(score, 4)


def _check_citations(
    answer: str,
    sources: list[dict],
    cfg: ValidatorConfig,
) -> tuple[list[ValidationIssue], float]:
    """Check 2 — In-text citation detection.

    Looks for:
    * "فتویٰ N" / "فتوی نمبر N" in-text patterns (model-generated)
    * Source file basenames appearing in the answer
    * Fatwa number metadata fields appearing verbatim

    Returns
    -------
    (issues, citation_score)   citation_score in [0, 1]
    """
    issues: list[ValidationIssue] = []

    # 2a. Pattern-based citation detection
    fatwa_refs = _FATWA_REF_RE.findall(answer)

    # 2b. Source filename references
    source_refs: list[str] = []
    for s in sources:
        src = (s.get("source_file") or s.get("source") or "").strip()
        if src:
            # Check stem (without extension) to be lenient
            stem = src.rsplit(".", 1)[0]
            if stem and stem.lower() in answer.lower():
                source_refs.append(src)

    # 2c. Fatwa number metadata in answer
    meta_refs: list[str] = []
    for s in sources:
        fn = str(s.get("fatwa_no") or "").strip()
        if fn and fn in answer:
            meta_refs.append(fn)

    total_citations = len(fatwa_refs) + len(source_refs) + len(meta_refs)
    n_sources       = len(sources)

    # Derive a [0, 1] score: at least one citation = 0.5 baseline,
    # capped by the fraction of sources somehow referenced.
    if total_citations == 0:
        citation_score = 0.0
    elif n_sources == 0:
        citation_score = 0.5
    else:
        # Count how many distinct sources are referenced in any way
        cited_sources = len(source_refs) + len(meta_refs)
        citation_score = min(1.0, 0.5 + 0.5 * (cited_sources / n_sources))

    if cfg.require_citation and total_citations == 0 and n_sources > 0:
        issues.append(ValidationIssue(
            code=Code.MISSING_CITATION,
            severity="warning",
            message=(
                "The answer does not contain any detectable fatwa citation "
                "(e.g. 'فتویٰ 1'). The model was instructed to cite sources."
            ),
            detail={
                "fatwa_ref_patterns":  0,
                "source_refs":         0,
                "meta_fatwa_refs":     0,
                "sources_available":   n_sources,
            },
        ))
    elif total_citations > 0 and 0 < n_sources > 1 and len(source_refs) + len(meta_refs) < n_sources:
        issues.append(ValidationIssue(
            code=Code.PARTIAL_CITATION,
            severity="info",
            message=(
                f"Answer cites some sources but not all "
                f"({len(source_refs) + len(meta_refs)}/{n_sources} sources referenced)."
            ),
            detail={
                "fatwa_ref_patterns": len(fatwa_refs),
                "source_refs":        source_refs,
                "meta_fatwa_refs":    meta_refs,
                "sources_available":  n_sources,
            },
        ))

    return issues, round(citation_score, 3)


def _check_urdu_quality(
    answer: str,
    cfg: ValidatorConfig,
) -> tuple[list[ValidationIssue], float, int]:
    """Check 3 — Urdu script quality.

    Returns
    -------
    (issues, urdu_ratio, latin_word_count)
    """
    issues: list[ValidationIssue] = []
    ratio        = _urdu_ratio(answer)
    latin_words  = _LATIN_WORD_RE.findall(answer)
    mojibake_hits = _MOJIBAKE_RE.findall(answer)

    # 3a. Overall script ratio
    if ratio < 0.10:
        issues.append(ValidationIssue(
            code=Code.WRONG_LANGUAGE,
            severity="error",
            message=(
                f"Answer has only {ratio:.1%} Arabic-script characters — "
                "it appears not to be in Urdu at all."
            ),
            detail={"urdu_ratio": round(ratio, 3)},
        ))
    elif ratio < cfg.min_urdu_ratio:
        issues.append(ValidationIssue(
            code=Code.MIXED_LANGUAGE,
            severity="warning",
            message=(
                f"Answer Urdu ratio {ratio:.1%} is below threshold "
                f"{cfg.min_urdu_ratio:.1%}. The response may be primarily "
                "in a non-Urdu script."
            ),
            detail={"urdu_ratio": round(ratio, 3), "min_urdu_ratio": cfg.min_urdu_ratio},
        ))

    # 3b. Excessive Latin words (English phrases inserted mid-answer)
    if len(latin_words) > cfg.max_latin_words:
        sample = latin_words[:10]
        issues.append(ValidationIssue(
            code=Code.MIXED_LANGUAGE,
            severity="warning",
            message=(
                f"Answer contains {len(latin_words)} Latin words "
                f"(max allowed: {cfg.max_latin_words}). "
                "The model may have mixed English and Urdu."
            ),
            detail={"latin_word_count": len(latin_words), "sample": sample},
        ))

    # 3c. Mojibake detection
    if len(mojibake_hits) >= cfg.mojibake_threshold:
        issues.append(ValidationIssue(
            code=Code.MOJIBAKE,
            severity="warning",
            message=(
                f"Possible encoding corruption detected "
                f"({len(mojibake_hits)} mojibake-like sequences found)."
            ),
            detail={"count": len(mojibake_hits), "sample": mojibake_hits[:5]},
        ))

    return issues, round(ratio, 3), len(latin_words)


def _check_hallucination_probe(
    answer: str,
    context_text: str,
    cfg: ValidatorConfig,
) -> tuple[list[ValidationIssue], float]:
    """Check 4 — Specific-fact fabrication probe.

    Detects numeric tokens (years, fatwa numbers, quantities) that appear in
    the answer but are *absent* from the retrieved context.  These are the
    most dangerous hallucinations — invented specific facts.

    Returns
    -------
    (issues, hallucination_risk_score)   score in [0, 1]
    """
    issues: list[ValidationIssue] = []

    # Normalise Eastern Arabic / Urdu digits (e.g. ۳۹ → 39) before comparing
    # so digit-form differences between the answer and retrieved context do
    # not create false-positive "fabricated" tokens.
    norm_answer  = _normalize_digits(answer)
    norm_context = _normalize_digits(context_text)

    answer_nums  = set(_NUMERIC_RE.findall(norm_answer))
    context_nums = set(_NUMERIC_RE.findall(norm_context))

    # Exclude trivially short or ambiguous tokens (single digits are noise)
    answer_nums  = {n for n in answer_nums  if len(n) >= 2}
    context_nums = {n for n in context_nums if len(n) >= 2}

    fabricated = answer_nums - context_nums

    if not answer_nums:
        risk_score = 0.0
    else:
        risk_score = len(fabricated) / len(answer_nums)

    # Require at least 2 fabricated tokens before raising the flag.
    # A single un-grounded number (e.g. a Quranic verse reference like "39")
    # would otherwise produce a misleading 100% risk from just 1/1.
    if risk_score > cfg.max_hallucination_risk and len(fabricated) >= 2:
        issues.append(ValidationIssue(
            code=Code.HIGH_HALLUCINATION_RISK,
            severity="warning",
            message=(
                f"{len(fabricated)} numeric value(s) in the answer were not "
                f"found in any retrieved fatwa. These may be fabricated. "
                f"Risk score: {risk_score:.1%}."
            ),
            detail={
                "fabricated_tokens": sorted(fabricated)[:10],
                "risk_score":        round(risk_score, 3),
                "answer_numerics":   sorted(answer_nums)[:10],
            },
        ))

    return issues, round(risk_score, 3)


def _check_answer_strength(
    answer: str,
    cfg: ValidatorConfig,
) -> list[ValidationIssue]:
    """Check 5 — Answer strength and completeness."""
    issues: list[ValidationIssue] = []
    stripped = answer.strip()

    if not stripped:
        issues.append(ValidationIssue(
            code=Code.EMPTY_ANSWER,
            severity="error",
            message="The answer is empty.",
            detail={"length": 0},
        ))
        return issues

    # Sentinel detection
    if stripped == cfg.sentinel.strip():
        issues.append(ValidationIssue(
            code=Code.SENTINEL_RETURNED,
            severity="warning",
            message=(
                "The model returned the no-answer sentinel "
                f"'{cfg.sentinel}'. "
                "No fatawa matched the query, or context was insufficient."
            ),
            detail={"sentinel": cfg.sentinel},
        ))
        return issues   # further length checks are not meaningful for sentinel

    word_count = len(stripped.split())

    if word_count < cfg.min_answer_words:
        issues.append(ValidationIssue(
            code=Code.WEAK_ANSWER,
            severity="warning",
            message=(
                f"Answer is very short ({word_count} words, "
                f"minimum: {cfg.min_answer_words}). "
                "It may be incomplete or a partial refusal."
            ),
            detail={"word_count": word_count, "min_words": cfg.min_answer_words},
        ))

    if word_count > cfg.max_answer_words:
        issues.append(ValidationIssue(
            code=Code.OVERLY_LONG,
            severity="warning",
            message=(
                f"Answer is unusually long ({word_count} words, "
                f"maximum: {cfg.max_answer_words}). "
                "Possible context bleed or repetitive generation."
            ),
            detail={"word_count": word_count, "max_words": cfg.max_answer_words},
        ))

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Primary public API
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    answer: str,
    sources: list[dict],
    *,
    config: ValidatorConfig | None = None,
) -> dict:
    """Validate a single RAG answer against its retrieved sources.

    Parameters
    ----------
    answer:
        The LLM-generated answer string.
    sources:
        List of source metadata dicts as returned by ``rag.query()``
        (``result["sources"]``).  Each dict may contain ``text``,
        ``question``, ``answer``, ``source_file``, ``fatwa_no``, etc.
    config:
        Validation thresholds.  Defaults to ``ValidatorConfig()``.

    Returns
    -------
    ::

        {
            "valid":   bool,
            "issues":  [{"code", "severity", "message", "detail"}, ...],
            "scores":  {
                "groundedness":       float,
                "citation_score":     float,
                "urdu_ratio":         float,
                "hallucination_risk": float,
                "answer_words":       int,
                "context_words":      int,
            },
            "answer_preview": str,
        }
    """
    cfg = config or ValidatorConfig()
    all_issues: list[ValidationIssue] = []

    # Build a single context string from all sources.
    # Include fatwa_no so model-cited fatwa numbers are grounded in the
    # retrieved metadata and not flagged as fabricated numeric tokens.
    context_parts: list[str] = []
    for s in sources:
        for fld in ("text", "question", "answer"):
            val = s.get(fld, "")
            if val:
                context_parts.append(val)
        fn = str(s.get("fatwa_no", "") or "")
        if fn:
            context_parts.append(fn)
    context_text = " ".join(context_parts)
    context_words = len(context_text.split())

    # ── Run all five checks ───────────────────────────────────────────────
    strength_issues = _check_answer_strength(answer, cfg)
    all_issues.extend(strength_issues)

    # If the answer is empty or sentinel, skip metrics that require real text
    is_sentinel = answer.strip() == cfg.sentinel.strip()
    is_empty    = not answer.strip()

    if is_empty:
        groundedness   = 0.0
        citation_score = 0.0
        urdu_ratio     = 0.0
        hall_risk      = 0.0
    else:
        ground_issues, groundedness = _check_groundedness(answer, context_text, cfg)
        all_issues.extend(ground_issues)

        cite_issues, citation_score = _check_citations(answer, sources, cfg)
        if not is_sentinel:   # no citation expected for sentinel answer
            all_issues.extend(cite_issues)

        lang_issues, urdu_ratio, _ = _check_urdu_quality(answer, cfg)
        all_issues.extend(lang_issues)

        hall_issues, hall_risk = _check_hallucination_probe(answer, context_text, cfg)
        all_issues.extend(hall_issues)

    # ── Determine overall validity ────────────────────────────────────────
    valid = not any(i.code in _ERROR_CODES for i in all_issues)

    # ── Produce output ────────────────────────────────────────────────────
    return {
        "valid":  valid,
        "issues": [i.to_dict() for i in all_issues],
        "scores": {
            "groundedness":       groundedness,
            "citation_score":     citation_score,
            "urdu_ratio":         urdu_ratio,
            "hallucination_risk": hall_risk,
            "answer_words":       len(answer.split()),
            "context_words":      context_words,
        },
        "answer_preview": answer[:200],
    }


def validate_rag_result(
    rag_result: dict,
    *,
    config: ValidatorConfig | None = None,
) -> dict:
    """Convenience wrapper: validate a dict returned by :func:`src.pipeline.rag.query`.

    The result dict must have at minimum:
    * ``"answer"`` — the generated answer string
    * ``"sources"`` — list of source metadata dicts

    Returns the same schema as :func:`validate`.
    """
    answer  = rag_result.get("answer", "")
    sources = rag_result.get("sources", [])
    return validate(answer, sources, config=config)


def validate_batch(
    items: list[dict],
    *,
    config: ValidatorConfig | None = None,
) -> list[dict]:
    """Validate a list of ``{"answer": str, "sources": [...]}`` dicts.

    Returns a list of validation report dicts in the same order as *items*.
    Each input dict may also be a full ``rag.query()`` result — any extra
    keys are silently ignored.

    Parameters
    ----------
    items:
        Each item must contain ``"answer"`` and ``"sources"`` keys.
    config:
        Shared validator config applied to all items.

    Returns
    -------
    list of validation report dicts (same schema as :func:`validate`).
    """
    cfg = config or ValidatorConfig()
    reports: list[dict] = []
    for item in items:
        report = validate(
            item.get("answer", ""),
            item.get("sources", []),
            config=cfg,
        )
        reports.append(report)
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_ICON = {"error": "✗", "warning": "⚠", "info": "ℹ"}
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

_W = 78

# ANSI colours (skipped when stdout is not a TTY)
import sys as _sys
_TTY = _sys.stdout.isatty()
def _c(code: str, t: str) -> str:
    return f"{code}{t}\033[0m" if _TTY else t
_RED   = "\033[91m"
_YELL  = "\033[93m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"


def format_report(
    report: dict,
    *,
    label: str = "",
    show_scores: bool = True,
) -> str:
    """Render a validation report as a human-readable text block."""
    lines: list[str] = []
    add = lines.append

    valid   = report.get("valid", False)
    issues  = sorted(
        report.get("issues", []),
        key=lambda i: _SEVERITY_ORDER.get(i["severity"], 99),
    )
    scores  = report.get("scores", {})
    preview = report.get("answer_preview", "")

    # Header
    status_str = _c(_GREEN + _BOLD, "VALID") if valid else _c(_RED + _BOLD, "INVALID")
    header = f"  [{status_str}]"
    if label:
        header += f"  {_c(_BOLD, label)}"
    add("─" * _W)
    add(header)
    if preview:
        add(f"  Answer  : {preview[:100]}{'…' if len(preview) > 100 else ''}")

    # Scores
    if show_scores and scores:
        def _pct(k: str) -> str:
            v = scores.get(k, 0.0)
            colour = _GREEN if v >= 0.6 else (_YELL if v >= 0.3 else _RED)
            return _c(colour, f"{v:.1%}")

        add(
            f"  Scores  : "
            f"grounding={_pct('groundedness')}  "
            f"citation={_pct('citation_score')}  "
            f"urdu={_pct('urdu_ratio')}  "
            f"halluc-risk={_pct('hallucination_risk')}  "
            f"words={scores.get('answer_words', 0)}"
        )

    # Issues
    if issues:
        add(f"  Issues ({len(issues)}):")
        for issue in issues:
            sev  = issue["severity"]
            icon = _SEVERITY_ICON.get(sev, "•")
            clr  = _RED if sev == "error" else (_YELL if sev == "warning" else _DIM)
            code = _c(clr + _BOLD, f"[{issue['code']}]")
            msg  = issue["message"][:120]
            add(f"    {icon} {code}  {msg}")
    else:
        add(f"  {_c(_GREEN, '  No issues found.')}")

    add("─" * _W)
    return "\n".join(lines)


def print_report(
    report: dict,
    *,
    label: str = "",
    show_scores: bool = True,
    file=None,
) -> None:
    """Print a formatted validation report to *file* (default: stdout)."""
    print(format_report(report, label=label, show_scores=show_scores), file=file or _sys.stdout)


def print_batch_report(
    reports: list[dict],
    *,
    labels: list[str] | None = None,
    show_scores: bool = True,
    file=None,
) -> None:
    """Print a batch of validation reports with an aggregate summary.

    Parameters
    ----------
    reports:
        List of dicts from :func:`validate` / :func:`validate_batch`.
    labels:
        Optional list of label strings parallel to *reports*.
    """
    out = file or _sys.stdout
    labels = labels or [f"Result {i + 1}" for i in range(len(reports))]

    print(_c(_BOLD, "═" * _W), file=out)
    print(_c(_BOLD, "  RAG OUTPUT VALIDATION REPORT"), file=out)
    print(_c(_BOLD, "═" * _W), file=out)

    for report, label in zip(reports, labels):
        print(format_report(report, label=label, show_scores=show_scores), file=out)

    # Summary table
    valid_count = sum(1 for r in reports if r.get("valid"))
    error_count = sum(
        1 for r in reports
        for i in r.get("issues", [])
        if i["severity"] == "error"
    )
    warn_count  = sum(
        1 for r in reports
        for i in r.get("issues", [])
        if i["severity"] == "warning"
    )
    mean_g = (
        sum(r["scores"]["groundedness"] for r in reports if "scores" in r) / len(reports)
        if reports else 0.0
    )
    mean_c = (
        sum(r["scores"]["citation_score"] for r in reports if "scores" in r) / len(reports)
        if reports else 0.0
    )
    mean_u = (
        sum(r["scores"]["urdu_ratio"] for r in reports if "scores" in r) / len(reports)
        if reports else 0.0
    )

    print("═" * _W, file=out)
    print(_c(_BOLD, "  SUMMARY"), file=out)
    print(f"  Total   : {len(reports)}   "
          f"{_c(_GREEN, str(valid_count))} valid   "
          f"{_c(_RED, str(len(reports) - valid_count))} invalid", file=out)
    print(f"  Issues  : {error_count} errors   {warn_count} warnings", file=out)
    print(f"  Avg scores — "
          f"grounding={mean_g:.1%}  "
          f"citation={mean_c:.1%}  "
          f"urdu={mean_u:.1%}", file=out)
    print("═" * _W, file=out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Validate RAG output(s) from stdin or a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Input formats accepted:

              Single result:
                {"answer": "...", "sources": [{"text": "...", ...}]}

              Batch (JSON array):
                [{"answer": "...", "sources": [...]}, ...]

              Full rag.query() result dict (pass directly):
                {"answer": "...", "sources": [...], "num_chunks": 3, "timings": {...}}
        """),
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        metavar="FILE",
        help="JSON file to validate. Reads from stdin when omitted.",
    )
    parser.add_argument(
        "--min-groundedness", type=float, default=0.08, metavar="F",
        help="Groundedness threshold (default: 0.08).",
    )
    parser.add_argument(
        "--min-urdu-ratio", type=float, default=0.35, metavar="F",
        help="Minimum Urdu script ratio (default: 0.35).",
    )
    parser.add_argument(
        "--no-citation-check", action="store_true",
        help="Disable the citation requirement check.",
    )
    parser.add_argument(
        "--report", metavar="PATH",
        help="Write JSON validation output to this file.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Skip human-readable output; only write JSON report.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    import textwrap  # noqa: PLC0415  (local import to keep top-level clean)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s  %(message)s",
        handlers=[logging.StreamHandler(_sys.stdout)],
    )

    # Read input
    if args.input_file:
        raw = open(args.input_file, encoding="utf-8").read()
    else:
        raw = _sys.stdin.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON — {exc}", file=_sys.stderr)
        _sys.exit(2)

    cfg = ValidatorConfig(
        min_groundedness=args.min_groundedness,
        min_urdu_ratio=args.min_urdu_ratio,
        require_citation=not args.no_citation_check,
    )

    # Determine single vs batch
    if isinstance(data, list):
        reports = validate_batch(data, config=cfg)
        labels  = [f"Result {i + 1}" for i in range(len(reports))]
        merged  = {"results": reports}
    else:
        reports = [validate(data.get("answer", ""), data.get("sources", []), config=cfg)]
        labels  = ["Result"]
        merged  = reports[0]

    if not args.quiet:
        print_batch_report(reports, labels=labels)

    if args.report:
        Path(args.report).write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  Validation JSON written → {args.report}")

    # Exit with non-zero code when any result is invalid
    _sys.exit(0 if all(r.get("valid") for r in reports) else 1)


if __name__ == "__main__":
    _cli()
