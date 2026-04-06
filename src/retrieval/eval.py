"""Retrieval quality evaluation for the Hybrid+Vectorless RAG system.

Runs a structured batch of labelled test queries through the hybrid retriever
and produces a human-readable report plus an optional JSON dump.

Each test query carries:
  * the Urdu question
  * an expected category (used for category-match scoring)
  * a list of expected keywords (Urdu words that *should* appear in
    at least one of the top-k results)

For every query the evaluator reports:
  * top-k results with fused scores, category, and source file
  * per-result keyword highlighting (which keywords were found + where)
  * per-result boolean flags: keyword_hit, category_hit
  * aggregate metrics: Precision@k, keyword coverage, mean reciprocal rank

Usage — CLI
-----------
::

    # Quick check with default queries against a pre-indexed corpus:
    python -m src.retrieval.eval

    # Custom query file (JSON list of EvalQuery dicts):
    python -m src.retrieval.eval --queries my_queries.json

    # BM25-only mode (no Pinecone / OpenAI calls):
    python -m src.retrieval.eval --bm25-only

    # Save JSON report:
    python -m src.retrieval.eval --report report.json

    # Control result count and logging:
    python -m src.retrieval.eval --top-k 5 --verbose

Usage — programmatic
--------------------
::

    from src.retrieval.eval import EvalQuery, run_evaluation, print_report

    queries = [
        EvalQuery(
            label="نماز نیت",
            question="نماز میں نیت کا طریقہ کیا ہے؟",
            expected_category="NAMAZ",
            expected_keywords=["نیت", "نماز", "دل"],
        ),
    ]
    report = run_evaluation(queries, top_k=5)
    print_report(report)
    report.save_json("report.json")
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Regex helpers ─────────────────────────────────────────────────────────────

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+"
)

# ANSI colour codes for terminal highlighting
_BOLD  = "\033[1m"
_RED   = "\033[91m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_YELLOW = "\033[93m"
_DIM   = "\033[2m"
_RESET = "\033[0m"

_COLOUR_ENABLED = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Apply ANSI escape code only when stdout is a TTY."""
    if not _COLOUR_ENABLED:
        return text
    return f"{code}{text}{_RESET}"


def _urdu_ratio(text: str) -> float:
    s = text.replace(" ", "")
    return len(_ARABIC_RE.findall(s)) / len(s) if s else 0.0


def _tokenize_urdu(text: str) -> set[str]:
    """Return a set of Arabic-script tokens (length >= 2) from *text*."""
    return {t for t in _ARABIC_RE.findall(text) if len(t) >= 2}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    """A single labelled retrieval test case.

    Attributes
    ----------
    label:
        Short human-readable identifier shown in the report.
    question:
        Raw Urdu query string.
    expected_category:
        The category name (e.g. ``"NAMAZ"``) that retrieved results should
        come from.  Matching is *case-insensitive*.  Set to ``""`` to skip
        category evaluation.
    expected_keywords:
        Urdu words that should appear in at least one of the top-k results.
        A keyword is considered "found" if it is a substring of the result's
        combined question + answer text (after Urdu normalisation).
        Set to ``[]`` to skip keyword evaluation.
    category_filter:
        When set, passes this value to the hybrid retriever as a hard metadata
        filter.  Useful for targeted evaluation within a single topic.
    """

    label: str
    question: str
    expected_category: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    category_filter: str | None = None   # forwarded to hybrid_search()


@dataclass
class KeywordMatch:
    """Describes where one keyword was (or wasn't) found in a result."""

    keyword: str
    found: bool
    in_question: bool = False
    in_answer: bool = False
    in_text: bool = False


@dataclass
class ResultRow:
    """Evaluation data for one retrieved document."""

    rank: int
    chunk_id: str
    score: float
    category: str
    source_file: str
    question_preview: str        # first 100 chars of stored question
    answer_preview: str          # first 150 chars of stored answer
    keyword_matches: list[KeywordMatch]
    category_hit: bool           # retrieved category == expected category
    keyword_hit: bool            # at least one expected keyword found
    all_keywords_hit: bool       # every expected keyword found


@dataclass
class QueryEval:
    """Aggregated evaluation result for one :class:`EvalQuery`."""

    query: EvalQuery
    results: list[ResultRow]
    elapsed_ms: float

    # ── Aggregate metrics ─────────────────────────────────────────────────

    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0

    @property
    def category_precision(self) -> float:
        """Fraction of results whose category matches the expected one."""
        if not self.query.expected_category or not self.results:
            return -1.0   # N/A
        hits = sum(1 for r in self.results if r.category_hit)
        return hits / len(self.results)

    @property
    def keyword_coverage(self) -> float:
        """Fraction of expected keywords found in *any* retrieved result."""
        if not self.query.expected_keywords:
            return -1.0   # N/A
        found = {
            km.keyword
            for row in self.results
            for km in row.keyword_matches
            if km.found
        }
        return len(found) / len(self.query.expected_keywords)

    @property
    def mrr(self) -> float:
        """Mean Reciprocal Rank (first result with a keyword_hit)."""
        if not self.query.expected_keywords:
            return -1.0   # N/A
        for row in self.results:
            if row.keyword_hit:
                return 1.0 / row.rank
        return 0.0

    @property
    def first_category_hit_rank(self) -> int | None:
        """Rank of the first result whose category matches (1-based), or None."""
        if not self.query.expected_category:
            return None
        for row in self.results:
            if row.category_hit:
                return row.rank
        return None


@dataclass
class EvalReport:
    """Top-level container for a batch evaluation run."""

    query_evals: list[QueryEval]
    top_k: int
    bm25_only: bool
    elapsed_total_ms: float

    # ── Corpus-level aggregates ───────────────────────────────────────────

    @property
    def mean_keyword_coverage(self) -> float:
        vals = [e.keyword_coverage for e in self.query_evals if e.keyword_coverage >= 0]
        return sum(vals) / len(vals) if vals else -1.0

    @property
    def mean_category_precision(self) -> float:
        vals = [e.category_precision for e in self.query_evals if e.category_precision >= 0]
        return sum(vals) / len(vals) if vals else -1.0

    @property
    def mean_mrr(self) -> float:
        vals = [e.mrr for e in self.query_evals if e.mrr >= 0]
        return sum(vals) / len(vals) if vals else -1.0

    @property
    def mean_top_score(self) -> float:
        scores = [e.top_score for e in self.query_evals]
        return sum(scores) / len(scores) if scores else 0.0

    def save_json(self, path: str | Path) -> None:
        """Serialise the full report to a JSON file."""
        path = Path(path)

        def _serialise(ev: QueryEval) -> dict:
            return {
                "label": ev.query.label,
                "question": ev.query.question,
                "expected_category": ev.query.expected_category,
                "expected_keywords": ev.query.expected_keywords,
                "elapsed_ms": round(ev.elapsed_ms, 1),
                "top_score": round(ev.top_score, 4),
                "category_precision": round(ev.category_precision, 3)
                    if ev.category_precision >= 0 else None,
                "keyword_coverage": round(ev.keyword_coverage, 3)
                    if ev.keyword_coverage >= 0 else None,
                "mrr": round(ev.mrr, 3) if ev.mrr >= 0 else None,
                "first_category_hit_rank": ev.first_category_hit_rank,
                "results": [
                    {
                        "rank": r.rank,
                        "score": round(r.score, 4),
                        "category": r.category,
                        "source_file": r.source_file,
                        "question_preview": r.question_preview,
                        "answer_preview": r.answer_preview,
                        "category_hit": r.category_hit,
                        "keyword_hit": r.keyword_hit,
                        "all_keywords_hit": r.all_keywords_hit,
                        "keyword_matches": [
                            {
                                "keyword": km.keyword,
                                "found": km.found,
                                "in_question": km.in_question,
                                "in_answer": km.in_answer,
                            }
                            for km in r.keyword_matches
                        ],
                    }
                    for r in ev.results
                ],
            }

        # Use a plain dict to avoid dataclass recursion issues
        data = {
            "top_k": self.top_k,
            "bm25_only": self.bm25_only,
            "elapsed_total_ms": round(self.elapsed_total_ms, 1),
            "mean_keyword_coverage": round(self.mean_keyword_coverage, 3)
                if self.mean_keyword_coverage >= 0 else None,
            "mean_category_precision": round(self.mean_category_precision, 3)
                if self.mean_category_precision >= 0 else None,
            "mean_mrr": round(self.mean_mrr, 3) if self.mean_mrr >= 0 else None,
            "mean_top_score": round(self.mean_top_score, 4),
            "queries": [
                {
                    "label": ev.query.label,
                    "question": ev.query.question,
                    "expected_category": ev.query.expected_category,
                    "expected_keywords": ev.query.expected_keywords,
                    "elapsed_ms": round(ev.elapsed_ms, 1),
                    "top_score": round(ev.top_score, 4),
                    "category_precision": round(ev.category_precision, 3)
                        if ev.category_precision >= 0 else None,
                    "keyword_coverage": round(ev.keyword_coverage, 3)
                        if ev.keyword_coverage >= 0 else None,
                    "mrr": round(ev.mrr, 3) if ev.mrr >= 0 else None,
                    "first_category_hit_rank": ev.first_category_hit_rank,
                    "results": [
                        {
                            "rank": r.rank,
                            "score": round(r.score, 4),
                            "category": r.category,
                            "source_file": r.source_file,
                            "question_preview": r.question_preview,
                            "answer_preview": r.answer_preview,
                            "category_hit": r.category_hit,
                            "keyword_hit": r.keyword_hit,
                            "all_keywords_hit": r.all_keywords_hit,
                            "keyword_matches": [
                                {
                                    "keyword": km.keyword,
                                    "found": km.found,
                                    "in_question": km.in_question,
                                    "in_answer": km.in_answer,
                                }
                                for km in r.keyword_matches
                            ],
                        }
                        for r in ev.results
                    ],
                }
                for ev in self.query_evals
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Report saved → %s", path)


# ── Built-in test query bank ──────────────────────────────────────────────────

DEFAULT_QUERIES: list[EvalQuery] = [
    # ── NAMAZ ─────────────────────────────────────────────────────────────
    EvalQuery(
        label="نماز — نیت",
        question="نماز پڑھتے وقت نیت کا طریقہ کیا ہے؟",
        expected_category="NAMAZ",
        expected_keywords=["نیت", "نماز", "دل"],
    ),
    EvalQuery(
        label="نماز — قصر سفر",
        question="سفر میں نماز قصر کب کرنا جائز ہے؟",
        expected_category="NAMAZ",
        expected_keywords=["قصر", "سفر", "نماز", "رکعت"],
    ),
    EvalQuery(
        label="نماز — سورۃ الفاتحہ",
        question="کیا نماز میں سورۃ الفاتحہ پڑھنا ضروری ہے؟",
        expected_category="NAMAZ",
        expected_keywords=["فاتحہ", "نماز", "واجب"],
    ),
    # ── WUDU ──────────────────────────────────────────────────────────────
    EvalQuery(
        label="وضو — ناقضات",
        question="وضو کن چیزوں سے ٹوٹ جاتا ہے؟",
        expected_category="WUDU",
        expected_keywords=["وضو", "پیشاب", "ریح", "نیند"],
    ),
    EvalQuery(
        label="وضو — غسل فرض",
        question="غسل کب فرض ہوتا ہے؟",
        expected_category="WUDU",
        expected_keywords=["غسل", "فرض", "جنابت"],
    ),
    # ── ZAKAT ─────────────────────────────────────────────────────────────
    EvalQuery(
        label="زکوٰۃ — نصاب",
        question="زکوٰۃ کا نصاب کتنا ہے؟",
        expected_category="ZAKAT",
        expected_keywords=["زکوٰۃ", "نصاب", "سونا", "چاندی"],
    ),
    EvalQuery(
        label="زکوٰۃ — مصارف",
        question="زکوٰۃ کن لوگوں کو دینا جائز ہے؟",
        expected_category="ZAKAT",
        expected_keywords=["زکوٰۃ", "فقراء", "مساکین"],
    ),
    # ── FAST ──────────────────────────────────────────────────────────────
    EvalQuery(
        label="روزہ — کفارہ",
        question="رمضان میں روزہ توڑنے کا کفارہ کیا ہے؟",
        expected_category="FAST",
        expected_keywords=["روزہ", "کفارہ", "رمضان"],
    ),
    EvalQuery(
        label="روزہ — سفر",
        question="سفر میں روزہ رکھنا ضروری ہے؟",
        expected_category="FAST",
        expected_keywords=["روزہ", "سفر", "قضا"],
    ),
    # ── DIVORCE ───────────────────────────────────────────────────────────
    EvalQuery(
        label="طلاق — طریقہ",
        question="طلاق دینے کا شرعی طریقہ کیا ہے؟",
        expected_category="DIVORCE",
        expected_keywords=["طلاق", "شرعی", "بیوی"],
    ),
    # ── INHERITANCE ───────────────────────────────────────────────────────
    EvalQuery(
        label="وراثت — تقسیم",
        question="اسلام میں وراثت کی تقسیم کیسے ہوتی ہے؟",
        expected_category="INHERITANCE",
        expected_keywords=["وراثت", "تقسیم", "حصہ"],
    ),
    # ── Edge / stress queries ─────────────────────────────────────────────
    EvalQuery(
        label="[edge] بہت مختصر",
        question="نماز",
        expected_category="NAMAZ",
        expected_keywords=["نماز"],
    ),
    EvalQuery(
        label="[edge] خارج از موضوع",
        question="پائتھن پروگرامنگ کیسے سیکھیں؟",
        expected_category="",      # no expected category
        expected_keywords=[],      # no specific keywords required
    ),
]


# ── Core evaluation logic ─────────────────────────────────────────────────────

def _check_keywords(
    result: dict,
    expected_keywords: list[str],
) -> list[KeywordMatch]:
    """Return one :class:`KeywordMatch` per expected keyword for *result*.

    Matching is substring-based after Urdu normalisation, so partial word
    matches (e.g. stemming) are caught naturally.
    """
    from src.preprocessing.urdu_normalizer import normalize_urdu  # noqa: PLC0415

    meta = result.get("metadata", {})
    norm_q = normalize_urdu(meta.get("question", ""))
    norm_a = normalize_urdu(meta.get("answer", ""))
    norm_t = normalize_urdu(result.get("text", ""))

    matches: list[KeywordMatch] = []
    for kw in expected_keywords:
        norm_kw = normalize_urdu(kw)
        in_q = norm_kw in norm_q
        in_a = norm_kw in norm_a
        in_t = norm_kw in norm_t
        matches.append(KeywordMatch(
            keyword=kw,
            found=in_q or in_a or in_t,
            in_question=in_q,
            in_answer=in_a,
            in_text=in_t,
        ))
    return matches


def _evaluate_one(
    eq: EvalQuery,
    raw_results: list[dict],
) -> list[ResultRow]:
    """Convert raw retrieval results into :class:`ResultRow` objects."""
    rows: list[ResultRow] = []
    for i, res in enumerate(raw_results, 1):
        meta = res.get("metadata", {})
        cat = meta.get("category", "").upper().strip()
        expected_cat = eq.expected_category.upper().strip()

        kw_matches = _check_keywords(res, eq.expected_keywords)
        kw_hit = any(km.found for km in kw_matches)
        all_kw_hit = all(km.found for km in kw_matches) if kw_matches else False

        rows.append(ResultRow(
            rank=i,
            chunk_id=meta.get("doc_id", "") or meta.get("id", "") or f"rank_{i}",
            score=res.get("score", 0.0),
            category=cat or "—",
            source_file=meta.get("source_file", "—"),
            question_preview=(meta.get("question", "") or "")[:100],
            answer_preview=(meta.get("answer", "") or "")[:150],
            keyword_matches=kw_matches,
            category_hit=(cat == expected_cat) if expected_cat else False,
            keyword_hit=kw_hit,
            all_keywords_hit=all_kw_hit,
        ))
    return rows


def run_evaluation(
    queries: list[EvalQuery] | None = None,
    *,
    top_k: int = 5,
    bm25_only: bool = False,
    bm25_corpus: Any = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> EvalReport:
    """Run all *queries* through the retriever and return an :class:`EvalReport`.

    Parameters
    ----------
    queries:
        List of :class:`EvalQuery` instances.  Defaults to
        :data:`DEFAULT_QUERIES`.
    top_k:
        Number of results to retrieve per query.
    bm25_only:
        When True, bypass Pinecone and run BM25-only retrieval.  Avoids
        any OpenAI or Pinecone API calls — useful for offline / CI testing.
    bm25_corpus:
        Pre-loaded :class:`~src.retrieval.bm25_index.BM25Corpus` instance.
        When *None* in bm25-only mode, the corpus is loaded or built
        automatically.
    dense_weight / sparse_weight:
        Override default hybrid fusion weights.

    Returns
    -------
    :class:`EvalReport`
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    if bm25_only:
        from src.retrieval.bm25_index import BM25Corpus  # noqa: PLC0415
        if bm25_corpus is None:
            bm25_corpus = BM25Corpus.load_or_build()
    else:
        from src.retrieval.hybrid_retriever import hybrid_search  # noqa: PLC0415
        # Load BM25 once and reuse across queries to avoid repeated disk I/O
        from src.retrieval.bm25_index import BM25Corpus  # noqa: PLC0415
        if bm25_corpus is None:
            bm25_corpus = BM25Corpus.load_or_build()

    query_evals: list[QueryEval] = []
    t_total = time.perf_counter()

    for eq in queries:
        logger.debug("Evaluating: %s", eq.label)
        t0 = time.perf_counter()

        if bm25_only:
            raw = bm25_corpus.search(eq.question, top_k=top_k)
            # Normalise BM25 output to hybrid_search dict format
            results_raw = [
                {
                    "text":     r.get("text", ""),
                    "score":    r.get("score", 0.0),
                    "metadata": r.get("metadata", {}),
                }
                for r in raw
            ]
        else:
            results_raw = hybrid_search(
                eq.question,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                bm25_corpus=bm25_corpus,
                category=eq.category_filter,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        rows = _evaluate_one(eq, results_raw)
        query_evals.append(QueryEval(query=eq, results=rows, elapsed_ms=elapsed_ms))
        logger.info("  [%s]  %d results in %.0fms", eq.label, len(rows), elapsed_ms)

    elapsed_total = (time.perf_counter() - t_total) * 1000
    return EvalReport(
        query_evals=query_evals,
        top_k=top_k,
        bm25_only=bm25_only,
        elapsed_total_ms=elapsed_total,
    )


# ── Report printer ────────────────────────────────────────────────────────────

_W = 80   # terminal width


def _hr(char: str = "─", width: int = _W) -> str:
    return char * width


def _score_bar(score: float, width: int = 20) -> str:
    """Render a compact ASCII score bar, e.g.  ████████░░░░  0.82"""
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{score:.3f}"
    return f"{bar} {pct}"


def _highlight_keywords(text: str, keywords: list[str]) -> str:
    """Wrap found keywords in ANSI yellow bold in *text* (TTY only)."""
    if not _COLOUR_ENABLED or not keywords:
        return text
    from src.preprocessing.urdu_normalizer import normalize_urdu  # noqa: PLC0415
    result = text
    for kw in keywords:
        # Use a case/normalisation-insensitive substring replace
        pattern = re.compile(re.escape(kw))
        result = pattern.sub(_c(_YELLOW + _BOLD, kw), result)
    return result


def _fmt_metric(value: float, label: str, na: str = "N/A") -> str:
    if value < 0:
        return f"{label}: {_c(_DIM, na)}"
    colour = _GREEN if value >= 0.6 else (_YELLOW if value >= 0.3 else _RED)
    return f"{label}: {_c(colour, f'{value:.1%}')}"


def print_report(report: EvalReport, *, file=None) -> None:
    """Write a human-readable evaluation report to *file* (default: stdout)."""
    out = file or sys.stdout

    def p(*args, **kwargs):
        print(*args, file=out, **kwargs)

    mode = "BM25-only" if report.bm25_only else "Hybrid (dense + BM25)"
    p()
    p(_c(_BOLD, _hr("═")))
    p(_c(_BOLD, f"  RETRIEVAL QUALITY REPORT   mode={mode}   top_k={report.top_k}"))
    p(_c(_BOLD, _hr("═")))
    p(f"  Queries evaluated  : {len(report.query_evals)}")
    p(f"  Total time         : {report.elapsed_total_ms:.0f}ms")
    p()
    p("  Corpus-level metrics")
    p(f"    {_fmt_metric(report.mean_keyword_coverage,  'Keyword coverage  (avg)')}")
    p(f"    {_fmt_metric(report.mean_category_precision,'Category precision (avg)')}")
    p(f"    {_fmt_metric(report.mean_mrr,               'MRR                (avg)')}")
    p(f"    Mean top score     : {report.mean_top_score:.4f}")
    p()

    for ev in report.query_evals:
        tq = ev.query
        p(_hr())
        label_str = _c(_BOLD + _CYAN, tq.label)
        p(f"  {label_str}")
        p(f"  Question  : {tq.question}")
        if tq.expected_category:
            p(f"  Expected  : category={tq.expected_category}"
              + (f"  keywords={tq.expected_keywords}" if tq.expected_keywords else ""))
        p(f"  Elapsed   : {ev.elapsed_ms:.0f}ms   results={len(ev.results)}")

        # Per-query metrics
        metrics_parts = []
        if ev.category_precision >= 0:
            metrics_parts.append(_fmt_metric(ev.category_precision, "Cat-P"))
        if ev.keyword_coverage >= 0:
            metrics_parts.append(_fmt_metric(ev.keyword_coverage, "KW-cov"))
        if ev.mrr >= 0:
            metrics_parts.append(_fmt_metric(ev.mrr, "MRR"))
        if metrics_parts:
            p("  Metrics   : " + "   ".join(metrics_parts))

        p()

        if not ev.results:
            p(_c(_RED, "    ⚠  No results returned."))
            p()
            continue

        for row in ev.results:
            # ── Score bar ─────────────────────────────────────────────────
            cat_colour = _GREEN if row.category_hit else (_YELLOW if not tq.expected_category else _RED)
            cat_str = _c(cat_colour, row.category)
            p(f"  {'─' * 4} Rank {row.rank}  {_score_bar(min(row.score, 1.0))}  "
              f"cat={cat_str}  src={row.source_file}")

            # ── Stored question ───────────────────────────────────────────
            if row.question_preview:
                highlighted_q = _highlight_keywords(row.question_preview, tq.expected_keywords)
                p(f"    Q: {highlighted_q}")

            # ── Stored answer preview ─────────────────────────────────────
            if row.answer_preview:
                highlighted_a = _highlight_keywords(row.answer_preview, tq.expected_keywords)
                wrapped = textwrap.fill(
                    highlighted_a, width=72,
                    initial_indent="    A: ",
                    subsequent_indent="       ",
                )
                p(wrapped)

            # ── Keyword match summary ─────────────────────────────────────
            if row.keyword_matches:
                kw_parts = []
                for km in row.keyword_matches:
                    location = ""
                    if km.found:
                        locs = []
                        if km.in_question: locs.append("Q")
                        if km.in_answer:   locs.append("A")
                        loc_str = "/".join(locs) if locs else "T"
                        location = f"({loc_str})"
                    icon = _c(_GREEN, "✓") if km.found else _c(_RED, "✗")
                    kw_parts.append(f"{icon}{km.keyword}{location}")
                p("    Keywords: " + "  ".join(kw_parts))

            p()

        # ── Query summary bar ─────────────────────────────────────────────
        cat_hits = sum(1 for r in ev.results if r.category_hit)
        kw_hits  = sum(1 for r in ev.results if r.keyword_hit)
        n = len(ev.results)
        p(f"  Summary: {cat_hits}/{n} category hits   "
          f"{kw_hits}/{n} keyword hits   "
          f"top_score={ev.top_score:.4f}")
        p()

    # ── Final scoreboard ──────────────────────────────────────────────────────
    p(_hr("═"))
    p(_c(_BOLD, "  SCOREBOARD"))
    p(_hr("═"))
    header = f"  {'Label':<32}  {'Cat-P':>7}  {'KW-cov':>7}  {'MRR':>6}  {'Top-score':>10}"
    p(_c(_DIM, header))
    p(_c(_DIM, "  " + "─" * (len(header) - 2)))
    for ev in report.query_evals:
        cat_p  = f"{ev.category_precision:.1%}" if ev.category_precision >= 0 else "  N/A "
        kw_cov = f"{ev.keyword_coverage:.1%}"   if ev.keyword_coverage >= 0   else "  N/A "
        mrr    = f"{ev.mrr:.3f}"                if ev.mrr >= 0                else "  N/A"
        ts     = f"{ev.top_score:.4f}"
        label  = ev.query.label[:32]
        p(f"  {label:<32}  {cat_p:>7}  {kw_cov:>7}  {mrr:>6}  {ts:>10}")

    p(_hr("─"))
    mean_cp  = f"{report.mean_category_precision:.1%}" if report.mean_category_precision >= 0 else "  N/A"
    mean_kw  = f"{report.mean_keyword_coverage:.1%}"   if report.mean_keyword_coverage >= 0   else "  N/A"
    mean_mrr = f"{report.mean_mrr:.3f}"                if report.mean_mrr >= 0               else " N/A"
    mean_ts  = f"{report.mean_top_score:.4f}"
    p(_c(_BOLD, f"  {'MEAN':<32}  {mean_cp:>7}  {mean_kw:>7}  {mean_mrr:>6}  {mean_ts:>10}"))
    p(_hr("═"))
    p()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_queries_from_json(path: Path) -> list[EvalQuery]:
    """Load a JSON array of EvalQuery dicts from *path*."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(raw).__name__}")
    queries = []
    for item in raw:
        queries.append(EvalQuery(
            label=item.get("label", "unlabelled"),
            question=item.get("question", ""),
            expected_category=item.get("expected_category", ""),
            expected_keywords=item.get("expected_keywords", []),
            category_filter=item.get("category_filter"),
        ))
    return queries


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval quality evaluator for the Hybrid+Vectorless RAG system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--queries", "-q",
        metavar="PATH",
        help="Path to a JSON file containing a list of EvalQuery dicts. "
             "Defaults to the built-in DEFAULT_QUERIES bank.",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int, default=5,
        metavar="N",
        help="Number of results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Use BM25 retrieval only — no OpenAI embeddings or Pinecone queries.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float, default=None,
        metavar="W",
        help="Override dense retrieval weight (0.0–1.0).",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float, default=None,
        metavar="W",
        help="Override sparse retrieval weight (0.0–1.0).",
    )
    parser.add_argument(
        "--report", "-r",
        metavar="PATH",
        help="Write the full evaluation report to this JSON file.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not args.verbose:
        for _lib in ("httpx", "httpcore", "openai", "pinecone", "urllib3"):
            logging.getLogger(_lib).setLevel(logging.WARNING)

    queries: list[EvalQuery] | None = None
    if args.queries:
        queries = _load_queries_from_json(Path(args.queries))
        logger.info("Loaded %d queries from %s", len(queries), args.queries)

    report = run_evaluation(
        queries,
        top_k=args.top_k,
        bm25_only=args.bm25_only,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )

    print_report(report)

    if args.report:
        report.save_json(args.report)
        print(f"  JSON report saved → {args.report}")


if __name__ == "__main__":
    _cli()
