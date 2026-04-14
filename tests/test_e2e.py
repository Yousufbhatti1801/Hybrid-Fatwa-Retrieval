#!/usr/bin/env python3
"""
test_e2e.py — End-to-end validation for the Hybrid+Vectorless RAG pipeline.

Validates the complete system with a 10-record mini dataset:

  Step 1  Build a mini BM25 index from fixture records
  Step 2  Embed the 10 records + upsert to a dedicated test Pinecone index
  Step 3  Run 3 queries (normal / edge-case / ambiguous) through guarded_query()
  Step 4  Print retrieved documents, scores, and final answers
  Step 5  Run assertions: no crashes · pipeline flow · valid Urdu output

Usage
-----
  # Full live run (requires OPENAI_API_KEY + PINECONE_API_KEY in .env):
  python test_e2e.py

  # Use a specific Pinecone index for the test (NOT your production index):
  python test_e2e.py --index-name fatawa-e2e-test

  # Mock mode — no API calls, tests flow + guardrail logic only:
  python test_e2e.py --mock

  # Keep test vectors in Pinecone after the run (default behaviour is to delete):
  python test_e2e.py --keep

  # Verbose / debug logging:
  python test_e2e.py --verbose

  # Save JSON test report to a file:
  python test_e2e.py --report out.json

Requirements
------------
  API keys in .env:    OPENAI_API_KEY  PINECONE_API_KEY
  pip install:         openai pinecone-client pydantic-settings rank-bm25 tqdm
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Force UTF-8 stdout/stderr on Windows (cp1252 cannot encode Urdu) ────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Parse args BEFORE importing any src module so env overrides take effect ───

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end test for the Hybrid+Vectorless RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--index-name",
        default="fatawa-e2e-test",
        metavar="NAME",
        help="Pinecone index for this test run (default: fatawa-e2e-test). "
             "Use a name that is DIFFERENT from your production index.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Skip Pinecone and OpenAI calls; verify flow + guardrail logic only.",
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete test vectors from Pinecone after the run.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    p.add_argument(
        "--bm25-cache",
        default=".bm25_e2e_test.pkl",
        metavar="PATH",
        help="Path for the mini BM25 pickle (default: .bm25_e2e_test.pkl).",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        help="Write a JSON test report to this file.",
    )
    return p.parse_args()


_args = _parse_args()

# ── Redirect Pinecone to the test index BEFORE any src import ─────────────────
if not _args.mock:
    os.environ["PINECONE_INDEX_NAME"] = _args.index_name

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG if _args.verbose else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-36s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Quiet noisy third-party loggers unless verbose
if not _args.verbose:
    for _noisy in ("httpx", "httpcore", "openai", "pinecone", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("test_e2e")

# ── Now it is safe to import src modules ─────────────────────────────────────
from src.config import get_settings          # noqa: E402

get_settings.cache_clear()                   # ensure the overridden env var is picked up

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — Mini dataset (10 records)
# ─────────────────────────────────────────────────────────────────────────────
# Each record follows the unified schema produced by dynamic_loader.py.
# The `text` field is built in the standard "سوال: … جواب: …" format.

_MINI_FATAWA: list[dict] = [
    # ── NAMAZ (3 records) ────────────────────────────────────────────────────
    {
        "id": "e2e_namaz_001",
        "question": "نماز کی نیت دل سے کافی ہے یا زبان سے بھی ادا کرنا ضروری ہے؟",
        "answer": (
            "نماز کی نیت کا اصل مقام دل ہے۔ زبان سے نیت کے الفاظ ادا کرنا ضروری نہیں، "
            "تاہم جائز ہے۔ فقہاء کے نزدیک دل کا ارادہ ہی کافی ہے۔"
        ),
        "category": "NAMAZ",
        "source_file": "e2e_namaz.csv",
        "folder": "TEST",
    },
    {
        "id": "e2e_namaz_002",
        "question": "کیا نماز میں سورۃ الفاتحہ پڑھنا فرض ہے؟",
        "answer": (
            "امام کے پیچھے مقتدی کے لیے امام کی قراءت کافی ہے، اسے سورۃ الفاتحہ پڑھنا "
            "ضروری نہیں۔ منفرد اور امام کے لیے ہر رکعت میں سورۃ الفاتحہ پڑھنا واجب ہے۔"
        ),
        "category": "NAMAZ",
        "source_file": "e2e_namaz.csv",
        "folder": "TEST",
    },
    {
        "id": "e2e_namaz_003",
        "question": "سفر میں نماز قصر کب کی جاتی ہے؟",
        "answer": (
            "جب کوئی شخص ۷۸ کلومیٹر یا اس سے زیادہ کا سفر کرے تو وہ چار رکعت والی فرض "
            "نماز کی بجائے دو رکعت پڑھ سکتا ہے۔ اس کو نماز قصر کہتے ہیں۔ یہ رخصت اس وقت "
            "تک ہے جب تک وہ کسی مقام پر پندرہ دن سے کم رکے۔"
        ),
        "category": "NAMAZ",
        "source_file": "e2e_namaz.csv",
        "folder": "TEST",
    },
    # ── WUDU (2 records) ─────────────────────────────────────────────────────
    {
        "id": "e2e_wudu_001",
        "question": "وضو ٹوٹ جانے کی کیا کیا وجوہات ہیں؟",
        "answer": (
            "وضو ٹوٹنے کی وجوہات: پیشاب یا پاخانہ آنا، ریح خارج ہونا، خون یا پیپ بہنا، "
            "قے آنا، نیند آنا جس میں جسم ڈھیلا ہو جائے، اور بے ہوشی۔ "
            "ان میں سے کوئی بھی صورت پیش آئے تو وضو ٹوٹ جاتا ہے۔"
        ),
        "category": "WUDU",
        "source_file": "e2e_wudu.csv",
        "folder": "TEST",
    },
    {
        "id": "e2e_wudu_002",
        "question": "غسل کب فرض ہوتا ہے؟",
        "answer": (
            "غسل فرض ہونے کے اسباب: جنابت، حیض کا خاتمہ، نفاس کا خاتمہ، اور اسلام قبول "
            "کرنا۔ ان حالات میں نماز پڑھنا جائز نہیں جب تک غسل نہ کیا جائے۔"
        ),
        "category": "WUDU",
        "source_file": "e2e_wudu.csv",
        "folder": "TEST",
    },
    # ── ZAKAT (2 records) ────────────────────────────────────────────────────
    {
        "id": "e2e_zakat_001",
        "question": "زکوٰۃ کا نصاب کتنا ہے اور کب ادا کرنا فرض ہوتا ہے؟",
        "answer": (
            "سونے کا نصاب ساڑھے سات تولہ (87.48 گرام) اور چاندی کا نصاب ساڑھے باون تولہ "
            "(612.36 گرام) ہے۔ جب مسلمان ایک سال تک نصاب کا مالک رہے تو ڈھائی فیصد زکوٰۃ "
            "فرض ہے۔"
        ),
        "category": "ZAKAT",
        "source_file": "e2e_zakat.csv",
        "folder": "TEST",
    },
    {
        "id": "e2e_zakat_002",
        "question": "زکوٰۃ کن کن لوگوں کو دی جا سکتی ہے؟",
        "answer": (
            "قرآن نے آٹھ مصارف زکوٰۃ بیان کیے: فقراء، مساکین، عاملین زکوٰۃ، مولفۃ القلوب، "
            "غلام آزاد کرانے کے لیے، مقروض، فی سبیل اللہ، اور مسافر۔ "
            "اپنے والدین، اولاد، اور شوہر یا بیوی کو زکوٰۃ دینا جائز نہیں۔"
        ),
        "category": "ZAKAT",
        "source_file": "e2e_zakat.csv",
        "folder": "TEST",
    },
    # ── FAST (2 records) ─────────────────────────────────────────────────────
    {
        "id": "e2e_fast_001",
        "question": "رمضان کا روزہ جان بوجھ کر توڑنے کا کیا کفارہ ہے؟",
        "answer": (
            "رمضان کا روزہ جان بوجھ کر توڑنے کا کفارہ ترتیب وار یہ ہے: ایک مسلمان غلام "
            "آزاد کرنا، اگر ممکن نہ ہو تو دو ماہ کے مسلسل روزے رکھنا، اگر یہ بھی ممکن نہ "
            "ہو تو ساٹھ مسکینوں کو کھانا کھلانا۔ یہ کفارہ صرف جماع سے روزہ توڑنے پر واجب ہے۔"
        ),
        "category": "FAST",
        "source_file": "e2e_fast.csv",
        "folder": "TEST",
    },
    {
        "id": "e2e_fast_002",
        "question": "سفر میں روزہ رکھنا ضروری ہے یا چھوڑا جا سکتا ہے؟",
        "answer": (
            "سفر میں روزہ چھوڑنے کی اجازت ہے لیکن بعد میں قضا کرنا ضروری ہے۔ اگر سفر میں "
            "مشقت ہو تو نہ رکھنا افضل ہے۔ شریعت نے مسافر کو یہ رخصت دی ہے۔"
        ),
        "category": "FAST",
        "source_file": "e2e_fast.csv",
        "folder": "TEST",
    },
    # ── ADHAN (1 record — adds variety + tests ambiguous water query) ─────────
    {
        "id": "e2e_adhan_001",
        "question": "وضو کے لیے پانی کا استعمال کب جائز ہے؟",
        "answer": (
            "وضو کے لیے پاک پانی استعمال کرنا ضروری ہے۔ بہتا ہوا پانی اور کھڑا پانی دونوں "
            "درست ہیں بشرطیکہ نجاست نہ ملی ہو۔ سمندر، دریا، بارش، اور کنویں کا پانی وضو "
            "کے لیے جائز ہے۔"
        ),
        "category": "WUDU",
        "source_file": "e2e_adhan.csv",
        "folder": "TEST",
    },
]

# Build the combined text field (standard pipeline format)
for _r in _MINI_FATAWA:
    _r["text"] = f"سوال: {_r['question']} جواب: {_r['answer']}"

_TEST_IDS = [r["id"] for r in _MINI_FATAWA]

# ─────────────────────────────────────────────────────────────────────────────
# TEST QUERIES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestQuery:
    label: str
    question: str
    scenario: str       # "normal" | "edge_case" | "ambiguous"
    expect_answer: bool # True = expect a real Urdu answer; False = expect sentinel


TEST_QUERIES: list[TestQuery] = [
    TestQuery(
        label="[1] Normal — نماز کی نیت",
        question="نماز پڑھتے وقت نیت کس طرح کی جاتی ہے؟",
        scenario="normal",
        expect_answer=True,
    ),
    TestQuery(
        label="[2] Edge case — out-of-domain query",
        question="پائتھن پروگرامنگ لینگویج سیکھنے کا بہترین طریقہ کیا ہے؟",
        scenario="edge_case",
        expect_answer=False,  # no Islamic fatwa about Python; guardrails should fire
    ),
    TestQuery(
        label="[3] Ambiguous — پانی",
        question="پانی",
        scenario="ambiguous",
        expect_answer=True,   # low confidence, but wudu records mention water
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: TestQuery
    retrieved: list[dict] = field(default_factory=list)
    answer: str = ""
    guardrail_hits: list[str] = field(default_factory=list)
    verdict_summaries: list[dict] = field(default_factory=list)
    passed_preflight: bool = True
    timings: dict = field(default_factory=dict)
    assertions: list[dict] = field(default_factory=list)   # {"name", "passed", "detail"}
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_W = 78  # terminal width for separator lines


def _hr(char: str = "─") -> str:
    return char * _W


def _box(title: str) -> str:
    pad = _W - 2 - len(title)
    return f"╔{'═' * (_W - 2)}╗\n║ {title}{' ' * max(0, pad - 1)}║\n╚{'═' * (_W - 2)}╝"


def _urdu_ratio(text: str) -> float:
    s = text.replace(" ", "")
    return len(_ARABIC_RE.findall(s)) / len(s) if s else 0.0


def _wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    """Wrap long text lines for display."""
    lines = []
    for para in text.split("\n"):
        wrapped = textwrap.fill(para, width=width, subsequent_indent=indent)
        lines.append(wrapped if wrapped else "")
    return "\n".join(lines)


def _print_retrieved(results: list[dict]) -> None:
    if not results:
        print("  (no documents retrieved)")
        return
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        score = r.get("score", 0.0)
        category = meta.get("category", "—")
        question = meta.get("question", "")[:80]
        source = meta.get("source_file", "—")
        print(f"  [{i}] score={score:.4f}  cat={category}  src={source}")
        if question:
            print(f"      Q: {question}")
    print()


def _print_query_result(qr: QueryResult, idx: int) -> None:
    print()
    print(_box(qr.query.label))
    print(f"  Scenario : {qr.query.scenario}")
    print(f"  Question : {qr.query.question}")
    print()
    print(f"  {'─' * 30} Retrieved Documents {'─' * 28}")
    _print_retrieved(qr.retrieved)

    print(f"  {'─' * 30} Final Answer {'─' * 35}")
    if qr.answer:
        urdu_pct = _urdu_ratio(qr.answer) * 100
        word_count = len(qr.answer.split())
        print(f"  Urdu ratio : {urdu_pct:.0f}%   Words : {word_count}")
        print()
        # Indent and wrap for readability
        for line in qr.answer.split("\n"):
            print(f"  {line}" if line.strip() else "")
    else:
        print("  (empty answer)")
    print()

    print(f"  {'─' * 30} Guardrails {'─' * 37}")
    print(f"  Passed preflight : {qr.passed_preflight}")
    print(f"  Guardrail hits   : {qr.guardrail_hits or 'none'}")
    for v in qr.verdict_summaries:
        icon = "✓" if v.get("passed") else "✗"
        reason = f"  — {v.get('reason', '')}" if not v.get("passed") else ""
        print(f"    {icon} {v.get('guard', '')}  {v.get('detail', '')}{reason}")
    print()

    # Timings
    t = qr.timings
    if t:
        parts = "  ".join(f"{k}={v}ms" for k, v in t.items())
        print(f"  Timings : {parts}")

    # Assertions
    print()
    print(f"  {'─' * 30} Assertions {'─' * 37}")
    all_ok = True
    for a in qr.assertions:
        icon = "PASS" if a["passed"] else "FAIL"
        detail = f"  ({a.get('detail', '')})" if a.get("detail") else ""
        print(f"  [{icon}]  {a['name']}{detail}")
        if not a["passed"]:
            all_ok = False
    print()

    if qr.error:
        print(f"  ERROR: {qr.error}")


def _print_summary(results: list[QueryResult]) -> None:
    print()
    print(_hr("═"))
    print("  TEST SUMMARY")
    print(_hr("═"))
    total_assertions = sum(len(r.assertions) for r in results)
    passed = sum(a["passed"] for r in results for a in r.assertions)
    failed = total_assertions - passed
    crashed = sum(1 for r in results if r.error)

    print(f"  Queries run   : {len(results)}")
    print(f"  Assertions    : {passed}/{total_assertions} passed")
    print(f"  Crashes       : {crashed}")
    print()

    for qr in results:
        fails = [a for a in qr.assertions if not a["passed"]]
        status = "PASS" if not fails and not qr.error else "FAIL"
        print(f"  [{status}]  {qr.query.label}")
        for f in fails:
            print(f"         ✗ {f['name']}: {f.get('detail', '')}")
    print()

    overall = "ALL TESTS PASSED" if failed == 0 and crashed == 0 else f"{failed} ASSERTION(S) FAILED  {crashed} CRASH(ES)"
    border = "✓" if failed == 0 and crashed == 0 else "✗"
    print(f"  {border}  {overall}")
    print(_hr("═"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build mini BM25
# ─────────────────────────────────────────────────────────────────────────────

def build_mini_bm25(cache_path: Path) -> Any:
    """Build a BM25Corpus from the 10 mini fatawa fixtures and cache to disk."""
    from src.retrieval.bm25_index import BM25Corpus  # noqa: PLC0415

    log.info("Building mini BM25 corpus (%d records)…", len(_MINI_FATAWA))
    corpus = BM25Corpus.build(_MINI_FATAWA)
    corpus.save(cache_path)
    log.info("Mini BM25 saved → %s", cache_path)
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Embed + upsert to test Pinecone index
# ─────────────────────────────────────────────────────────────────────────────

def setup_test_index() -> tuple[Any, dict]:
    """Embed all 10 mini fatawa and upsert them to the test Pinecone index.

    Returns ``(index_handle, upsert_stats)``.
    """
    from src.embedding.embedder import embed_texts       # noqa: PLC0415
    from src.indexing.pinecone_store import (            # noqa: PLC0415
        init_index,
        build_sparse_vector,
        _make_metadata_from_dict,
    )
    from src.indexing.sparse import build_sparse_vector  # noqa: PLC0415 (re-export)

    settings = get_settings()
    log.info("Creating / connecting to test index '%s'…", settings.pinecone_index_name)
    index = init_index()
    log.info("Test index ready.")

    # Embed all texts in one batch (10 records — well within rate limits)
    texts = [r["text"] for r in _MINI_FATAWA]
    log.info("Embedding %d records via %s…", len(texts), settings.embedding_model)
    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    embed_ms = (time.perf_counter() - t0) * 1000
    log.info("Embedded %d records in %.0fms", len(embeddings), embed_ms)

    # Build Pinecone vector dicts
    vectors = []
    for record, dense_vec in zip(_MINI_FATAWA, embeddings):
        meta = _make_metadata_from_dict({
            **record,
            "doc_id":       record["id"],
            "chunk_index":  0,
            "total_chunks": 1,
            "length_flag":  "normal",
            "date":         "",
            "reference":    "",
        })
        vectors.append({
            "id":            record["id"],
            "values":        dense_vec,
            "sparse_values": build_sparse_vector(record["text"]),
            "metadata":      meta,
        })

    # Upsert (Pinecone minimum batch is 1 for test scale — use a namespace to isolate)
    log.info("Upserting %d vectors to index '%s'…", len(vectors), settings.pinecone_index_name)
    t0 = time.perf_counter()
    # Upsert in one batch (10 vectors is tiny)
    upsert_resp = index.upsert(vectors=vectors)
    upsert_ms = (time.perf_counter() - t0) * 1000
    upserted = getattr(upsert_resp, "upserted_count", len(vectors))
    log.info("Upserted %d vectors in %.0fms", upserted, upsert_ms)

    # Brief pause to allow Pinecone to make vectors queryable
    log.info("Waiting 5s for index to become queryable…")
    time.sleep(5)

    return index, {"upserted": upserted, "embed_ms": round(embed_ms, 1), "upsert_ms": round(upsert_ms, 1)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 (MOCK) — Fake retrieval results
# ─────────────────────────────────────────────────────────────────────────────

def mock_hybrid_search(question: str, bm25_corpus: Any, top_k: int = 5) -> list[dict]:
    """BM25-only search used in mock mode (no Pinecone call)."""
    hits = bm25_corpus.search(question, top_k=top_k)
    results = []
    for h in hits:
        if h["score"] > 0:
            results.append({
                "text":     h.get("text", ""),
                "score":    h["score"],
                "metadata": h.get("metadata", {}),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run a single test query
# ─────────────────────────────────────────────────────────────────────────────

def run_query_live(tq: TestQuery, bm25_corpus: Any) -> QueryResult:
    """Run one test query through the full guarded pipeline (live mode)."""
    from src.pipeline.guardrails import (   # noqa: PLC0415
        GuardrailConfig,
        guarded_query,
    )

    qr = QueryResult(query=tq)
    # Relax thresholds slightly for the 10-record test corpus
    # (retrieval scores from a tiny index are naturally lower than production)
    cfg = GuardrailConfig(
        min_context_score=0.05,   # production default: 0.10
        min_top_score=0.10,       # production default: 0.20
        min_overlap_ratio=0.05,   # production default: 0.10
        min_urdu_ratio=0.30,      # production default: 0.40
    )

    try:
        t0 = time.perf_counter()
        result = guarded_query(
            tq.question,
            config=cfg,
            bm25_corpus=bm25_corpus,
            top_k=5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        qr.retrieved = [
            {"text": s.get("text", ""), "score": 0.0, "metadata": s}
            for s in result.sources
        ]
        qr.answer = result.answer
        qr.guardrail_hits = result.guardrail_hits
        qr.passed_preflight = result.passed
        qr.timings = {**result.timings, "total_ms": round(elapsed_ms, 1)}
        qr.verdict_summaries = [
            {
                "guard":  v.guard,
                "passed": v.passed,
                "reason": v.reason,
                "detail": str(v.detail),
            }
            for v in result.verdicts
        ]
    except Exception as exc:
        log.exception("Query '%s' raised an exception", tq.question)
        qr.error = str(exc)

    return qr


def run_query_mock(tq: TestQuery, bm25_corpus: Any) -> QueryResult:
    """Run one test query in mock mode: BM25 retrieval + guardrail checks (no LLM)."""
    from src.pipeline.guardrails import (   # noqa: PLC0415
        GuardrailConfig,
        ContextGuard, ConfidenceGuard,
        HallucinationGuard, LanguageGuard, LengthGuard,
    )

    qr = QueryResult(query=tq)
    cfg = GuardrailConfig(
        min_context_score=0.05,
        min_top_score=0.10,
        min_overlap_ratio=0.05,
        min_urdu_ratio=0.30,
    )

    try:
        # Retrieval (BM25 only — no Pinecone)
        retrieved = mock_hybrid_search(tq.question, bm25_corpus)
        qr.retrieved = retrieved

        # Pre-flight guards
        ctx_v = ContextGuard(cfg).check(retrieved)
        conf_v = ConfidenceGuard(cfg).check(retrieved)
        verdicts = [ctx_v, conf_v]
        hits: list[str] = []
        if not ctx_v.passed:
            hits.append(ctx_v.guard)
        if not conf_v.passed:
            hits.append(conf_v.guard)

        if hits:
            # Pre-flight failed — no LLM call
            from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL  # noqa: PLC0415
            answer = NO_ANSWER_SENTINEL
            qr.passed_preflight = False
        else:
            # Simulate a clean Urdu answer (no real LLM in mock mode)
            best = retrieved[0] if retrieved else {}
            stored_answer = best.get("metadata", {}).get("answer", "")
            answer = stored_answer or "مناسب جواب دستیاب نہیں۔"

        # Post-generation guards
        hall_v = HallucinationGuard(cfg).check(answer, retrieved)
        verdicts.append(hall_v)
        if not hall_v.passed:
            hits.append(hall_v.guard)
            from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL  # noqa: PLC0415
            answer = NO_ANSWER_SENTINEL

        lang_v, answer = LanguageGuard(cfg).check(answer)
        verdicts.append(lang_v)
        if not lang_v.passed:
            hits.append(lang_v.guard)

        len_v, answer = LengthGuard(cfg).check(answer)
        verdicts.append(len_v)
        if len_v.reason:
            hits.append(len_v.guard)

        qr.answer = answer
        qr.guardrail_hits = hits
        qr.verdict_summaries = [
            {
                "guard":  v.guard,
                "passed": v.passed,
                "reason": v.reason,
                "detail": str(v.detail),
            }
            for v in verdicts
        ]
        qr.timings = {"mock_ms": 0}

    except Exception as exc:
        log.exception("Mock query '%s' raised an exception", tq.question)
        qr.error = str(exc)

    return qr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Assertions
# ─────────────────────────────────────────────────────────────────────────────

def _assert(qr: QueryResult, name: str, condition: bool, detail: str = "") -> None:
    qr.assertions.append({"name": name, "passed": condition, "detail": detail})


def verify_query_result(qr: QueryResult) -> None:
    """Run all assertions for one query result."""
    tq = qr.query

    # ── Universal assertions (apply to every query) ──────────────────────────
    _assert(qr, "no_crash", qr.error == "",
            detail=qr.error[:120] if qr.error else "")

    _assert(qr, "answer_not_none", qr.answer is not None,
            detail=f"answer={repr(qr.answer)[:60]}")

    _assert(qr, "answer_not_empty_string", len(qr.answer.strip()) > 0,
            detail=f"len={len(qr.answer)}")

    _assert(qr, "answer_has_urdu_chars",
            _urdu_ratio(qr.answer) > 0.0,
            detail=f"urdu_ratio={_urdu_ratio(qr.answer):.2f}")

    _assert(qr, "verdicts_returned",
            len(qr.verdict_summaries) >= 2,
            detail=f"count={len(qr.verdict_summaries)}")

    _assert(qr, "guardrail_hits_is_list",
            isinstance(qr.guardrail_hits, list))

    if not qr.error:
        # timings dict was populated
        _assert(qr, "timings_populated",
                len(qr.timings) > 0,
                detail=str(qr.timings))

    # ── Scenario-specific assertions ─────────────────────────────────────────
    if tq.scenario == "normal":
        _assert(qr, "normal:has_retrieved_docs",
                len(qr.retrieved) > 0,
                detail=f"retrieved={len(qr.retrieved)}")
        _assert(qr, "normal:passed_preflight",
                qr.passed_preflight,
                detail="preflight_guards_rejected")
        _assert(qr, "normal:answer_urdu_dominant",
                _urdu_ratio(qr.answer) >= 0.20,
                detail=f"urdu_ratio={_urdu_ratio(qr.answer):.2f}")
        _assert(qr, "normal:answer_length_reasonable",
                5 <= len(qr.answer.split()) <= 1000,
                detail=f"words={len(qr.answer.split())}")

    elif tq.scenario == "edge_case":
        from src.pipeline.prompt_builder import NO_ANSWER_SENTINEL  # noqa: PLC0415
        # For out-of-domain queries: either the guardrails fired OR the LLM
        # returned the sentinel.  Both are acceptable outcomes.
        is_sentinel = qr.answer.strip() == NO_ANSWER_SENTINEL.strip()
        guards_fired = len(qr.guardrail_hits) > 0
        _assert(qr, "edge_case:sentinel_or_guardrail",
                is_sentinel or guards_fired or not qr.passed_preflight,
                detail=f"sentinel={is_sentinel}, hits={qr.guardrail_hits}, preflight={qr.passed_preflight}")

    elif tq.scenario == "ambiguous":
        # Ambiguous queries must not crash and must return a non-empty answer
        _assert(qr, "ambiguous:no_crash", qr.error == "")
        _assert(qr, "ambiguous:answer_returned",
                len(qr.answer.strip()) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def cleanup(index: Any, bm25_cache: Path, keep: bool) -> None:
    """Delete test vectors from Pinecone and remove the temporary BM25 cache."""
    if keep:
        log.info("--keep flag set; skipping cleanup.")
        return

    # Delete test vectors
    try:
        log.info("Deleting %d test vectors from Pinecone…", len(_TEST_IDS))
        index.delete(ids=_TEST_IDS)
        log.info("Test vectors deleted.")
    except Exception as exc:
        log.warning("Could not delete test vectors: %s", exc)

    # Remove temp BM25 file
    if bm25_cache.exists():
        bm25_cache.unlink(missing_ok=True)
        log.info("Removed temp BM25 cache: %s", bm25_cache)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """Orchestrate all test steps.  Returns 0 on success, 1 on any failure."""
    args = _args
    bm25_cache = Path(args.bm25_cache)
    mock_mode: bool = args.mock
    settings = get_settings()

    print()
    print(_box("Hybrid+Vectorless RAG  —  End-to-End Test"))
    mode_label = "MOCK (no API calls)" if mock_mode else f"LIVE  →  index: {settings.pinecone_index_name}"
    print(f"  Mode       : {mode_label}")
    print(f"  Records    : {len(_MINI_FATAWA)} mini fatawa fixtures")
    print(f"  Queries    : {len(TEST_QUERIES)}")
    print(f"  BM25 cache : {bm25_cache}")
    print(_hr())

    # ── Step 1: Build mini BM25 ───────────────────────────────────────────────
    print()
    print("  STEP 1 / 3  —  Building mini BM25 index")
    print(_hr())
    t0 = time.perf_counter()
    bm25_corpus = build_mini_bm25(bm25_cache)
    print(f"  BM25 built in {(time.perf_counter() - t0) * 1000:.0f}ms  ({len(_MINI_FATAWA)} docs)")

    # ── Step 2: Embed + index (live) or skip (mock) ────────────────────────────
    pinecone_index = None
    if mock_mode:
        print()
        print("  STEP 2 / 3  —  Pinecone indexing  [SKIPPED — mock mode]")
        print(_hr())
    else:
        print()
        print("  STEP 2 / 3  —  Embedding + indexing to Pinecone")
        print(_hr())
        t0 = time.perf_counter()
        try:
            pinecone_index, upsert_stats = setup_test_index()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  Indexed {upsert_stats['upserted']} vectors in {elapsed:.0f}ms")
            print(f"  embed_ms={upsert_stats['embed_ms']}  upsert_ms={upsert_stats['upsert_ms']}")
        except Exception as exc:
            log.error("Pinecone setup failed: %s", exc)
            print(f"\n  FATAL: Could not set up test Pinecone index: {exc}")
            print("  Hint: Check PINECONE_API_KEY in .env and that your plan supports")
            print(f"  index creation for '{settings.pinecone_index_name}'.")
            return 1

    # ── Step 3 + 4 + 5: Run queries, print, assert ────────────────────────────
    print()
    print("  STEP 3 / 3  —  Running test queries")
    print(_hr())

    all_results: list[QueryResult] = []

    for i, tq in enumerate(TEST_QUERIES, 1):
        print()
        log.info("Running query %d/%d: %s", i, len(TEST_QUERIES), tq.label)

        if mock_mode:
            qr = run_query_mock(tq, bm25_corpus)
        else:
            qr = run_query_live(tq, bm25_corpus)

        # Step 5: Verify assertions
        verify_query_result(qr)

        # Step 4: Print
        _print_query_result(qr, i)
        all_results.append(qr)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(all_results)

    # ── Optional JSON report ──────────────────────────────────────────────────
    if args.report:
        report_path = Path(args.report)
        report = {
            "mode": "mock" if mock_mode else "live",
            "index_name": settings.pinecone_index_name if not mock_mode else "n/a",
            "fixture_count": len(_MINI_FATAWA),
            "queries": [
                {
                    "label": qr.query.label,
                    "scenario": qr.query.scenario,
                    "question": qr.query.question,
                    "answer_preview": qr.answer[:200],
                    "guardrail_hits": qr.guardrail_hits,
                    "passed_preflight": qr.passed_preflight,
                    "assertions": qr.assertions,
                    "error": qr.error,
                    "timings": qr.timings,
                    "retrieved_count": len(qr.retrieved),
                    "top_score": max((r.get("score", 0.0) for r in qr.retrieved), default=0.0),
                }
                for qr in all_results
            ],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON report written → {report_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if pinecone_index is not None:
        cleanup(pinecone_index, bm25_cache, keep=args.keep)
    elif bm25_cache.exists() and not args.keep:
        bm25_cache.unlink(missing_ok=True)

    # ── Exit code ─────────────────────────────────────────────────────────────
    failed = sum(
        1
        for qr in all_results
        for a in qr.assertions
        if not a["passed"]
    )
    crashed = sum(1 for qr in all_results if qr.error)
    return 0 if (failed == 0 and crashed == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
