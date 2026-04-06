"""Stress test for the Hybrid+Vectorless RAG pipeline.

Simulates up to 500 k fatawa records without any external API calls
(OpenAI / Pinecone are replaced by DryRunContext mocks) and measures:

  - Batching performance  (throughput: records / second per stage)
  - Memory usage          (peak heap via tracemalloc; RSS via psutil if available)
  - Processing time       (wall-clock per stage, sub-stage breakdowns)

Stages profiled
---------------
  A  — Synthetic corpus generation
  B  — Urdu normalisation (``normalize_urdu``)
  C  — Chunking (``preprocess_records``), by text-length class
  D  — Embedding pipeline batch-size sweep (``embed_chunks``)
  E  — SQLite checkpoint: sequential write then read-back
  F  — BM25 corpus build + query latency at scale
  G  — Mock Pinecone upsert (batch-size sweep)
  H  — End-to-end pipeline smoke at configured dataset size
  I  — Full-pipeline 100 k end-to-end (configurable via --size)

Output
------
  - Formatted table to stdout (always)
  - Optional JSON report via --report FILE
  - Bottleneck section with severity classification
  - Concrete optimisation suggestions keyed to measured bottlenecks

Usage
-----
::

    python stress_test.py                              # default: 100 000 records
    python stress_test.py --size 10000                 # faster smoke run
    python stress_test.py --size 500000 --stages ABCDE # ingest + embed only
    python stress_test.py --batch-sweep                # extra batch-size sweep
    python stress_test.py --report stress_report.json
    python stress_test.py --profile                    # cProfile top-20 hotspots
    python stress_test.py --size 100000 --verbose      # DEBUG logging
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Optional-dependency stubs
# Must be injected into sys.modules BEFORE any src.* import so that
# import chains that touch pinecone or rank_bm25 don't raise ModuleNotFoundError.
# ─────────────────────────────────────────────────────────────────────────────

import io
import sys
from unittest.mock import MagicMock

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

if "pinecone" not in sys.modules:
    _pinecone_stub = MagicMock()
    _pinecone_stub.Pinecone = MagicMock
    _pinecone_stub.ServerlessSpec = MagicMock
    sys.modules["pinecone"] = _pinecone_stub


class _NdArrayLike:
    """A list wrapper that exposes the .tolist() method expected by BM25Corpus."""
    __slots__ = ("_data",)

    def __init__(self, data: list) -> None:
        self._data = data

    def tolist(self) -> list:
        return self._data


class _BM25OkapiStub:
    """Minimal BM25Okapi replacement that returns deterministic pseudo-scores.

    Used when rank_bm25 is not installed.  Scores are derived from the number
    of query tokens that appear as substrings of the document text — cheap
    but sufficient for throughput benchmarking.
    """

    def __init__(self, corpus: list[list[str]]) -> None:
        self._corpus = corpus          # list of token-lists
        self._n = len(corpus)

    def get_scores(self, query_tokens: list[str]) -> _NdArrayLike:
        import re as _re  # noqa: PLC0415
        q_set = set(query_tokens)
        scores: list[float] = []
        for tokens in self._corpus:
            if not tokens or not q_set:
                scores.append(0.0)
                continue
            overlap = len(q_set.intersection(tokens))
            scores.append(overlap / max(len(q_set), 1))
        return _NdArrayLike(scores)

    def get_top_n(self, query_tokens: list[str], docs: list, n: int = 5) -> list:
        return docs[:n]


if "rank_bm25" not in sys.modules:
    _rank_bm25_stub = MagicMock()
    _rank_bm25_stub.BM25Okapi = _BM25OkapiStub
    sys.modules["rank_bm25"] = _rank_bm25_stub

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import cProfile
import gc
import io
import json
import logging
import math
import os
import pickle
import pstats
import random
import struct
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger("stress_test")

# ─────────────────────────────────────────────────────────────────────────────
# Optional psutil  (for RSS memory; gracefully absent)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORIES      = ["NAMAZ", "WUDU", "ZAKAT", "FAST", "DIVORCE", "INHERITANCE",
                     "HAJJ", "JIHAD", "NIKKAH", "FOOD", "ADHAN", "TAUHEED"]
_SOURCES         = ["Banuri", "fatwaqa", "IslamQA", "urdufatwa"]
_SUBCATEGORIES   = ["salat", "wudu", "zakat", "saum", "talaaq", "miras",
                     "haj", "jihad", "nikah", "halal", "adhan", "tawhid"]

# Short / medium / long Urdu text fragments used to build synthetic records
_URDU_FRAGMENTS  = [
    "نماز میں نیت دل کا ارادہ ہے۔",
    "وضو کرنا فرض ہے۔ پانی سے ہاتھ منہ اور پاؤں دھوئیں۔",
    "زکوٰۃ ادا کرنا ہر مالدار مسلمان پر فرض ہے اور اس کے آٹھ مصارف ہیں۔",
    "رمضان کا روزہ فرض ہے اور اس کی قضا لازم ہے اگر کسی وجہ سے چھوٹ جائے۔",
    "حج اسلام کا پانچواں رکن ہے اور ہر صاحبِ استطاعت مسلمان پر عمر میں ایک بار فرض ہے۔",
    "نکاح کے لیے ایجاب و قبول اور گواہوں کی موجودگی ضروری ہے۔",
    "طلاق دینا شوہر کا حق ہے لیکن اسے بغیر ضرورت استعمال کرنا مکروہ ہے۔",
    "میراث کی تقسیم قرآن کریم کے احکام کے مطابق کی جائے گی۔",
    "ذبیحہ کے لیے بسم اللہ پڑھنا اور حلقوم کاٹنا شرط ہے۔",
    "صدقۃ الفطر ہر مسلمان پر واجب ہے جو نصاب کا مالک ہو۔",
    "قرآن کریم خاتم النبیین حضرت محمد صلی اللہ علیہ وسلم پر نازل ہوا۔",
    "توحید اسلام کی بنیاد ہے — لا الٰہ الا اللہ محمد رسول اللہ۔",
    "اذان میں اللہ اکبر چار بار اور کلمہ شہادت دو دو بار کہا جاتا ہے۔",
    "تیمم اس وقت جائز ہے جب پانی موجود نہ ہو یا پانی کا استعمال مضر ہو۔",
    "جہاد فی سبیل اللہ فرض کفایہ ہے اور اس کے شرائط فقہاء نے بیان کیے ہیں۔",
]

# Default batch sizes for the embedding sweep
_DEFAULT_BATCH_SIZES = [16, 32, 64, 128, 256, 512]
_DEFAULT_PINECONE_BATCH_SIZES = [50, 100, 200, 300, 500]

# ─────────────────────────────────────────────────────────────────────────────
# Result data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageMetric:
    stage_id:          str
    stage_name:        str
    n_records:         int
    elapsed_s:         float
    records_per_sec:   float
    peak_heap_mb:      float          # tracemalloc peak
    rss_delta_mb:      float          # psutil RSS delta (0 if unavailable)
    sub_metrics:       dict           = field(default_factory=dict)
    notes:             list[str]      = field(default_factory=list)

    @property
    def ms_per_record(self) -> float:
        return (self.elapsed_s * 1000 / self.n_records) if self.n_records else 0.0


@dataclass
class StressReport:
    timestamp:          str
    dataset_size:       int
    total_elapsed_s:    float
    stages:             list[StageMetric]   = field(default_factory=list)
    bottlenecks:        list[dict]          = field(default_factory=list)
    optimizations:      list[str]           = field(default_factory=list)
    platform_info:      dict                = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: timing + memory
# ─────────────────────────────────────────────────────────────────────────────

def _rss_mb() -> float:
    if not _PSUTIL:
        return 0.0
    try:
        return _psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    except Exception:
        return 0.0


class _Profiled:
    """Context manager that records wall-clock time, tracemalloc peak, and RSS delta."""

    def __init__(self, label: str) -> None:
        self._label     = label
        self.elapsed_s  = 0.0
        self.heap_mb    = 0.0
        self.rss_delta  = 0.0

    def __enter__(self) -> "_Profiled":
        gc.collect()
        tracemalloc.start()
        self._rss0  = _rss_mb()
        self._t0    = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_s = time.perf_counter() - self._t0
        _, peak        = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.heap_mb   = round(peak / 1_048_576, 2)
        self.rss_delta = round(_rss_mb() - self._rss0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic corpus generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_text(length_class: str, rng: random.Random) -> str:
    """Build a synthetic Urdu text string of the requested length class."""
    repeats = {"short": rng.randint(1, 3), "medium": rng.randint(5, 15), "long": rng.randint(20, 50)}[length_class]
    frags   = [rng.choice(_URDU_FRAGMENTS) for _ in range(repeats)]
    return " ".join(frags)


def generate_corpus(
    n: int,
    *,
    seed: int = 42,
    short_frac: float = 0.60,
    medium_frac: float = 0.30,
) -> list[dict]:
    """Return *n* synthetic records in the unified pipeline dict format.

    Length distribution matches a realistic fatawa corpus:
    - 60 % short  (1–3 sentences)
    - 30 % medium (5–15 sentences)
    - 10 % long   (20–50 sentences)
    """
    rng = random.Random(seed)

    short_n  = int(n * short_frac)
    medium_n = int(n * medium_frac)
    long_n   = n - short_n - medium_n

    specs: list[str] = (
        ["short"]  * short_n  +
        ["medium"] * medium_n +
        ["long"]   * long_n
    )
    rng.shuffle(specs)

    records: list[dict] = []
    for i, cls in enumerate(specs):
        q    = _make_text("short", rng)
        a    = _make_text(cls, rng)
        cat  = rng.choice(_CATEGORIES)
        src  = rng.choice(_SOURCES)
        doc_id = f"stress_{i:08x}"
        records.append({
            "id":          doc_id,
            "question":    q,
            "answer":      a,
            "text":        f"سوال: {q} جواب: {a}",
            "category":    cat,
            "source_file": f"{src}_{cat.lower()}.csv",
            "folder":      f"{src}-ExtractedData-Output",
            "date":        None,
            "reference":   None,
            "metadata": {
                "question":    q,
                "answer":      a,
                "category":    cat,
                "source_file": f"{src}_{cat.lower()}.csv",
                "doc_id":      doc_id,
                "folder":      f"{src}-ExtractedData-Output",
                "fatwa_no":    str(i),
            },
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Stage A — Synthetic corpus generation
# ─────────────────────────────────────────────────────────────────────────────

def stage_A(n: int, verbose: bool) -> StageMetric:
    logger.info("[A] Generating %s synthetic records…", f"{n:,}")

    with _Profiled("A") as p:
        corpus = generate_corpus(n)

    rps = n / p.elapsed_s if p.elapsed_s else float("inf")
    logger.info("[A] Done  %.2fs  %.0f rec/s  heap=%.1f MB", p.elapsed_s, rps, p.heap_mb)

    sizes = [len(r["text"]) for r in corpus]
    return StageMetric(
        stage_id="A", stage_name="Corpus generation",
        n_records=n, elapsed_s=round(p.elapsed_s, 3),
        records_per_sec=round(rps), peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
        sub_metrics={
            "avg_text_chars": round(sum(sizes) / len(sizes)),
            "max_text_chars": max(sizes),
            "total_chars_MB": round(sum(sizes) / 1_048_576, 2),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage B — Urdu normalisation throughput
# ─────────────────────────────────────────────────────────────────────────────

def stage_B(corpus: list[dict], verbose: bool) -> StageMetric:
    from src.preprocessing.urdu_normalizer import normalize_urdu   # noqa: PLC0415

    n = len(corpus)
    logger.info("[B] Normalising %s records…", f"{n:,}")

    with _Profiled("B") as p:
        for rec in corpus:
            normalize_urdu(rec["text"])

    rps = n / p.elapsed_s if p.elapsed_s else float("inf")
    logger.info("[B] Done  %.2fs  %.0f rec/s  heap=%.1f MB", p.elapsed_s, rps, p.heap_mb)

    return StageMetric(
        stage_id="B", stage_name="Urdu normalisation",
        n_records=n, elapsed_s=round(p.elapsed_s, 3),
        records_per_sec=round(rps), peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage C — Chunking throughput (short / medium / long separately)
# ─────────────────────────────────────────────────────────────────────────────

def stage_C(corpus: list[dict], verbose: bool) -> StageMetric:
    from src.preprocessing.chunker import preprocess_records   # noqa: PLC0415

    n     = len(corpus)
    logger.info("[C] Chunking %s documents…", f"{n:,}")

    with _Profiled("C") as p:
        chunks = list(preprocess_records(corpus))

    rps = n / p.elapsed_s if p.elapsed_s else float("inf")
    ratio = len(chunks) / n if n else 0

    logger.info("[C] Done  %.2fs  %.0f rec/s  chunks=%s  ratio=%.2f  heap=%.1f MB",
                p.elapsed_s, rps, f"{len(chunks):,}", ratio, p.heap_mb)

    return StageMetric(
        stage_id="C", stage_name="Chunking",
        n_records=n, elapsed_s=round(p.elapsed_s, 3),
        records_per_sec=round(rps), peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
        sub_metrics={
            "total_chunks": len(chunks),
            "chunk_expand_ratio": round(ratio, 3),
            "avg_text_len": round(
                sum(len(c.get("text", "")) for c in chunks) / len(chunks)
            ) if chunks else 0,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage D — Embedding batch-size sweep
# ─────────────────────────────────────────────────────────────────────────────

def stage_D(
    corpus: list[dict],
    batch_sizes: list[int],
    verbose: bool,
    do_sweep: bool,
) -> StageMetric:
    """Use the real embed_chunks() pipeline with mock embed_texts (DryRunContext).

    Measures raw throughput (records embedded per second) for different
    batch_size values to find the optimal batching configuration.
    """
    from src.embedding.pipeline import embed_chunks   # noqa: PLC0415

    # Run only a 5 k slice for the sweep so it finishes quickly
    sample  = corpus[:5_000] if do_sweep else corpus[:min(len(corpus), 20_000)]
    n       = len(sample)

    sweep_results: dict[int, dict] = {}
    best_bs = batch_sizes[0]
    best_rps = 0.0

    sizes_to_run = batch_sizes if do_sweep else [64]

    for bs in sizes_to_run:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            with _Profiled(f"D-bs{bs}") as p:
                count = sum(
                    1 for _ in embed_chunks(
                        iter(sample),
                        checkpoint_path=db_path,
                        batch_size=bs,
                        show_progress=verbose,
                    )
                )
            rps = count / p.elapsed_s if p.elapsed_s else float("inf")
            sweep_results[bs] = {
                "batch_size":  bs,
                "embedded":    count,
                "elapsed_s":   round(p.elapsed_s, 3),
                "rec_per_sec": round(rps),
                "heap_MB":     p.heap_mb,
            }
            if rps > best_rps:
                best_rps = rps
                best_bs  = bs
            logger.info("[D] bs=%-4d  %s recs  %.2fs  %.0f rec/s  heap=%.1f MB",
                        bs, f"{count:,}", p.elapsed_s, rps, p.heap_mb)
        finally:
            db_path.unlink(missing_ok=True)

    # Report the best result as the primary metric
    best   = sweep_results[best_bs]
    return StageMetric(
        stage_id="D", stage_name="Embedding (batch-size sweep)",
        n_records=n, elapsed_s=best["elapsed_s"],
        records_per_sec=best["rec_per_sec"], peak_heap_mb=best["heap_MB"],
        rss_delta_mb=0.0,
        sub_metrics={
            "sweep": sweep_results,
            "optimal_batch_size": best_bs,
            "note": "Best batch size shown as primary",
        },
        notes=[f"Optimal batch size: {best_bs}  ({best['rec_per_sec']:,} rec/s)"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage E — SQLite checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def stage_E(n: int, verbose: bool) -> StageMetric:
    """Benchmark the EmbeddingCheckpoint write and read-back paths directly.

    Writes *n* fake 3072-dim float32 vectors and then reads them all back.
    This isolates pure I/O cost without the embedding CPU overhead.
    """
    from src.embedding.pipeline import EmbeddingCheckpoint   # noqa: PLC0415

    DIM        = 3072
    BATCH_SIZE = 256

    logger.info("[E] SQLite checkpoint: %s records × %d-dim…", f"{n:,}", DIM)

    # ── pre-generate random bytes (outside the measured region) ──────────
    all_rows: list[dict] = []
    rng = random.Random(0)
    for i in range(n):
        # Use random bytes packed as float32 — avoids compute overhead in loop
        raw = bytes(rng.randint(0, 255) for _ in range(DIM * 4))
        all_rows.append({
            "id":        f"stress_e_{i:08x}",
            "embedding": list(struct.unpack(f"<{DIM}f", raw)),
            "metadata":  {"category": "NAMAZ", "fatwa_no": str(i)},
        })

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    try:
        with _Profiled("E-write") as pw:
            with EmbeddingCheckpoint(db_path) as ckpt:
                for i in range(0, n, BATCH_SIZE):
                    ckpt.write_batch(all_rows[i : i + BATCH_SIZE])

        write_rps = n / pw.elapsed_s if pw.elapsed_s else float("inf")
        write_mb  = (n * DIM * 4) / 1_048_576

        with _Profiled("E-read") as pr:
            with EmbeddingCheckpoint(db_path) as ckpt:
                read_count = sum(1 for _ in ckpt.iter_all())

        read_rps = read_count / pr.elapsed_s if pr.elapsed_s else float("inf")

        logger.info("[E] write  %.2fs  %.0f rec/s  (%.1f MB raw vectors)",
                    pw.elapsed_s, write_rps, write_mb)
        logger.info("[E] read   %.2fs  %.0f rec/s", pr.elapsed_s, read_rps)

        return StageMetric(
            stage_id="E", stage_name="SQLite checkpoint I/O",
            n_records=n, elapsed_s=round(pw.elapsed_s + pr.elapsed_s, 3),
            records_per_sec=round(write_rps),
            peak_heap_mb=pw.heap_mb, rss_delta_mb=pw.rss_delta,
            sub_metrics={
                "write_rec_per_sec": round(write_rps),
                "read_rec_per_sec":  round(read_rps),
                "write_elapsed_s":   round(pw.elapsed_s, 3),
                "read_elapsed_s":    round(pr.elapsed_s, 3),
                "raw_vector_MB":     round(write_mb, 1),
                "batch_size":        BATCH_SIZE,
            },
        )
    finally:
        db_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage F — BM25 corpus build + query latency
# ─────────────────────────────────────────────────────────────────────────────

def stage_F(corpus: list[dict], verbose: bool) -> StageMetric:
    from src.retrieval.bm25_index import BM25Corpus   # noqa: PLC0415

    n       = len(corpus)
    queries = [
        "نماز پڑھتے وقت نیت کا حکم",
        "زکوٰۃ کا نصاب کتنا ہے",
        "روزہ کی حالت میں کون سی چیزیں مکروہ ہیں",
        "وضو ٹوٹنے کی وجوہات",
        "طلاق کے احکام",
    ]

    logger.info("[F] Building BM25 index over %s docs…", f"{n:,}")

    with _Profiled("F-build") as pb:
        bm25_corpus = BM25Corpus.build(corpus)

    build_rps = n / pb.elapsed_s if pb.elapsed_s else float("inf")
    logger.info("[F] Build  %.2fs  %.0f rec/s  heap=%.1f MB",
                pb.elapsed_s, build_rps, pb.heap_mb)

    # ── Query latency benchmark ───────────────────────────────────────────
    N_QUERY_TRIALS = 50
    query_times: list[float] = []

    for _ in range(N_QUERY_TRIALS):
        q = random.choice(queries)
        t0 = time.perf_counter()
        bm25_corpus.search(q, top_k=10)
        query_times.append(time.perf_counter() - t0)

    avg_q_ms  = 1000 * sum(query_times) / len(query_times)
    p50_q_ms  = 1000 * sorted(query_times)[len(query_times) // 2]
    p99_q_ms  = 1000 * sorted(query_times)[int(len(query_times) * 0.99)]

    logger.info("[F] Query p50=%.1f ms  p99=%.1f ms  avg=%.1f ms",
                p50_q_ms, p99_q_ms, avg_q_ms)

    # ── Pickle serialisation ──────────────────────────────────────────────
    import pickle   # noqa: PLC0415
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pkl_path = Path(f.name)

    try:
        with _Profiled("F-pickle") as pp:
            with open(pkl_path, "wb") as fh:
                pickle.dump(bm25_corpus, fh, protocol=pickle.HIGHEST_PROTOCOL)
            pkl_bytes = pkl_path.stat().st_size

        with _Profiled("F-unpickle") as pu:
            with open(pkl_path, "rb") as fh:
                _ = pickle.load(fh)   # noqa: S301

        logger.info("[F] Pickle size=%.1f MB  write=%.2fs  load=%.2fs",
                    pkl_bytes / 1_048_576, pp.elapsed_s, pu.elapsed_s)
    finally:
        pkl_path.unlink(missing_ok=True)

    return StageMetric(
        stage_id="F", stage_name="BM25 build + query",
        n_records=n, elapsed_s=round(pb.elapsed_s, 3),
        records_per_sec=round(build_rps), peak_heap_mb=pb.heap_mb,
        rss_delta_mb=pb.rss_delta,
        sub_metrics={
            "build_rec_per_sec": round(build_rps),
            "query_avg_ms":      round(avg_q_ms, 2),
            "query_p50_ms":      round(p50_q_ms, 2),
            "query_p99_ms":      round(p99_q_ms, 2),
            "n_query_trials":    N_QUERY_TRIALS,
            "pickle_size_MB":    round(pkl_bytes / 1_048_576, 2),
            "pickle_write_s":    round(pp.elapsed_s, 3),
            "pickle_load_s":     round(pu.elapsed_s, 3),
        },
        notes=[
            f"Query p50 {p50_q_ms:.1f} ms / p99 {p99_q_ms:.1f} ms @ {n:,} docs",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage G — Mock Pinecone upsert batch-size sweep
# ─────────────────────────────────────────────────────────────────────────────

def stage_G(
    n: int,
    batch_sizes: list[int],
    verbose: bool,
    do_sweep: bool,
) -> StageMetric:
    """Benchmark the complete upsert path (build vector records + upsert calls).

    Uses the mock Pinecone index from DryRunContext so no network is involved.
    Measures the per-record overhead of metadata serialisation, sparse vector
    construction, and the in-memory upsert bookkeeping.
    """
    from src.indexing.sparse import build_sparse_vector   # noqa: PLC0415
    from src.dry_run import _MockPineconeIndex            # noqa: PLC0415

    DIM   = 3072
    rng   = random.Random(1)
    sweep: dict[int, dict] = {}

    sizes_to_run = batch_sizes if do_sweep else [100]
    sample_n     = min(n, 5_000 if do_sweep else 20_000)

    for bs in sizes_to_run:
        index = _MockPineconeIndex()
        vectors: list[dict] = []

        raw_vecs = [
            [rng.gauss(0, 1) for _ in range(DIM)]
            for _ in range(sample_n)
        ]
        # Normalise
        for v in raw_vecs:
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            for j in range(len(v)):
                v[j] /= norm

        # Build vector payloads
        def _mk_vec(i: int) -> dict:
            txt     = rng.choice(_URDU_FRAGMENTS)
            sparse  = build_sparse_vector(txt)
            return {
                "id":            f"stress_g_{i:08x}",
                "values":        raw_vecs[i],
                "sparse_values": {"indices": sparse["indices"], "values": sparse["values"]},
                "metadata": {
                    "question":    txt,
                    "answer":      txt,
                    "category":    rng.choice(_CATEGORIES),
                    "source_file": "stress.csv",
                    "doc_id":      f"stress_g_{i:08x}",
                    "fatwa_no":    str(i),
                    "text":        txt,
                },
            }

        with _Profiled(f"G-bs{bs}") as p:
            for i in range(0, sample_n, bs):
                batch     = [_mk_vec(j) for j in range(i, min(i + bs, sample_n))]
                index.upsert(batch)

        rps = sample_n / p.elapsed_s if p.elapsed_s else float("inf")
        sweep[bs] = {"batch_size": bs, "rec_per_sec": round(rps),
                     "elapsed_s": round(p.elapsed_s, 3), "heap_MB": p.heap_mb}
        logger.info("[G] bs=%-4d  %.2fs  %.0f rec/s  heap=%.1f MB",
                    bs, p.elapsed_s, rps, p.heap_mb)

    best_bs   = max(sweep, key=lambda b: sweep[b]["rec_per_sec"])
    best      = sweep[best_bs]
    return StageMetric(
        stage_id="G", stage_name="Pinecone upsert (batch-size sweep)",
        n_records=sample_n, elapsed_s=best["elapsed_s"],
        records_per_sec=best["rec_per_sec"], peak_heap_mb=best["heap_MB"],
        rss_delta_mb=0.0,
        sub_metrics={"sweep": sweep, "optimal_batch_size": best_bs},
        notes=[f"Optimal Pinecone batch size: {best_bs}  ({best['rec_per_sec']:,} rec/s)"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage H — Hybrid retrieval throughput (guarded_query)
# ─────────────────────────────────────────────────────────────────────────────

def stage_H(corpus: list[dict], verbose: bool) -> StageMetric:
    """Measure end-to-end guarded_query throughput over the mock corpus.

    Runs 20 representative queries (including an out-of-domain one) and
    records per-query wall-clock time + guardrail outcomes.
    """
    from src.retrieval.bm25_index import BM25Corpus        # noqa: PLC0415
    from src.pipeline.guardrails import GuardrailConfig, guarded_query  # noqa: PLC0415

    queries = [
        "نماز پڑھتے وقت نیت کا طریقہ کیا ہے",
        "زکوٰۃ کا نصاب کتنا ہے اور کتنی رقم پر فرض ہے",
        "وضو کرنے کا صحیح طریقہ",
        "روزہ رکھنے کی شرائط کیا ہیں",
        "طلاق کے احکام اور اسکے اقسام",
        "حج فرض ہونے کی شرائط",
        "Python programming language",   # out-of-domain probe
        "نکاح کے لیے گواہوں کی تعداد",
        "ذبیحہ کے لیے کیا شرائط ہیں",
        "میراث کی تقسیم کا طریقہ",
        "صدقۃ الفطر کا نصاب",
        "پانی کی اقسام اور طہارت",
        "اذان میں کتنی بار اللہ اکبر کہا جاتا ہے",
        "توحید کا مطلب کیا ہے",
        "جہاد فی سبیل اللہ کی تعریف",
        "قرآن کریم کی تلاوت کے فضائل",
        "عید کی نماز کا وقت",
        "تیمم کا طریقہ",
        "حائضہ عورت کے لیے احکام",
        "سفر میں نماز قصر",
    ]

    cfg = GuardrailConfig(
        min_context_score=0.03,
        min_top_score=0.03,
        min_overlap_ratio=0.03,
        min_urdu_ratio=0.15,
    )

    # Build small BM25 corpus from a 1 k slice  (faster index build)
    bm25 = BM25Corpus.build(corpus[:1_000])

    query_times:     list[float] = []
    guardrail_hits:  int         = 0
    preflight_fails: int         = 0

    logger.info("[H] Running %d queries through guarded_query()…", len(queries))

    with _Profiled("H") as p:
        for q in queries:
            t0 = time.perf_counter()
            gr = guarded_query(q, config=cfg, bm25_corpus=bm25, top_k=5)
            query_times.append(time.perf_counter() - t0)
            if gr.guardrail_hits:
                guardrail_hits += 1
            if not gr.passed:
                preflight_fails += 1

    avg_ms = 1000 * sum(query_times) / len(query_times)
    p50_ms = 1000 * sorted(query_times)[len(query_times) // 2]
    p99_ms = 1000 * sorted(query_times)[-1]

    logger.info("[H] avg=%.1f ms  p50=%.1f ms  p99=%.1f ms  "
                "guardrail_hits=%d  preflight_fails=%d",
                avg_ms, p50_ms, p99_ms, guardrail_hits, preflight_fails)

    return StageMetric(
        stage_id="H", stage_name="Hybrid retrieval (guarded_query)",
        n_records=len(queries), elapsed_s=round(p.elapsed_s, 3),
        records_per_sec=round(len(queries) / p.elapsed_s) if p.elapsed_s else 0,
        peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
        sub_metrics={
            "n_queries":        len(queries),
            "avg_ms":           round(avg_ms, 1),
            "p50_ms":           round(p50_ms, 1),
            "p99_ms":           round(p99_ms, 1),
            "guardrail_hits":   guardrail_hits,
            "preflight_fails":  preflight_fails,
        },
        notes=[f"Query p50 {p50_ms:.1f} ms  p99 {p99_ms:.1f} ms"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage I — Full end-to-end pipeline (100 k records)
# ─────────────────────────────────────────────────────────────────────────────

def stage_I(corpus: list[dict], verbose: bool) -> StageMetric:
    """Single-pass throughput: normalise → chunk → embed → upsert.

    Processes every record once in a streaming fashion (no list materialisation
    between stages) to simulate the real production ingest path.
    """
    from src.preprocessing.urdu_normalizer import normalize_urdu   # noqa: PLC0415
    from src.preprocessing.chunker          import preprocess_records  # noqa: PLC0415
    from src.embedding.pipeline             import embed_chunks    # noqa: PLC0415
    from src.dry_run                        import _MockPineconeIndex  # noqa: PLC0415
    from src.indexing.sparse                import build_sparse_vector  # noqa: PLC0415
    import math as _math   # noqa: PLC0415

    n      = len(corpus)
    index  = _MockPineconeIndex()
    PINCONE_BS = 100

    logger.info("[I] Full end-to-end pass over %s records…", f"{n:,}")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    stage_times: dict[str, float] = defaultdict(float)
    chunk_count = 0
    embed_count = 0
    upsert_count = 0
    pinecone_buf: list[dict] = []

    def _flush_pinecone(buf: list[dict]) -> None:
        nonlocal upsert_count
        index.upsert(buf)
        upsert_count += len(buf)

    try:
        with _Profiled("I-total") as p:

            # ── Chunking ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            chunks = list(preprocess_records(corpus))
            stage_times["chunk_s"] = time.perf_counter() - t0
            chunk_count = len(chunks)

            # ── Embedding ────────────────────────────────────────────────
            t0 = time.perf_counter()
            embedded_iter = embed_chunks(
                iter(chunks),
                checkpoint_path=db_path,
                batch_size=64,
                show_progress=verbose,
            )

            for rec in embedded_iter:
                embed_count += 1
                # ── Upsert ───────────────────────────────────────────────
                emb  = rec["embedding"]
                norm = _math.sqrt(sum(v * v for v in emb)) or 1.0
                emb  = [v / norm for v in emb]
                txt  = (rec.get("metadata") or {}).get("text", "")
                sp   = build_sparse_vector(txt)
                pinecone_buf.append({
                    "id":            rec["id"],
                    "values":        emb,
                    "sparse_values": {"indices": sp["indices"], "values": sp["values"]},
                    "metadata":      rec.get("metadata", {}),
                })
                if len(pinecone_buf) >= PINCONE_BS:
                    _flush_pinecone(pinecone_buf)
                    pinecone_buf.clear()

            stage_times["embed_upsert_s"] = time.perf_counter() - t0

        if pinecone_buf:
            _flush_pinecone(pinecone_buf)

    finally:
        db_path.unlink(missing_ok=True)

    total_s = p.elapsed_s
    rps     = n / total_s if total_s else float("inf")

    logger.info("[I] Done  %.2fs  %.0f rec/s  chunks=%s  embeds=%s  upserts=%s",
                total_s, rps, f"{chunk_count:,}", f"{embed_count:,}", f"{upsert_count:,}")

    return StageMetric(
        stage_id="I", stage_name="Full end-to-end ingest",
        n_records=n, elapsed_s=round(total_s, 3),
        records_per_sec=round(rps), peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
        sub_metrics={
            "chunk_s":       round(stage_times["chunk_s"], 3),
            "embed_upsert_s": round(stage_times["embed_upsert_s"], 3),
            "total_chunks":  chunk_count,
            "total_embeds":  embed_count,
            "total_upserts": upsert_count,
            "chunk_pct":     round(100 * stage_times["chunk_s"] / total_s, 1) if total_s else 0,
            "embed_pct":     round(100 * stage_times["embed_upsert_s"] / total_s, 1) if total_s else 0,
        },
    )


def _iter_pkl_embeddings(cache_path: Path) -> Generator[dict, None, None]:
    """Yield embedding records from a streamed pickle cache file."""
    with cache_path.open("rb") as fh:
        header = pickle.load(fh)
        if not isinstance(header, dict) or header.get("format") != "stress-embeddings-v1":
            raise ValueError(f"Unsupported embedding cache format in {cache_path}")

        while True:
            try:
                rec = pickle.load(fh)
            except EOFError:
                break
            if isinstance(rec, dict) and "id" in rec and "embedding" in rec:
                yield rec


def _cache_and_iter_embeddings(
    records: Iterable[dict],
    cache_path: Path,
) -> Generator[dict, None, None]:
    """Stream records while writing them to a pickle cache once."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(
            {
                "format": "stress-embeddings-v1",
                "created_by": "stress_test.stage_I",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for rec in records:
            pickle.dump(rec, fh, protocol=pickle.HIGHEST_PROTOCOL)
            yield rec


def stage_I_with_cache(
    corpus: list[dict],
    verbose: bool,
    embed_cache_pkl: Path | None,
) -> StageMetric:
    """Stage I wrapper that reuses a persisted embedding PKL cache when present."""
    from src.preprocessing.chunker          import preprocess_records  # noqa: PLC0415
    from src.embedding.pipeline             import embed_chunks    # noqa: PLC0415
    from src.dry_run                        import _MockPineconeIndex  # noqa: PLC0415
    from src.indexing.sparse                import build_sparse_vector  # noqa: PLC0415
    import math as _math   # noqa: PLC0415

    n      = len(corpus)
    index  = _MockPineconeIndex()
    PINCONE_BS = 100

    logger.info("[I] Full end-to-end pass over %s records…", f"{n:,}")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    stage_times: dict[str, float] = defaultdict(float)
    chunk_count = 0
    embed_count = 0
    upsert_count = 0
    pinecone_buf: list[dict] = []

    def _flush_pinecone(buf: list[dict]) -> None:
        nonlocal upsert_count
        index.upsert(buf)
        upsert_count += len(buf)

    try:
        with _Profiled("I-total") as p:
            t0 = time.perf_counter()
            chunks = list(preprocess_records(corpus))
            stage_times["chunk_s"] = time.perf_counter() - t0
            chunk_count = len(chunks)

            t0 = time.perf_counter()
            if embed_cache_pkl and embed_cache_pkl.exists():
                logger.info("[I] Reusing embedding cache from %s", embed_cache_pkl)
                embedded_iter = _iter_pkl_embeddings(embed_cache_pkl)
            else:
                fresh_iter = embed_chunks(
                    iter(chunks),
                    checkpoint_path=db_path,
                    batch_size=64,
                    show_progress=verbose,
                )
                if embed_cache_pkl:
                    logger.info("[I] Writing embedding cache to %s", embed_cache_pkl)
                    embedded_iter = _cache_and_iter_embeddings(fresh_iter, embed_cache_pkl)
                else:
                    embedded_iter = fresh_iter

            for rec in embedded_iter:
                embed_count += 1
                emb  = rec["embedding"]
                norm = _math.sqrt(sum(v * v for v in emb)) or 1.0
                emb  = [v / norm for v in emb]
                txt  = (rec.get("metadata") or {}).get("text", "")
                sp   = build_sparse_vector(txt)
                pinecone_buf.append({
                    "id":            rec["id"],
                    "values":        emb,
                    "sparse_values": {"indices": sp["indices"], "values": sp["values"]},
                    "metadata":      rec.get("metadata", {}),
                })
                if len(pinecone_buf) >= PINCONE_BS:
                    _flush_pinecone(pinecone_buf)
                    pinecone_buf.clear()

            stage_times["embed_upsert_s"] = time.perf_counter() - t0

        if pinecone_buf:
            _flush_pinecone(pinecone_buf)

    finally:
        db_path.unlink(missing_ok=True)

    total_s = p.elapsed_s
    rps     = n / total_s if total_s else float("inf")

    logger.info("[I] Done  %.2fs  %.0f rec/s  chunks=%s  embeds=%s  upserts=%s",
                total_s, rps, f"{chunk_count:,}", f"{embed_count:,}", f"{upsert_count:,}")

    notes = [f"chunks={chunk_count:,} embeds={embed_count:,} upserts={upsert_count:,}"]
    if embed_cache_pkl:
        notes.append(f"embed_cache={embed_cache_pkl}")

    return StageMetric(
        stage_id="I", stage_name="Full end-to-end ingest",
        n_records=n, elapsed_s=round(total_s, 3),
        records_per_sec=round(rps), peak_heap_mb=p.heap_mb, rss_delta_mb=p.rss_delta,
        sub_metrics={
            "chunk_s":       round(stage_times["chunk_s"], 3),
            "embed_upsert_s": round(stage_times["embed_upsert_s"], 3),
            "total_chunks":  chunk_count,
            "total_embeds":  embed_count,
            "total_upserts": upsert_count,
            "chunk_pct":     round(100 * stage_times["chunk_s"] / total_s, 1) if total_s else 0,
            "embed_pct":     round(100 * stage_times["embed_upsert_s"] / total_s, 1) if total_s else 0,
        },
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck analysis
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_THRESHOLDS = {
    "CRITICAL": 500,     # < 500 rec/s → critical
    "HIGH":     2_000,   # < 2 000 rec/s → high
    "MEDIUM":   10_000,  # < 10 000 rec/s → medium
    # otherwise → LOW / negligible
}


def analyse_bottlenecks(stages: list[StageMetric]) -> list[dict]:
    bottlenecks: list[dict] = []
    for s in stages:
        if s.records_per_sec <= 0:
            continue
        rps = s.records_per_sec
        if rps < _SEVERITY_THRESHOLDS["CRITICAL"]:
            sev = "CRITICAL"
        elif rps < _SEVERITY_THRESHOLDS["HIGH"]:
            sev = "HIGH"
        elif rps < _SEVERITY_THRESHOLDS["MEDIUM"]:
            sev = "MEDIUM"
        else:
            sev = "LOW"

        if sev in ("CRITICAL", "HIGH", "MEDIUM"):
            bottlenecks.append({
                "stage_id":       s.stage_id,
                "stage_name":     s.stage_name,
                "severity":       sev,
                "records_per_sec": s.records_per_sec,
                "elapsed_s":      s.elapsed_s,
                "peak_heap_mb":   s.peak_heap_mb,
            })

    bottlenecks.sort(key=lambda b: b["records_per_sec"])
    return bottlenecks


# ─────────────────────────────────────────────────────────────────────────────
# Optimization suggestions (data-driven)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizations(stages: list[StageMetric], bottlenecks: list[dict]) -> list[str]:
    """Return concrete, actionable optimization suggestions based on results."""
    stage_map = {s.stage_id: s for s in stages}
    opts: list[str] = []

    # ── Embedding batch size ────────────────────────────────────────────────
    d = stage_map.get("D")
    if d and "optimal_batch_size" in d.sub_metrics:
        obs  = d.sub_metrics["optimal_batch_size"]
        sweep = d.sub_metrics.get("sweep", {})
        if sweep:
            max_rps  = max(v["rec_per_sec"] for v in sweep.values())
            min_rps  = min(v["rec_per_sec"] for v in sweep.values())
            gain_pct = round(100 * (max_rps - min_rps) / min_rps) if min_rps else 0
            if gain_pct > 10:
                opts.append(
                    f"[D] Set embed batch_size={obs} in embed_chunks() — "
                    f"yields {gain_pct}% throughput gain vs. worst batch size."
                )

    # ── SQLite write throughput ─────────────────────────────────────────────
    e = stage_map.get("E")
    if e:
        write_rps = e.sub_metrics.get("write_rec_per_sec", 0)
        if write_rps < 5_000:
            opts.append(
                "[E] SQLite write throughput is low "
                f"({write_rps:,} rec/s). "
                "Enable WAL mode: conn.execute('PRAGMA journal_mode=WAL') + "
                "PRAGMA synchronous=NORMAL;  Expected 3-5× improvement."
            )
        if e.sub_metrics.get("raw_vector_MB", 0) > 500:
            opts.append(
                "[E] Checkpoint DB will exceed 500 MB for this dataset. "
                "Consider storing float16 vectors (half-precision) — "
                "halves I/O and disk without significant retrieval quality loss."
            )

    # ── BM25 build time ─────────────────────────────────────────────────────
    f = stage_map.get("F")
    if f:
        build_rps = f.sub_metrics.get("build_rec_per_sec", 0)
        p99_ms    = f.sub_metrics.get("query_p99_ms", 0.0)
        if build_rps < 5_000:
            opts.append(
                f"[F] BM25 index build is slow ({build_rps:,} rec/s). "
                "Pre-build the index in a background process and persist with pickle. "
                "For corpora > 100 k docs use sparse arrays (scipy.sparse) "
                "instead of rank_bm25 dense lists — 2-4× faster build."
            )
        if p99_ms > 200:
            opts.append(
                f"[F] BM25 query p99 = {p99_ms:.0f} ms — too slow for real-time. "
                "Pre-filter by category label before BM25 search "
                "(reduces the search space by ~8×)."
            )
        pkl_size = f.sub_metrics.get("pickle_size_MB", 0)
        pickle_load_s = f.sub_metrics.get("pickle_load_s", 0)
        if pkl_size > 200:
            opts.append(
                f"[F] BM25 pickle is {pkl_size:.0f} MB. "
                f"Loading takes {pickle_load_s:.1f}s. "
                "Partition the corpus by category and build 14 smaller indexes "
                "(one per category folder) — reduces each partition to ~7% of the total."
            )

    # ── Chunking ────────────────────────────────────────────────────────────
    c = stage_map.get("C")
    if c:
        ratio = c.sub_metrics.get("chunk_expand_ratio", 1.0)
        if ratio > 2.5:
            opts.append(
                f"[C] Chunk expansion ratio = {ratio:.1f}× — long answers are being "
                "fragmented heavily. Increase MAX_TOKENS from 500 to 750-1000 "
                "to reduce chunk count and downstream embedding cost."
            )
        if c.records_per_sec < 5_000:
            opts.append(
                f"[C] Chunker throughput low ({c.records_per_sec:,} rec/s). "
                "The sentence-split regex fires on every record. "
                "Pre-compile _SENTENCE_SPLIT_RE at module load (already done), "
                "but consider batch tokenisation with multiprocessing.Pool "
                "for > 200 k records."
            )

    # ── Memory ──────────────────────────────────────────────────────────────
    high_heap = [s for s in stages if s.peak_heap_mb > 512]
    if high_heap:
        names = ", ".join(f"[{s.stage_id}] {s.stage_name}" for s in high_heap)
        opts.append(
            f"High heap usage (> 512 MB) in: {names}. "
            "Convert list-materialising loops to generators — e.g. replace "
            "list(preprocess_records(docs)) with iterating directly so the "
            "chunker output streams through the embedder without full materialisation."
        )

    # ── Full pipeline time estimate ──────────────────────────────────────────
    i = stage_map.get("I")
    if i:
        if i.records_per_sec > 0:
            real_embed_rps = i.records_per_sec   # dry-run embed is instant vs. real API
            seconds_per_1m = 1_000_000 / real_embed_rps
            # Real OpenAI embed: ~2000 rec/s at 3072-dim with 2 parallel threads
            real_api_rps = 2_000
            real_seconds_1m = 1_000_000 / real_api_rps
            opts.append(
                f"[I] Pipeline processes {i.records_per_sec:,} rec/s in dry-run. "
                f"With real OpenAI embeddings (~{real_api_rps:,} rec/s rate-limited), "
                f"1 M records ≈ {real_seconds_1m/3600:.1f} hrs. "
                "Use embed_chunks() checkpointing — it resumes after any crash. "
                "Run 3-4 parallel ingest processes with disjoint CSV shards to "
                "maximise API throughput within your OpenAI rate-limit tier."
            )

    # ── Pinecone upsert ──────────────────────────────────────────────────────
    g = stage_map.get("G")
    if g and "optimal_batch_size" in g.sub_metrics:
        obs  = g.sub_metrics["optimal_batch_size"]
        sweep = g.sub_metrics.get("sweep", {})
        if sweep and len(sweep) > 1:
            opts.append(
                f"[G] Set Pinecone upsert batch_size={obs} "
                f"({g.records_per_sec:,} rec/s). "
                "The Pinecone serverless endpoint loves 100-300 vector batches; "
                "below 50 incurs per-request overhead; above 500 risks timeouts."
            )

    if not opts:
        opts.append("No significant bottlenecks detected at the configured dataset size.")

    return opts


# ─────────────────────────────────────────────────────────────────────────────
# Platform info
# ─────────────────────────────────────────────────────────────────────────────

def _platform_info() -> dict:
    import platform, sys   # noqa: PLC0415, E401
    info: dict = {
        "python":   sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if _PSUTIL:
        vm = _psutil.virtual_memory()
        info["ram_total_GB"] = round(vm.total / 1_073_741_824, 1)
        info["ram_avail_GB"] = round(vm.available / 1_073_741_824, 1)
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

_SEV_COLOUR = {"CRITICAL": "\033[91m", "HIGH": "\033[93m",
               "MEDIUM": "\033[94m", "LOW": "\033[92m"}
_RESET       = "\033[0m"
_IS_TTY      = sys.stdout.isatty()


def _colour(txt: str, sev: str) -> str:
    if not _IS_TTY:
        return txt
    return f"{_SEV_COLOUR.get(sev, '')}{txt}{_RESET}"


def _bar(value: float, max_value: float, width: int = 20) -> str:
    if max_value <= 0:
        return "─" * width
    filled = min(width, round(width * value / max_value))
    return "█" * filled + "░" * (width - filled)


def print_report(report: StressReport) -> None:
    W = 90
    print()
    print("═" * W)
    print(f"  STRESS TEST REPORT  —  {report.timestamp}")
    print(f"  Dataset: {report.dataset_size:,} records  |  "
          f"Total elapsed: {report.total_elapsed_s:.1f}s")
    pinfo = report.platform_info
    print(f"  Platform: {pinfo.get('platform', '?')}  |  "
          f"Python {pinfo.get('python', '?')}  |  "
          f"CPUs: {pinfo.get('cpu_count', '?')}  |  "
          f"RAM: {pinfo.get('ram_total_GB', '?')} GB")
    print("═" * W)

    # ── Stage table ──────────────────────────────────────────────────────────
    max_rps = max((s.records_per_sec for s in report.stages), default=1)
    fmt = "  {:<2}  {:<38}  {:>8}  {:>10}  {:>9}  {:>8}  {}"
    print(fmt.format("ID", "Stage", "n_records", "rec/s", "elapsed_s",
                     "heap_MB", "bar"))
    print("  " + "─" * (W - 2))
    for s in report.stages:
        bar = _bar(s.records_per_sec, max_rps, 18)
        print(fmt.format(
            s.stage_id,
            s.stage_name[:38],
            f"{s.n_records:,}",
            f"{s.records_per_sec:,}",
            f"{s.elapsed_s:.3f}s",
            f"{s.peak_heap_mb:.1f}",
            bar,
        ))
        # Print sub-metric notes
        for note in s.notes:
            print(f"       ↳ {note}")

    # ── Stage D sweep table ──────────────────────────────────────────────────
    d = next((s for s in report.stages if s.stage_id == "D"), None)
    if d and len(d.sub_metrics.get("sweep", {})) > 1:
        print()
        print("  Embedding batch-size sweep:")
        print(f"  {'batch_size':>12}  {'rec/s':>10}  {'elapsed_s':>10}  {'heap_MB':>9}")
        print("  " + "─" * 50)
        for bs, v in sorted(d.sub_metrics["sweep"].items()):
            marker = " ◀ optimal" if bs == d.sub_metrics.get("optimal_batch_size") else ""
            print(f"  {bs:>12}  {v['rec_per_sec']:>10,}  {v['elapsed_s']:>9.3f}s  "
                  f"{v.get('heap_MB', 0):>8.1f}{marker}")

    # ── Stage G sweep table ──────────────────────────────────────────────────
    g = next((s for s in report.stages if s.stage_id == "G"), None)
    if g and len(g.sub_metrics.get("sweep", {})) > 1:
        print()
        print("  Pinecone upsert batch-size sweep:")
        print(f"  {'batch_size':>12}  {'rec/s':>10}  {'elapsed_s':>10}")
        print("  " + "─" * 40)
        for bs, v in sorted(g.sub_metrics["sweep"].items()):
            marker = " ◀ optimal" if bs == g.sub_metrics.get("optimal_batch_size") else ""
            print(f"  {bs:>12}  {v['rec_per_sec']:>10,}  {v['elapsed_s']:>9.3f}s{marker}")

    # ── Full e2e sub-stage breakdown ─────────────────────────────────────────
    i = next((s for s in report.stages if s.stage_id == "I"), None)
    if i and i.sub_metrics:
        s = i.sub_metrics
        print()
        print("  End-to-end sub-stage breakdown:")
        print(f"    Chunking   : {s.get('chunk_s',0):.3f}s  ({s.get('chunk_pct',0)}%)")
        print(f"    Embed+Upsert: {s.get('embed_upsert_s',0):.3f}s  ({s.get('embed_pct',0)}%)")
        print(f"    Chunks produced: {s.get('total_chunks',0):,}  "
              f"Embeds: {s.get('total_embeds',0):,}  Upserts: {s.get('total_upserts',0):,}")

    # ── BM25 query latency ───────────────────────────────────────────────────
    f = next((s for s in report.stages if s.stage_id == "F"), None)
    if f and f.sub_metrics:
        s = f.sub_metrics
        print()
        print("  BM25 query latency:")
        print(f"    p50={s.get('query_p50_ms',0):.1f} ms  "
              f"p99={s.get('query_p99_ms',0):.1f} ms  "
              f"avg={s.get('query_avg_ms',0):.1f} ms")
        print(f"    Pickle size: {s.get('pickle_size_MB',0):.1f} MB  "
              f"load: {s.get('pickle_load_s',0):.2f}s")

    # ── Bottlenecks ──────────────────────────────────────────────────────────
    if report.bottlenecks:
        print()
        print("─" * W)
        print("  BOTTLENECKS")
        print("─" * W)
        for b in report.bottlenecks:
            sev_label = _colour(f"[{b['severity']}]", b["severity"])
            print(f"  {sev_label}  [{b['stage_id']}] {b['stage_name']}"
                  f"  {b['records_per_sec']:,} rec/s  heap={b['peak_heap_mb']:.1f} MB")
    else:
        print()
        print("  No significant bottlenecks detected.")

    # ── Optimizations ────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  OPTIMIZATION SUGGESTIONS")
    print("─" * W)
    for i_opt, opt in enumerate(report.optimizations, 1):
        # Word-wrap at 86 chars
        words = opt.split()
        line  = f"  {i_opt:>2}. "
        for w in words:
            if len(line) + len(w) + 1 > 88:
                print(line)
                line = "      " + w + " "
            else:
                line += w + " "
        print(line)

    print()
    print("═" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

_STAGE_MAP = {
    "A": ("Corpus generation",         False),
    "B": ("Urdu normalisation",         False),
    "C": ("Chunking",                   False),
    "D": ("Embedding sweep",            True),   # requires dry-run
    "E": ("SQLite checkpoint I/O",      True),
    "F": ("BM25 build + query",         False),
    "G": ("Pinecone upsert sweep",      True),
    "H": ("Hybrid retrieval",           True),
    "I": ("Full end-to-end ingest",     True),
}

_DEFAULT_STAGES = "ABCDEFGHI"


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Stress test the RAG pipeline (no API calls).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--size", "-n", type=int, default=100_000, metavar="N",
                        help="Number of synthetic records (default: 100 000).")
    parser.add_argument("--stages", default=_DEFAULT_STAGES, metavar="IDS",
                        help=f"Stage IDs to run, e.g. 'ABCF' (default: {_DEFAULT_STAGES}). "
                             "A=generate B=normalise C=chunk D=embed E=sqlite "
                             "F=bm25 G=pinecone H=retrieval I=e2e")
    parser.add_argument("--batch-sweep", action="store_true",
                        help="Run full batch-size sweep for stages D and G.")
    parser.add_argument("--report", metavar="FILE",
                        help="Write JSON report to FILE.")
    parser.add_argument("--profile", action="store_true",
                        help="Run cProfile over the full test and print top-20 hotspots.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging.")
    parser.add_argument(
        "--embed-cache-pkl",
        type=Path,
        default=Path(".stress_stage_i_embeddings.pkl"),
        metavar="PATH",
        help="Reusable embedding cache for stage I (default: .stress_stage_i_embeddings.pkl).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for lib in ("httpx", "httpcore", "urllib3", "pinecone", "rank_bm25"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    stages_to_run = [s.upper() for s in args.stages if s.upper() in _STAGE_MAP]
    if not stages_to_run:
        logger.error("No valid stage IDs in --stages %r.  Valid: %s",
                     args.stages, list(_STAGE_MAP))
        sys.exit(1)

    pr: cProfile.Profile | None = None
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    from src.dry_run import DryRunContext   # noqa: PLC0415

    t_global = time.perf_counter()

    with DryRunContext():
        completed_stages: list[StageMetric] = []

        # ── A: always first if included ───────────────────────────────────
        corpus: list[dict] = []

        if "A" in stages_to_run:
            m = stage_A(args.size, args.verbose)
            completed_stages.append(m)
            corpus = generate_corpus(args.size)   # re-generate for use below
        else:
            logger.info("Skipping stage A — generating minimal corpus for downstream stages…")
            corpus = generate_corpus(min(args.size, 1_000))

        # We need a full-size corpus for stages B/C/F/I
        # If A was skipped but those stages are requested, generate quietly
        if any(s in stages_to_run for s in "BCFI") and len(corpus) < args.size and "A" not in stages_to_run:
            logger.info("Generating %s records for downstream stages…", f"{args.size:,}")
            corpus = generate_corpus(args.size)

        # ── B ─────────────────────────────────────────────────────────────
        if "B" in stages_to_run:
            completed_stages.append(stage_B(corpus, args.verbose))

        # ── C ─────────────────────────────────────────────────────────────
        if "C" in stages_to_run:
            completed_stages.append(stage_C(corpus, args.verbose))

        # ── D ─────────────────────────────────────────────────────────────
        if "D" in stages_to_run:
            completed_stages.append(
                stage_D(corpus, _DEFAULT_BATCH_SIZES, args.verbose, args.batch_sweep)
            )

        # ── E ─────────────────────────────────────────────────────────────
        if "E" in stages_to_run:
            # Benchmark with 5 k records (deterministic, fast)
            e_n = min(args.size, 5_000)
            completed_stages.append(stage_E(e_n, args.verbose))

        # ── F ─────────────────────────────────────────────────────────────
        if "F" in stages_to_run:
            completed_stages.append(stage_F(corpus, args.verbose))

        # ── G ─────────────────────────────────────────────────────────────
        if "G" in stages_to_run:
            completed_stages.append(
                stage_G(args.size, _DEFAULT_PINECONE_BATCH_SIZES, args.verbose, args.batch_sweep)
            )

        # ── H ─────────────────────────────────────────────────────────────
        if "H" in stages_to_run:
            completed_stages.append(stage_H(corpus, args.verbose))

        # ── I ─────────────────────────────────────────────────────────────
        if "I" in stages_to_run:
            completed_stages.append(
                stage_I_with_cache(corpus, args.verbose, args.embed_cache_pkl)
            )

    total_elapsed = time.perf_counter() - t_global

    if pr:
        pr.disable()
        print()
        print("─" * 80)
        print("  cProfile — top 20 functions by cumulative time")
        print("─" * 80)
        stream = io.StringIO()
        ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
        ps.print_stats(20)
        print(stream.getvalue())

    bottlenecks = analyse_bottlenecks(completed_stages)
    opts        = build_optimizations(completed_stages, bottlenecks)

    from datetime import datetime, timezone   # noqa: PLC0415
    report = StressReport(
        timestamp       = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        dataset_size    = args.size,
        total_elapsed_s = round(total_elapsed, 2),
        stages          = completed_stages,
        bottlenecks     = bottlenecks,
        optimizations   = opts,
        platform_info   = _platform_info(),
    )

    print_report(report)

    if args.report:
        out = Path(args.report)
        out.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  JSON report written → {out}")
        print()


if __name__ == "__main__":
    _cli()
