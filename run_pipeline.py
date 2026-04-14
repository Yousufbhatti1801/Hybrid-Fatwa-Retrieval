"""Fatawa RAG — unified pipeline entry point.

Three modes selected with ``--mode``:

    full    Run the complete ingest pipeline (stages 1-6) then drop into
            the interactive query REPL.

    index   Run only the ingest / indexing pipeline (stages 1-6).
            Useful for nightly re-index jobs or initial data loading.

    query   Skip ingest entirely and open the interactive query REPL
            (or answer a single question when --question is supplied).
            Requires a populated Pinecone index.

Usage
-----
::

    # Full pipeline then REPL
    python run_pipeline.py --mode full

    # Ingest/index only (no query)
    python run_pipeline.py --mode index

    # Ingest a specific subset of stages, then exit
    python run_pipeline.py --mode index --stages 5,6

    # Interactive query REPL
    python run_pipeline.py --mode query

    # Single question — print answer and exit
    python run_pipeline.py --mode query --question "نماز قصر کے احکام"

    # Single question with guardrails enabled
    python run_pipeline.py --mode query --question "سوال" --guardrails

    # Dry-run (no external API calls — uses mock embeddings/retrieval)
    python run_pipeline.py --mode full --dry-run
    python run_pipeline.py --mode query --dry-run --question "نماز"

    # Write JSON summary of index run to a file
    python run_pipeline.py --mode index --summary-json run_summary.json

Index options
-------------
    --data-root   PATH   Root of fatawa CSV tree       [default: data]
    --checkpoint  PATH   SQLite embedding checkpoint   [default: embed_checkpoint.db]
    --batch-size  INT    Vectors per API/upsert call   [default: 100]
    --stages      LIST   Comma-separated stage numbers [default: 1,2,3,4,5,6]

Query options
-------------
    --question    TEXT   Run a single non-interactive query; skip the REPL
    --top-k       INT    Retrieve this many chunks                [default: settings]
    --category    TEXT   Filter results to a single fatawa category
    --guardrails         Wrap the query with all 5 safety guardrails
    --stream             Stream the LLM answer token-by-token
    --validate           Run the output validator and print its report

Global options
--------------
    --mode        full|index|query   Pipeline mode (required)
    --dry-run                        No external API calls; uses mock backend
    --log-level   DEBUG|INFO|WARNING [default: INFO]
    --summary-json PATH              Write index-stage JSON summary here
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import textwrap
import time
from pathlib import Path
from typing import Generator

# Default data root: sibling ``data/`` folder next to this repo.
_DATA_ROOT_DEFAULT = Path(__file__).resolve().parent.parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Keep noisy third-party loggers quiet unless the user asked for DEBUG
    if level.upper() != "DEBUG":
        for _lib in ("httpx", "httpcore", "urllib3", "openai._base_client", "pinecone"):
            logging.getLogger(_lib).setLevel(logging.WARNING)


logger = logging.getLogger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Stage result
# ─────────────────────────────────────────────────────────────────────────────

class StageResult:
    """Lightweight value object returned by each pipeline stage."""

    def __init__(self, stage: int, name: str) -> None:
        self.stage = stage
        self.name = name
        self.skipped = False
        self.elapsed_s: float = 0.0
        self.summary: dict = {}
        self.error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def __repr__(self) -> str:
        if self.skipped:
            return f"[SKIP] Stage {self.stage} — {self.name}"
        if self.error:
            return f"[FAIL] Stage {self.stage} — {self.name}  error={self.error}"
        return f"[ OK ] Stage {self.stage} — {self.name} ({self.elapsed_s:.1f}s)"


# ─────────────────────────────────────────────────────────────────────────────
# INDEX MODE — ingest pipeline stages 1–6
# ─────────────────────────────────────────────────────────────────────────────

def stage_1_csv_analysis(data_root: Path, dry_run: bool) -> tuple[StageResult, dict]:
    result = StageResult(1, "CSV Analysis")
    t0 = time.perf_counter()
    if dry_run:
        logger.info("[Stage 1] DRY-RUN — scan_and_analyse('%s')", data_root)
        result.skipped = True
        return result, {}
    try:
        logger.info("[Stage 1] Scanning CSVs under '%s' …", data_root)
        from src.analysis.schema_analyzer import scan_and_analyse  # noqa: PLC0415
        analysis = scan_and_analyse(str(data_root))
        n = len(analysis.get("files", []))
        result.elapsed_s = time.perf_counter() - t0
        result.summary = {
            "files_analysed": n,
            "common_columns": analysis.get("summary", {}).get("common_columns", []),
        }
        logger.info("[Stage 1] %d files analysed in %.1fs", n, result.elapsed_s)
        return result, analysis
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 1] Failed: %s", exc)
        return result, {}


def stage_2_schema_inference(analysis: dict, dry_run: bool) -> tuple[StageResult, list[dict]]:
    result = StageResult(2, "Schema Inference")
    t0 = time.perf_counter()
    file_reports = analysis.get("files", [])
    if dry_run:
        logger.info("[Stage 2] DRY-RUN — infer_all() on %d files", len(file_reports))
        result.skipped = True
        return result, []
    if not file_reports:
        logger.warning("[Stage 2] No file reports from Stage 1 — skipping.")
        result.skipped = True
        return result, []
    try:
        logger.info("[Stage 2] Inferring schema mappings for %d files …", len(file_reports))
        from src.analysis.schema_mapper import infer_all  # noqa: PLC0415
        mappings = infer_all(file_reports)
        result.elapsed_s = time.perf_counter() - t0
        result.summary = {
            "mappings_inferred":    len(mappings),
            "files_with_question":  sum(1 for m in mappings if m.get("mapping", {}).get("question")),
            "files_with_answer":    sum(1 for m in mappings if m.get("mapping", {}).get("answer")),
        }
        logger.info(
            "[Stage 2] %d mappings in %.1fs  (q=%d a=%d)",
            result.summary["mappings_inferred"], result.elapsed_s,
            result.summary["files_with_question"], result.summary["files_with_answer"],
        )
        return result, mappings
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 2] Failed: %s", exc)
        return result, []


def stage_3_data_normalisation(
    data_root: Path, mappings: list[dict], dry_run: bool
) -> tuple[StageResult, Generator]:
    result = StageResult(3, "Data Normalisation")
    t0 = time.perf_counter()
    if dry_run:
        logger.info("[Stage 3] DRY-RUN — stream_corpus('%s')", data_root)
        result.skipped = True
        return result, iter([])
    try:
        logger.info("[Stage 3] Initialising corpus stream from '%s' …", data_root)
        from src.ingestion.dynamic_loader import stream_corpus  # noqa: PLC0415
        corpus_gen = stream_corpus(str(data_root),
                                   precomputed_mappings=mappings or None)
        result.elapsed_s = time.perf_counter() - t0
        result.summary = {"stream": "ready", "mappings_used": len(mappings)}
        logger.info("[Stage 3] Stream ready in %.1fs (lazy — consumed in Stage 4).",
                    result.elapsed_s)
        return result, corpus_gen
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 3] Failed: %s", exc)
        return result, iter([])


def stage_4_preprocessing(corpus_gen: Generator, dry_run: bool) -> tuple[StageResult, Generator]:
    result = StageResult(4, "Preprocessing")
    t0 = time.perf_counter()
    if dry_run:
        logger.info("[Stage 4] DRY-RUN — preprocess_records()")
        result.skipped = True
        return result, iter([])
    try:
        logger.info("[Stage 4] Wrapping corpus stream with preprocess_records() …")
        from src.preprocessing.chunker import preprocess_records  # noqa: PLC0415
        chunks_seen = [0]

        def _counted():
            for chunk in preprocess_records(corpus_gen):
                chunks_seen[0] += 1
                if chunks_seen[0] % 5_000 == 0:
                    logger.info("[Stage 4] %d chunks preprocessed …", chunks_seen[0])
                yield chunk

        result.elapsed_s = time.perf_counter() - t0
        result.summary = {"stream": "ready", "_counter": chunks_seen}
        logger.info("[Stage 4] Preprocessing stream ready in %.1fs (lazy).",
                    result.elapsed_s)
        return result, _counted()
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 4] Failed: %s", exc)
        return result, iter([])


def stage_5_embedding(
    chunk_gen: Generator, checkpoint_path: Path, batch_size: int, dry_run: bool
) -> tuple[StageResult, int]:
    result = StageResult(5, "Embedding Generation")
    t0 = time.perf_counter()
    if dry_run:
        logger.info("[Stage 5] DRY-RUN — embed_chunks() → '%s'", checkpoint_path)
        result.skipped = True
        return result, 0
    try:
        logger.info("[Stage 5] Embedding (checkpoint='%s', batch=%d) …",
                    checkpoint_path, batch_size)
        from src.embedding.pipeline import embed_chunks  # noqa: PLC0415
        records = list(embed_chunks(
            chunk_gen,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            show_progress=True,
        ))
        n = len(records)
        result.elapsed_s = time.perf_counter() - t0
        result.summary = {"total_embedded": n, "checkpoint": str(checkpoint_path)}
        logger.info("[Stage 5] %d vectors embedded in %.1fs.", n, result.elapsed_s)
        return result, n
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 5] Failed: %s", exc)
        return result, 0


def stage_6_pinecone_indexing(
    checkpoint_path: Path, batch_size: int, dry_run: bool
) -> StageResult:
    result = StageResult(6, "Pinecone Indexing")
    t0 = time.perf_counter()
    if dry_run:
        logger.info("[Stage 6] DRY-RUN — upsert_records() from '%s'", checkpoint_path)
        result.skipped = True
        return result
    if not checkpoint_path.exists():
        result.error = f"Checkpoint '{checkpoint_path}' not found — run Stage 5 first."
        logger.error("[Stage 6] %s", result.error)
        return result
    try:
        logger.info("[Stage 6] Uploading vectors from '%s' to Pinecone …", checkpoint_path)
        from src.embedding.pipeline import load_checkpoint        # noqa: PLC0415
        from src.indexing.pinecone_store import upsert_records    # noqa: PLC0415
        summary = upsert_records(
            load_checkpoint(checkpoint_path),
            batch_size=batch_size,
            skip_existing=True,
            show_progress=True,
        )
        result.elapsed_s = time.perf_counter() - t0
        result.summary = summary
        logger.info(
            "[Stage 6] upserted=%d  skipped=%d  in %.1fs.",
            summary.get("upserted", 0), summary.get("skipped", 0), result.elapsed_s,
        )
        return result
    except Exception as exc:
        result.error = str(exc)
        logger.error("[Stage 6] Failed: %s", exc)
        return result


def run_index(
    data_root: Path = _DATA_ROOT_DEFAULT,
    checkpoint_path: Path = Path("embed_checkpoint.db"),
    batch_size: int = 100,
    stages: set[int] = frozenset({1, 2, 3, 4, 5, 6}),
    dry_run: bool = False,
) -> list[StageResult]:
    """Execute the requested ingest pipeline stages and return stage results."""
    t_start = time.perf_counter()

    _banner("INGEST PIPELINE", extra={
        "stages":       sorted(stages),
        "data_root":    str(data_root),
        "checkpoint":   str(checkpoint_path),
        "batch_size":   batch_size,
        "dry_run":      dry_run,
    })

    def _skip(n: int, name: str) -> StageResult:
        r = StageResult(n, name)
        r.skipped = True
        logger.info("[Stage %d] Skipped.", n)
        return r

    # Stage 1
    r1, analysis   = stage_1_csv_analysis(data_root, dry_run) if 1 in stages \
                     else (_skip(1, "CSV Analysis"), {})

    # Stage 2
    r2, mappings   = stage_2_schema_inference(analysis, dry_run) if 2 in stages \
                     else (_skip(2, "Schema Inference"), [])

    # Stage 3
    r3, corpus_gen = stage_3_data_normalisation(data_root, mappings, dry_run) if 3 in stages \
                     else (_skip(3, "Data Normalisation"), iter([]))

    # Stage 4
    r4, chunk_gen  = stage_4_preprocessing(corpus_gen, dry_run) if 4 in stages \
                     else (_skip(4, "Preprocessing"), iter([]))

    # Stage 5  (drives stages 3 + 4 via generator pull)
    r5, _n         = stage_5_embedding(chunk_gen, checkpoint_path, batch_size, dry_run) if 5 in stages \
                     else (_skip(5, "Embedding Generation"), 0)

    # Update Stage 4 counter now that the stream is exhausted
    if hasattr(r4, "summary") and "_counter" in r4.summary:
        r4.summary["total_chunks"] = r4.summary.pop("_counter")[0]

    # Stage 6
    r6             = stage_6_pinecone_indexing(checkpoint_path, batch_size, dry_run) if 6 in stages \
                     else _skip(6, "Pinecone Indexing")

    results = [r1, r2, r3, r4, r5, r6]
    _print_stage_summary(results, time.perf_counter() - t_start)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# QUERY MODE — interactive REPL or single-question
# ─────────────────────────────────────────────────────────────────────────────

_W  = 70
_SEP  = "─" * _W
_SEP2 = "═" * _W


def _print_answer(answer: str) -> None:
    _safe_print(f"\n{_SEP2}")
    _safe_print("  جواب (Answer)")
    _safe_print(_SEP2)
    for line in answer.splitlines():
        _safe_print(textwrap.fill(line, width=_W + 10) if line.strip() else "")
    _safe_print(_SEP2)


def _print_sources(sources: list[dict], n: int = 3) -> None:
    top = sources[:n]
    if not top:
        _safe_print("\n  (کوئی ماخذ دستیاب نہیں — no sources available)")
        return
    _safe_print(f"\n  ماخذ (Top {len(top)} Sources)")
    _safe_print(_SEP)
    for i, meta in enumerate(top, 1):
        question = (meta.get("question") or "").strip()
        if len(question) > 120:
            question = question[:120].rsplit(" ", 1)[0] + "…"
        _safe_print(f"  [{i}] category : {meta.get('category', '—')}")
        _safe_print(f"       source   : {meta.get('source') or meta.get('source_file', '—')}")
        if meta.get("fatwa_no"):
            _safe_print(f"       fatwa_no : {meta['fatwa_no']}")
        if question:
            _safe_print(f"       question : {question}")
        _safe_print("")


def _print_timings(timings: dict) -> None:
    _safe_print(_SEP)
    parts = []
    for key, label in [
        ("normalise_ms", "normalise"),
        ("retrieve_ms",  "retrieve"),
        ("trim_ms",      "trim"),
        ("generate_ms",  "generate"),
        ("guardrail_ms", "guardrails"),
        ("total_ms",     "total"),
    ]:
        if key in timings:
            parts.append(f"{label} {timings[key]:.0f}ms")
    _safe_print("  " + "  |  ".join(parts))
    _safe_print(_SEP)


def _print_guardrail_hits(hits: list[str], passed: bool) -> None:
    if not hits:
        return
    status = "PASSED" if passed else "BLOCKED"
    _safe_print(f"\n  Guardrails [{status}]  hits: {', '.join(hits)}")


def _safe_print(text: str) -> None:
    """Print with UTF-8 fallback on Windows CP1252/pipe terminals."""
    try:
        print(text)
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Ensure fallback stays pure ASCII so legacy consoles can always render it.
        try:
            print(text.encode("ascii", errors="replace").decode("ascii"))
        except Exception:
            pass
    except (OSError, ValueError):
        # stdout is a closed pipe or in an invalid state — silently ignore.
        pass


def run_query_once(
    question: str,
    *,
    top_k: int | None,
    category: str | None,
    guardrails: bool,
    stream: bool,
    validate: bool,
) -> int:
    """Execute a single query; return 0 on success, 1 on error."""
    _safe_print(f"\n{_SEP}")
    _safe_print(f"  سوال: {question.strip()}")
    _safe_print(_SEP)

    try:
        if guardrails:
            from src.pipeline.guardrails import GuardrailConfig, guarded_query  # noqa: PLC0415
            cfg = GuardrailConfig()
            result = guarded_query(
                question,
                config=cfg,
                top_k=top_k,
                category=category,
            )
            _print_answer(result.answer)
            _print_sources(result.sources)
            _print_guardrail_hits(result.guardrail_hits, result.passed)
            _print_timings(result.timings)
            if validate:
                _run_validation(result.answer, result.sources)
        elif stream:
            from src.pipeline.rag import query  # noqa: PLC0415
            tokens = query(question, top_k=top_k, stream=True)
            _safe_print("")
            for tok in tokens:
                print(tok, end="", flush=True)
            print()
            _safe_print(_SEP2)
        else:
            from src.pipeline.rag import query  # noqa: PLC0415
            result = query(question, top_k=top_k)
            _print_answer(result["answer"])
            _print_sources(result["sources"])
            _print_timings(result["timings"])
            if validate:
                _run_validation(result["answer"], result["sources"])
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        _safe_print(f"\n  [ERROR] {exc}")
        return 1


def _run_validation(answer: str, sources: list[dict]) -> None:
    try:
        from src.pipeline.output_validator import validate, print_report  # noqa: PLC0415
        report = validate(answer, sources)
        _safe_print(f"\n  Validation: {'VALID' if report['valid'] else 'INVALID'}")
        print_report(report)
    except Exception as exc:
        logger.warning("Output validator unavailable: %s", exc)


def run_query_repl(
    *,
    top_k: int | None,
    category: str | None,
    guardrails: bool,
    validate: bool,
) -> None:
    """Start an interactive query REPL."""
    _safe_print(_SEP2)
    _safe_print("  اسلامی فتاویٰ RAG سسٹم — Islamic Fatawa RAG System")
    if guardrails:
        _safe_print("  Guardrails: ON")
    if category:
        _safe_print(f"  Category filter: {category}")
    _safe_print("  Type 'exit' or press Ctrl-C to quit.")
    _safe_print(_SEP2)

    while True:
        try:
            try:
                raw = input("\n  Question: ").strip()
            except UnicodeEncodeError:
                raw = input("\n  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            _safe_print("\n  Goodbye.")
            break
        except (OSError, ValueError):
            # stdout/stdin in invalid state (e.g. PowerShell pipe with 2>&1).
            _safe_print("\n  Goodbye.")
            break
        if not raw or raw.lower() in {"exit", "quit", "q"}:
            _safe_print("  خدا حافظ — Goodbye.")
            break
        run_query_once(
            raw,
            top_k=top_k,
            category=category,
            guardrails=guardrails,
            stream=False,
            validate=validate,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str, extra: dict | None = None) -> None:
    _safe_print("=" * _W)
    _safe_print(f"  {title}")
    if extra:
        for k, v in extra.items():
            _safe_print(f"    {k:<18} {v}")
    _safe_print("=" * _W)


def _print_stage_summary(results: list[StageResult], total_s: float) -> None:
    _safe_print("=" * _W)
    _safe_print(f"  PIPELINE SUMMARY  |  total={total_s:.1f}s")
    for r in results:
        if r.skipped:
            status = "SKIP"
        elif r.error:
            status = f"FAIL  {r.error[:50]}"
        else:
            status = f" OK   ({r.elapsed_s:.1f}s)"
        _safe_print(f"  Stage {r.stage}  {r.name:<30}  {status}")
        if r.summary and not r.skipped and not r.error:
            for k, v in r.summary.items():
                if not k.startswith("_"):
                    _safe_print(f"           {k:<28}  {v}")
    _safe_print("=" * _W)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Fatawa RAG — unified pipeline entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode", "-m",
        required=True,
        choices=["full", "index", "query"],
        metavar="MODE",
        help="Pipeline mode: full | index | query  (required)",
    )

    # ── Index options ────────────────────────────────────────────────────────
    idx = p.add_argument_group("Index options (--mode index / full)")
    idx.add_argument("--data-root",   type=Path, default=_DATA_ROOT_DEFAULT,
                     metavar="PATH",  help="Root of fatawa CSV tree    [../data (sibling of repo)]")
    idx.add_argument("--checkpoint",  type=Path, default=Path("embed_checkpoint.db"),
                     metavar="PATH",  help="SQLite embedding checkpoint [embed_checkpoint.db]")
    idx.add_argument("--batch-size",  type=int,  default=100,
                     metavar="INT",   help="Vectors per API/upsert call [100]")
    idx.add_argument("--stages",      type=str,  default="1,2,3,4,5,6",
                     metavar="LIST",  help="Comma-separated stages      [1,2,3,4,5,6]")
    idx.add_argument("--summary-json", type=Path, default=None,
                     metavar="PATH",  help="Write JSON stage summary here")

    # ── Query options ─────────────────────────────────────────────────────────
    qry = p.add_argument_group("Query options (--mode query / full)")
    qry.add_argument("--question", "-q", type=str, default=None,
                     metavar="TEXT",  help="Single question (non-interactive)")
    qry.add_argument("--top-k",    type=int, default=None,
                     metavar="INT",   help="Chunks to retrieve (overrides settings)")
    qry.add_argument("--category", type=str, default=None,
                     metavar="CAT",   help="Filter results to one fatawa category")
    qry.add_argument("--guardrails", action="store_true",
                     help="Wrap query with all 5 safety guardrails")
    qry.add_argument("--stream",     action="store_true",
                     help="Stream LLM answer token-by-token")
    qry.add_argument("--validate",   action="store_true",
                     help="Run output validator and print its report")

    # ── Global ────────────────────────────────────────────────────────────────
    p.add_argument("--dry-run",    action="store_true",
                   help="No external API calls; use mock backend")
    p.add_argument("--log-level",  type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   metavar="LEVEL", help="Logging verbosity [INFO]")

    return p.parse_args(argv)


def _parse_stages(raw: str) -> set[int]:
    try:
        stages = {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        raise SystemExit(
            f"--stages must be comma-separated integers, got: '{raw}'"
        ) from None
    invalid = stages - {1, 2, 3, 4, 5, 6}
    if invalid:
        raise SystemExit(f"Invalid stage numbers: {sorted(invalid)}. Valid: 1–6.")
    return stages


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def _stub_missing_deps() -> None:
    """Pre-inject MagicMock stubs for optional packages not installed locally.

    Only stubs packages that are genuinely absent from the environment so
    that production runs (where the real packages are installed) are never
    affected.  Required for dry-run mode in dev environments that lack the
    full production dependency set.
    """
    import importlib.util
    from unittest.mock import MagicMock

    _stubs: dict[str, list[str]] = {
        "pinecone": ["pinecone.data", "pinecone.data.index"],
        "rank_bm25": [],
        "tqdm": ["tqdm.auto"],
    }
    for top, sub_mods in _stubs.items():
        if importlib.util.find_spec(top) is None:
            stub = MagicMock()
            sys.modules.setdefault(top, stub)
            for sub in sub_mods:
                sys.modules.setdefault(sub, MagicMock())
            logger.debug("[dry_run] sys.modules stub injected: %s", top)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # Reconfigure stdout/stderr to UTF-8 on Windows so Urdu text and
    # box-drawing characters never raise UnicodeEncodeError in pipelines.
    import io as _io
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    # Activate dry-run backend before any src.* import touches external deps
    if args.dry_run:
        _stub_missing_deps()
        from src.dry_run import DryRunContext  # noqa: PLC0415
        ctx = DryRunContext()
        ctx.__enter__()
        logger.info("Dry-run mode active — no external API calls will be made.")

    try:
        _run(args)
    finally:
        if args.dry_run:
            ctx.__exit__(None, None, None)


def _run(args: argparse.Namespace) -> None:
    mode   = args.mode
    stages = _parse_stages(args.stages)

    # ── INDEX mode ────────────────────────────────────────────────────────────
    if mode in ("full", "index"):
        results = run_index(
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            batch_size=args.batch_size,
            stages=stages,
            dry_run=args.dry_run,
        )

        # Optionally write JSON summary
        if args.summary_json:
            payload = [
                {
                    "stage":     r.stage,
                    "name":      r.name,
                    "skipped":   r.skipped,
                    "elapsed_s": round(r.elapsed_s, 2),
                    "error":     r.error,
                    "summary":   {k: v for k, v in r.summary.items()
                                  if not k.startswith("_")},
                }
                for r in results
            ]
            args.summary_json.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Stage summary written to '%s'", args.summary_json)

        failed = [r for r in results if r.error]
        if failed and mode == "index":
            sys.exit(1)

    # ── QUERY mode ────────────────────────────────────────────────────────────
    if mode in ("full", "query"):
        if args.question:
            rc = run_query_once(
                args.question,
                top_k=args.top_k,
                category=args.category,
                guardrails=args.guardrails,
                stream=args.stream,
                validate=args.validate,
            )
            sys.exit(rc)
        else:
            try:
                run_query_repl(
                    top_k=args.top_k,
                    category=args.category,
                    guardrails=args.guardrails,
                    validate=args.validate,
                )
            except Exception as _repl_exc:
                logger.debug("REPL exited with unexpected error: %s", _repl_exc)
            sys.exit(0)


if __name__ == "__main__":
    main()
