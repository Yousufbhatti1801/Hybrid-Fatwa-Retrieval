"""Main pipeline orchestrator — connects every component of the RAG system.

Stage graph
-----------
  1 (CSV Analysis) → 2 (Schema Inference) → 3 (Data Loading)
  → 4 (Preprocessing) → 5 (Embedding) → 6 (Pinecone Indexing)

Key features
------------
* Intermediate outputs persisted to ``--work-dir`` so stages 1 and 2 can
  be skipped automatically on re-runs (analysis.json, mappings.json).
* Pipeline state tracked in ``pipeline_state.json`` — completed stages are
  auto-skipped; use ``--force`` to re-run them.
* Per-stage ``try/except`` — a failing stage is marked FAILED and logged
  with a full traceback; downstream stages that can still run will proceed.
* Stage 5 (embedding) resumes via its SQLite checkpoint automatically.
* Stage 6 (Pinecone) is idempotent via ``skip_existing=True``.
* Every stage is importable as a standalone function.

Usage
-----
Run everything (auto-resumes on re-run)::

    python orchestrator.py

Force-rerun all stages::

    python orchestrator.py --force

Run only stages 5 and 6 (uses saved analysis + mappings from work-dir)::

    python orchestrator.py --stages 5,6

Dry-run (no external API calls)::

    python orchestrator.py --dry-run

Write a JSON summary after completion::

    python orchestrator.py --summary-json report.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)-38s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: int
    name: str
    status: str = "not_run"   # not_run | completed | skipped | failed
    elapsed_s: float = 0.0
    summary: dict = field(default_factory=dict)
    error: str = ""

    def mark_skipped(self, reason: str = "") -> None:
        self.status = "skipped"
        self.summary["skip_reason"] = reason

    def mark_failed(self, exc: Exception) -> None:
        self.status = "failed"
        self.error = traceback.format_exc()
        self.summary["error"] = str(exc)
        logger.error(
            "[Stage %d] FAILED — %s\n%s", self.stage, exc, self.error
        )

    def mark_completed(self, t0: float, summary: dict) -> None:
        self.status = "completed"
        self.elapsed_s = round(time.perf_counter() - t0, 2)
        self.summary.update(summary)

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "skipped")


# ---------------------------------------------------------------------------
# Pipeline state (persisted as JSON between runs)
# ---------------------------------------------------------------------------

class PipelineState:
    """Reads and writes ``pipeline_state.json`` inside the work directory."""

    _FILENAME = "pipeline_state.json"

    def __init__(self, work_dir: Path) -> None:
        self._path = work_dir / self._FILENAME
        self._data: dict = self._load()

    # ── I/O ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Could not read state file %s — starting fresh.", self._path)
        return {"stages": {}}

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Stage queries ─────────────────────────────────────────────────────

    def is_completed(self, stage: int) -> bool:
        return self._data.get("stages", {}).get(str(stage), {}).get("status") == "completed"

    def record(self, result: StageResult) -> None:
        self._data.setdefault("stages", {})[str(result.stage)] = {
            "status":    result.status,
            "elapsed_s": result.elapsed_s,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary":   {k: v for k, v in result.summary.items() if k != "chunks_counter"},
        }
        self.save()


# ---------------------------------------------------------------------------
# Intermediate output helpers
# ---------------------------------------------------------------------------

_ANALYSIS_FILE = "analysis.json"
_MAPPINGS_FILE = "mappings.json"


def _save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug("Saved intermediate output → %s", path)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("Loaded intermediate output ← %s", path)
        return data
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Stage 1 — CSV Analysis
# ---------------------------------------------------------------------------

def run_stage_1(
    data_root: Path,
    work_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> tuple[StageResult, dict]:
    """Scan all CSVs under *data_root* and produce schema reports.

    Output is saved to ``work_dir/analysis.json``.  On subsequent runs the
    saved file is reloaded instead of re-scanning (unless *force* is True).

    Returns
    -------
    (StageResult, analysis_dict)
    """
    result = StageResult(1, "CSV Analysis")
    out_path = work_dir / _ANALYSIS_FILE

    # ── Auto-skip ─────────────────────────────────────────────────────────
    cached = _load_json(out_path)
    if not force and cached is not None and (state is None or state.is_completed(1)):
        logger.info("[Stage 1] Auto-skipped — loaded from %s", out_path)
        result.mark_skipped(f"loaded from {out_path}")
        return result, cached

    if dry_run:
        logger.info("[Stage 1] DRY-RUN — would scan '%s'", data_root)
        result.mark_skipped("dry-run")
        return result, {}

    t0 = time.perf_counter()
    logger.info("[Stage 1] Scanning CSVs under '%s' …", data_root)
    try:
        from src.analysis.schema_analyzer import scan_and_analyse  # noqa: PLC0415

        analysis = scan_and_analyse(str(data_root))
        n_files = len(analysis.get("files", []))
        _save_json(out_path, analysis)
        result.mark_completed(t0, {"files_analysed": n_files, "output": str(out_path)})
        logger.info("[Stage 1] Done — %d files analysed in %.1fs", n_files, result.elapsed_s)
    except Exception as exc:
        result.mark_failed(exc)
        analysis = {}

    if state:
        state.record(result)
    return result, analysis


# ---------------------------------------------------------------------------
# Stage 2 — Schema Inference
# ---------------------------------------------------------------------------

def run_stage_2(
    analysis: dict,
    work_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> tuple[StageResult, list[dict]]:
    """Infer role→column mappings from the stage-1 analysis.

    Output is saved to ``work_dir/mappings.json``.  On subsequent runs the
    saved file is used if stage 1 was not re-run (unless *force* is True).

    Returns
    -------
    (StageResult, mappings_list)
    """
    result = StageResult(2, "Schema Inference")
    out_path = work_dir / _MAPPINGS_FILE

    # ── Auto-skip ─────────────────────────────────────────────────────────
    cached = _load_json(out_path)
    if not force and cached is not None and (state is None or state.is_completed(2)):
        logger.info("[Stage 2] Auto-skipped — loaded from %s", out_path)
        result.mark_skipped(f"loaded from {out_path}")
        return result, cached

    if dry_run:
        logger.info("[Stage 2] DRY-RUN — would call infer_all()")
        result.mark_skipped("dry-run")
        return result, []

    file_reports = analysis.get("files", [])
    if not file_reports:
        logger.warning("[Stage 2] No file reports from Stage 1 — skipping.")
        result.mark_skipped("no file reports from Stage 1")
        return result, []

    t0 = time.perf_counter()
    logger.info("[Stage 2] Inferring schema mappings for %d files …", len(file_reports))
    try:
        from src.analysis.schema_mapper import infer_all  # noqa: PLC0415

        raw_mappings = infer_all(file_reports)
        # infer_all may return dataclass instances — serialise them
        mappings: list[dict] = [
            asdict(m) if hasattr(m, "__dataclass_fields__") else m
            for m in raw_mappings
        ]
        _save_json(out_path, mappings)
        result.mark_completed(
            t0,
            {
                "mappings_inferred": len(mappings),
                "files_with_question": sum(
                    1 for m in mappings if m.get("mapping", {}).get("question")
                ),
                "output": str(out_path),
            },
        )
        logger.info(
            "[Stage 2] Done — %d mappings inferred in %.1fs", len(mappings), result.elapsed_s
        )
    except Exception as exc:
        result.mark_failed(exc)
        mappings = []

    if state:
        state.record(result)
    return result, mappings


# ---------------------------------------------------------------------------
# Stage 3 — Data Loading (returns a generator — not persisted)
# ---------------------------------------------------------------------------

def run_stage_3(
    data_root: Path,
    mappings: list[dict],
    *,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> tuple[StageResult, Generator]:
    """Set up the streaming corpus generator.

    This stage is *always* executed (generator setup costs nothing).  The
    actual I/O happens lazily during Stage 4/5 iteration.

    Returns
    -------
    (StageResult, corpus_generator)
    """
    result = StageResult(3, "Data Loading")

    if dry_run:
        logger.info("[Stage 3] DRY-RUN — would build stream_corpus('%s')", data_root)
        result.mark_skipped("dry-run")
        return result, iter([])

    t0 = time.perf_counter()
    logger.info("[Stage 3] Initialising corpus stream from '%s' …", data_root)
    try:
        from src.ingestion.dynamic_loader import stream_corpus  # noqa: PLC0415

        corpus_gen = stream_corpus(str(data_root), precomputed_mappings=mappings or None)
        result.mark_completed(
            t0, {"stream": "ready", "mappings_provided": len(mappings)}
        )
        logger.info(
            "[Stage 3] Stream ready in %.2fs (I/O deferred to Stage 4/5).", result.elapsed_s
        )
    except Exception as exc:
        result.mark_failed(exc)
        corpus_gen = iter([])

    if state:
        state.record(result)
    return result, corpus_gen


# ---------------------------------------------------------------------------
# Stage 4 — Preprocessing (wraps Stage 3 generator)
# ---------------------------------------------------------------------------

def run_stage_4(
    corpus_gen: Generator,
    *,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> tuple[StageResult, Generator]:
    """Wrap the corpus generator with normalisation + chunking.

    Like Stage 3, this is lazy — no data is processed until Stage 5 consumes
    the returned generator.

    Returns
    -------
    (StageResult, chunk_generator)
    """
    result = StageResult(4, "Preprocessing")

    if dry_run:
        logger.info("[Stage 4] DRY-RUN — would wrap stream with preprocess_records()")
        result.mark_skipped("dry-run")
        return result, iter([])

    t0 = time.perf_counter()
    logger.info("[Stage 4] Wrapping corpus stream with preprocess_records() …")
    try:
        from src.preprocessing.chunker import preprocess_records  # noqa: PLC0415

        chunks_seen = [0]  # mutable counter for the logging closure

        def _counted(gen: Generator) -> Generator:
            for chunk in preprocess_records(gen):
                chunks_seen[0] += 1
                if chunks_seen[0] % 5_000 == 0:
                    logger.info("[Stage 4] Preprocessed %d chunks so far …", chunks_seen[0])
                yield chunk

        chunk_gen = _counted(corpus_gen)
        result.mark_completed(t0, {"stream": "ready", "chunks_counter": chunks_seen})
        logger.info("[Stage 4] Preprocessing stream ready in %.2fs.", result.elapsed_s)
    except Exception as exc:
        result.mark_failed(exc)
        chunk_gen = iter([])
        chunks_seen = [0]

    if state:
        state.record(result)
    return result, chunk_gen


# ---------------------------------------------------------------------------
# Stage 5 — Embedding Generation
# ---------------------------------------------------------------------------

def run_stage_5(
    chunk_gen: Generator,
    checkpoint_path: Path,
    batch_size: int,
    *,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> tuple[StageResult, list[dict]]:
    """Embed all preprocessed chunks via OpenAI with SQLite checkpointing.

    Automatically resumes from ``checkpoint_path`` — already-embedded IDs
    are replayed instantly without API calls.

    Returns
    -------
    (StageResult, embedded_records_list)
    """
    result = StageResult(5, "Embedding Generation")

    if dry_run:
        logger.info(
            "[Stage 5] DRY-RUN — would call embed_chunks() → '%s'", checkpoint_path
        )
        result.mark_skipped("dry-run")
        return result, []

    t0 = time.perf_counter()
    logger.info(
        "[Stage 5] Embedding chunks (checkpoint='%s', batch_size=%d) …",
        checkpoint_path,
        batch_size,
    )
    try:
        from src.embedding.pipeline import embed_chunks  # noqa: PLC0415

        records: list[dict] = list(
            embed_chunks(
                chunk_gen,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                show_progress=True,
            )
        )
        result.mark_completed(
            t0,
            {
                "total_embedded": len(records),
                "checkpoint": str(checkpoint_path),
            },
        )
        logger.info(
            "[Stage 5] Done — %d vectors ready in %.1fs.", len(records), result.elapsed_s
        )
    except Exception as exc:
        result.mark_failed(exc)
        records = []

    if state:
        state.record(result)
    return result, records


# ---------------------------------------------------------------------------
# Stage 6 — Pinecone Indexing
# ---------------------------------------------------------------------------

def run_stage_6(
    checkpoint_path: Path,
    batch_size: int,
    *,
    dry_run: bool = False,
    state: PipelineState | None = None,
) -> StageResult:
    """Upload all embedded vectors from the checkpoint into Pinecone.

    Uses ``skip_existing=True`` so re-running is safe and idempotent.

    Returns
    -------
    StageResult
    """
    result = StageResult(6, "Pinecone Indexing")

    if dry_run:
        logger.info(
            "[Stage 6] DRY-RUN — would call upsert_records() from '%s'", checkpoint_path
        )
        result.mark_skipped("dry-run")
        return result

    if not checkpoint_path.exists():
        msg = f"Checkpoint '{checkpoint_path}' not found — run Stage 5 first."
        logger.error("[Stage 6] %s", msg)
        result.mark_skipped(msg)
        return result

    t0 = time.perf_counter()
    logger.info("[Stage 6] Uploading vectors from '%s' to Pinecone …", checkpoint_path)
    try:
        from src.embedding.pipeline import load_checkpoint  # noqa: PLC0415
        from src.indexing.pinecone_store import upsert_records  # noqa: PLC0415

        upsert_summary = upsert_records(
            load_checkpoint(checkpoint_path),
            batch_size=batch_size,
            skip_existing=True,
            show_progress=True,
        )
        result.mark_completed(t0, upsert_summary)
        logger.info(
            "[Stage 6] Done — upserted=%d skipped=%d in %.1fs.",
            upsert_summary.get("upserted", 0),
            upsert_summary.get("skipped", 0),
            result.elapsed_s,
        )
    except Exception as exc:
        result.mark_failed(exc)

    if state:
        state.record(result)
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrate(
    data_root: Path = Path("data"),
    work_dir: Path = Path(".pipeline_cache"),
    batch_size: int = 100,
    requested_stages: set[int] | None = None,
    force: bool = False,
    fail_fast: bool = False,
    dry_run: bool = False,
) -> list[StageResult]:
    """Run the full ingest pipeline.

    Parameters
    ----------
    data_root:
        Root directory containing the fatawa CSV source folders.
    work_dir:
        Directory for intermediate files (``analysis.json``, ``mappings.json``,
        ``pipeline_state.json``, embedding checkpoint).  Created if absent.
    batch_size:
        Used by Stage 5 (embedding) and Stage 6 (Pinecone upsert).
    requested_stages:
        If ``None``, all six stages are candidates; completed stages are
        auto-skipped.  If provided, only those stages are executed (generators
        for stages 3/4 are always set up when any of stages 4/5/6 is requested).
    force:
        Disable auto-skip — re-run all requested stages even if previously
        completed.
    fail_fast:
        Abort immediately when a stage fails instead of continuing.
    dry_run:
        Log what would run without calling OpenAI or Pinecone.

    Returns
    -------
    list[StageResult]
        One entry per stage (1–6) in execution order.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = work_dir / "embed_checkpoint.db"
    state = PipelineState(work_dir)

    all_stages = {1, 2, 3, 4, 5, 6}
    run_set = requested_stages if requested_stages is not None else all_stages

    t_pipeline = time.perf_counter()
    results: list[StageResult] = []

    logger.info("=" * 72)
    logger.info(
        "FATAWA INGEST PIPELINE  stages=%s  force=%s  dry_run=%s  fail_fast=%s",
        sorted(run_set), force, dry_run, fail_fast,
    )
    logger.info("  data_root  : %s", data_root)
    logger.info("  work_dir   : %s", work_dir)
    logger.info("  checkpoint : %s", checkpoint_path)
    logger.info("  batch_size : %d", batch_size)
    logger.info("=" * 72)

    # Shared data flowing between stages
    analysis: dict = {}
    mappings: list[dict] = []
    corpus_gen: Generator = iter([])
    chunk_gen: Generator = iter([])

    def _should_run(stage: int) -> bool:
        """True when stage is in the run set and not auto-skipped."""
        if stage not in run_set:
            return False
        if not force and state.is_completed(stage) and stage in (1, 2):
            return False          # stages 3-6 are cheap/idempotent — always run
        return True

    def _abort_if_fast(r: StageResult) -> bool:
        if fail_fast and r.status == "failed":
            logger.critical("[Orchestrator] fail_fast=True — aborting after Stage %d failure.", r.stage)
            return True
        return False

    # ── Stage 1 ────────────────────────────────────────────────────────────
    if 1 in run_set:
        r1, analysis = run_stage_1(
            data_root, work_dir, force=force, dry_run=dry_run, state=state
        )
    else:
        # Still try to load cached analysis so Stage 2 has something to work with
        cached = _load_json(work_dir / _ANALYSIS_FILE)
        r1, analysis = StageResult(1, "CSV Analysis"), cached or {}
        r1.mark_skipped("not in requested stages")
        logger.info("[Stage 1] Skipped (not in run set).")
    results.append(r1)
    if _abort_if_fast(r1):
        return results

    # ── Stage 2 ────────────────────────────────────────────────────────────
    if 2 in run_set:
        r2, mappings = run_stage_2(
            analysis, work_dir, force=force, dry_run=dry_run, state=state
        )
    else:
        cached = _load_json(work_dir / _MAPPINGS_FILE)
        r2, mappings = StageResult(2, "Schema Inference"), cached or []
        r2.mark_skipped("not in requested stages")
        logger.info("[Stage 2] Skipped (not in run set) — loaded %d cached mappings.",
                    len(mappings))
    results.append(r2)
    if _abort_if_fast(r2):
        return results

    # ── Stages 3+4 are always set up when any of 4/5/6 are requested ──────
    need_stream = bool(run_set & {3, 4, 5, 6})

    if need_stream and 3 in run_set:
        r3, corpus_gen = run_stage_3(data_root, mappings, dry_run=dry_run, state=state)
    elif need_stream:
        # Set up generator silently (no API calls) so downstream stages work
        r3 = StageResult(3, "Data Loading")
        if not dry_run:
            from src.ingestion.dynamic_loader import stream_corpus
            corpus_gen = stream_corpus(str(data_root), precomputed_mappings=mappings or None)
        r3.mark_skipped("not in requested stages (generator still initialised)")
        logger.info("[Stage 3] Skipped but stream initialised for downstream stages.")
    else:
        r3 = StageResult(3, "Data Loading")
        r3.mark_skipped("not in requested stages")
        logger.info("[Stage 3] Skipped.")
    results.append(r3)
    if _abort_if_fast(r3):
        return results

    if need_stream and 4 in run_set:
        r4, chunk_gen = run_stage_4(corpus_gen, dry_run=dry_run, state=state)
    elif need_stream:
        r4 = StageResult(4, "Preprocessing")
        if not dry_run:
            from src.preprocessing.chunker import preprocess_records
            chunk_gen = preprocess_records(corpus_gen)
        r4.mark_skipped("not in requested stages (chunker still wrapping stream)")
        logger.info("[Stage 4] Skipped but preprocessing stream wrapping for downstream stages.")
    else:
        r4 = StageResult(4, "Preprocessing")
        r4.mark_skipped("not in requested stages")
        logger.info("[Stage 4] Skipped.")
    results.append(r4)
    if _abort_if_fast(r4):
        return results

    # ── Stage 5 ────────────────────────────────────────────────────────────
    if 5 in run_set:
        r5, _embedded = run_stage_5(
            chunk_gen, checkpoint_path, batch_size, dry_run=dry_run, state=state
        )
        # Finalise Stage 4 chunk counter now that the stream is exhausted
        if "chunks_counter" in r4.summary:
            r4.summary["total_chunks"] = r4.summary["chunks_counter"][0]
            del r4.summary["chunks_counter"]
    else:
        r5 = StageResult(5, "Embedding Generation")
        r5.mark_skipped("not in requested stages")
        logger.info("[Stage 5] Skipped.")
    results.append(r5)
    if _abort_if_fast(r5):
        return results

    # ── Stage 6 ────────────────────────────────────────────────────────────
    if 6 in run_set:
        r6 = run_stage_6(checkpoint_path, batch_size, dry_run=dry_run, state=state)
    else:
        r6 = StageResult(6, "Pinecone Indexing")
        r6.mark_skipped("not in requested stages")
        logger.info("[Stage 6] Skipped.")
    results.append(r6)

    # ── Pre-warm BM25 cache so first query is fast ─────────────────────────
    if not dry_run and any(r.ok for r in results if r.stage in (5, 6)):
        try:
            logger.info("[Post-pipeline] Pre-warming BM25 cache …")
            from src.retrieval.hybrid_retriever import _get_bm25_corpus
            _get_bm25_corpus()
            logger.info("[Post-pipeline] BM25 cache ready.")
        except Exception as exc:
            logger.warning("[Post-pipeline] BM25 pre-warm failed: %s", exc)

    # ── Final summary ───────────────────────────────────────────────────────
    total_s = time.perf_counter() - t_pipeline
    _STATUS_ICON = {
        "completed": "✓",
        "skipped":   "−",
        "failed":    "✗",
        "not_run":   " ",
    }
    logger.info("=" * 72)
    logger.info("PIPELINE COMPLETE  total_time=%.1fs", total_s)
    for r in results:
        icon = _STATUS_ICON.get(r.status, "?")
        logger.info(
            "  [%s] Stage %d  %-32s  %s  (%.1fs)",
            icon, r.stage, r.name, r.status.upper(), r.elapsed_s,
        )
        if r.status == "failed":
            logger.info("        error: %s", r.summary.get("error", ""))
        elif r.status == "completed":
            for k, v in r.summary.items():
                if k not in ("chunks_counter", "error"):
                    logger.info("        %-28s  %s", k, v)
    logger.info("=" * 72)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fatawa ingest pipeline orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python orchestrator.py                        # full run / auto-resume\n"
            "  python orchestrator.py --force                # re-run everything\n"
            "  python orchestrator.py --stages 5,6           # embed + index only\n"
            "  python orchestrator.py --dry-run              # validate config\n"
            "  python orchestrator.py --stages 1,2 --force   # refresh schema cache\n"
        ),
    )
    p.add_argument("--data-root",  type=Path, default=Path("data"),
                   metavar="PATH", help="Fatawa CSV root directory  [default: data]")
    p.add_argument("--work-dir",   type=Path, default=Path(".pipeline_cache"),
                   metavar="PATH", help="Intermediate file cache     [default: .pipeline_cache]")
    p.add_argument("--batch-size", type=int,  default=100,
                   metavar="INT",  help="Vectors per API/upsert call [default: 100]")
    p.add_argument("--stages",     type=str,  default=None,
                   metavar="LIST", help="Comma-separated stages to force-run (e.g. 5,6)")
    p.add_argument("--force",       action="store_true",
                   help="Re-run requested stages even if previously completed")
    p.add_argument("--fail-fast",   action="store_true",
                   help="Abort on first stage failure instead of continuing")
    p.add_argument("--dry-run",     action="store_true",
                   help="Log what would run without calling any external APIs")
    p.add_argument("--log-level",  type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   metavar="LEVEL", help="Logging verbosity [default: INFO]")
    p.add_argument("--summary-json", type=Path, default=None,
                   metavar="PATH", help="Write a JSON run summary to this file")
    return p.parse_args(argv)


def _parse_stages(raw: str) -> set[int]:
    try:
        stages = {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        raise SystemExit(f"--stages must be comma-separated integers, got: '{raw}'")
    invalid = stages - {1, 2, 3, 4, 5, 6}
    if invalid:
        raise SystemExit(f"Invalid stage numbers: {sorted(invalid)}. Valid range: 1–6.")
    return stages


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    configure_logging(args.log_level)

    requested = _parse_stages(args.stages) if args.stages else None

    results = orchestrate(
        data_root=args.data_root,
        work_dir=args.work_dir,
        batch_size=args.batch_size,
        requested_stages=requested,
        force=args.force,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )

    if args.summary_json:
        summary = [
            {
                "stage":     r.stage,
                "name":      r.name,
                "status":    r.status,
                "elapsed_s": r.elapsed_s,
                "summary":   {k: v for k, v in r.summary.items() if k != "chunks_counter"},
                "error":     r.error,
            }
            for r in results
        ]
        args.summary_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Run summary written to '%s'", args.summary_json)

    failed = [r for r in results if r.status == "failed"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
