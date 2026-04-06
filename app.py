"""
app.py — Flask web frontend for the Islamic Fatawa RAG System.

Usage
-----
    # Dry-run (no API keys needed — uses mock backend):
    python app.py --dry-run

    # Production (requires .env with OPENAI_API_KEY + PINECONE_API_KEY):
    python app.py

    # Custom port:
    python app.py --port 8080

    # With guardrails always on:
    python app.py --guardrails
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── Force UTF-8 I/O on Windows ───────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONUTF8", "1")

# ── Arg parse (before any src import) ────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Islamic Fatawa RAG — Web UI")
    p.add_argument("--dry-run",    action="store_true", help="Use mock backend (no API calls)")
    p.add_argument("--guardrails", action="store_true", default=True,
                   help="Always apply safety guardrails (default: on)")
    p.add_argument("--no-guardrails", dest="guardrails", action="store_false",
                   help="Disable guardrails")
    p.add_argument("--port",  type=int, default=5000, help="Server port [5000]")
    p.add_argument("--host",  type=str, default="127.0.0.1", help="Bind address [127.0.0.1]")
    p.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve [5]")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"], help="Log verbosity")
    return p.parse_args()


_args = _parse()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, _args.log_level),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
for _lib in ("httpx", "httpcore", "openai", "pinecone", "urllib3", "werkzeug"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger("app")

# ── Optional-dep stubs for dry-run ────────────────────────────────────────────
if _args.dry_run:
    from unittest.mock import MagicMock
    for _pkg, _subs in [
        ("pinecone", ["pinecone.data"]),
        ("rank_bm25", []),
        ("tqdm", ["tqdm.auto"]),
    ]:
        if importlib.util.find_spec(_pkg) is None:
            sys.modules.setdefault(_pkg, MagicMock())
            for _s in _subs:
                sys.modules.setdefault(_s, MagicMock())

# ── Load .env ────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Src imports ───────────────────────────────────────────────────────────────
from src.config import get_settings

if _args.dry_run:
    from src.dry_run import DryRunContext
    _dry_ctx = DryRunContext()
    _dry_ctx.__enter__()
    log.info("Dry-run mode active — no external API calls.")

# Import pipeline after dry-run patches are in place
from src.pipeline.guardrails import GuardrailConfig, guarded_query
from src.pipeline.rag import query as rag_query
from src.pipeline.output_validator import validate as ov_validate
from src.preprocessing.urdu_normalizer import normalize_urdu

# ── Flask ─────────────────────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

CATEGORIES = [
    "ALL", "NAMAZ", "WUDU", "ZAKAT", "FAST", "HAJJ",
    "NIKKAH", "DIVORCE", "INHERITANCE", "FOOD", "JIHAD",
    "TAUHEED", "FORGIVING", "ADHAN", "OTHER",
]

_CAT_ICONS = {
    "ALL": "🕌", "NAMAZ": "🕋", "WUDU": "💧", "ZAKAT": "💰", "FAST": "🌙",
    "HAJJ": "🕌", "NIKKAH": "💍", "DIVORCE": "📜", "INHERITANCE": "⚖️",
    "FOOD": "🍽️", "JIHAD": "🌿", "TAUHEED": "☪️", "FORGIVING": "🤝",
    "ADHAN": "📢", "OTHER": "📖",
}

_CAT_UR = {
    "ALL": "تمام", "NAMAZ": "نماز", "WUDU": "وضو", "ZAKAT": "زکوٰۃ",
    "FAST": "روزہ", "HAJJ": "حج", "NIKKAH": "نکاح", "DIVORCE": "طلاق",
    "INHERITANCE": "وراثت", "FOOD": "کھانا", "JIHAD": "جہاد",
    "TAUHEED": "توحید", "FORGIVING": "معافی", "ADHAN": "اذان", "OTHER": "دیگر",
}

@app.template_filter('cat_icon')
def cat_icon(cat): return _CAT_ICONS.get(cat, "📖")

@app.template_filter('cat_display')
def cat_display(cat): return _CAT_UR.get(cat, cat)

# Expose filters as globals so they work inside Jinja2 calls in the template
app.jinja_env.globals['cat_icon']    = cat_icon
app.jinja_env.globals['cat_display'] = cat_display

_guard_cfg = GuardrailConfig(
    min_context_score=0.03,
    min_top_score=0.03,
    min_overlap_ratio=0.03,
    min_urdu_ratio=0.15,
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        categories=CATEGORIES,
        dry_run=_args.dry_run,
        guardrails_on=_args.guardrails,
    )


@app.route("/api/query", methods=["POST"])
def api_query():
    data     = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    category = (data.get("category") or "").strip().upper()
    top_k    = int(data.get("top_k") or _args.top_k)
    use_grd  = bool(data.get("guardrails", _args.guardrails))
    validate  = bool(data.get("validate", True))

    if not question:
        return jsonify({"error": "سوال خالی ہے — Question is empty."}), 400

    cat_filter = category if category and category != "ALL" else None

    t0 = time.perf_counter()
    try:
        if use_grd:
            gr = guarded_query(
                question,
                config=_guard_cfg,
                top_k=top_k,
                category=cat_filter,
            )
            answer       = gr.answer
            sources      = gr.sources
            blocked      = not gr.passed
            guard_hits   = gr.guardrail_hits
            num_chunks   = gr.num_chunks
        else:
            result       = rag_query(question, top_k=top_k)
            answer       = result["answer"]
            sources      = result.get("sources", [])
            blocked      = False
            guard_hits   = []
            num_chunks   = result.get("num_chunks", len(sources))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Output validation
        val_report = None
        if validate and not blocked:
            try:
                val_report = ov_validate(answer, sources)
            except Exception:
                pass

        # Normalise sources for JSON
        clean_sources = []
        for s in sources[:5]:
            meta = s.get("metadata", s) if isinstance(s, dict) else {}
            clean_sources.append({
                "category":    meta.get("category", "—"),
                "source_file": meta.get("source_file", meta.get("source", "—")),
                "question":    meta.get("question", ""),
                "fatwa_no":    meta.get("fatwa_no", ""),
                "score":       round(s.get("score", 0.0), 4) if isinstance(s, dict) else 0.0,
            })

        return jsonify({
            "answer":       answer,
            "sources":      clean_sources,
            "blocked":      blocked,
            "guard_hits":   guard_hits,
            "num_chunks":   num_chunks,
            "elapsed_ms":   elapsed_ms,
            "validation":   _fmt_validation(val_report),
            "dry_run":      _args.dry_run,
        })

    except Exception as exc:
        log.exception("Query failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


def _fmt_validation(report) -> dict | None:
    if not report:
        return None
    scores = report.get("scores", {})
    issues = [i.get("code", "") for i in report.get("issues", []) if i.get("severity") != "ok"]
    return {
        "valid":      report.get("valid", True),
        "grounding":  round(scores.get("groundedness", 0) * 100, 1),
        "urdu":       round(scores.get("urdu_ratio", 0) * 100, 1),
        "halluc":     round(scores.get("hallucination_risk", 0) * 100, 1),
        "issues":     issues,
    }


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if _warmup_ready else "warming_up",
        "dry_run": _args.dry_run,
        "ready": _warmup_ready,
    })


# ── Warm-up: pre-load expensive objects in a background thread ───────────────

import threading

_warmup_ready = False


def _warmup():
    """Pre-load BM25 corpus, Pinecone index handle, and OpenAI clients
    so the first query doesn't pay the cold-start penalty."""
    global _warmup_ready
    t0 = time.perf_counter()
    log.info("Warming up caches (background)…")

    if _args.dry_run:
        log.info("  (dry-run — skipping BM25 / Pinecone warm-up)")
    else:
        # BM25 corpus (loaded from pickle or built once)
        try:
            from src.retrieval.hybrid_retriever import _get_bm25_corpus
            _get_bm25_corpus()
            log.info("  ✓ BM25 corpus cached")
        except Exception as exc:
            log.warning("  ✗ BM25 warm-up skipped: %s", exc)

        # Pinecone index handle
        try:
            from src.indexing.pinecone_store import init_index
            init_index()
            log.info("  ✓ Pinecone index handle cached")
        except Exception as exc:
            log.warning("  ✗ Pinecone warm-up skipped: %s", exc)

    # OpenAI clients (embedding + chat)
    try:
        from src.embedding.embedder import _get_client
        _get_client()
        log.info("  ✓ OpenAI embedding client cached")
    except Exception as exc:
        log.warning("  ✗ OpenAI embedding warm-up skipped: %s", exc)

    try:
        from src.pipeline.rag import _get_chat_client
        _get_chat_client()
        log.info("  ✓ OpenAI chat client cached")
    except Exception as exc:
        log.warning("  ✗ OpenAI chat warm-up skipped: %s", exc)

    _warmup_ready = True
    log.info("Warm-up done in %.1fs", time.perf_counter() - t0)


# Only start warm-up in the actual server process, not the reloader
if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not _args.debug:
    threading.Thread(target=_warmup, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Islamic Fatawa RAG Web UI on http://%s:%d", _args.host, _args.port)
    app.run(host=_args.host, port=_args.port, debug=_args.debug)
