"""Smoke test for Islam360Retriever.retrieve_fast.

Runs the end-to-end fast path WITHOUT calling the LLM synthesis step
(monkey-patches it to a stub) so we can see the BM25 ranking quickly
and deterministically — no OpenAI quota burn, no waiting on chat.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("src.islam360.retrieve").setLevel(logging.INFO)

import src.islam360.retrieve as r  # noqa: E402

# Stub out the synthesis call so the test is fully offline / instant.
r._synthesize_answer = lambda q, c: f"[stub answer over {len(c)} chunks]"
# Disable the LLM Roman→Urdu translator too — let the deterministic map
# handle conversion. This makes the test 100% offline.
import src.config as _cfg  # noqa: E402
_cfg.get_settings.cache_clear()
_cfg.get_settings().islam360_fast_canonicalise_roman = False  # type: ignore[attr-defined]

queries = [
    "namaz ka treeqa",
    "riba ka hukum",
    "talaq ke baray me bata do",
    "نماز کا طریقہ",
    "موزوں پر مسح کا حکم",
    "Is taking a home loan with interest haram?",  # English (will rely on map)
]

retr = r.Islam360Retriever()

for q in queries:
    print("=" * 78)
    print(f"USER QUERY: {q!r}")
    print("=" * 78)
    out = retr.retrieve_fast(q, top_k=5)
    log = out.get("log", {})
    print(f"  canonical: {log.get('canonical_query', '')!r}")
    print(f"  no_match : {out.get('no_match')}")
    print(f"  top hits :")
    for i, c in enumerate(out.get("sources", []), 1):
        meta = c.get("metadata") or {}
        qtxt = (meta.get("question") or "")[:90]
        print(f"    {i}. score={c['score']:.2f} (full={c['full_score']:.2f}, q={c['q_score']:.2f})")
        print(f"       Q: {qtxt}")
    print()
