#!/usr/bin/env python3
"""CLI: Islam360 RAG query (rewrite → hybrid → re-rank → answer).

Example::

    python run_islam360_query.py "What is the ruling on zakat on gold?"
    python run_islam360_query.py "زکوٰۃ کا نصاب کیا ہے؟"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so Urdu/Arabic text can be printed
# without UnicodeEncodeError (default cp1252 cannot encode the Arabic block).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


def main() -> None:
    p = argparse.ArgumentParser(description="Islam360 fatwa query (1536-d index)")
    p.add_argument("question", nargs="?", default="", help="Question text")
    p.add_argument("--no-llm-rewrite", action="store_true", help="Rule-based rewrite only")
    args = p.parse_args()
    q = (args.question or "").strip()
    if not q:
        q = input("Question: ").strip()
    if not q:
        print("Empty question.", file=sys.stderr)
        sys.exit(1)

    from src.islam360.retrieve import Islam360Retriever

    ret = Islam360Retriever()
    out = ret.retrieve(q, use_llm_rewrite=not args.no_llm_rewrite)

    rw = out["rewritten_query"]
    print("\n--- Rewritten query ---")
    print(rw.optimized_query)
    print("\n--- Log (JSON) ---")
    print(json.dumps(out.get("log", {}), ensure_ascii=False, indent=2))
    print("\n--- Answer ---")
    print(out["answer"])
    print("\n--- Top sources ---")
    for s in out.get("sources") or []:
        m = s.get("metadata") or {}
        print(
            f"  id={s.get('id')}  final={s.get('final_score')}  "
            f"rerank={s.get('rerank_score')}  cat={m.get('category')}"
        )
        print(f"    Q: {str(m.get('question',''))[:120]}...")


if __name__ == "__main__":
    main()
