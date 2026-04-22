"""Direct BM25-only retrieval test against the Islam360 cache.

Bypasses ALL higher layers — no LLM rewrite, no dense retrieval, no
rerank, no fusion, no synthesis.  Prints the top-5 raw BM25 hits per
query so we can see what the sparse index actually returns.
"""
import io
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.WARNING)

# Force UTF-8 output on Windows.
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.retrieval.bm25_index import BM25Corpus  # noqa: E402

corpus = BM25Corpus.load(Path(".bm25_islam360_cache.pkl"))
print(f"Loaded: {len(corpus)} docs")
print()

queries = [
    ("Raw roman",   "namaz ka treeqa"),
    ("Raw Urdu",    "نماز کا طریقہ"),
    ("Full Urdu",   "نماز پڑھنے کا صحیح طریقہ کیا ہے"),
    ("Test #1 was", "موزوں پر مسح کا حکم کیا ہے"),
]

for label, q in queries:
    print(f"=== {label}: {q!r} ===")
    hits = corpus.search(q, top_k=5)
    if not hits:
        print("  (no hits)")
    for i, h in enumerate(hits, 1):
        qtext = h["metadata"].get("question", "")[:110]
        score = h["score"]
        print(f"  {i}. score={score:.2f}  Q: {qtext}")
    print()
