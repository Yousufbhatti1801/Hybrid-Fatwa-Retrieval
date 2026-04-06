"""Debug script: identify BM25 build bottleneck."""
import time, json, sqlite3, sys

sys.stdout.reconfigure(encoding="utf-8")

print("Step 1: Loading from checkpoint DB (streaming)...")
t0 = time.perf_counter()
conn = sqlite3.connect("embed_checkpoint.db")
conn.execute("PRAGMA journal_mode=WAL")
cursor = conn.execute("SELECT id, metadata FROM embeddings")
docs = []
for row_id, meta_json in cursor:
    meta = json.loads(meta_json) if meta_json else {}
    q = meta.get("question", "")
    a = meta.get("answer", "")
    docs.append({"id": row_id, "text": f"سوال: {q} جواب: {a}", "question": q, "answer": a})
conn.close()
print(f"  Loaded+parsed {len(docs)} docs in {time.perf_counter()-t0:.1f}s")

print("Step 2: Fast tokenizing (str.split)...")
t2 = time.perf_counter()
tokenized = [[w for w in d["text"].split() if len(w) > 1] for d in docs]
print(f"  Tokenized {len(tokenized)} docs in {time.perf_counter()-t2:.1f}s")

print("Step 3: Fitting BM25Okapi...")
t3 = time.perf_counter()
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(tokenized)
print(f"  BM25 fitted in {time.perf_counter()-t3:.1f}s")

print(f"\nTotal: {time.perf_counter()-t0:.1f}s")
