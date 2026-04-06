import requests, json

r = requests.post(
    "http://127.0.0.1:5000/api/query",
    json={"question": "کیا بیوی کو طلاق دینے کے بعد رجوع کیا جا سکتا ہے؟", "top_k": 5, "guardrails": False},
)
d = r.json()
print("=" * 60)
print(f"  Status:   {r.status_code}")
print(f"  Elapsed:  {d.get('elapsed_ms')}ms")
print(f"  Chunks:   {d.get('num_chunks')}")
print(f"  Blocked:  {d.get('blocked')}")
print(f"  DryRun:   {d.get('dry_run')}")
print("=" * 60)
print()
print("ANSWER:")
print("-" * 60)
print(d.get("answer", ""))
print()
print("SOURCES:")
print("-" * 60)
for i, s in enumerate(d.get("sources", []), 1):
    print(f"  [{i}] {s['category']} | score={s['score']} | {s['source_file']}")
    if s.get("question"):
        print(f"      Q: {s['question'][:120]}")
print()
v = d.get("validation")
if v:
    print("VALIDATION:")
    print(f"  Valid={v['valid']}  Grounding={v['grounding']}%  Urdu={v['urdu']}%  Halluc={v['halluc']}%")
    if v.get("issues"):
        print(f"  Issues: {v['issues']}")
