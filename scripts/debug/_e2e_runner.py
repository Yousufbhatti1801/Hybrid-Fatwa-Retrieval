"""E2E test runner — executed by the evaluation harness."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# ── Force UTF-8 stdout/stderr on Windows (cp1252 cannot encode Urdu) ────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Stub missing optional packages before any src.* import ──────────────────
for _pkg, _subs in [
    ("pinecone", ["pinecone.data"]),
    ("rank_bm25", []),
    ("tqdm", ["tqdm.auto"]),
]:
    if importlib.util.find_spec(_pkg) is None:
        sys.modules.setdefault(_pkg, MagicMock())
        for _s in _subs:
            sys.modules.setdefault(_s, MagicMock())

from src.dry_run import DryRunContext  # noqa: E402
from src.pipeline.rag import query  # noqa: E402
from src.pipeline.guardrails import GuardrailConfig, guarded_query  # noqa: E402
from src.pipeline.output_validator import validate  # noqa: E402
from src.preprocessing.urdu_normalizer import normalize_urdu  # noqa: E402
from src.preprocessing.chunker import preprocess_record  # noqa: E402
from src.analysis.schema_analyzer import scan_and_analyse  # noqa: E402
from src.analysis.schema_mapper import infer_all  # noqa: E402

# ── Test queries ─────────────────────────────────────────────────────────────
QUERIES = [
    ("نماز_نیت",   "نماز میں نیت کا کیا حکم ہے؟",          "NAMAZ"),
    ("نماز_قصر",  "سفر میں نماز قصر کا طریقہ کیا ہے؟",    "NAMAZ"),
    ("وضو",        "وضو کن چیزوں سے ٹوٹتا ہے؟",             "WUDU"),
    ("تیمم",       "تیمم کب جائز ہے؟",                      "WUDU"),
    ("زکوٰۃ_نصاب", "زکوٰۃ کا نصاب کیا ہے؟",                "ZAKAT"),
    ("زکوٰۃ_مصارف","زکوٰۃ کے مصارف کیا ہیں؟",              "ZAKAT"),
    ("روزہ_فرض",   "رمضان کا روزہ کس پر فرض ہے؟",          "FAST"),
    ("روزہ_موانع", "روزے کی حالت میں کیا چیزیں روزہ توڑتی ہیں؟", "FAST"),
    ("off_topic",  "hello tell me lottery numbers please",  None),
    ("خالی_سوال",  "   ",                                   None),
]

report: dict = {
    "pipeline_status": "ok",
    "issues_found": [],
    "recommendations": [],
    "sections": {},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Module import connectivity
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION A — Module Import Connectivity")
print("=" * 70)

import_checks = {
    "config.settings":           "src.config.settings",
    "ingestion.loader":          "src.ingestion.loader",
    "ingestion.scanner":         "src.ingestion.scanner",
    "ingestion.dynamic_loader":  "src.ingestion.dynamic_loader",
    "preprocessing.chunker":     "src.preprocessing.chunker",
    "preprocessing.normalizer":  "src.preprocessing.urdu_normalizer",
    "embedding.embedder":        "src.embedding.embedder",
    "embedding.pipeline":        "src.embedding.pipeline",
    "indexing.pinecone_store":   "src.indexing.pinecone_store",
    "indexing.sparse":           "src.indexing.sparse",
    "retrieval.hybrid_retriever":"src.retrieval.hybrid_retriever",
    "retrieval.bm25_index":      "src.retrieval.bm25_index",
    "retrieval.eval":            "src.retrieval.eval",
    "pipeline.rag":              "src.pipeline.rag",
    "pipeline.prompt_builder":   "src.pipeline.prompt_builder",
    "pipeline.context_trimmer":  "src.pipeline.context_trimmer",
    "pipeline.guardrails":       "src.pipeline.guardrails",
    "pipeline.output_validator": "src.pipeline.output_validator",
    "analysis.schema_analyzer":  "src.analysis.schema_analyzer",
    "analysis.schema_mapper":    "src.analysis.schema_mapper",
    "dry_run":                   "src.dry_run",
}

import_results = {}
for label, mod in import_checks.items():
    try:
        importlib.import_module(mod)
        import_results[label] = "ok"
        print(f"  [OK]   {label}")
    except Exception as e:
        import_results[label] = f"FAIL: {e}"
        print(f"  [FAIL] {label}: {e}")
        report["issues_found"].append({
            "section": "A_imports",
            "severity": "error",
            "module": label,
            "message": str(e),
        })
        report["pipeline_status"] = "fail"

report["sections"]["A_imports"] = import_results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — Data ingestion: schema analysis + mapping
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION B — Data Ingestion (Schema Analysis + Mapping)")
print("=" * 70)

ingestion_result: dict = {}
try:
    t0 = time.perf_counter()
    analysis = scan_and_analyse("data")
    files = analysis.get("files", [])
    elapsed = time.perf_counter() - t0
    print(f"  scan_and_analyse: {len(files)} files in {elapsed:.2f}s")

    if files:
        mappings = infer_all(files)
        q_mapped = sum(1 for m in mappings if m.get("mapping", {}).get("question"))
        a_mapped = sum(1 for m in mappings if m.get("mapping", {}).get("answer"))
        print(f"  infer_all:        {len(mappings)} mappings  (q={q_mapped}  a={a_mapped})")
        ingestion_result = {
            "files_found": len(files),
            "mappings": len(mappings),
            "question_mapped": q_mapped,
            "answer_mapped": a_mapped,
        }
        unmapped = [
            m.get("file", "?")
            for m in mappings
            if not m.get("mapping", {}).get("question")
        ]
        if unmapped:
            msg = f"{len(unmapped)} files have no 'question' mapping: {unmapped[:5]}"
            print(f"  [WARN] {msg}")
            report["issues_found"].append({
                "section": "B_ingestion",
                "severity": "warning",
                "message": msg,
            })
    else:
        print("  [WARN] No CSV files found under data/")
        report["issues_found"].append({
            "section": "B_ingestion",
            "severity": "warning",
            "message": "No CSV files found under data/",
        })
        ingestion_result = {"files_found": 0}
except Exception as e:
    print(f"  [FAIL] {e}")
    ingestion_result = {"error": str(e)}
    report["issues_found"].append({
        "section": "B_ingestion",
        "severity": "error",
        "message": str(e),
    })
    report["pipeline_status"] = "fail"

report["sections"]["B_ingestion"] = ingestion_result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — Preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION C — Preprocessing (Urdu normalizer + Chunker)")
print("=" * 70)

TEST_RECORDS = [
    {
        "question": "نماز میں نیت کا کیا حکم ہے اور نیت کی کیا اہمیت ہے؟",
        "answer": (
            "نماز کی نیت فرض ہے۔ نیت کا محل دل ہے نہ زبان۔ "
            "فقہاءِ احناف کے نزدیک نماز شروع کرتے وقت دل سے ارادہ ضروری ہے۔"
        ),
        "category": "NAMAZ",
        "source_file": "salat_output.csv",
    },
    {
        "question": "وضو کن چیزوں سے ٹوٹتا ہے اور کن حالات میں تجدید وضو لازم ہے؟",
        "answer": (
            "وضو پیشاب، پاخانہ، ریح، خون بہنے، گہری نیند، اور بے ہوشی سے ٹوٹتا ہے۔ "
            "ان میں سے کوئی بھی ہو تو نئے سرے سے وضو کرنا ضروری ہے۔"
        ),
        "category": "WUDU",
        "source_file": "wudu_output.csv",
    },
]

preprocess_result: dict = {}
try:
    from src.preprocessing.chunker import preprocess_records  # noqa: E402

    chunks = list(preprocess_records(iter(TEST_RECORDS)))
    print(f"  preprocess_records: {len(chunks)} chunks from {len(TEST_RECORDS)} records")

    for i, ch in enumerate(chunks[:3]):
        text = ch["text"] if isinstance(ch, dict) else getattr(ch, "text", str(ch))
        print(f"  chunk[{i}] len={len(text)}  preview={text[:60].replace(chr(10),' ')!r}")

    # Test normalizer
    raw = "نمازکی نیّت کا طریقہ  کیا ہے؟"
    normalised = normalize_urdu(raw)
    norm_ok = len(normalised) > 0 and normalised != raw
    print(f"  normalize_urdu: in={len(raw)}  out={len(normalised)}  changed={norm_ok}")

    preprocess_result = {
        "input_records": len(TEST_RECORDS),
        "output_chunks": len(chunks),
        "normalizer_ok": norm_ok,
    }
    if len(chunks) == 0:
        report["issues_found"].append({
            "section": "C_preprocess",
            "severity": "error",
            "message": "preprocess_records produced 0 chunks for valid test records",
        })
        report["pipeline_status"] = "fail"

except Exception as e:
    print(f"  [FAIL] {e}")
    preprocess_result = {"error": str(e)}
    report["issues_found"].append({
        "section": "C_preprocess",
        "severity": "error",
        "message": str(e),
    })
    report["pipeline_status"] = "fail"

report["sections"]["C_preprocess"] = preprocess_result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — Embedding + mock Pinecone indexing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION D — Embedding + Pinecone Indexing (dry-run)")
print("=" * 70)

embedding_result: dict = {}
with DryRunContext():
    try:
        from src.embedding.embedder import embed_single, embed_texts  # noqa: E402
        from src.indexing.pinecone_store import init_index  # noqa: E402
        from src.indexing.sparse import build_sparse_vector  # noqa: E402

        # Embedding smoke test
        vec1 = embed_single("نماز کی نیت")
        assert len(vec1) == 3072, f"embed_single dim={len(vec1)}, expected 3072"
        print(f"  embed_single: dim={len(vec1)}  norm={sum(v*v for v in vec1)**.5:.4f}")

        vecs = embed_texts(["نماز کی نیت", "وضو کے احکام", "زکوٰۃ کا نصاب"])
        assert len(vecs) == 3
        print(f"  embed_texts:  {len(vecs)} vectors  mean_dim={sum(len(v) for v in vecs)//len(vecs)}")

        # Pinecone index smoke test
        idx = init_index()
        upsert_vectors = [
            {"id": f"test_{i}", "values": embed_single(f"test text {i}"),
             "metadata": {"category": "NAMAZ", "text": f"test {i}"}}
            for i in range(5)
        ]
        resp = idx.upsert(upsert_vectors)
        print(f"  pinecone.upsert: {resp.upserted_count} vectors")
        stats = idx.describe_index_stats()
        print(f"  pinecone.stats:  total_vectors={stats.total_vector_count}  dim={stats.dimension}")

        # Sparse vector smoke test
        sv = build_sparse_vector("نماز کی نیت فرض ہے")
        print(f"  sparse_vector:   indices={len(sv.get('indices', []))}  values={len(sv.get('values', []))}")

        embedding_result = {
            "embed_dim": len(vec1),
            "embed_texts_count": len(vecs),
            "upserted_count": resp.upserted_count,
            "sparse_indices": len(sv.get("indices", [])),
        }
    except Exception as e:
        print(f"  [FAIL] {e}")
        embedding_result = {"error": str(e)}
        report["issues_found"].append({
            "section": "D_embedding",
            "severity": "error",
            "message": str(e),
        })
        report["pipeline_status"] = "fail"

report["sections"]["D_embedding"] = embedding_result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION E — Retrieval quality (all categories)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION E — RAG Query + Retrieval Quality")
print("=" * 70)

query_results = []
with DryRunContext():
    for label, q, expected_cat in QUERIES:
        t0 = time.perf_counter()
        try:
            r = query(q, top_k=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            ans = r["answer"]
            sources = r["sources"]
            sentinel_hit = "مناسب جواب" in ans
            # sources items are flat dicts (no nested 'metadata' key)
            top_cat = sources[0].get("category", "?") if sources else "?"
            cat_ok = (expected_cat is None) or (top_cat == expected_cat)
            ans_words = len(ans.split())

            qr = {
                "label": label,
                "ok": True,
                "elapsed_ms": round(elapsed_ms, 1),
                "answer_words": ans_words,
                "sources_returned": len(sources),
                "sentinel_triggered": sentinel_hit,
                "top_category": top_cat,
                "category_match": cat_ok,
                "preview": ans[:90].replace("\n", " "),
            }
            status = "OK  " if cat_ok else "WARN"
            flag   = "sentinel" if sentinel_hit else ""
            print(f"  [{status}] {label:<18} cat={top_cat:<8} words={ans_words:<4} {flag}")

            if not cat_ok and expected_cat:
                # In dry-run mode the mock corpus has 4 NAMAZ docs vs 2 per
                # other category, so occasional category mismatches are expected
                # from the keyword ranker — record as info, not an error.
                report["issues_found"].append({
                    "section": "E_retrieval",
                    "severity": "info",
                    "message": (
                        f"Query '{label}': expected category {expected_cat}, "
                        f"got {top_cat} (dry-run mock corpus bias — "
                        "verify against real index)"
                    ),
                })
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            qr = {"label": label, "ok": False, "error": str(e)}
            report["issues_found"].append({
                "section": "E_retrieval",
                "severity": "error",
                "message": f"Query '{label}' raised: {e}",
            })
            report["pipeline_status"] = "fail"

        query_results.append(qr)

report["sections"]["E_retrieval"] = {
    "total_queries": len(query_results),
    "ok":   sum(1 for r in query_results if r.get("ok")),
    "fail": sum(1 for r in query_results if not r.get("ok")),
    "category_mismatches": sum(
        1 for r in query_results
        if r.get("ok") and not r.get("category_match") and QUERIES[query_results.index(r)][2]
    ),
    "results": query_results,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION F — Guardrails on all queries
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION F — Guardrails")
print("=" * 70)

guardrail_results = []
cfg = GuardrailConfig()
with DryRunContext():
    for label, q, expected_cat in QUERIES:
        try:
            gr = guarded_query(q, config=cfg, top_k=5)
            hits_str = ", ".join(gr.guardrail_hits) if gr.guardrail_hits else "none"
            print(
                f"  {'PASS' if gr.passed else 'BLOCK'} {label:<18}"
                f" hits=[{hits_str}]  words={len(gr.answer.split())}"
            )
            guardrail_results.append({
                "label": label,
                "passed": gr.passed,
                "guardrail_hits": gr.guardrail_hits,
                "answer_words": len(gr.answer.split()),
            })
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            guardrail_results.append({"label": label, "passed": False, "error": str(e)})
            report["issues_found"].append({
                "section": "F_guardrails",
                "severity": "error",
                "message": f"guarded_query '{label}' raised: {e}",
            })
            report["pipeline_status"] = "fail"

blocks = [r for r in guardrail_results if not r.get("passed")]
report["sections"]["F_guardrails"] = {
    "total": len(guardrail_results),
    "passed": sum(1 for r in guardrail_results if r.get("passed")),
    "blocked": len(blocks),
    "blocked_labels": [r["label"] for r in blocks],
    "results": guardrail_results,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION G — Output validator on every query response
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION G — Output Validator")
print("=" * 70)

validator_results = []
with DryRunContext():
    for label, q, _ in QUERIES:
        try:
            r = query(q, top_k=5)
            vr = validate(r["answer"], r["sources"])
            scores = vr["scores"]
            status = "VALID  " if vr["valid"] else "INVALID"
            issue_codes = [i["code"] for i in vr["issues"]]
            print(
                f"  [{status}] {label:<18}"
                f" ground={scores['groundedness']:.2f}"
                f" cit={scores['citation_score']:.2f}"
                f" urdu={scores['urdu_ratio']:.2f}"
                f" hall={scores['hallucination_risk']:.2f}"
                + (f"  issues={issue_codes}" if issue_codes else "")
            )
            validator_results.append({
                "label": label,
                "valid": vr["valid"],
                "scores": scores,
                "issues": issue_codes,
            })
            for issue in vr["issues"]:
                if issue["severity"] == "error":
                    report["issues_found"].append({
                        "section": "G_validator",
                        "severity": "warning",
                        "message": f"Query '{label}': validator issue {issue['code']}: {issue['message']}",
                    })
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            validator_results.append({"label": label, "valid": False, "error": str(e)})
            report["issues_found"].append({
                "section": "G_validator",
                "severity": "error",
                "message": f"validate '{label}' raised: {e}",
            })
            report["pipeline_status"] = "fail"

report["sections"]["G_validator"] = {
    "total": len(validator_results),
    "valid": sum(1 for r in validator_results if r.get("valid")),
    "invalid": sum(1 for r in validator_results if not r.get("valid") and not r.get("error")),
    "errors": sum(1 for r in validator_results if r.get("error")),
    "results": validator_results,
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION H — Retrieval eval metrics (MRR, Precision@k, KW-coverage)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION H — Retrieval Eval Metrics")
print("=" * 70)

eval_result: dict = {}
with DryRunContext():
    try:
        from src.retrieval.eval import EvalQuery, run_evaluation  # noqa: E402

        eval_queries = [
            EvalQuery(
                label="نماز نیت",
                question="نماز میں نیت کا کیا حکم ہے؟",
                expected_category="NAMAZ",
                expected_keywords=["نیت", "نماز", "فرض"],
            ),
            EvalQuery(
                label="نماز قصر",
                question="سفر میں نماز قصر کا طریقہ",
                expected_category="NAMAZ",
                expected_keywords=["قصر", "سفر", "رکعت"],
            ),
            EvalQuery(
                label="وضو",
                question="وضو کن چیزوں سے ٹوٹتا ہے",
                expected_category="WUDU",
                expected_keywords=["وضو", "پیشاب", "نیند"],
            ),
            EvalQuery(
                label="زکوٰۃ نصاب",
                question="زکوٰۃ کا نصاب کیا ہے",
                expected_category="ZAKAT",
                expected_keywords=["نصاب", "زکوٰۃ", "تولہ"],
            ),
            EvalQuery(
                label="روزہ",
                question="رمضان کا روزہ کس پر فرض ہے",
                expected_category="FAST",
                expected_keywords=["روزہ", "رمضان", "فرض"],
            ),
        ]

        # Pass a dummy bm25_corpus so run_evaluation skips BM25Corpus.load_or_build(),
        # which would otherwise try to pickle the MagicMock'd rank_bm25.BM25Okapi.
        _dummy_corpus = MagicMock()
        eval_report = run_evaluation(eval_queries, top_k=5, bm25_corpus=_dummy_corpus)
        # EvalReport exposes mean_mrr / mean_category_precision / mean_keyword_coverage
        mrr  = eval_report.mean_mrr
        catp = eval_report.mean_category_precision
        kwc  = eval_report.mean_keyword_coverage
        n_q  = len(eval_report.query_evals)
        print(f"  MRR:           {mrr:.3f}")
        print(f"  Category-P@k:  {catp:.3f}")
        print(f"  KW-coverage:   {kwc:.3f}")
        print(f"  queries run:   {n_q}")
        eval_result = {
            "total_queries":        n_q,
            "mrr":                  round(mrr,  3),
            "category_precision":   round(catp, 3),
            "keyword_coverage":     round(kwc,  3),
        }

        if mrr < 0.5:
            report["issues_found"].append({
                "section": "H_eval",
                "severity": "warning",
                "message": f"MRR={mrr:.3f} is below acceptable threshold (0.50) — check retrieval weights",
            })
        if kwc < 0.5:
            report["issues_found"].append({
                "section": "H_eval",
                "severity": "warning",
                "message": f"KW-coverage={kwc:.3f} is below threshold (0.50) — consider BM25 re-index",
            })
    except Exception as e:
        print(f"  [FAIL] {e}")
        eval_result = {"error": str(e)}
        report["issues_found"].append({
            "section": "H_eval",
            "severity": "error",
            "message": str(e),
        })
        report["pipeline_status"] = "fail"

report["sections"]["H_eval"] = eval_result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION I — Orchestrator smoke test
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION I — Orchestrator")
print("=" * 70)

orch_result: dict = {}
with DryRunContext():
    try:
        import orchestrator as orch  # noqa: E402

        results = orch.orchestrate(
            data_root=Path("data"),
            work_dir=Path("_e2e_orch_work"),
            requested_stages={1, 2, 3, 4, 5, 6},
            dry_run=True,
            force=True,
        )
        statuses = {r.stage: r.status for r in results}
        print(f"  stages returned: {sorted(statuses.keys())}")
        for stage, status in sorted(statuses.items()):
            print(f"  Stage {stage}: {status}")
        orch_result = {"stages": statuses}
        failed_stages = [s for s, st in statuses.items() if st == "failed"]
        if failed_stages:
            report["issues_found"].append({
                "section": "I_orchestrator",
                "severity": "error",
                "message": f"Orchestrator stages failed: {failed_stages}",
            })
            report["pipeline_status"] = "fail"
    except Exception as e:
        print(f"  [FAIL] {e}")
        orch_result = {"error": str(e)}
        report["issues_found"].append({
            "section": "I_orchestrator",
            "severity": "error",
            "message": str(e),
        })
        report["pipeline_status"] = "fail"

report["sections"]["I_orchestrator"] = orch_result
infos    = [i for i in report["issues_found"] if i["severity"] == "info"]


# ─────────────────────────────────────────────────────────────────────────────
# Generate recommendations
# ─────────────────────────────────────────────────────────────────────────────
errors   = [i for i in report["issues_found"] if i["severity"] == "error"]
warnings = [i for i in report["issues_found"] if i["severity"] == "warning"]

# Always-present best-practice recommendations
report["recommendations"] = [
    "Populate real API keys (OPENAI_API_KEY, PINECONE_API_KEY) in .env before production use.",
    "Run `python orchestrator.py --stages 5,6` after adding new CSVs to refresh the index.",
    "Schedule nightly re-index: `python run_pipeline.py --mode index --summary-json daily.json`.",
    "Monitor retrieval quality periodically with `python -m src.retrieval.eval --report eval.json`.",
    "Use `--guardrails` for all user-facing query endpoints.",
    "Review output validator reports for MISSING_CITATION or WEAK_GROUNDEDNESS flags.",
]

if report["sections"].get("B_ingestion", {}).get("files_found", 0) == 0:
    report["recommendations"].insert(0,
        "Place fatawa CSV files under data/<DataSource>/<CATEGORY>/ and re-run ingestion.")

if errors:
    report["pipeline_status"] = "fail"
    report["recommendations"].insert(0,
        f"Fix {len(errors)} critical error(s) before deploying (see issues_found).")

# ───────────────────────────────────────────
print(f"  info            : {len(infos)}")
# Print + persist final report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL REPORT")
print("=" * 70)
print(f"  pipeline_status : {report['pipeline_status'].upper()}")
print(f"  errors          : {len(errors)}")
print(f"  warnings        : {len(warnings)}")
if report["issues_found"]:
    print("\n  Issues:")
    for iss in report["issues_found"]:
        sev = iss["severity"].upper()
        print(f"    [{sev}] [{iss['section']}] {iss['message'][:100]}")
print("\n  Recommendations:")
for rec in report["recommendations"][:6]:
    print(f"    • {rec[:95]}")

# Strip large sub-arrays from the JSON we keep (keep only summaries)
slim = {
    "pipeline_status": report["pipeline_status"],
    "issues_found":    report["issues_found"],
    "recommendations": report["recommendations"],
    "summary": {
        "A_imports":      {"all_ok": all(v == "ok" for v in import_results.values())},
        "B_ingestion":    {k: v for k, v in report["sections"].get("B_ingestion", {}).items()
                          if k != "error"},
        "C_preprocess":   report["sections"].get("C_preprocess", {}),
        "D_embedding":    report["sections"].get("D_embedding", {}),
        "E_retrieval":    {k: v for k, v in report["sections"].get("E_retrieval", {}).items()
                          if k != "results"},
        "F_guardrails":   {k: v for k, v in report["sections"].get("F_guardrails", {}).items()
                          if k != "results"},
        "G_validator":    {k: v for k, v in report["sections"].get("G_validator", {}).items()
                          if k != "results"},
        "H_eval":         report["sections"].get("H_eval", {}),
        "I_orchestrator": report["sections"].get("I_orchestrator", {}),
    },
}

with open("_e2e_report.json", "w", encoding="utf-8") as fh:
    json.dump(slim, fh, ensure_ascii=False, indent=2)
print("\n  Full report saved to _e2e_report.json")
print("=" * 70)

sys.exit(0 if report["pipeline_status"] == "ok" else 1)
