"""Pipeline debug & validation utility.

Runs a lightweight end-to-end dry-run of Stages 1–4 (no embeddings, no
Pinecone) and prints richly formatted diagnostic output so you can verify
correctness *before* spending API budget on indexing.

What it checks
--------------
1. Schema mappings   — detected column→role assignments per CSV, with
                       confidence scores and a warning when a mandatory
                       role (question / answer) is missing.
2. Normalised fatawas — prints N sample records from the dynamic loader,
                        before and after Urdu normalisation, so you can
                        spot encoding problems or empty fields.
3. Chunked output    — shows the chunks produced for the same sample records,
                       with token estimates and length flags.
4. Urdu text integrity — runs character-level checks across a larger random
                         sample to detect mojibake, ASCII-only records,
                         empty answer fields, and short/degenerate text.

Usage
-----
Show the first 3 samples from every CSV::

    python debug_pipeline.py

Show 10 samples, search only in the NAMAZ category folder::

    python debug_pipeline.py --samples 10 --category NAMAZ

Write a JSON report instead of printing to the terminal::

    python debug_pipeline.py --json-out debug_report.json

Limit integrity checks to 500 records (faster)::

    python debug_pipeline.py --integrity-sample 500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import unicodedata
from pathlib import Path
from typing import Any

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,          # suppress library noise
    format="%(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("debug")

# Rich Unicode box-drawing helpers
_SEP  = "─" * 72
_BOLD = "\033[1m"
_DIM  = "\033[2m"
_RED  = "\033[31m"
_YEL  = "\033[33m"
_GRN  = "\033[32m"
_RST  = "\033[0m"

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RST}" if _USE_COLOR else text


def _header(title: str) -> None:
    print(f"\n{_c(_BOLD, _SEP)}")
    print(_c(_BOLD, f"  {title}"))
    print(_c(_BOLD, _SEP))


def _ok(msg: str)   -> None: print(_c(_GRN, f"  ✓  {msg}"))
def _warn(msg: str) -> None: print(_c(_YEL, f"  ⚠  {msg}"))
def _err(msg: str)  -> None: print(_c(_RED, f"  ✗  {msg}"))
def _info(msg: str) -> None: print(f"     {msg}")


# ── Arabic/Urdu script detection ──────────────────────────────────────────────

_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def _urdu_ratio(text: str) -> float:
    """Fraction of non-space chars that are Arabic-script."""
    stripped = text.replace(" ", "")
    if not stripped:
        return 0.0
    return len(_ARABIC_RE.findall(stripped)) / len(stripped)


def _is_urdu(text: str, threshold: float = 0.25) -> bool:
    return _urdu_ratio(text) >= threshold


# ── Section 1: Schema mappings ────────────────────────────────────────────────

def check_schema_mappings(
    data_root: Path,
    category: str | None,
    report: dict,
) -> None:
    _header("SECTION 1 — Schema Mappings")

    from src.analysis.schema_analyzer import scan_and_analyse
    from src.analysis.schema_mapper import infer_all

    print(f"  Scanning '{data_root}' …")
    analysis = scan_and_analyse(str(data_root))
    file_reports = analysis.get("files", [])

    if not file_reports:
        _err("No CSV files found under the data root.")
        report["schema_mappings"] = {"error": "no files found"}
        return

    if category:
        file_reports = [
            f for f in file_reports
            if category.upper() in f.get("path", "").upper()
        ]
        print(f"  Filtered to category '{category}': {len(file_reports)} files")

    mappings = infer_all(file_reports)
    mapping_index: dict[str, dict] = {m["path"]: m for m in mappings}

    total = len(mappings)
    missing_q  = [m for m in mappings if not m.get("mapping", {}).get("question")]
    missing_a  = [m for m in mappings if not m.get("mapping", {}).get("answer")]

    _info(f"Files checked      : {total}")
    if missing_q:
        _warn(f"Missing 'question' role : {len(missing_q)} file(s)")
    else:
        _ok("All files have a 'question' mapping")
    if missing_a:
        _warn(f"Missing 'answer' role   : {len(missing_a)} file(s)")
    else:
        _ok("All files have an 'answer' mapping")

    print()
    for m in mappings[:20]:  # cap display to avoid walls of text
        path_label = Path(m["path"]).name
        mapping = m.get("mapping", {})
        q_col   = mapping.get("question") or _c(_RED, "—")
        a_col   = mapping.get("answer")   or _c(_RED, "—")
        cat_col = mapping.get("category") or "—"
        confs   = m.get("confidence", {})
        q_conf  = f"{confs.get('question', 0):.2f}" if confs.get("question") else "—"
        a_conf  = f"{confs.get('answer', 0):.2f}"   if confs.get("answer")   else "—"
        print(
            f"  {_c(_DIM, path_label):<45}  "
            f"Q={q_col} ({q_conf})  "
            f"A={a_col} ({a_conf})  "
            f"cat={cat_col}"
        )

    if total > 20:
        _info(f"  … and {total - 20} more (use --json-out to see all)")

    report["schema_mappings"] = {
        "total_files": total,
        "missing_question": [m["path"] for m in missing_q],
        "missing_answer":   [m["path"] for m in missing_a],
        "mappings": [
            {
                "path":       m["path"],
                "mapping":    m.get("mapping", {}),
                "confidence": m.get("confidence", {}),
            }
            for m in mappings
        ],
    }


# ── Section 2: Normalised fatawa samples ──────────────────────────────────────

def check_normalised_samples(
    data_root: Path,
    n_samples: int,
    category: str | None,
    report: dict,
) -> list[dict]:
    """Print N sample records showing raw vs normalised text."""
    _header("SECTION 2 — Normalised Fatawa Samples")

    from src.ingestion.dynamic_loader import stream_corpus
    from src.preprocessing.urdu_normalizer import normalize_urdu

    print(f"  Streaming up to {n_samples} records …")

    collected: list[dict] = []
    for rec in stream_corpus(str(data_root)):
        if category and category.upper() not in rec.get("category", "").upper():
            continue
        collected.append(rec)
        if len(collected) >= n_samples:
            break

    if not collected:
        _err("No records matched. Check --data-root and --category.")
        report["normalised_samples"] = []
        return []

    _ok(f"Collected {len(collected)} sample record(s)")
    sample_data: list[dict] = []

    for i, rec in enumerate(collected, 1):
        print(f"\n  {'─'*66}")
        print(f"  Record {i}/{len(collected)}  |  id={rec.get('id', '?')}  "
              f"|  category={rec.get('category', '?')}  "
              f"|  source={rec.get('source_file', '?')}")
        print(f"  {'─'*66}")

        raw_q     = rec.get("question", "")
        raw_a     = rec.get("answer",   "")
        norm_q    = normalize_urdu(raw_q)
        norm_a    = normalize_urdu(raw_a)

        # Question
        print(f"  {_c(_BOLD, 'QUESTION (raw):')}")
        _info(raw_q[:300] + ("…" if len(raw_q) > 300 else ""))
        print(f"  {_c(_BOLD, 'QUESTION (normalised):')}")
        _info(norm_q[:300] + ("…" if len(norm_q) > 300 else ""))

        # Answer
        print(f"  {_c(_BOLD, 'ANSWER (raw):')}")
        _info(raw_a[:300] + ("…" if len(raw_a) > 300 else ""))
        print(f"  {_c(_BOLD, 'ANSWER (normalised):')}")
        _info(norm_a[:300] + ("…" if len(norm_a) > 300 else ""))

        # Quick integrity notes
        q_ratio = _urdu_ratio(norm_q)
        a_ratio = _urdu_ratio(norm_a)
        if q_ratio < 0.15:
            _warn(f"Question Urdu ratio low: {q_ratio:.0%}")
        if a_ratio < 0.15:
            _warn(f"Answer Urdu ratio low: {a_ratio:.0%}")
        if not norm_q.strip():
            _err("Question is EMPTY after normalisation")
        if not norm_a.strip():
            _err("Answer is EMPTY after normalisation")

        sample_data.append({
            "id":           rec.get("id"),
            "category":     rec.get("category"),
            "source_file":  rec.get("source_file"),
            "question_raw": raw_q[:500],
            "answer_raw":   raw_a[:500],
            "question_norm": norm_q[:500],
            "answer_norm":   norm_a[:500],
            "q_urdu_ratio": round(q_ratio, 3),
            "a_urdu_ratio": round(a_ratio, 3),
        })

    report["normalised_samples"] = sample_data
    return collected


# ── Section 3: Chunked outputs ────────────────────────────────────────────────

def check_chunked_outputs(
    records: list[dict],
    report: dict,
) -> None:
    _header("SECTION 3 — Chunked Outputs")

    if not records:
        _warn("No records to chunk (Section 2 returned nothing).")
        report["chunked_outputs"] = []
        return

    from src.preprocessing.chunker import preprocess_record

    chunk_report: list[dict] = []

    for i, rec in enumerate(records, 1):
        chunks = preprocess_record(rec)
        n = len(chunks)

        print(f"\n  Record {i}  |  id={rec.get('id', '?')}  →  {n} chunk(s)")

        if n == 0:
            _err("  preprocess_record() produced ZERO chunks — record discarded")

        for j, c in enumerate(chunks, 1):
            meta  = c.get("metadata", {})
            tok   = meta.get("token_estimate", "?")
            flag  = meta.get("length_flag", "?")
            text  = c.get("text", "")
            preview = text[:120].replace("\n", " ")

            flag_col = (
                _c(_YEL, flag) if flag == "too_short"
                else _c(_RED, flag) if flag == "too_long"
                else flag
            )
            print(
                f"    Chunk {j}/{n}  "
                f"id={c.get('id', '?')}  "
                f"tokens≈{tok}  "
                f"flag={flag_col}"
            )
            _info(f""{preview}{"…" if len(text) > 120 else ""}"")

        chunk_report.append({
            "doc_id":       rec.get("id"),
            "num_chunks":   n,
            "chunk_ids":    [c.get("id") for c in chunks],
            "token_estimates": [
                c.get("metadata", {}).get("token_estimate") for c in chunks
            ],
            "length_flags": [
                c.get("metadata", {}).get("length_flag") for c in chunks
            ],
        })

    total_chunks = sum(r["num_chunks"] for r in chunk_report)
    _ok(f"Total chunks from {len(records)} sample record(s): {total_chunks}")
    report["chunked_outputs"] = chunk_report


# ── Section 4: Urdu text integrity ────────────────────────────────────────────

def check_urdu_integrity(
    data_root: Path,
    integrity_sample: int,
    category: str | None,
    report: dict,
) -> None:
    _header("SECTION 4 — Urdu Text Integrity")

    from src.ingestion.dynamic_loader import stream_corpus
    from src.preprocessing.urdu_normalizer import normalize_urdu

    print(f"  Sampling up to {integrity_sample} records for integrity checks …")

    stats: dict[str, Any] = {
        "total":              0,
        "empty_question":     0,
        "empty_answer":       0,
        "low_urdu_question":  0,   # Urdu ratio < 0.25
        "low_urdu_answer":    0,
        "ascii_only_question": 0,
        "ascii_only_answer":   0,
        "possible_mojibake":  0,
        "short_answer":       0,   # < 10 words
        "long_answer":        0,   # > 3000 words (potential CSV bleed)
        "q_urdu_ratios":      [],
        "a_urdu_ratios":      [],
        "answer_word_counts": [],
    }
    examples: dict[str, list[str]] = {
        "empty_question":      [],
        "empty_answer":        [],
        "low_urdu_question":   [],
        "ascii_only_question": [],
        "possible_mojibake":   [],
        "short_answer":        [],
    }
    _MAX_EXAMPLES = 3  # cap per issue type

    # Mojibake detector — common symptom: many replacement chars or Latin
    # words mixed into what should be pure Urdu
    _MOJIBAKE_RE = re.compile(r"[Ã¢â€œ\ufffd]{2,}")

    for rec in stream_corpus(str(data_root)):
        if category and category.upper() not in rec.get("category", "").upper():
            continue
        if stats["total"] >= integrity_sample:
            break

        stats["total"] += 1
        doc_id = rec.get("id", "?")

        raw_q = rec.get("question", "")
        raw_a = rec.get("answer",   "")
        norm_q = normalize_urdu(raw_q)
        norm_a = normalize_urdu(raw_a)

        qr = _urdu_ratio(norm_q)
        ar = _urdu_ratio(norm_a)
        stats["q_urdu_ratios"].append(qr)
        stats["a_urdu_ratios"].append(ar)

        wc_a = len(norm_a.split())
        stats["answer_word_counts"].append(wc_a)

        # ── individual checks ──────────────────────────────────────────────
        if not norm_q.strip():
            stats["empty_question"] += 1
            if len(examples["empty_question"]) < _MAX_EXAMPLES:
                examples["empty_question"].append(doc_id)

        if not norm_a.strip():
            stats["empty_answer"] += 1
            if len(examples["empty_answer"]) < _MAX_EXAMPLES:
                examples["empty_answer"].append(doc_id)

        if norm_q and qr < 0.25:
            stats["low_urdu_question"] += 1
            if norm_q.isascii() and len(norm_q) > 5:
                stats["ascii_only_question"] += 1
                if len(examples["ascii_only_question"]) < _MAX_EXAMPLES:
                    examples["ascii_only_question"].append(f"{doc_id}: {norm_q[:60]}")
            if len(examples["low_urdu_question"]) < _MAX_EXAMPLES:
                examples["low_urdu_question"].append(
                    f"{doc_id} (ratio={qr:.2f}): {norm_q[:60]}"
                )

        if norm_a and ar < 0.25:
            stats["low_urdu_answer"] += 1
        if norm_a.isascii() and len(norm_a) > 5:
            stats["ascii_only_answer"] += 1

        if _MOJIBAKE_RE.search(norm_q) or _MOJIBAKE_RE.search(norm_a):
            stats["possible_mojibake"] += 1
            if len(examples["possible_mojibake"]) < _MAX_EXAMPLES:
                combined = (norm_q + " " + norm_a)[:80]
                examples["possible_mojibake"].append(f"{doc_id}: {combined}")

        if 0 < wc_a < 10:
            stats["short_answer"] += 1
            if len(examples["short_answer"]) < _MAX_EXAMPLES:
                examples["short_answer"].append(f"{doc_id}: {norm_a[:60]}")
        if wc_a > 3000:
            stats["long_answer"] += 1

    n = stats["total"]
    if n == 0:
        _err("No records matched for integrity check.")
        report["urdu_integrity"] = {"error": "no records"}
        return

    # ── Aggregate stats ────────────────────────────────────────────────────
    avg_qr = sum(stats["q_urdu_ratios"]) / n
    avg_ar = sum(stats["a_urdu_ratios"]) / n
    avg_wc = sum(stats["answer_word_counts"]) / n

    def _pct(k: str) -> str:
        return f"{stats[k]}/{n} ({stats[k]/n:.1%})"

    _info(f"Records sampled          : {n}")
    print()

    # Urdu ratio
    if avg_qr >= 0.6:
        _ok(f"Avg question Urdu ratio  : {avg_qr:.1%}")
    else:
        _warn(f"Avg question Urdu ratio  : {avg_qr:.1%}  (expected ≥ 60 %)")

    if avg_ar >= 0.6:
        _ok(f"Avg answer Urdu ratio    : {avg_ar:.1%}")
    else:
        _warn(f"Avg answer Urdu ratio    : {avg_ar:.1%}  (expected ≥ 60 %)")

    _info(f"Avg answer word count    : {avg_wc:.0f} words")
    print()

    # Issue summary
    checks = [
        ("empty_question",      "Empty questions",           False),
        ("empty_answer",        "Empty answers",             False),
        ("low_urdu_question",   "Low-Urdu questions (<25%)", True),
        ("low_urdu_answer",     "Low-Urdu answers (<25%)",   True),
        ("ascii_only_question", "ASCII-only questions",      False),
        ("ascii_only_answer",   "ASCII-only answers",        False),
        ("possible_mojibake",   "Possible mojibake",         False),
        ("short_answer",        "Very short answers (<10w)", True),
        ("long_answer",         "Very long answers (>3000w)",True),
    ]

    any_issue = False
    for key, label, is_warn in checks:
        count = stats[key]
        if count == 0:
            _ok(f"{label:<40}: 0")
        else:
            any_issue = True
            msg = f"{label:<40}: {_pct(key)}"
            _warn(msg) if is_warn else _err(msg)
            for ex in examples.get(key, []):
                _info(f"    e.g. {ex}")

    if not any_issue:
        _ok("All integrity checks passed!")

    # Persist to report
    report["urdu_integrity"] = {
        "records_sampled":      n,
        "avg_q_urdu_ratio":     round(avg_qr, 3),
        "avg_a_urdu_ratio":     round(avg_ar, 3),
        "avg_answer_word_count": round(avg_wc, 1),
        "issues": {
            k: {
                "count":    stats[k],
                "pct":      round(stats[k] / n, 4),
                "examples": examples.get(k, []),
            }
            for k, _, _ in checks
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug & validate the ingest pipeline before indexing",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        metavar="PATH",
        help="Root directory of the fatawa CSV tree  [default: data]",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=3,
        metavar="INT",
        help="Records to show in Sections 2 & 3     [default: 3]",
    )
    p.add_argument(
        "--integrity-sample",
        type=int,
        default=1000,
        metavar="INT",
        help="Records to check in Section 4          [default: 1000]",
    )
    p.add_argument(
        "--category",
        type=str,
        default=None,
        metavar="STR",
        help="Filter by category folder name (e.g. NAMAZ, ZAKAT)",
    )
    p.add_argument(
        "--sections",
        type=str,
        default="1,2,3,4",
        metavar="LIST",
        help="Comma-separated sections to run        [default: 1,2,3,4]",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full JSON report to this file",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global _USE_COLOR
    args = _parse_args(argv)

    if args.no_color:
        _USE_COLOR = False

    try:
        sections = {int(s.strip()) for s in args.sections.split(",")}
    except ValueError:
        sys.exit("--sections must be comma-separated integers, e.g. 1,2,3,4")

    invalid = sections - {1, 2, 3, 4}
    if invalid:
        sys.exit(f"Invalid section numbers: {sorted(invalid)}. Valid: 1–4.")

    report: dict[str, Any] = {
        "data_root":         str(args.data_root),
        "samples":           args.samples,
        "integrity_sample":  args.integrity_sample,
        "category_filter":   args.category,
    }

    print(_c(_BOLD, "\nFATAWA PIPELINE DEBUG UTILITY"))
    print(f"  data_root  : {args.data_root}")
    print(f"  samples    : {args.samples}")
    print(f"  integrity  : {args.integrity_sample} records")
    print(f"  category   : {args.category or '(all)'}")
    print(f"  sections   : {sorted(sections)}")

    # ── Run requested sections ─────────────────────────────────────────────
    if 1 in sections:
        check_schema_mappings(args.data_root, args.category, report)

    records: list[dict] = []
    if 2 in sections:
        records = check_normalised_samples(
            args.data_root, args.samples, args.category, report
        )

    if 3 in sections:
        check_chunked_outputs(records, report)

    if 4 in sections:
        check_urdu_integrity(
            args.data_root, args.integrity_sample, args.category, report
        )

    # ── JSON output ────────────────────────────────────────────────────────
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  JSON report written to '{args.json_out}'")

    print(f"\n{_c(_BOLD, _SEP)}")
    print(_c(_BOLD, "  Debug complete."))
    print(_c(_BOLD, _SEP) + "\n")


if __name__ == "__main__":
    main()
