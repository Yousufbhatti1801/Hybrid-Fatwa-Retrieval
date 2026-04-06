"""Schema analyser for the Fatawa CSV corpus.

Scans all 4 source folders, samples each CSV (first 100 rows by default),
and produces structured per-file reports plus a cross-dataset summary that
highlights common columns, inconsistent naming, and missing fields.

Usage
-----
    python -m src.analysis.schema_analyzer          # prints JSON to stdout
    python -m src.analysis.schema_analyzer --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

# ── Urdu detection ────────────────────────────────────────────────────────────
# A string is considered Urdu / Arabic-script if ≥ 30 % of its non-space
# characters fall in the Arabic Unicode block.
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_URDU_THRESHOLD = 0.30


def _is_urdu(text: str) -> bool:
    clean = text.replace(" ", "")
    if not clean:
        return False
    ratio = len(_ARABIC_RE.findall(clean)) / len(clean)
    return ratio >= _URDU_THRESHOLD


def _detect_language(values: list[str]) -> str:
    """Return 'urdu', 'mixed', or 'latin' based on a sample of non-null values."""
    non_empty = [v for v in values if isinstance(v, str) and v.strip()][:20]
    if not non_empty:
        return "empty"
    urdu_count = sum(1 for v in non_empty if _is_urdu(v))
    ratio = urdu_count / len(non_empty)
    if ratio >= 0.7:
        return "urdu"
    if ratio >= 0.2:
        return "mixed"
    return "latin"


# ── Row-count estimation ──────────────────────────────────────────────────────

def _estimate_row_count(path: Path) -> int:
    """Fast line-count approximation (does not read full content into memory)."""
    try:
        with open(path, "rb") as fh:
            count = sum(1 for _ in fh)
        return max(0, count - 1)   # subtract header row
    except OSError:
        return -1


# ── Per-column analysis ───────────────────────────────────────────────────────

def _analyse_column(series: pd.Series) -> dict:
    total = len(series)
    null_count = series.isna().sum() + (series == "").sum()
    null_ratio = round(null_count / total, 4) if total else 1.0

    non_null = series.dropna().astype(str)
    non_null = non_null[non_null.str.strip() != ""]

    # Up to 3 unique sample values
    sample_values: list[str] = []
    for val in non_null.unique()[:3]:
        truncated = val.strip()
        if len(truncated) > 120:
            truncated = truncated[:120] + "…"
        sample_values.append(truncated)

    language = _detect_language(non_null.tolist())

    return {
        "name":         series.name,
        "dtype":        str(series.dtype),
        "null_ratio":   null_ratio,
        "sample_values": sample_values,
        "language":     language,
    }


# ── Per-file report ───────────────────────────────────────────────────────────

def analyse_file(path: Path, sample_rows: int = 100) -> dict:
    """Return a structured schema report for a single CSV file.

    Only reads the first *sample_rows* rows — never the full dataset.
    """
    parts = path.parts
    # Expected layout: …/<source>/<CATEGORY>/xxx_output.csv
    source   = parts[-3] if len(parts) >= 3 else "unknown"
    category = parts[-2] if len(parts) >= 2 else "unknown"

    report: dict[str, Any] = {
        "file_name":         path.name,
        "folder":            source,
        "category":          category,
        "path":              str(path),
        "row_count_estimate": _estimate_row_count(path),
        "columns":           [],
        "error":             None,
    }

    try:
        df = pd.read_csv(
            path,
            encoding="utf-8",
            dtype=str,
            nrows=sample_rows,
            low_memory=False,
        )
    except Exception as exc:
        report["error"] = str(exc)
        return report

    report["sampled_rows"] = len(df)
    report["columns"] = [_analyse_column(df[col]) for col in df.columns]
    return report


# ── Cross-dataset summary ─────────────────────────────────────────────────────

def _build_summary(reports: list[dict]) -> dict:
    """Compare all file reports and highlight schema consistency issues."""
    valid = [r for r in reports if r["error"] is None]
    if not valid:
        return {"error": "No valid files found."}

    # Column name sets per file
    col_sets: list[set[str]] = [
        {c["name"] for c in r["columns"]} for r in valid
    ]

    # Columns present in every file
    common_columns = sorted(set.intersection(*col_sets)) if col_sets else []

    # All unique column names across all files
    all_columns: set[str] = set().union(*col_sets)

    # Columns missing from at least one file
    inconsistent: dict[str, list[str]] = {}
    for col in sorted(all_columns):
        missing_in = [
            r["file_name"] for r, cs in zip(valid, col_sets) if col not in cs
        ]
        if missing_in:
            inconsistent[col] = missing_in

    # Per-source column name variants (catch casing / spelling differences)
    all_lower: dict[str, set[str]] = {}
    for cs in col_sets:
        for col in cs:
            key = col.strip().lower()
            all_lower.setdefault(key, set()).add(col)
    name_variants = {
        k: sorted(v) for k, v in all_lower.items() if len(v) > 1
    }

    return {
        "total_files":       len(reports),
        "valid_files":       len(valid),
        "errored_files":     len(reports) - len(valid),
        "common_columns":    common_columns,
        "all_columns":       sorted(all_columns),
        "missing_per_column": inconsistent,   # col → list of files missing it
        "column_name_variants": name_variants, # catches casing inconsistencies
    }


# ── Main scanner ─────────────────────────────────────────────────────────────

DATA_SOURCES = [
    "Banuri-ExtractedData-Output",
    "fatwaqa-ExtractedData-Output",
    "IslamQA-ExtractedData-Output",
    "urdufatwa-ExtractedData-Output",
]


def scan_and_analyse(
    data_root: Path | str = "data",
    sample_rows: int = 100,
) -> dict:
    """Scan all source folders, analyse each CSV, and return a full report.

    Parameters
    ----------
    data_root:
        Root directory that contains the four source sub-folders.
    sample_rows:
        Number of rows to sample per file (default 100).

    Returns
    -------
    ::

        {
            "files":   [<per-file report>, ...],
            "summary": {<cross-dataset summary>},
        }
    """
    root = Path(data_root)
    reports: list[dict] = []

    for source in DATA_SOURCES:
        source_dir = root / source
        if not source_dir.is_dir():
            continue
        for csv_path in sorted(source_dir.rglob("*_output.csv")):
            reports.append(analyse_file(csv_path, sample_rows=sample_rows))

    summary = _build_summary(reports)
    return {"files": reports, "summary": summary}


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse CSV schema across the fatawa corpus."
    )
    parser.add_argument(
        "--data-root", default="data",
        help="Root folder containing the 4 source directories (default: data)"
    )
    parser.add_argument(
        "--sample", type=int, default=100,
        help="Rows to sample per file (default: 100)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Write JSON report to this file instead of stdout"
    )
    args = parser.parse_args()

    result = scan_and_analyse(data_root=args.data_root, sample_rows=args.sample)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Report saved to {args.out}  ({len(result['files'])} files analysed)")
    else:
        print(output)


if __name__ == "__main__":
    _cli()
