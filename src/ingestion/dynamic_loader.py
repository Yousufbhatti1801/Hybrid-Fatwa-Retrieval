"""Dynamic CSV loader driven by inferred schema mappings.

This module does NOT assume fixed column names.  It accepts the role
mappings produced by ``src.analysis.schema_mapper`` and applies them at
read-time to build a unified record format.

Unified output schema
---------------------
Every record yielded by this module follows this shape::

    {
        "id":          str,      # SHA-256 stable ID (16-char hex)
        "query":       str,      # raw query-like column if present, else ""
        "question":    str,      # mapped question column, or ""
        "answer":      str,      # mapped answer column, or ""
        "text":        str,      # "سوال: {question} جواب: {answer}"
        "category":    str,      # mapped category column OR path-derived folder
        "source_file": str,      # CSV filename
        "folder":      str,      # parent source directory name
        "date":        str|None, # mapped date column if present
        "reference":   str|None, # mapped reference/URL column if present
    }

Usage — streaming (memory-safe for 100k+ rows)
----------------------------------------------
    from src.ingestion.dynamic_loader import stream_corpus

    for record in stream_corpus(data_root="data"):
        print(record["question"][:80])

Usage — batch
-------------
    from src.ingestion.dynamic_loader import load_corpus_batched

    for batch in load_corpus_batched(data_root="data", batch_size=500):
        embed(batch)   # process each batch

Usage — load everything into memory (small datasets / testing)
--------------------------------------------------------------
    from src.ingestion.dynamic_loader import load_corpus
    records = load_corpus(data_root="data")
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Generator, Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Encodings tried in order; covers the vast majority of Urdu CSVs
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1256", "latin-1")

# Roles used to extract values from each row
_ROLES = ("question", "answer", "category", "date", "reference")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    """Try multiple encodings; return DataFrame or None on failure."""
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
                dtype=str,
                low_memory=False,
                on_bad_lines="skip",
            )
            return df.fillna("")
        except UnicodeDecodeError:
            continue
        except Exception:
            logger.exception("Failed to read %s", path)
            return None
    logger.error("Could not decode %s with any known encoding", path)
    return None


def _get_value(row, col: str | None) -> str:
    """Safely pull a string value from a DataFrame row."""
    if col is None:
        return ""
    try:
        val = row[col]
        return str(val).strip() if val and str(val).strip() not in ("nan", "None") else ""
    except (KeyError, TypeError):
        return ""


def _get_query_value(row) -> str:
    """Best-effort extraction of a query-like field from raw row columns.

    This is independent of role-mapping so files with both `query` and
    `question` columns can preserve both values in the unified record.
    """
    query_aliases = {
        "query", "queries", "question", "questions",
        "sawal", "sawaal", "سوال",
    }
    try:
        for col in row.index:
            if str(col).strip().lower() in query_aliases:
                val = str(row[col]).strip()
                if val and val not in ("nan", "None"):
                    return val
    except Exception:
        return ""
    return ""


def _build_record(
    row,
    mapping: dict[str, str | None],
    path_category: str,
    source_folder: str,
    file_name: str,
    row_index: int,
) -> dict | None:
    """Convert one DataFrame row into a unified record dict.

    Returns None for empty rows (both question and answer are blank).
    """
    question  = _get_value(row, mapping.get("question"))
    query     = _get_query_value(row)
    answer    = _get_value(row, mapping.get("answer"))

    # Skip rows that carry no usable text
    if not question and not answer:
        return None

    # Category: prefer mapped column value; fall back to folder-derived name
    category_col = mapping.get("category")
    category = _get_value(row, category_col) if category_col else ""
    if not category:
        category = path_category

    date      = _get_value(row, mapping.get("date"))      or None
    reference = _get_value(row, mapping.get("reference")) or None

    rec_id = _stable_id(
        source_folder,
        file_name,
        str(row_index),
        question[:40],
    )

    return {
        "id":          rec_id,
        "query":       query,
        "question":    question,
        "answer":      answer,
        "text":        f"سوال: {question} جواب: {answer}",
        "category":    category,
        "source_file": file_name,
        "folder":      source_folder,
        "date":        date,
        "reference":   reference,
    }


# ── Mapping resolution ────────────────────────────────────────────────────────

def _resolve_mapping(
    path: Path,
    precomputed: dict[str, dict] | None,
    sample_rows: int,
) -> dict[str, str | None]:
    """Return a role→column mapping for *path*.

    Tries (in order):
    1. Pre-computed mapping dict keyed by file name or full path string.
    2. On-the-fly inference via ``schema_mapper.infer_mapping``.
    """
    if precomputed:
        key = path.name
        entry = precomputed.get(key) or precomputed.get(str(path))
        if entry:
            return entry.get("mapping", {})

    # Lazy import to keep this module independent of analysis package
    from src.analysis.schema_analyzer import analyse_file
    from src.analysis.schema_mapper import infer_mapping

    report  = analyse_file(path, sample_rows=sample_rows)
    fm      = infer_mapping(report)
    return fm.mapping


# ── Core record generator ─────────────────────────────────────────────────────

def _stream_file(
    path: Path,
    mapping: dict[str, str | None],
) -> Generator[dict, None, None]:
    """Yield unified records from a single CSV file using *mapping*."""
    parts = path.parts
    source_folder = parts[-3] if len(parts) >= 3 else path.parent.parent.name
    path_category = parts[-2] if len(parts) >= 2 else ""
    file_name     = path.name

    df = _read_csv_safe(path)
    if df is None:
        return

    # Validate that mapped columns actually exist; clear missing ones
    actual_cols = set(df.columns)
    clean_mapping: dict[str, str | None] = {}
    for role, col in mapping.items():
        if col and col not in actual_cols:
            logger.warning(
                "%s: mapped column '%s' for role '%s' not found — skipping role",
                file_name, col, role,
            )
            clean_mapping[role] = None
        else:
            clean_mapping[role] = col

    yielded = 0
    for idx, row in df.iterrows():
        record = _build_record(
            row, clean_mapping, path_category, source_folder, file_name, idx
        )
        if record is not None:
            yield record
            yielded += 1

    logger.info("Streamed %d records from %s", yielded, path)


# ── Data-source discovery ─────────────────────────────────────────────────────

_DATA_SOURCES = [
    "Banuri-ExtractedData-Output",
    "fatwaqa-ExtractedData-Output",
    "IslamQA-ExtractedData-Output",
    "urdufatwa-ExtractedData-Output",
]


def _discover_csvs(data_root: Path) -> list[Path]:
    paths: list[Path] = []
    for source in _DATA_SOURCES:
        src_dir = data_root / source
        if src_dir.is_dir():
            paths.extend(sorted(src_dir.rglob("*_output.csv")))
    return paths


# ── Public streaming API ──────────────────────────────────────────────────────

def stream_corpus(
    data_root: str | Path = "data",
    precomputed_mappings: list[dict] | None = None,
    sample_rows: int = 100,
) -> Generator[dict, None, None]:
    """Yield every record from the full corpus one at a time.

    Memory usage is O(one CSV) at a time — safe for 100k+ total rows.

    Parameters
    ----------
    data_root:
        Root directory containing the 4 source sub-folders.
    precomputed_mappings:
        Output of ``infer_all()`` (list of file-mapping dicts).  When
        provided, schema inference is skipped.  Pass None to infer on-the-fly.
    sample_rows:
        Rows to sample when inferring schema on-the-fly (default 100).
    """
    root = Path(data_root)
    csv_paths = _discover_csvs(root)
    logger.info("Discovered %d CSV files under %s", len(csv_paths), root)

    # Build a lookup dict from pre-computed mappings if supplied
    lookup: dict[str, dict] | None = None
    if precomputed_mappings:
        lookup = {m["file_name"]: m for m in precomputed_mappings if "file_name" in m}

    for path in csv_paths:
        mapping = _resolve_mapping(path, lookup, sample_rows)
        yield from _stream_file(path, mapping)


def load_corpus_batched(
    data_root: str | Path = "data",
    batch_size: int = 500,
    precomputed_mappings: list[dict] | None = None,
    sample_rows: int = 100,
) -> Generator[list[dict], None, None]:
    """Yield the corpus in batches of *batch_size* records.

    Suitable for feeding directly into embedding / upsert loops::

        for batch in load_corpus_batched(batch_size=100):
            pairs = generate_embeddings_from_dicts(batch)
            upsert(pairs)
    """
    batch: list[dict] = []
    for record in stream_corpus(data_root, precomputed_mappings, sample_rows):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_corpus(
    data_root: str | Path = "data",
    precomputed_mappings: list[dict] | None = None,
    sample_rows: int = 100,
) -> list[dict]:
    """Load the full corpus into memory and return as a list.

    For large corpora prefer ``stream_corpus`` or ``load_corpus_batched``.
    """
    records = list(stream_corpus(data_root, precomputed_mappings, sample_rows))
    logger.info("Loaded %d records total", len(records))
    return records
