"""Load and normalise CSV fatwa files into a unified document format."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.ingestion.scanner import scan_corpus

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = {"Url_Link", "Query", "FatwahNo", "Question", "Answer"}


@dataclass
class FatwaDocument:
    """Single fatwa loaded from a CSV row."""

    doc_id: str
    source: str          # e.g. "Banuri-ExtractedData-Output"
    category: str        # e.g. "NAMAZ"
    subcategory: str     # e.g. "salat"
    url: str
    query: str
    fatwa_no: str
    question: str
    answer: str
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenated question + answer used for embedding."""
        return f"{self.question}\n\n{self.answer}"


def _stable_id(source: str, fatwa_no: str, url: str) -> str:
    raw = f"{source}::{fatwa_no}::{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_df(path: Path) -> pd.DataFrame | None:
    """Read a CSV with UTF-8 encoding; return None if unreadable or missing columns."""
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str, low_memory=False)
    except Exception:
        logger.exception("Failed to read %s", path)
        return None

    if not EXPECTED_COLUMNS.issubset(set(df.columns)):
        logger.warning("Skipping %s – unexpected columns: %s", path, list(df.columns))
        return None

    return df


def load_csv(path: Path) -> list[FatwaDocument]:
    """Parse a single CSV file into FatwaDocument objects (used by the pipeline)."""
    parts = path.parts
    source = parts[-3]
    category = parts[-2]
    subcategory = path.stem.replace("_output", "")

    df = _read_df(path)
    if df is None:
        return []

    # Fill NaN with empty string so .str operations are safe
    df = df.fillna("")

    docs: list[FatwaDocument] = []
    for row in df.itertuples(index=False):
        question = str(row.Question).strip()
        answer = str(row.Answer).strip()
        if not question and not answer:
            continue

        url = str(row.Url_Link).strip()
        fatwa_no = str(row.FatwahNo).strip()

        docs.append(
            FatwaDocument(
                doc_id=_stable_id(source, fatwa_no, url),
                source=source,
                category=category,
                subcategory=subcategory,
                url=url,
                query=str(row.Query).strip(),
                fatwa_no=fatwa_no,
                question=question,
                answer=answer,
            )
        )

    logger.info("Loaded %d documents from %s", len(docs), path)
    return docs


def load_all() -> list[FatwaDocument]:
    """Discover and load all CSVs from the corpus."""
    csv_files = scan_corpus()
    logger.info("Found %d CSV files", len(csv_files))
    all_docs: list[FatwaDocument] = []
    for csv_path in csv_files:
        all_docs.extend(load_csv(csv_path))
    logger.info("Total documents loaded: %d", len(all_docs))
    return all_docs


# ── Dict-based API (matches the requested output schema) ─────────────────────

def load_csv_as_dicts(path: Path) -> list[dict]:
    """Return rows from a single CSV as plain dicts with a combined text field.

    Output schema per record:
        {
            "id":          str,   # SHA-256 stable ID
            "text":        str,   # "سوال: {question} جواب: {answer}"
            "question":    str,
            "answer":      str,
            "category":    str,   # parent folder (e.g. "NAMAZ")
            "source_file": str,   # CSV filename
            "folder":      str,   # data-source folder (e.g. "Banuri-ExtractedData-Output")
            "source_name": str,   # human-readable source (e.g. "Banuri Institute")
        }
    """
    from src.preprocessing.chunker import get_source_display_name  # lazy import

    parts = path.parts
    source = parts[-3]
    category = parts[-2]

    df = _read_df(path)
    if df is None:
        return []

    df = df.fillna("")

    records: list[dict] = []
    for row in df.itertuples(index=False):
        question = str(row.Question).strip()
        answer = str(row.Answer).strip()
        if not question and not answer:
            continue

        url = str(row.Url_Link).strip()
        fatwa_no = str(row.FatwahNo).strip()

        records.append(
            {
                "id": _stable_id(source, fatwa_no, url),
                "text": f"سوال: {question} جواب: {answer}",
                "question": question,
                "answer": answer,
                "category": category,
                "source_file": path.name,
                "folder": source,
                "source_name": get_source_display_name(source),
            }
        )

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def load_all_as_dicts() -> list[dict]:
    """Load the entire corpus and return records in the flat dict schema.

    Scales to 100k+ rows — CSVs are read with pandas and processed
    file-by-file so memory stays bounded.
    """
    csv_files = scan_corpus()
    logger.info("Found %d CSV files", len(csv_files))
    all_records: list[dict] = []
    for csv_path in csv_files:
        all_records.extend(load_csv_as_dicts(csv_path))
    logger.info("Total records loaded: %d", len(all_records))
    return all_records
