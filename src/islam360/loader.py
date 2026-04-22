"""Load Islam360 fatwa records from ``data/islam-360-fatwa-data`` (CSV / JSON)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.islam360.documents import (
    build_embedding_text,
    build_index_text,
    build_metadata,
    stable_id,
)

logger = logging.getLogger(__name__)

# Column aliases (case-insensitive) → canonical field
_QUESTION_ALIASES = frozenset(
    {"question", "query", "sawal", "سوال", "q", "fatwa_question", "title"}
)
_ANSWER_ALIASES = frozenset({"answer", "response", "جواب", "a", "fatwa", "reply"})
_CATEGORY_ALIASES = frozenset({"category", "topic", "section", "zimni", "cat"})
_SCHOLAR_ALIASES = frozenset({"scholar", "mufti", "author", "alim", "sheikh"})
_LANG_ALIASES = frozenset({"language", "lang", "locale"})
# URL / canonical-link column names (Banuri CSVs use ``Url_Link``, others
# may use ``url`` / ``link`` / ``reference``).
_URL_ALIASES = frozenset(
    {"url", "url_link", "urllink", "link", "href", "source_url", "reference", "ref"}
)
# Fatwa-number column names (Banuri CSVs use ``FatwahNo``).
_FATWA_NO_ALIASES = frozenset(
    {"fatwa_no", "fatwahno", "fatwa_number", "fatwaid", "fatwa_id", "no", "number"}
)


def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _pick_column(df: pd.DataFrame, aliases: set[str]) -> str | None:
    cols = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        if a in cols:
            return cols[a]
    for c in df.columns:
        if _norm_col(c) in aliases:
            return c
    return None


def _read_table(path: Path) -> pd.DataFrame | None:
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            return pd.read_csv(path, encoding="utf-8", dtype=str, low_memory=False)
        if suf in (".json", ".jsonl"):
            if suf == ".jsonl":
                return pd.read_json(path, lines=True, dtype=False)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict) and "records" in data:
                return pd.DataFrame(data["records"])
            return pd.DataFrame([data])
    except Exception:
        logger.exception("Failed to read %s", path)
        return None
    return None


def load_islam360_records(
    data_dir: Path | None = None,
) -> list[dict]:
    """Load all supported files under *data_dir* into flat dict records.

    Each record is suitable for :func:`build_bm25_docs` / embedding ingest::

        {
            "id": str,
            "text": str,       # Question + Answer combined
            "question": str,
            "answer": str,
            "category": str,
            "scholar": str,
            "language": str,
            "source_file": str,
        }
    """
    settings = get_settings()
    root = Path(data_dir or settings.islam360_data_dir)
    if not root.is_dir():
        logger.warning("Islam360 data directory does not exist: %s", root)
        return []

    # Deferred import to avoid a circular dep (url_index imports from
    # loader for the column aliases).
    from src.islam360.url_index import _sect_and_source_for_path

    records: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".csv", ".json", ".jsonl"):
            continue
        df = _read_table(path)
        if df is None or df.empty:
            continue
        df = df.fillna("")

        cq = _pick_column(df, _QUESTION_ALIASES)
        ca = _pick_column(df, _ANSWER_ALIASES)
        if not cq or not ca:
            logger.warning(
                "Skipping %s — need Question+Answer columns (found q=%s a=%s)",
                path, cq, ca,
            )
            continue
        cc = _pick_column(df, _CATEGORY_ALIASES)
        cs = _pick_column(df, _SCHOLAR_ALIASES)
        cl = _pick_column(df, _LANG_ALIASES)
        cu = _pick_column(df, _URL_ALIASES)
        cn = _pick_column(df, _FATWA_NO_ALIASES)

        # Sect / source are derived once per file from the parent folder
        # name (``Banuri-*`` → deobandi/banuri, ``urdufatwa-*`` → ...).
        sect, source = _sect_and_source_for_path(path)
        folder_name = path.parent.name

        for _, row in df.iterrows():
            question = str(row[cq] if cq in row else "").strip()
            answer = str(row[ca] if ca in row else "").strip()
            if not question and not answer:
                continue
            cat = str(row[cc]).strip() if cc and cc in row else ""
            scholar = str(row[cs]).strip() if cs and cs in row else ""
            lang = str(row[cl]).strip() if cl and cl in row else "ur"
            if not lang:
                lang = "ur"
            url = str(row[cu]).strip() if cu and cu in row else ""
            fatwa_no = str(row[cn]).strip() if cn and cn in row else ""

            sid = stable_id(question, answer, f"{path.name}:{question[:40]}")
            records.append(
                {
                    "id": sid,
                    # ``text`` = Question + Answer  → indexed by BM25 (sparse)
                    "text": build_index_text(question, answer),
                    # ``embedding_text`` = Question only → fed to OpenAI
                    # embeddings so the dense vector is symmetric with the
                    # user's query (also a question).
                    "embedding_text": build_embedding_text(question, answer),
                    "question": question,
                    "answer": answer,
                    "category": cat,
                    "scholar": scholar,
                    "language": lang,
                    "source_file": path.name,
                    "folder": folder_name,
                    "sect": sect,
                    "source": source,
                    "url": url,
                    "fatwa_no": fatwa_no,
                }
            )

    logger.info("Loaded %d Islam360 records from %s", len(records), root)
    return records


def records_to_bm25_docs(records: list[dict]) -> list[dict]:
    """Convert loader records to BM25Corpus document dicts."""
    out: list[dict] = []
    for r in records:
        meta = build_metadata(
            category=r.get("category", ""),
            scholar=r.get("scholar", ""),
            language=r.get("language", "ur"),
            source_file=r.get("source_file", ""),
            question=r.get("question", ""),
            answer=r.get("answer", ""),
            sect=r.get("sect", ""),
            source=r.get("source", ""),
            folder=r.get("folder", ""),
        )
        out.append(
            {
                "id": r["id"],
                "text": r["text"],
                "question": r["question"],
                "answer": r["answer"],
                "category": meta["category"],
                "source_file": r.get("source_file", ""),
                "folder": meta["folder"],
                "source_name": meta["source_name"],
                "maslak": "",
                "sect": meta.get("sect", ""),
                "source": meta.get("source", ""),
                "scholar": meta["scholar"],
                "language": meta["language"],
                "corpus_source": "islam360",
                "url": r.get("url", ""),
                "fatwa_no": r.get("fatwa_no", ""),
            }
        )
    return out
