"""Sidecar Islam360 metadata lookup (url / fatwa_no / sect / source).

The active BM25 pickle cache was built before we had sect-level metadata,
so we keep a tiny sidecar map

    stable_id  ─►  { "url": ..., "fatwa_no": ..., "sect": ..., "source": ... }

built directly from the raw CSVs under
``settings.islam360_data_dir``. The map is cached on disk as a pickle so
the one-time CSV walk (~500 MB) only happens on first load.

Sect / source taxonomy (authoritative)
--------------------------------------

============================  ==========  ===================
Folder                         Sect        Source code
============================  ==========  ===================
``Banuri-ExtractedData-*``     deobandi    ``banuri``
``urdufatwa-ExtractedData-*``  barelvi     ``urdu_fatwa``
``IslamQA-ExtractedData-*``    ahle_hadith ``ahle_hadith_1``
``fatwaqa-ExtractedData-*``    ahle_hadith ``ahle_hadith_2``
============================  ==========  ===================

Every fatwa inherits its sect/source from the folder its CSV lives in.
This is deliberate: mixing per-row labels across sources is what caused
the earlier "all results come from Banuri" contamination bug.
"""

from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.islam360.documents import stable_id
from src.islam360.loader import (
    _ANSWER_ALIASES,
    _FATWA_NO_ALIASES,
    _QUESTION_ALIASES,
    _URL_ALIASES,
    _norm_col,
    _pick_column,
    _read_table,
)

logger = logging.getLogger(__name__)

# ── Sect / source taxonomy ────────────────────────────────────────────────────

# Canonical sect codes exposed to the rest of the system.
SECT_DEOBANDI = "deobandi"
SECT_BARELVI = "barelvi"
SECT_AHLE_HADITH = "ahle_hadith"

# Source → sect (used for filter reconstruction).
SOURCE_TO_SECT: dict[str, str] = {
    "banuri":         SECT_DEOBANDI,
    "urdu_fatwa":     SECT_BARELVI,
    "ahle_hadith_1":  SECT_AHLE_HADITH,
    "ahle_hadith_2":  SECT_AHLE_HADITH,
}

# Sect → allowed sources (used when enforcing filter before retrieval).
SECT_TO_SOURCES: dict[str, frozenset[str]] = {
    SECT_DEOBANDI:    frozenset({"banuri"}),
    SECT_BARELVI:     frozenset({"urdu_fatwa"}),
    SECT_AHLE_HADITH: frozenset({"ahle_hadith_1", "ahle_hadith_2"}),
}

ALL_SECTS: tuple[str, ...] = (SECT_DEOBANDI, SECT_BARELVI, SECT_AHLE_HADITH)


def _sect_and_source_for_path(path: Path) -> tuple[str, str]:
    """Derive ``(sect, source)`` from a CSV's parent folder name.

    Returns ``("", "")`` if the path does not live under a known source
    folder — in that case the row is left un-labelled (no sect filter
    will ever match it, which is the safe default).
    """
    for part in path.parts:
        low = part.lower()
        # Use substring so that capitalisation variants (``Banuri`` vs
        # ``banuri``) and any ``-ExtractedData-Output`` / ``-raw`` suffix
        # match uniformly.
        if "banuri" in low:
            return SECT_DEOBANDI, "banuri"
        if "urdufatwa" in low:
            return SECT_BARELVI, "urdu_fatwa"
        if "islamqa" in low:
            return SECT_AHLE_HADITH, "ahle_hadith_1"
        if "fatwaqa" in low:
            return SECT_AHLE_HADITH, "ahle_hadith_2"
    return "", ""


_lock = threading.Lock()
_lookup: dict[str, dict[str, str]] | None = None


def _cache_path() -> Path:
    """Sidecar cache lives next to the BM25 pickle."""
    s = get_settings()
    bm25 = Path(s.islam360_bm25_cache_path)
    return bm25.with_name(".islam360_url_lookup.pkl")


def _build_from_csvs() -> dict[str, dict[str, str]]:
    """Walk the raw Islam360 CSVs and build the id → metadata map.

    Uses the exact same ``stable_id(question, answer, f"{path.name}:{question[:40]}")``
    formula as :mod:`src.islam360.loader`, so the keys line up with ids
    already baked into the BM25 corpus.

    Unlike the previous implementation, this records an entry for EVERY
    row (not just rows with a URL) — the sect/source labels are needed
    to drive the per-sect retrieval filter. Rows where the parent folder
    cannot be mapped to a known source are still recorded, but with
    blank ``sect``/``source`` (they'll never pass a sect filter).
    """
    s = get_settings()
    root = Path(s.islam360_data_dir)
    if not root.is_dir():
        logger.warning("Islam360 data dir missing for URL lookup: %s", root)
        return {}

    out: dict[str, dict[str, str]] = {}
    n_files = 0
    n_rows = 0
    n_with_url = 0
    per_source: dict[str, int] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".csv", ".json", ".jsonl"):
            continue
        df = _read_table(path)
        if df is None or df.empty:
            continue
        df = df.fillna("")

        # Same multi-column fan-out trick used for URL backfill: loader's
        # ``_pick_column`` iterates a frozenset with non-deterministic
        # order, so when a CSV has BOTH ``Query`` and ``Question`` we
        # might disagree with the BM25 cache on which one won. We index
        # every (question-column × answer-column) combination so either
        # pairing resolves.
        q_cols = [c for c in df.columns if _norm_col(c) in _QUESTION_ALIASES]
        a_cols = [c for c in df.columns if _norm_col(c) in _ANSWER_ALIASES]
        if not q_cols or not a_cols:
            continue
        cu = _pick_column(df, _URL_ALIASES)
        cn = _pick_column(df, _FATWA_NO_ALIASES)

        sect, source = _sect_and_source_for_path(path)
        n_files += 1
        if source:
            per_source[source] = per_source.get(source, 0) + len(df)

        for _, row in df.iterrows():
            url = str(row[cu]).strip() if cu and cu in row else ""
            fatwa_no = str(row[cn]).strip() if cn and cn in row else ""
            for qc in q_cols:
                for ac in a_cols:
                    q = str(row[qc] if qc in row else "").strip()
                    a = str(row[ac] if ac in row else "").strip()
                    if not q and not a:
                        continue
                    sid = stable_id(q, a, f"{path.name}:{q[:40]}")
                    existing = out.get(sid)
                    if existing:
                        # Don't downgrade a populated entry; only fill
                        # blanks (a later pairing may have an empty URL
                        # column even though the earlier one didn't).
                        if not existing.get("url") and url:
                            existing["url"] = url
                            n_with_url += 1
                        if not existing.get("fatwa_no") and fatwa_no:
                            existing["fatwa_no"] = fatwa_no
                        # sect/source are path-derived, never overwrite
                        continue
                    out[sid] = {
                        "url": url,
                        "fatwa_no": fatwa_no,
                        "sect": sect,
                        "source": source,
                    }
                    n_rows += 1
                    if url:
                        n_with_url += 1

    logger.info(
        "Built Islam360 metadata lookup: %d rows (%d with URL) across %d files | per-source: %s",
        n_rows, n_with_url, n_files, per_source,
    )
    return out


def _load_or_build() -> dict[str, dict[str, str]]:
    path = _cache_path()
    if path.exists():
        try:
            with open(path, "rb") as f:
                data: Any = pickle.load(f)
            if isinstance(data, dict):
                # Sanity check: if sidecar was built before we added
                # sect/source, force a rebuild so filtering works.
                sample_val = next(iter(data.values()), None)
                if isinstance(sample_val, dict) and "sect" in sample_val:
                    logger.info(
                        "Loaded Islam360 metadata lookup cache: %d ids (%s)",
                        len(data), path,
                    )
                    return data
                logger.info(
                    "Sidecar at %s is missing sect/source — rebuilding", path
                )
        except Exception as exc:
            logger.warning(
                "Could not read metadata lookup cache %s: %s — rebuilding", path, exc
            )

    data = _build_from_csvs()
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved Islam360 metadata lookup cache to %s", path)
    except OSError as exc:
        logger.warning("Could not save metadata lookup cache: %s", exc)
    return data


def get_url_lookup() -> dict[str, dict[str, str]]:
    """Return the (cached) stable_id → metadata map.

    The first call builds the map (reading every CSV under
    ``settings.islam360_data_dir``); subsequent calls are free.

    Each entry carries four fields:
        * ``url`` — canonical link for cross-verification (may be blank)
        * ``fatwa_no`` — site-local fatwa number (may be blank)
        * ``sect`` — one of ``deobandi`` / ``barelvi`` / ``ahle_hadith``
        * ``source`` — one of ``banuri`` / ``urdu_fatwa`` /
          ``ahle_hadith_1`` / ``ahle_hadith_2``
    """
    global _lookup
    if _lookup is not None:
        return _lookup
    with _lock:
        if _lookup is None:
            _lookup = _load_or_build()
    return _lookup


def enrich_metadata(chunk_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Inject ``url``, ``fatwa_no``, ``sect``, ``source`` into *meta*.

    Only fills blanks — if metadata already carries these fields (e.g.
    a future BM25 rebuild persists them natively) we leave them alone.
    """
    info = get_url_lookup().get(chunk_id)
    if not info:
        return meta
    if info.get("url") and not meta.get("url"):
        meta["url"] = info["url"]
    if info.get("fatwa_no") and not meta.get("fatwa_no"):
        meta["fatwa_no"] = info["fatwa_no"]
    if info.get("sect") and not meta.get("sect"):
        meta["sect"] = info["sect"]
    if info.get("source") and not meta.get("source"):
        meta["source"] = info["source"]
    return meta


def get_sect_for_id(chunk_id: str) -> str:
    """Fast path for sect filter checks — returns ``""`` if unknown."""
    info = get_url_lookup().get(chunk_id)
    if not info:
        return ""
    return info.get("sect") or ""


def get_source_for_id(chunk_id: str) -> str:
    """Fast path for source filter checks — returns ``""`` if unknown."""
    info = get_url_lookup().get(chunk_id)
    if not info:
        return ""
    return info.get("source") or ""
