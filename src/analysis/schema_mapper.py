"""Automatic schema-role inference for the Fatawa CSV corpus.

Given the output of ``schema_analyzer.scan_and_analyse()`` (or a list of
per-file reports), this module maps each CSV's raw column names to five
standardised roles:

    question  | answer  | category  | date  | reference

Algorithm
---------
For each column we compute a confidence score in [0.0, 1.0] composed of:

1. Name score   — how closely the column name matches known aliases
                  (English, romanised Urdu, and pure Urdu headings).
2. Content score — how well the sampled values *look* like the target role
                  (Urdu script ratio, length distribution, URL pattern…).

The column with the highest combined score wins each role slot.
If the best score is below a minimum threshold the role is set to None.

Usage
-----
    from src.analysis.schema_mapper import infer_mapping, infer_all

    # from a pre-computed scan result
    scan = scan_and_analyse("data")
    mappings = infer_all(scan["files"])

    # or directly from a file report
    mapping = infer_mapping(file_report)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Role definitions ──────────────────────────────────────────────────────────

# Minimum combined confidence to assign a role (below → None)
_MIN_CONFIDENCE = 0.25

# Weight split between name-based and content-based evidence
_NAME_WEIGHT    = 0.65
_CONTENT_WEIGHT = 0.35


@dataclass
class RoleMapping:
    column: str | None
    confidence: float          # [0.0, 1.0]
    name_score: float
    content_score: float


@dataclass
class FileMapping:
    file_name: str
    folder: str
    category: str
    mapping: dict[str, str | None]                  # role → column name / None
    confidence: dict[str, float]                    # role → score
    detail: dict[str, dict] = field(default_factory=dict)  # full RoleMapping per role


# ── Alias tables ─────────────────────────────────────────────────────────────
# Each entry is (alias_pattern, score).  Patterns are matched
# case-insensitively against the stripped column name.

_EXACT_WEIGHT  = 1.0
_STRONG_WEIGHT = 0.85
_WEAK_WEIGHT   = 0.55

_ROLE_ALIASES: dict[str, list[tuple[str, float]]] = {
    "question": [
        # Exact / near-exact
        (r"^question$",         _EXACT_WEIGHT),
        (r"^questions$",        _EXACT_WEIGHT),
        (r"^sawal$",            _EXACT_WEIGHT),   # romanised Urdu
        (r"^sawaal$",           _EXACT_WEIGHT),
        (r"^سوال$",             _EXACT_WEIGHT),   # Urdu
        # Strong
        (r"^ques(tion)?",       _STRONG_WEIGHT),
        (r"^q$",                _STRONG_WEIGHT),
        (r"^query$",            _STRONG_WEIGHT),
        (r"question",           _WEAK_WEIGHT),
        (r"sawal",              _WEAK_WEIGHT),
        (r"سوال",               _WEAK_WEIGHT),
    ],
    "answer": [
        (r"^answer$",           _EXACT_WEIGHT),
        (r"^answers$",          _EXACT_WEIGHT),
        (r"^jawab$",            _EXACT_WEIGHT),
        (r"^jawaab$",           _EXACT_WEIGHT),
        (r"^جواب$",             _EXACT_WEIGHT),
        (r"^ans$",              _STRONG_WEIGHT),
        (r"^fatwa_?text$",      _STRONG_WEIGHT),
        (r"^fatwa$",            _STRONG_WEIGHT),
        (r"^فتویٰ$",            _STRONG_WEIGHT),
        (r"answer",             _WEAK_WEIGHT),
        (r"jawab",              _WEAK_WEIGHT),
        (r"جواب",               _WEAK_WEIGHT),
    ],
    "category": [
        (r"^category$",         _EXACT_WEIGHT),
        (r"^cat$",              _STRONG_WEIGHT),
        (r"^topic$",            _STRONG_WEIGHT),
        (r"^subject$",          _STRONG_WEIGHT),
        (r"^type$",             _STRONG_WEIGHT),
        (r"^zumra$",            _STRONG_WEIGHT),
        (r"^زمرہ$",             _EXACT_WEIGHT),
        (r"^موضوع$",            _STRONG_WEIGHT),
        (r"^section$",          _WEAK_WEIGHT),
        (r"cat(egory)?",        _WEAK_WEIGHT),
        (r"topic",              _WEAK_WEIGHT),
    ],
    "date": [
        (r"^date$",             _EXACT_WEIGHT),
        (r"^tarikh$",           _EXACT_WEIGHT),
        (r"^تاریخ$",            _EXACT_WEIGHT),
        (r"^created_?at$",      _STRONG_WEIGHT),
        (r"^published(_at)?$",  _STRONG_WEIGHT),
        (r"^timestamp$",        _STRONG_WEIGHT),
        (r"date",               _WEAK_WEIGHT),
        (r"tarikh",             _WEAK_WEIGHT),
        (r"تاریخ",              _WEAK_WEIGHT),
    ],
    "reference": [
        (r"^url(_link)?$",      _EXACT_WEIGHT),
        (r"^url$",              _EXACT_WEIGHT),
        (r"^link$",             _EXACT_WEIGHT),
        (r"^source$",           _STRONG_WEIGHT),
        (r"^maakhaz$",          _STRONG_WEIGHT),
        (r"^ماخذ$",             _EXACT_WEIGHT),
        (r"^حوالہ$",            _STRONG_WEIGHT),
        (r"^reference$",        _STRONG_WEIGHT),
        (r"^ref$",              _STRONG_WEIGHT),
        (r"^fatwahno$",         _STRONG_WEIGHT),
        (r"^fatwa_?no$",        _STRONG_WEIGHT),
        (r"url",                _WEAK_WEIGHT),
        (r"source",             _WEAK_WEIGHT),
        (r"ref(erence)?",       _WEAK_WEIGHT),
    ],
}

# Pre-compile all patterns once
_COMPILED_ALIASES: dict[str, list[tuple[re.Pattern, float]]] = {
    role: [(re.compile(pat, re.IGNORECASE), score) for pat, score in aliases]
    for role, aliases in _ROLE_ALIASES.items()
}


# ── Name scoring ──────────────────────────────────────────────────────────────

def _name_score(col_name: str, role: str) -> float:
    """Return the best pattern match score for *col_name* against *role*."""
    col = col_name.strip()
    best = 0.0
    for pattern, score in _COMPILED_ALIASES[role]:
        if pattern.search(col):
            best = max(best, score)
    return best


# ── Content scoring ───────────────────────────────────────────────────────────

_URL_RE   = re.compile(r"https?://|www\.")
_DATE_RE  = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def _urdu_ratio(text: str) -> float:
    clean = text.replace(" ", "")
    if not clean:
        return 0.0
    return len(_ARABIC_RE.findall(clean)) / len(clean)


def _avg_urdu_ratio(samples: list[str]) -> float:
    vals = [v for v in samples if isinstance(v, str) and v.strip()]
    if not vals:
        return 0.0
    return sum(_urdu_ratio(v) for v in vals) / len(vals)


def _avg_len(samples: list[str]) -> float:
    vals = [v for v in samples if isinstance(v, str)]
    return sum(len(v) for v in vals) / len(vals) if vals else 0.0


def _url_ratio(samples: list[str]) -> float:
    vals = [v for v in samples if isinstance(v, str)]
    if not vals:
        return 0.0
    return sum(1 for v in vals if _URL_RE.search(v)) / len(vals)


def _date_ratio(samples: list[str]) -> float:
    vals = [v for v in samples if isinstance(v, str)]
    if not vals:
        return 0.0
    return sum(1 for v in vals if _DATE_RE.search(v)) / len(vals)


def _content_score(samples: list[str], role: str) -> float:
    """Heuristic content-based confidence for a given role."""
    if not samples:
        return 0.0

    urdu  = _avg_urdu_ratio(samples)
    avglen = _avg_len(samples)
    url_r  = _url_ratio(samples)
    date_r = _date_ratio(samples)

    if role == "question":
        # Urdu text, medium length (30–300 chars typical), ends with ؟ sometimes
        urdu_score = min(urdu / 0.5, 1.0)
        len_score  = 1.0 if 30 <= avglen <= 400 else max(0.0, 1.0 - abs(avglen - 200) / 500)
        ends_q     = sum(1 for v in samples if isinstance(v, str) and v.strip().endswith("؟")) / max(len(samples), 1)
        return min(1.0, 0.4 * urdu_score + 0.3 * len_score + 0.3 * ends_q)

    if role == "answer":
        # Urdu text, long (typically >100 chars)
        urdu_score = min(urdu / 0.5, 1.0)
        len_score  = 1.0 if avglen >= 100 else avglen / 100
        return min(1.0, 0.5 * urdu_score + 0.5 * len_score)

    if role == "category":
        # Short values, low uniqueness (repeated categories)
        uniq = len({v for v in samples if isinstance(v, str) and v.strip()})
        len_score  = 1.0 if avglen <= 30 else max(0.0, 1.0 - (avglen - 30) / 100)
        uniq_score = 1.0 if uniq <= 20 else max(0.0, 1.0 - (uniq - 20) / 80)
        return min(1.0, 0.5 * len_score + 0.5 * uniq_score)

    if role == "date":
        return min(1.0, date_r + 0.1)

    if role == "reference":
        # URL-like or alphanumeric ID
        id_ratio = sum(
            1 for v in samples
            if isinstance(v, str) and (
                _URL_RE.search(v)
                or re.match(r"^[\w\-\.\/]+$", v.strip())
            )
        ) / max(len(samples), 1)
        return min(1.0, 0.6 * url_r + 0.4 * id_ratio)

    return 0.0


# ── Per-column scoring ────────────────────────────────────────────────────────

def _score_column(col_info: dict, role: str) -> RoleMapping:
    name    = col_info["name"]
    samples = col_info.get("sample_values", [])

    ns = _name_score(name, role)
    cs = _content_score(samples, role)
    combined = _NAME_WEIGHT * ns + _CONTENT_WEIGHT * cs

    return RoleMapping(
        column=name,
        confidence=round(combined, 4),
        name_score=round(ns, 4),
        content_score=round(cs, 4),
    )


# ── Per-file inference ────────────────────────────────────────────────────────

def infer_mapping(file_report: dict) -> FileMapping:
    """Infer role → column mapping for a single file report.

    Parameters
    ----------
    file_report:
        One element from ``scan_and_analyse()["files"]``.

    Returns
    -------
    A ``FileMapping`` with best-match columns and confidence scores.
    """
    columns: list[dict] = file_report.get("columns", [])
    roles   = ["question", "answer", "category", "date", "reference"]

    best: dict[str, RoleMapping | None] = {r: None for r in roles}

    for col_info in columns:
        for role in roles:
            rm = _score_column(col_info, role)
            current = best[role]
            if current is None or rm.confidence > current.confidence:
                best[role] = rm

    # Apply minimum threshold — roles with low confidence → None
    mapping:    dict[str, str | None]  = {}
    confidence: dict[str, float]       = {}
    detail:     dict[str, dict]        = {}

    for role in roles:
        rm = best[role]
        if rm is None or rm.confidence < _MIN_CONFIDENCE:
            mapping[role]    = None
            confidence[role] = rm.confidence if rm else 0.0
            detail[role]     = asdict(rm) if rm else {}
        else:
            mapping[role]    = rm.column
            confidence[role] = rm.confidence
            detail[role]     = asdict(rm)

        logger.debug(
            "%s | role=%-10s col=%-20s conf=%.2f",
            file_report.get("file_name", "?"),
            role,
            mapping[role],
            confidence[role],
        )

    return FileMapping(
        file_name=file_report.get("file_name", ""),
        folder=file_report.get("folder", ""),
        category=file_report.get("category", ""),
        mapping=mapping,
        confidence=confidence,
        detail=detail,
    )


# ── Batch inference ───────────────────────────────────────────────────────────

def infer_all(file_reports: list[dict]) -> list[dict]:
    """Infer mappings for every file in a scan result.

    Returns a list of plain dicts (JSON-serialisable), each shaped::

        {
            "file_name":  str,
            "folder":     str,
            "category":   str,
            "mapping": {
                "question":  str | None,
                "answer":    str | None,
                "category":  str | None,
                "date":      str | None,
                "reference": str | None,
            },
            "confidence": {
                "question":  float,
                ...
            },
            "detail": {
                "question":  {"column": ..., "confidence": ..., ...},
                ...
            }
        }
    """
    results = []
    for report in file_reports:
        if report.get("error"):
            results.append({
                "file_name": report.get("file_name", ""),
                "folder":    report.get("folder", ""),
                "error":     report["error"],
            })
            continue
        fm = infer_mapping(report)
        results.append(asdict(fm))
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse, json
    from src.analysis.schema_analyzer import scan_and_analyse

    parser = argparse.ArgumentParser(
        description="Infer schema roles for every CSV in the fatawa corpus."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--out", default=None, help="Save JSON to file")
    args = parser.parse_args()

    scan   = scan_and_analyse(data_root=args.data_root, sample_rows=args.sample)
    result = infer_all(scan["files"])
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Schema mappings saved to {args.out}  ({len(result)} files)")
    else:
        print(output)


if __name__ == "__main__":
    _cli()
