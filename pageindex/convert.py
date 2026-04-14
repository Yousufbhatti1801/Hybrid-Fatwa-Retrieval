"""Convert the existing CSV fatawa corpus to PageIndex inputs.

Outputs:
  pageindex/data/fatawa_index.md     — hierarchical markdown headings
                                       (# Fatawa Index → ## School →
                                        ### Category → #### Topic →
                                        ##### Fatwa)
  pageindex/data/fatawa_lookup.json  — flat dict keyed by composite ID
                                       containing the full Q&A record

Reuses ``src.ingestion.dynamic_loader.stream_corpus`` so we don't
duplicate any CSV-parsing logic. Run once after CSVs change::

    python -m pageindex.convert
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Make src.* importable when running as `python -m pageindex.convert`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingestion.dynamic_loader import stream_corpus  # noqa: E402
from src.preprocessing.chunker import (                  # noqa: E402
    SOURCE_DISPLAY_NAMES,
    SOURCE_MASLAK,
)
from pageindex.subgroup import (                          # noqa: E402
    CACHE_PATH as SUBGROUP_CACHE_PATH,
    assign_fatwa_to_group,
    _cache_key as _subgroup_cache_key,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────────────────────────────────

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_MD_PATH = _PKG_DIR / "data" / "fatawa_index.md"
DEFAULT_LOOKUP_PATH = _PKG_DIR / "data" / "fatawa_lookup.json"

# ──────────────────────────────────────────────────────────────────────────
# School ordering (matches the existing Flask UI per-school cards)
# ──────────────────────────────────────────────────────────────────────────

SCHOOL_ORDER = ["Banuri", "fatwaqa", "IslamQA", "urdufatwa"]

# Maps the long folder name in the corpus to the short school_id used
# everywhere else (matches plan.md and the existing app.py SCHOOL_MASLAK
# table at app.py:469).
_FOLDER_TO_SCHOOL_ID = {
    "Banuri-ExtractedData-Output":    "Banuri",
    "fatwaqa-ExtractedData-Output":   "fatwaqa",
    "IslamQA-ExtractedData-Output":   "IslamQA",
    "urdufatwa-ExtractedData-Output": "urdufatwa",
}


# ──────────────────────────────────────────────────────────────────────────
# Super-group clustering for large categories (especially OTHER)
# ──────────────────────────────────────────────────────────────────────────
# When a (school_id, category) pair has too many subtopics (>15), we add
# an intermediate H4 heading level that groups the subtopics into 6-8
# thematic "super-groups". This makes the LLM's Step A pick from 8
# clear clusters instead of 260 opaque subtopics.
#
# Structure of each entry:
#   SUPER_GROUP_MAP[(school_id, category)] = [
#       (group_key, bilingual_label, set_of_subtopics_OR_urdu_keyword_list),
#   ]
#
# For Banuri/fatwaqa the match is exact set membership (transliterated names).
# For IslamQA/urdufatwa the match is keyword containment in Urdu topic names.
# ──────────────────────────────────────────────────────────────────────────

# Banuri/OTHER — 36 subtopics → 7 super-groups
_BANURI_OTHER = [
    ("FINANCIAL", "مالی معاملات — Financial & Commercial",
     {"muamlat", "ijaarah", "shirkat", "muzaaribat", "rahan", "shufaa",
      "musaaqaat", "muzaareat", "maash-o-amwaal"}),
    ("SOCIAL", "حقوق و معاشرت — Social & Conduct",
     {"huqooq-muashret", "mamnooaat-mubahat", "razaat", "khunsaa"}),
    ("KNOWLEDGE", "علوم و فنون — Religious Knowledge",
     {"uloom-funoon", "ebadat", "tareekh-o-siyar", "tanqeeh"}),
    ("FUNERAL", "جنائز — Death & Funeral",
     {"janaiz"}),
    ("LEGAL", "حدود و قضاء — Legal & Judicial",
     {"hudood", "jinayaat", "qaza-o-iftaa", "shahaadat", "ikrah"}),
    ("PROPERTY", "ایمان نذر وقف — Oaths, Property & Endowments",
     {"ayman-o-nuzoor", "wakalat", "wadeeat", "waqf", "ihya-elmawaat",
      "gasab", "luqtaa-aur-laqeet"}),
    ("MISC", "قربانی و متفرقات — Sacrifice & Miscellaneous",
     {"uzhiyah", "aabeq-o-mafqood", "aariyat", "dawaa", "itaaq",
      "siyasat-o-hijrat"}),
]

# fatwaqa/OTHER — 26 subtopics → 6 super-groups
_FATWAQA_OTHER = [
    ("FINANCIAL", "مالی معاملات — Financial & Commercial",
     {"khareed-o-farokht", "ijarah", "shirkat", "muzaribat",
      "qarz-hiba-rahan", "iqtisad", "tijarat", "mahnama-ahkam-e-tijarat",
      "waqf"}),
    ("SOCIAL", "خاندانی و سماجی — Social & Family",
     {"auraton-ke-masail", "bachon-ke-naam", "huqooq-ul-ibad", "razaat"}),
    ("KNOWLEDGE", "علوم دینیہ — Religious Knowledge",
     {"quran-aur-hadees", "fazail-o-seerat", "mamoolat-e-ahlesunnat",
      "sunnatain-aur-adab", "masnoon-duayein"}),
    ("LEGAL", "ایمان و قضاء — Oaths & Legal",
     {"qasam-aur-mannat", "saza-o-qaza", "kafir-aur-murtad", "luqatah"}),
    ("FUNERAL", "موت و قربانی — Death & Sacrifice",
     {"mayyat", "qurbani-aur-aqeeqah"}),
    ("MISC", "متفرقات — Miscellaneous",
     {"mutafariqat", "mukhtasar-jawabat"}),
]

# IslamQA/OTHER (142 subtopics) + urdufatwa/OTHER (260 subtopics) use
# keyword-based matching because the topic names are Urdu. Each list
# entry is (group_key, label, [urdu_keywords]). A subtopic is matched
# to the FIRST group where any keyword appears in the subtopic name.
_URDU_OTHER_CLUSTERS = [
    ("FINANCIAL", "مالی معاملات — Financial & Commercial",
     ["سود", "بینک", "انشورنس", "تکافل", "سرمایہ", "تجار", "خرید", "فروخت",
      "قرض", "اجرت", "شراکت", "لین دین", "معاملات", "معاشی", "مالی",
      "کریڈٹ", "کرنسی", "ٹیکس", "آڑہت", "اجارہ", "وکالت", "ٹھیکہ",
      "ملٹی نیشنل", "نظام بینکاری", "مصارف", "اسٹاک", "جوا", "سٹہ",
      "رشوت", "نفقہ", "ضمان", "جعالہ"]),
    ("SOCIAL", "حقوق و معاشرت — Social & Conduct",
     ["معاشرت", "معاشرتی", "اجتماعی", "اصلاح معاشرہ", "حقوق", "والدین",
      "پڑوسی", "پرورش", "نفسیات", "نفسیاتی", "ملازمت", "آداب", "امر بالمعروف",
      "دوستی", "عشق", "اولاد", "تربیت", "خانگی", "رشتہ دار", "سماجی",
      "خواتین", "نسوان", "اطفال", "عورت", "مرد و عورت", "خوارج"]),
    ("QURAN_HADITH", "قرآن و حدیث — Quran & Hadith Sciences",
     ["قرآن", "حدیث", "تفسیر", "تجوید", "تحقیق", "تراجم", "ترجمہ",
      "اصول فقہ", "اصول حدیث", "علوم قرآن", "علوم حدیث", "مصطلح",
      "فہم", "شرح", "فوائد", "مصاحف", "احادیث", "کتب حدیث", "کتب وصحائف",
      "فضائل قرآن", "تدبر", "اعجاز", "لغات", "قراءات", "گرامر", "قواعد"]),
    ("AQEEDAH", "عقائد و مسالک — Creed & Sects",
     ["تقابل", "ادیان", "مسالک", "بدعت", "بدعی", "عقائد", "شرکیہ",
      "اہل تشیع", "بریلوی", "دیوبندی", "حنفی", "سلفی", "باطنیت",
      "معتزلہ", "یہودیت", "عیسائیت", "ہندو", "بدھ", "آتش", "سکھ",
      "جدیدیت", "انتہا", "نفاق", "تصوف", "نقشبندیہ", "چشتیہ",
      "قادریہ", "شاذلیہ", "سہروردیہ", "رفاعیہ", "وحدت", "تناسخ",
      "دہریت", "سوشلزم", "قادیانیت", "ارتداد", "تقدیر", "قدر",
      "عقیدہ", "منہج", "جدیدیت"]),
    ("HISTORY", "تاریخ و سیرت — History & Biography",
     ["تاریخ", "سیرت", "تذکر", "انبیا", "مشاہیر", "صحابہ", "صحابیات",
      "قصص", "خلافت", "شخصیات", "سلف", "تدوین"]),
    ("WORSHIP", "عبادات و رسوم — Worship & Rituals",
     ["عبادات", "احکام و مسائل", "اوقات", "مساجد", "امامت", "اذکار",
      "ادعیہ", "نذر", "وقف", "قربانی", "واجبات", "فرائض", "فضائل",
      "وجوب", "کفارہ", "ولایت", "اعتکاف", "عقیقہ", "دم", "جھاڑ",
      "دعا", "فطرت", "نو زائیدہ", "قبر", "حشر", "جنت", "جہنم",
      "قیامت", "ایصال", "جنازہ", "موت", "میت"]),
    ("HEALTH_LIFESTYLE", "صحت و طرز زندگی — Health & Lifestyle",
     ["علاج", "ایلو", "ہومیو", "لباس", "ستر", "حجاب", "موسیقی",
      "کھیل", "مصوری", "تصویر", "میڈیا", "فلم", "ڈرامہ",
      "تمدن", "تہذیب", "استحاضہ", "نفاس", "حیض", "نجاست",
      "بناؤ", "سنگھار", "طب", "صحت", "پانی", "غسل",
      "خواب", "حکمت", "رسوم"]),
    ("MISC", "متفرقات — Miscellaneous",
     ["متفرقات", "جدید مسائل", "اسلام اور", "مختلف", "منہج",
      "حکم شرعی", "مباحات", "محرمات", "مفسدات"]),
]


def _make_exact_map(clusters: list) -> dict[str, tuple[str, str]]:
    """Build a {subtopic_name → (group_key, label)} mapping for
    schools with exact transliterated subtopic names (Banuri, fatwaqa).
    """
    out: dict[str, tuple[str, str]] = {}
    for gkey, label, members in clusters:
        for m in members:
            out[m] = (gkey, label)
    return out


_BANURI_OTHER_MAP = _make_exact_map(_BANURI_OTHER)
_FATWAQA_OTHER_MAP = _make_exact_map(_FATWAQA_OTHER)


def _assign_supergroup(school_id: str, category: str, subtopic: str) -> tuple[str, str] | None:
    """Return ``(group_key, bilingual_label)`` for a subtopic, or None
    if the (school, category) is not clustered.

    Only fires for ``category == "OTHER"`` on the four schools.
    """
    if category != "OTHER":
        return None

    if school_id == "Banuri":
        return _BANURI_OTHER_MAP.get(subtopic)
    if school_id == "fatwaqa":
        return _FATWAQA_OTHER_MAP.get(subtopic)

    # IslamQA + urdufatwa: keyword containment matching
    if school_id in ("IslamQA", "urdufatwa"):
        lower = subtopic.lower()
        for gkey, label, keywords in _URDU_OTHER_CLUSTERS:
            if any(kw in lower for kw in keywords):
                return (gkey, label)
        # Unmatched → MISC
        return ("MISC", "متفرقات — Miscellaneous")

    return None


def _school_id(folder: str) -> str:
    return _FOLDER_TO_SCHOOL_ID.get(folder, folder)


def _school_label(folder: str) -> str:
    """Human-readable Urdu/English label for a school."""
    name = SOURCE_DISPLAY_NAMES.get(folder, folder)
    maslak = SOURCE_MASLAK.get(folder, "")
    return f"{name} — {maslak}" if maslak else name


def _subtopic(record: dict) -> str:
    """Extract a subtopic name from the source CSV filename.

    e.g. ``talaaq_output.csv`` → ``talaaq``.
    """
    stem = Path(record.get("source_file", "")).stem
    return re.sub(r"_output$", "", stem) or "general"


def _build_path_category_map(data_root: Path) -> dict[tuple[str, str], str]:
    """Walk the filesystem to map each CSV to its **directory-derived**
    category (e.g. ``NAMAZ``, ``DIVORCE``).

    This is needed because the dynamic_loader's schema_mapper
    mis-classifies the ``FatwahNo`` column as "category" for some
    sources (e.g. Banuri), so ``rec["category"]`` ends up being the
    fatwa reference number, not the real topic category. The actual
    category lives in the parent directory name (e.g.
    ``Banuri-ExtractedData-Output/DIVORCE/talaaq_output.csv``).

    Returns a dict keyed by ``(folder, source_file)``.
    """
    out: dict[tuple[str, str], str] = {}
    for source in _FOLDER_TO_SCHOOL_ID:
        src_dir = data_root / source
        if not src_dir.is_dir():
            continue
        for csv_path in src_dir.rglob("*_output.csv"):
            # category = parent directory name (e.g. "NAMAZ")
            cat = csv_path.parent.name if csv_path.parent != src_dir else "OTHER"
            out[(source, csv_path.name)] = cat.strip() or "OTHER"
    return out


# ──────────────────────────────────────────────────────────────────────────
# Markdown sanitisation
# ──────────────────────────────────────────────────────────────────────────

# A markdown heading must be a single line. We strip newlines and any
# leading '#' characters from titles to avoid breaking the parser, and
# collapse runs of whitespace.
_BAD_TITLE_CHARS = re.compile(r"[\r\n]+")
_MULTI_WS = re.compile(r"\s+")


def _clean_title(text: str, fallback: str = "—") -> str:
    if not text:
        return fallback
    text = _BAD_TITLE_CHARS.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    text = text.lstrip("#").strip()
    return text[:200] or fallback


# ──────────────────────────────────────────────────────────────────────────
# Composite ID
# ──────────────────────────────────────────────────────────────────────────

def _composite_id(school_id: str, category: str, subtopic: str,
                  fatwa_no: str, row_idx: int) -> str:
    """Stable, deterministic ID matching plan.md's format.

    Format::
        {school_id}__{category}__{subtopic}__{fatwa_no}_{row_idx}
    """
    safe_fatwa = (fatwa_no or "no_id").replace("\n", " ").strip()
    return f"{school_id}__{category}__{subtopic}__{safe_fatwa}_{row_idx}"


# ──────────────────────────────────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────────────────────────────────

def convert(
    md_path: Path = DEFAULT_MD_PATH,
    lookup_path: Path = DEFAULT_LOOKUP_PATH,
    data_root: str | None = None,
) -> dict:
    """Stream the corpus and emit ``fatawa_index.md`` + ``fatawa_lookup.json``.

    Returns a small summary dict with row counts per school.
    """
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    # Group records by (school_id, category, subtopic) so the markdown
    # tree comes out clean. We hold one school in memory at a time so this
    # stays manageable for ~330k records (~1 GB peak).
    grouped: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    lookup: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    skipped = 0

    logger.info("Streaming corpus…")
    # Resolve data_root from settings if not explicitly provided
    if data_root is None:
        from src.config import get_settings  # noqa: PLC0415
        data_root = str(get_settings().data_root)

    # Pre-compute path-based category lookup — dynamic_loader's
    # rec["category"] is unreliable for some sources (see helper docstring).
    path_cat = _build_path_category_map(Path(data_root))
    logger.info("Path-category map built: %d (folder, file) pairs",
                len(path_cat))

    for row_idx, rec in enumerate(stream_corpus(data_root=data_root)):
        folder = rec.get("folder", "")
        school_id = _school_id(folder)
        if school_id not in SCHOOL_ORDER:
            skipped += 1
            continue

        # Prefer the path-derived category; fall back to rec["category"]
        # only if we couldn't find the file on disk (shouldn't happen).
        category = path_cat.get((folder, rec.get("source_file", "")), "")
        if not category:
            category = (rec.get("category") or "OTHER").strip() or "OTHER"
        subtopic = _subtopic(rec)
        fatwa_no = (rec.get("query") or "").strip() \
            or (rec.get("question") or "").strip()[:80] \
            or f"row_{row_idx}"

        cid = _composite_id(school_id, category, subtopic, fatwa_no, row_idx)

        # The flat lookup record matches plan.md's schema (lines 76-94)
        lookup[cid] = {
            "school_id":    school_id,
            "school":       SOURCE_MASLAK.get(folder, ""),
            "school_label": _school_label(folder),
            "category":     category,
            "subtopic":     subtopic,
            "fatwa_no":     fatwa_no,
            "query_text":   rec.get("query", ""),
            "question_text": rec.get("question", ""),
            "answer_text":  rec.get("answer", ""),
            "url":          rec.get("reference") or "",
        }

        grouped[school_id][category][subtopic].append({
            "id":       cid,
            "fatwa_no": fatwa_no,
            "question": rec.get("question", "")[:160],
        })
        counts[school_id] += 1

        if (row_idx + 1) % 10_000 == 0:
            logger.info("  …%d records processed", row_idx + 1)

    logger.info("Total records: %d (skipped %d)", sum(counts.values()), skipped)

    # ── Write the markdown tree ──────────────────────────────────────────
    # For clustered categories (e.g. OTHER), we insert an extra H4
    # "super-group" level:
    #   ### OTHER                          (H3 = category)
    #   #### مالی معاملات — Financial      (H4 = super-group, NEW)
    #   ##### muamlat                      (H5 = subtopic, was H4)
    #   ###### بینک سے قرض لینے کا حکم    (H6 = fatwa, was H5)
    #
    # For non-clustered categories the hierarchy stays at 3 levels:
    #   ### DIVORCE                        (H3 = category)
    #   #### talaaq                        (H4 = subtopic)
    #   ##### fatwa title                  (H5 = fatwa)
    # ── Load subgroup cache (if available) ─────────────────────────────
    sg_cache: dict = {}
    if SUBGROUP_CACHE_PATH.exists():
        try:
            sg_cache = json.loads(SUBGROUP_CACHE_PATH.read_text(encoding="utf-8"))
            logger.info("Loaded subgroup cache: %d entries", len(sg_cache))
        except Exception as exc:
            logger.warning("Could not load subgroup cache: %s", exc)
    else:
        logger.info("No subgroup cache found — large subtopics won't be split. "
                     "Run `python -m pageindex.subgroup` first for best accuracy.")

    # Helper: write a list of fatwas under a subtopic, splitting into
    # sub-groups if the subgroup cache has an entry for this subtopic.
    def _write_fatwas(
        f, school_id: str, category: str, subtopic: str,
        fatwas: list[dict], heading_prefix: str, fatwa_prefix: str,
    ) -> None:
        """Write fatwas to the markdown file.

        If the subgroup cache has sub-groups for this (school, category,
        subtopic), split the fatwas into sub-groups at the same heading
        level. Otherwise, write them flat.

        ``heading_prefix`` is the ``#`` string for the subtopic level
        (e.g. ``"#####"``). ``fatwa_prefix`` is one level deeper.
        """
        cache_key = _subgroup_cache_key(school_id, category, subtopic)
        entry = sg_cache.get(cache_key)

        if not entry or not entry.get("groups") or len(fatwas) < 50:
            # No sub-groups — write the subtopic heading then fatwas flat
            f.write(f"{heading_prefix} {_clean_title(subtopic)}\n\n")
            for fw in sorted(fatwas, key=lambda x: x["fatwa_no"]):
                title = _clean_title(fw["fatwa_no"])
                f.write(f"{fatwa_prefix} {title}\n")
                f.write(f"<!-- id:{fw['id']} -->\n\n")
            return

        # Split fatwas into sub-groups
        groups = entry["groups"]
        buckets: list[list[dict]] = [[] for _ in groups]
        for fw in fatwas:
            idx = assign_fatwa_to_group(fw["fatwa_no"], groups)
            buckets[idx].append(fw)

        for gi, group in enumerate(groups):
            bucket = buckets[gi]
            if not bucket:
                continue
            label = f"{subtopic} — {group['name_ur']} — {group['name_en']}"
            f.write(f"{heading_prefix} {_clean_title(label)}\n\n")
            for fw in sorted(bucket, key=lambda x: x["fatwa_no"]):
                title = _clean_title(fw["fatwa_no"])
                f.write(f"{fatwa_prefix} {title}\n")
                f.write(f"<!-- id:{fw['id']} -->\n\n")

    logger.info("Writing markdown tree → %s", md_path)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Fatawa Index\n\n")
        for school_id in SCHOOL_ORDER:
            if school_id not in grouped:
                continue
            folder = next(
                k for k, v in _FOLDER_TO_SCHOOL_ID.items() if v == school_id
            )
            f.write(f"## {_clean_title(_school_label(folder), school_id)}\n\n")

            for category in sorted(grouped[school_id].keys()):
                f.write(f"### {_clean_title(category)}\n\n")

                # Check if this category should be clustered (super-groups)
                subtopics = grouped[school_id][category]
                sg_buckets: dict[str, dict[str, list]] = {}
                unclustered: dict[str, list] = {}

                for subtopic in sorted(subtopics.keys()):
                    sg = _assign_supergroup(school_id, category, subtopic)
                    if sg:
                        _gkey, glabel = sg
                        sg_buckets.setdefault(glabel, {})[subtopic] = subtopics[subtopic]
                    else:
                        unclustered[subtopic] = subtopics[subtopic]

                if sg_buckets:
                    # Clustered: H4 super-group → H5 subtopic (or sub-group) → H6 fatwa
                    for glabel in sorted(sg_buckets.keys()):
                        f.write(f"#### {_clean_title(glabel)}\n\n")
                        for subtopic in sorted(sg_buckets[glabel].keys()):
                            _write_fatwas(
                                f, school_id, category, subtopic,
                                sg_buckets[glabel][subtopic],
                                heading_prefix="#####",
                                fatwa_prefix="######",
                            )

                if unclustered:
                    # Unclustered: H4 subtopic (or sub-group) → H5 fatwa
                    for subtopic in sorted(unclustered.keys()):
                        _write_fatwas(
                            f, school_id, category, subtopic,
                            unclustered[subtopic],
                            heading_prefix="####",
                            fatwa_prefix="#####",
                        )

    # ── Write the flat lookup JSON ───────────────────────────────────────
    logger.info("Writing flat lookup → %s (%d entries)",
                lookup_path, len(lookup))
    with lookup_path.open("w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False)

    summary = {
        "total":           len(lookup),
        "skipped":         skipped,
        "by_school":       dict(counts),
        "md_path":         str(md_path),
        "lookup_path":     str(lookup_path),
    }
    logger.info("Conversion complete: %s", summary)
    return summary


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="pageindex.convert",
        description="Convert the fatawa CSV corpus to PageIndex inputs.",
    )
    p.add_argument("--md-path",     type=Path, default=DEFAULT_MD_PATH)
    p.add_argument("--lookup-path", type=Path, default=DEFAULT_LOOKUP_PATH)
    p.add_argument("--data-root",   type=str,  default=None,
                   help="Override DATA_ROOT (default: settings.data_root)")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = convert(
        md_path=args.md_path,
        lookup_path=args.lookup_path,
        data_root=args.data_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
