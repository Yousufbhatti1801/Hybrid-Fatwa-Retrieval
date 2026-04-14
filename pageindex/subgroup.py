"""One-time LLM classification: split large subtopics into sub-groups.

For each subtopic with >THRESHOLD fatwas, this script:
1. Samples fatwa titles from the flat lookup JSON
2. Asks gpt-4o-mini to identify 5-15 themed sub-categories
3. Caches the LLM response in ``subgroup_cache.json``
4. On subsequent runs, only processes uncached subtopics

The cache is consumed by ``convert.py`` at tree-build time to split
large subtopics into smaller, navigable sub-groups.

Usage::

    python -m pageindex.subgroup                    # classify uncached
    python -m pageindex.subgroup --force            # redo all
    python -m pageindex.subgroup --threshold 200    # lower threshold

Cost: ~84 LLM calls × ~$0.001 = ~$0.10 total (one-time).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI  # noqa: E402
from src.config import get_settings  # noqa: E402

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
LOOKUP_PATH = _PKG_DIR / "data" / "fatawa_lookup.json"
CACHE_PATH = _PKG_DIR / "data" / "subgroup_cache.json"

THRESHOLD = 300  # subtopics with more fatwas than this get split
SAMPLE_SIZE = 150  # fatwa titles sampled per subtopic for LLM
MAX_GROUPS = 15
MIN_GROUPS = 5

# ──────────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert Islamic fiqh librarian. "
    "Always return strict JSON. No commentary."
)

_USER_TEMPLATE = (
    "You are classifying Islamic fatwa titles from the topic "
    '"{subtopic}" in the {school_label} fatwa collection.\n\n'
    "Based on the {n} sample titles below, identify {min_g}-{max_g} "
    "distinct thematic sub-categories that would help a researcher "
    "find a specific fatwa quickly. Each sub-category should be "
    "clearly distinct and cover a meaningful subset.\n\n"
    "Sample titles:\n{samples}\n\n"
    "Return STRICT JSON:\n"
    '{{"groups": [\n'
    '  {{"name_ur": "اردو نام", "name_en": "English Name", '
    '"keywords": ["keyword1", "keyword2", ...]}},\n'
    "  ...\n"
    "]}}\n\n"
    "Rules:\n"
    "- Each group must have 3-10 Urdu keywords that would match fatwa "
    "titles belonging to that group\n"
    "- Keywords should be common words/phrases that appear in the titles\n"
    "- Include both Urdu and transliterated English terms where relevant\n"
    "- The last group should be a catch-all 'عمومی — General' for "
    "titles that don't fit elsewhere\n"
    "- name_ur and name_en should be short (2-4 words each)"
)

# School labels for the prompt
_SCHOOL_LABELS = {
    "Banuri": "Banuri Institute (Deobandi)",
    "fatwaqa": "FatwaQA (Ahle Hadees)",
    "IslamQA": "IslamQA (Ahle Hadees)",
    "urdufatwa": "UrduFatwa (Barelvi)",
}


def _cache_key(school_id: str, category: str, subtopic: str) -> str:
    return f"{school_id}::{category}::{subtopic}"


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _classify_subtopic(
    school_id: str,
    category: str,
    subtopic: str,
    titles: list[str],
    client: OpenAI,
    model: str,
) -> list[dict]:
    """Call the LLM to classify a sample of fatwa titles into sub-groups.

    Returns a list of ``{"name_ur", "name_en", "keywords"}`` dicts.
    """
    # Sample and shuffle for diversity
    sample = random.sample(titles, min(SAMPLE_SIZE, len(titles)))
    sample_text = "\n".join(f"{i+1}. {t[:120]}" for i, t in enumerate(sample))

    user_msg = _USER_TEMPLATE.format(
        subtopic=subtopic,
        school_label=_SCHOOL_LABELS.get(school_id, school_id),
        n=len(sample),
        min_g=MIN_GROUPS,
        max_g=MAX_GROUPS,
        samples=sample_text,
    )

    try:
        comp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=800,
            timeout=20,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        parsed = json.loads(comp.choices[0].message.content or "{}")
        groups = parsed.get("groups", [])
        if not isinstance(groups, list) or len(groups) < 2:
            logger.warning("LLM returned <2 groups for %s/%s/%s",
                           school_id, category, subtopic)
            return []
        # Validate structure
        valid = []
        for g in groups:
            if (isinstance(g, dict)
                    and g.get("name_ur") and g.get("name_en")
                    and isinstance(g.get("keywords"), list)
                    and len(g["keywords"]) >= 1):
                valid.append({
                    "name_ur": str(g["name_ur"]).strip(),
                    "name_en": str(g["name_en"]).strip(),
                    "keywords": [str(k).strip() for k in g["keywords"]
                                 if str(k).strip()],
                })
        return valid
    except Exception as exc:
        logger.error("LLM call failed for %s/%s/%s: %s",
                     school_id, category, subtopic, exc)
        return []


def assign_fatwa_to_group(
    title: str, groups: list[dict]
) -> int:
    """Return the index of the first matching group for a fatwa title.

    Uses keyword containment. Returns len(groups)-1 (the last group,
    expected to be "General") if no keyword matches.
    """
    lower = title.lower()
    for i, g in enumerate(groups):
        if any(kw.lower() in lower for kw in g["keywords"]):
            return i
    # No match → last group (General catch-all)
    return len(groups) - 1


def run(
    threshold: int = THRESHOLD,
    force: bool = False,
) -> dict:
    """Run the sub-group classification for all large subtopics.

    Returns a summary dict.
    """
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Lookup not found at {LOOKUP_PATH}. "
            "Run `python -m pageindex.convert` first."
        )

    logger.info("Loading lookup from %s …", LOOKUP_PATH)
    lookup = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d records", len(lookup))

    # Group fatwa titles by (school, category, subtopic)
    by_subtopic: dict[str, list[str]] = defaultdict(list)
    for _cid, rec in lookup.items():
        key = _cache_key(rec["school_id"], rec["category"], rec["subtopic"])
        title = rec.get("query_text") or rec.get("fatwa_no") or ""
        if title:
            by_subtopic[key].append(title)

    # Filter to large subtopics
    large = {k: v for k, v in by_subtopic.items() if len(v) >= threshold}
    logger.info("Found %d subtopics with >= %d fatwas (of %d total subtopics)",
                len(large), threshold, len(by_subtopic))

    cache = _load_cache()
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    model = settings.chat_model

    classified = 0
    skipped = 0
    failed = 0

    for key, titles in sorted(large.items()):
        if not force and key in cache:
            skipped += 1
            continue

        parts = key.split("::")
        school_id, category, subtopic = parts[0], parts[1], parts[2]
        logger.info("Classifying %s (%d titles) …", key, len(titles))

        groups = _classify_subtopic(
            school_id, category, subtopic, titles, client, model
        )
        if groups:
            cache[key] = {
                "groups": groups,
                "total_fatwas": len(titles),
            }
            classified += 1
            group_summary = ", ".join(
                f"{g['name_en']}({len(g['keywords'])}kw)" for g in groups
            )
            logger.info("  → %d groups: %s", len(groups), group_summary)
        else:
            failed += 1
            logger.warning("  → FAILED, no groups returned")

        # Save after each classification so progress isn't lost on crash
        _save_cache(cache)

    summary = {
        "total_large_subtopics": len(large),
        "classified": classified,
        "skipped_cached": skipped,
        "failed": failed,
        "cache_entries": len(cache),
        "cache_path": str(CACHE_PATH),
    }
    logger.info("Done: %s", summary)
    return summary


# ──────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="pageindex.subgroup",
        description="One-time LLM classification of large subtopics.",
    )
    p.add_argument("--threshold", type=int, default=THRESHOLD,
                   help=f"Min fatwas to trigger split [default: {THRESHOLD}]")
    p.add_argument("--force", action="store_true",
                   help="Re-classify even if cached")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    summary = run(threshold=args.threshold, force=args.force)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
