"""LLM-driven tree navigation over the parsed PageIndex tree.

This is the **vectorless retrieval** path. The caller first normalises and
expands the query (``pipeline_pageindex`` + ``query_enrich``).  For each
madhab, we run (LLM) tree navigation and two ranking stages:

  1. ``_pick_categories_and_topics`` — pick categories + topics in the
     compact tree.
  2. ``_pick_fatwas`` (Step B) — keyword-scored candidates (optionally
     using Q+A from the flat lookup, not the title alone) then an LLM
     re-rank on full masʾala text.
  3. ``_refine_cid_order`` (pass 2) — optional second LLM that re-orders
     a short list of composite ids for precision.

The chosen fatwa is then resolved against ``fatawa_lookup.json`` (a
plain dict, no LLM) and a relevance percent is computed via keyword
overlap. The four schools are searched in parallel via a
``ThreadPoolExecutor`` so wall-clock latency is the latency of one
school's two-call descent (~3-5 s).

The tree, the lookup, and the per-query results are all cached in
module-level state so warm queries are cheap.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

# Make src.* importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI                              # noqa: E402
from src.config import get_settings                    # noqa: E402

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Constants & paths
# ──────────────────────────────────────────────────────────────────────────

_PKG_DIR        = Path(__file__).resolve().parent
_DATA_DIR       = _PKG_DIR / "data"
_INDEX_DIR      = _PKG_DIR / "index"
LOOKUP_PATH     = _DATA_DIR / "fatawa_lookup.json"
DOC_ID_PATH     = _INDEX_DIR / "doc_id.txt"

ALL_SCHOOLS = ["Banuri", "fatwaqa", "IslamQA", "urdufatwa"]

# Per-school display label + maslak (matches existing chunker.SOURCE_MASLAK).
_SCHOOL_LABEL = {
    "Banuri":    "Banuri Town — Deobandi",
    "fatwaqa":   "FatwaQA — Ahle Hadees",
    "IslamQA":   "IslamQA — Ahle Hadees",
    "urdufatwa": "UrduFatwa — Barelvi",
}

_SCHOOL_MASLAK = {
    "Banuri":    "Deobandi",
    "fatwaqa":   "Ahle Hadees",
    "IslamQA":   "Ahle Hadees",
    "urdufatwa": "Barelvi",
}

# Bound the candidate list size for the Step B LLM call.
_MAX_FATWA_CANDIDATES = 100

# How many top-ranked fatawa to return per school (primary + alternates).
_DEFAULT_TOP_N_PER_SCHOOL = 4

# Per-school result is cached for the lifetime of the worker process so
# repeated queries are free. Keyed by (core_question, school_id).
_NAV_CACHE_SIZE = 256


# ──────────────────────────────────────────────────────────────────────────
# Lazy resource loaders (tree, lookup, OpenAI client)
# ──────────────────────────────────────────────────────────────────────────

_TREE: dict | None = None        # parsed JSON of the active doc
_LOOKUP: dict[str, dict] | None = None
_SCHOOL_INDEX: dict[str, dict] | None = None  # school_id → school subtree


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def _cat_title_matches_hint(title: str, hint: str) -> bool:
    """Match category node title to ``category_hint`` (supports ``NAMAZ — …``)."""
    if not title or not hint:
        return False
    h = hint.strip().upper()
    raw = title.strip()
    t = raw.upper()
    if t == h or t.startswith(h + " "):
        return True
    for sep in ("—", "\u2013", "-"):
        if sep in raw:
            first = raw.split(sep, 1)[0].strip().upper()
            if first == h:
                return True
    return False


def _lookup_haystack_for_scoring(
    rec: dict, leaf_title: str,
) -> str:
    """Match terms against masʾala + reference lines only.

    Including long ``answer_text`` was the main source of *false* lexical
    hits: generic fiqh discussion mentions many terms unrelated to the
    user's specific question.
    """
    best_q = _best_question_line(rec, leaf_title)
    return " ".join(
        p
        for p in (
            leaf_title,
            best_q,
            rec.get("query_text", ""),
        )
        if p
    ).lower()


_BOILERPLATE_SNIPPETS = (
    "dar-ul-ifta",
    "darulifta",
    "www.",
    "feedback@",
    "اَلْجَوَابُ",
    "الجواب",
    "بِسْمِ اللہ",
    "بسم الله",
    "وَاللہُ اَعْلَم",
    "والله اعلم",
)


def _is_boilerplate_question_line(text: str) -> bool:
    """Heuristic guard against template boilerplate posing as a question."""
    t = " ".join((text or "").split()).strip().lower()
    if not t:
        return True
    if len(t) < 12 and "؟" not in t and "?" not in t:
        return True
    return any(snippet in t for snippet in _BOILERPLATE_SNIPPETS)


def _best_question_line(rec: dict, fallback_title: str) -> str:
    """Pick the most question-like text from lookup fields."""
    candidates = [
        (rec.get("question_text") or "").strip(),
        (rec.get("query_text") or "").strip(),
        (fallback_title or "").strip(),
    ]
    for c in candidates:
        if c and not _is_boilerplate_question_line(c):
            return c
    for c in candidates:
        if c:
            return c
    return ""


_U2_STOP2 = frozenset(
    {"کے", "میں", "سے", "کا", "کی", "نہ", "وہ", "یہ", "تو", "کو", "پر",
     "is", "of", "in", "or", "to", "a", "it"},
)
_UR2 = re.compile(r"[\u0600-\u06FF\u0750-\u077Fa-zA-Z]{2,}")


def _urdu_scoring_tokens(text: str, *, cap: int = 18) -> list[str]:
    """Tokenize for *matching*; 2+ codepoints (3+ was dropping real terms)."""
    if not (text or "").strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for t in _UR2.findall((text or "").lower()):
        if t in _U2_STOP2 and len(t) < 3:
            continue
        if t in seen or len(t) < 2:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= cap:
            break
    return out


def _raw_phrase_substring_bonus(hay: str, raw: str) -> int:
    """When the user asked a long specific phrase, reward leaves whose text
    contains the same 2+ word n-grams or a long sub-span (common for Urdu fiqh
    fatawa where keyword bags fail).
    """
    if not (raw and hay):
        return 0
    h = re.sub(r"[\n\r\t]+", " ", hay.lower().strip())
    r = re.sub(r"[\n\r\t]+", " ", raw.lower().strip())
    b = 0
    if len(r) > 4 and r in h:
        b += 20
    parts = [p for p in r.split() if len(p) > 1]
    for i in range(len(parts) - 1):
        w2 = f"{parts[i]} {parts[i+1]}"
        if len(w2) > 3 and w2 in h:
            b += 8
    for i in range(len(parts) - 2):
        w3 = f"{parts[i]} {parts[i+1]} {parts[i+2]}"
        if len(w3) > 6 and w3 in h:
            b += 10
    return b


def _tight_scoring_tokens(
    core_q: str,
    full_kw: list[str] | None,
    raw_user: str = "",
) -> list[str]:
    """Tight token set + user's exact wording for bucket pre-scoring."""
    merged: list[str] = []
    seen: set[str] = set()

    def _ad(t: str) -> None:
        t = t.strip().lower()
        if len(t) < 2 or t in seen:
            return
        if len(t) < 3 and t in _U2_STOP2:
            return
        seen.add(t)
        merged.append(t)

    for t in _urdu_scoring_tokens(raw_user, cap=10):
        _ad(t)
    for t in _urdu_scoring_tokens(core_q, cap=10):
        _ad(t)
    for k in (full_kw or []):
        _ad((k or "").lower())
    return merged[:18]


def _load_tree() -> dict:
    """Read the active tree JSON from ``pageindex/index/``.

    Cached after first load. Idempotent and thread-safe (single
    assignment under the GIL).
    """
    global _TREE, _SCHOOL_INDEX
    if _TREE is not None:
        return _TREE

    if not DOC_ID_PATH.exists():
        raise FileNotFoundError(
            f"PageIndex tree not built yet. Expected {DOC_ID_PATH}. "
            "Run `python -m pageindex.convert && "
            "python -m pageindex.ingest_pageindex` first."
        )
    doc_id = DOC_ID_PATH.read_text(encoding="utf-8").strip()
    tree_path = _INDEX_DIR / f"{doc_id}.json"
    logger.info("Loading PageIndex tree from %s", tree_path)
    with tree_path.open("r", encoding="utf-8") as f:
        _TREE = json.load(f)

    # Build a school_id → school_node lookup once.
    # The tree has a single root "# Fatawa Index" with 4 school children
    # at H2 level, so we check both top-level and one level down.
    structure = _TREE.get("structure", [])
    candidates = list(structure)
    for node in structure:
        candidates.extend(node.get("nodes") or [])
    si: dict[str, dict] = {}
    for school_node in candidates:
        title = school_node.get("title", "")
        for sid in ALL_SCHOOLS:
            if sid.lower() in title.lower():
                si[sid] = school_node
                break
    _SCHOOL_INDEX = si
    logger.info("Tree has %d top-level schools: %s",
                len(structure), list(si.keys()))
    return _TREE


def _load_lookup() -> dict[str, dict]:
    global _LOOKUP
    if _LOOKUP is not None:
        return _LOOKUP
    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Fatawa lookup not built yet. Expected {LOOKUP_PATH}. "
            "Run `python -m pageindex.convert` first."
        )
    logger.info("Loading fatawa lookup from %s …", LOOKUP_PATH)
    with LOOKUP_PATH.open("r", encoding="utf-8") as f:
        _LOOKUP = json.load(f)
    logger.info("Loaded %d fatawa records into memory", len(_LOOKUP))
    return _LOOKUP


_COMPACT_TREES: dict[str, tuple[str, dict]] = {}


def preload() -> None:
    """Force-load tree, lookup, and pre-compute compact tree text.

    Called by app.py warmup thread.
    """
    _load_tree()
    _load_lookup()
    # Pre-compute the compact tree text for each school — it's static
    # and expensive enough that we don't want to rebuild it per-query.
    for sid in ALL_SCHOOLS:
        node = _get_school_subtree(sid)
        if node:
            _COMPACT_TREES[sid] = _build_compact_school_tree(node)


# ──────────────────────────────────────────────────────────────────────────
# Tree helpers
# ──────────────────────────────────────────────────────────────────────────

def _get_school_subtree(school_id: str) -> dict | None:
    _load_tree()
    return (_SCHOOL_INDEX or {}).get(school_id)


def _is_supergroup_node(node: dict) -> bool:
    """Return True if ``node`` is a super-group (i.e., its children
    are subtopics that themselves have fatwa children, rather than
    being fatwa leaves directly)."""
    children = node.get("nodes") or []
    if not children:
        return False
    first = children[0]
    # A super-group's children are subtopics (with `nodes` list of fatwas);
    # a normal topic's children are fatwa leaves (with `fatwa_id`).
    return bool(first.get("nodes")) and not first.get("fatwa_id")


def _count_leaves(node: dict) -> int:
    """Count all fatwa leaves under ``node`` at any depth."""
    children = node.get("nodes") or []
    if not children:
        return 1 if node.get("fatwa_id") else 0
    return sum(_count_leaves(c) for c in children)


def _is_subgroup_title(title: str) -> bool:
    """Return True if a node title is a sub-group split (contains ' — ')."""
    return " — " in title and title.count(" — ") >= 1


def _build_compact_school_tree(school_node: dict) -> tuple[str, dict[str, dict]]:
    """Return (compact_text, id_index).

    Handles three types of tree nodes:

    1. **Super-groups** (e.g., OTHER → Financial): shown as aggregated
       one-liners with fatwa count + sample subtopic names.

    2. **Sub-group splits** (e.g., `muamlat — Crypto`, `muamlat — Banking`):
       grouped back into their parent subtopic and shown as a single
       aggregated line listing all sub-group labels. The LLM picks the
       parent subtopic; Step A.5 drills into the sub-groups.

    3. **Normal small topics** (<300 fatwas, no split): shown with
       fatwa count + 2 sample titles.

    This keeps the compact tree small (~2-3k tokens per school) even
    when sub-groups expand the actual tree to thousands of nodes.
    """
    lines: list[str] = []
    id_index: dict[str, dict] = {}

    for cat_node in school_node.get("nodes", []) or []:
        cat_id = cat_node.get("node_id") or "?"
        cat_title = cat_node.get("title", "")
        id_index[cat_id] = cat_node
        lines.append(f"[{cat_id}] {cat_title}")

        children = cat_node.get("nodes", []) or []

        # ── Detect sub-group splits: group children that share a common
        #    subtopic prefix (e.g., "muamlat — X", "muamlat — Y") back
        #    into their parent subtopic for compact display.
        grouped_subgroups: dict[str, list[dict]] = {}  # prefix → [child nodes]
        standalone: list[dict] = []

        for child in children:
            ctitle = child.get("title", "")
            if _is_subgroup_title(ctitle):
                prefix = ctitle.split(" — ")[0].strip()
                grouped_subgroups.setdefault(prefix, []).append(child)
            else:
                standalone.append(child)

        # Emit grouped sub-groups as aggregated one-liners
        for prefix, sg_nodes in sorted(grouped_subgroups.items()):
            # Use the FIRST sub-group's node_id as the representative;
            # index ALL sub-group nodes so the LLM can pick any.
            first_id = sg_nodes[0].get("node_id") or "?"
            total_fatawa = sum(_count_leaves(n) for n in sg_nodes)
            sub_labels = [
                n.get("title", "").split(" — ", 1)[-1].strip()[:35]
                for n in sg_nodes if n.get("title")
            ]
            # Register each sub-group node in the id_index
            for n in sg_nodes:
                nid = n.get("node_id") or "?"
                id_index[nid] = n
            # Show as: [first_id] prefix (N fatawa, K sub-groups: label1, label2, ...)
            labels_str = ", ".join(sub_labels[:5])
            if len(sub_labels) > 5:
                labels_str += f", +{len(sub_labels)-5} more"
            lines.append(
                f"  [{first_id}] {prefix} ({total_fatawa} fatawa, "
                f"{len(sg_nodes)} sub-groups: {labels_str})"
            )

        # Emit standalone children (super-groups or normal topics)
        for child in standalone:
            cid = child.get("node_id") or "?"
            ctitle = child.get("title", "")
            id_index[cid] = child

            if _is_supergroup_node(child):
                subtopics = child.get("nodes") or []
                n_sub = len(subtopics)
                n_fatawa = _count_leaves(child)
                samples = [s.get("title", "")[:40] for s in subtopics[:3]]
                sample_str = ", ".join(samples)
                lines.append(
                    f"  [{cid}] {ctitle} ({n_fatawa} fatawa, "
                    f"{n_sub} subtopics: {sample_str})"
                )
            else:
                leaves = child.get("nodes") or []
                n_fatawa = len(leaves)
                lines.append(f"  [{cid}] {ctitle} — {n_fatawa} fatawa")
                # Only show sample titles for small topics (not sub-groups)
                if leaves and n_fatawa <= 300:
                    step = max(1, n_fatawa // 3)
                    for i in (0, step):
                        if i < n_fatawa:
                            title = (leaves[i].get("title") or "").strip()
                            if title:
                                lines.append(f"      · {title[:90]}")

    return "\n".join(lines), id_index


# ──────────────────────────────────────────────────────────────────────────
# LLM call wrappers — both use json_object response_format + temp 0
# ──────────────────────────────────────────────────────────────────────────

_PICK_CT_SYSTEM = (
    "You are an Urdu Islamic fiqh research assistant. "
    "Always reply with strict JSON, no commentary."
)

_PICK_CT_USER = (
    "صارف کے اصل الفاظ (as typed / original):\n{raw}\n\n"
    "تحلیل شدہ سوال (analysed / standard Urdu line):\n{q}\n\n"
    "Optional category hint: {hint}\n\n"
    "نیچے {school_label} کی categories اور ان کے اندر topics/groups ہیں۔\n"
    "Some categories have themed 'super-groups' shown as parenthetical "
    "entries (e.g. Financial, Social). Topics may have opaque Arabic-"
    "transliterated names; use the sample titles/subtopic names to judge.\n"
    "Tree:\n{tree}\n\n"
    "**You MUST return at least one topic_node_id.** "
    "Pick the node_ids of the categories/groups/topics most relevant to "
    "the user's question. Pick at most 3 categories and 6 topics/groups.\n\n"
    'Return STRICT JSON: {{"category_node_ids": ["…"], '
    '"topic_node_ids": ["…"], "reasoning_brief": "<one Urdu sentence>"}}\n'
    "Return only node_ids that appear in the tree above."
)


def _pick_categories_and_topics(
    school_id: str,
    school_node: dict,
    core_question: str,
    category_hint: str | None,
    user_raw: str = "",
) -> dict:
    # Use pre-computed compact tree if available (populated during preload)
    if school_id in _COMPACT_TREES:
        compact, id_index = _COMPACT_TREES[school_id]
    else:
        compact, id_index = _build_compact_school_tree(school_node)
    if not compact:
        return {"category_node_ids": [], "topic_node_ids": [],
                "id_index": id_index, "reasoning_brief": "empty subtree"}

    settings = get_settings()
    user_msg = _PICK_CT_USER.format(
        raw=(user_raw or core_question)[:1500],
        q=core_question,
        hint=category_hint or "none",
        school_label=_SCHOOL_LABEL.get(school_id, school_id),
        tree=compact,
    )
    try:
        comp = _client().chat.completions.create(
            model=settings.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=250,
            timeout=12,
            messages=[
                {"role": "system", "content": _PICK_CT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        llm_json = comp.choices[0].message.content or "{}"
        parsed = json.loads(llm_json)
    except Exception as exc:
        logger.warning("[%s] Step A LLM call failed: %s", school_id, exc)
        parsed = {}

    cat_ids = [str(x) for x in (parsed.get("category_node_ids") or [])][:3]
    topic_ids = [str(x) for x in (parsed.get("topic_node_ids") or [])][:6]
    # Validate against the actual tree — drop hallucinated ids.
    cat_ids = [c for c in cat_ids if c in id_index]
    topic_ids = [t for t in topic_ids if t in id_index]
    if not topic_ids:
        logger.warning(
            "[%s] Step A returned no valid topic_node_ids "
            "(parsed=%s, raw_topic_ids=%s)",
            school_id,
            list(parsed.keys()),
            parsed.get("topic_node_ids"),
        )
    return {
        "category_node_ids": cat_ids,
        "topic_node_ids":    topic_ids,
        "id_index":          id_index,
        "reasoning_brief":   parsed.get("reasoning_brief", ""),
    }


# ──────────────────────────────────────────────────────────────────────────
# Step A.5: drill into a super-group's subtopics (fires only for
# clustered categories like OTHER). Tiny prompt (~500 tokens).
# ──────────────────────────────────────────────────────────────────────────

_PICK_SUB_USER = (
    "صارف (اصل): {raw}\n\n"
    "تحلیل: {q}\n\n"
    'نیچے "{group_label}" گروپ کے اندر موجود subtopics کی فہرست ہے:\n'
    "{subtopics}\n\n"
    "Pick at most 3 subtopic_node_ids most relevant to the question.\n"
    'Return STRICT JSON: {{"subtopic_node_ids": ["…"]}}'
)


def _pick_subtopics_within_group(
    school_id: str,
    group_node: dict,
    core_question: str,
    user_raw: str = "",
) -> list[dict]:
    """Step A.5 — narrow a super-group to its most relevant subtopics.

    Returns a list of subtopic tree nodes (each with .nodes = fatwa leaves).
    """
    subtopics = group_node.get("nodes") or []
    if len(subtopics) <= 5:
        # Small enough to use entirely — no LLM call needed
        return subtopics

    # Build a compact list of subtopics with fatwa count
    lines: list[str] = []
    id_map: dict[str, dict] = {}
    for sub in subtopics:
        sid = sub.get("node_id") or "?"
        stitle = sub.get("title", "")
        n = len(sub.get("nodes") or [])
        id_map[sid] = sub
        # Show 1 sample fatwa title for context
        sample = ""
        leaves = sub.get("nodes") or []
        if leaves:
            sample = f"  — e.g. {(leaves[0].get('title') or '')[:70]}"
        lines.append(f"[{sid}] {stitle} ({n} fatawa){sample}")

    settings = get_settings()
    user_msg = _PICK_SUB_USER.format(
        raw=(user_raw or core_question)[:1500],
        q=core_question,
        group_label=group_node.get("title", ""),
        subtopics="\n".join(lines),
    )
    try:
        comp = _client().chat.completions.create(
            model=settings.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=150,
            timeout=10,
            messages=[
                {"role": "system", "content": _PICK_CT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        llm_text = comp.choices[0].message.content or "{}"
        parsed = json.loads(llm_text)
    except Exception as exc:
        logger.warning("[%s] Step A.5 failed: %s", school_id, exc)
        parsed = {}

    picked_ids = [str(x) for x in (parsed.get("subtopic_node_ids") or [])][:3]
    picked_ids = [x for x in picked_ids if x in id_map]

    if picked_ids:
        return [id_map[x] for x in picked_ids]

    # Fallback: return all subtopics (let keyword pre-filter handle it)
    logger.warning("[%s] Step A.5 returned no ids; using all %d subtopics",
                   school_id, len(subtopics))
    return subtopics


_PICK_FATWA_SYSTEM = _PICK_CT_SYSTEM

_PICK_FATWA_USER = (
    "صارف (اصل): {raw}\n\n"
    "تحلیل: {q}\n"
    "Madhab: {school_label}\n\n"
    "نیچے فتاویٰ کے امیدوار — ہر سطر: [f-id] + مسئلہ/سوال (tree یا حوالہ)۔\n"
    f"(تقریباً {_MAX_FATWA_CANDIDATES} سے کم؛ پہلے کلیدی الفاظ/مکمل جملے سے"
    " ترتیب دیے گئے)\n"
    "You are a **semantic re-ranker** — use each line's *question* text.\n"
    "Do **not** pick a fatwa about a *different* fiqh issue just because a word"
    " overlaps. Wrong-topic matches must be ranked last or omitted.\n"
    "{titles}\n\n"
    "Task: **Rank the best {top_n} fatwas** for this **exact** user masʾala (same"
    " hukm / same issue). **You MUST return f-ids only from the list.** "
    'Return STRICT JSON: {{"ranked_fatwa_ids": ["f0007", "f0012"], '
    '"confidence": "high|medium|low"}}. '
    "If every candidate is off-topic, return {[]} — do not force a wrong result."
)


def _gather_fatwa_candidates(
    topic_nodes: list[dict],
    query_keywords: list[str] | None = None,
    user_raw_for_phrase: str = "",
) -> tuple[list[dict], dict[str, str]]:
    """Walk all leaves and return the top ``_MAX_FATWA_CANDIDATES`` by
    lexical match: when enabled, use question+answer from the flat
    lookup, not the tree title alone, plus expanded search terms.
    """
    settings = get_settings()
    use_lookup = bool(settings.pageindex_lookup_scoring)
    lookup = _load_lookup() if use_lookup else None

    keywords = [k.lower() for k in (query_keywords or []) if len(k) > 1]
    scored: list[tuple[int, str, str]] = []  # (score, title, composite_id)
    for topic in topic_nodes:
        for leaf in topic.get("nodes", []) or []:
            cid = leaf.get("fatwa_id")
            if not cid:
                continue
            title = (leaf.get("title") or "")[:500]
            if use_lookup and lookup and cid in lookup:
                rec = lookup[cid]
                hay = _lookup_haystack_for_scoring(rec, title)
                title_for_rank = _best_question_line(rec, title)
                qline = (rec.get("question_text") or "")
                qline_l = qline.lower()
                qtext_l = (rec.get("query_text") or "").lower()
            else:
                hay = title.lower()
                title_for_rank = title
                qline_l = ""
                qtext_l = ""

            if keywords:
                score = 0
                for k in keywords:
                    if k in qline_l:
                        score += 4
                    elif k in qtext_l:
                        score += 3
                    elif k in hay:
                        score += 2
                # Longer tokens are usually more specific in fiqh queries.
                score += sum(1 for k in keywords if len(k) >= 5 and k in hay)
            else:
                score = 0
            score += _raw_phrase_substring_bonus(hay, user_raw_for_phrase)
            if use_lookup and lookup and cid in lookup and _is_boilerplate_question_line(qline):
                score -= 2
            scored.append((score, title_for_rank, cid))

    if not scored:
        return [], {}

    scored.sort(key=lambda x: (-x[0], x[1]))

    # If every candidate has zero lexical support, avoid forcing a random
    # off-topic branch into Step B.
    best_score = scored[0][0]
    if keywords and best_score <= 0:
        return [], {}

    if best_score > 0:
        floor = max(1, int(best_score * 0.25))
        pruned = [row for row in scored if row[0] >= floor]
        if len(pruned) >= min(24, _MAX_FATWA_CANDIDATES):
            scored = pruned

    top = scored[:_MAX_FATWA_CANDIDATES]

    candidates: list[dict] = []
    fid_to_cid: dict[str, str] = {}
    for i, (_score, title, cid) in enumerate(top, 1):
        fid = f"f{i:04d}"
        fid_to_cid[fid] = cid
        candidates.append({"fid": fid, "title": title})
    return candidates, fid_to_cid


def _pick_fatwas(
    school_id: str,
    core_question: str,
    topic_nodes: list[dict],
    query_keywords: list[str] | None = None,
    top_n: int = _DEFAULT_TOP_N_PER_SCHOOL,
    user_raw: str = "",
) -> list[str]:
    """LLM re-rank pass over the pre-filtered candidates.

    Returns up to ``top_n`` composite fatwa ids ordered from most
    to least relevant (as judged by the LLM acting on top of the
    keyword-overlap pre-filter). Empty list on total failure.
    """
    candidates, fid_to_cid = _gather_fatwa_candidates(
        topic_nodes,
        query_keywords=query_keywords,
        user_raw_for_phrase=(user_raw or core_question)[:2000],
    )
    if not candidates:
        return []

    lookup = _load_lookup()
    lines: list[str] = []
    for c in candidates:
        fid = c["fid"]
        cid = fid_to_cid.get(fid) or ""
        rec = (lookup or {}).get(cid) if cid else None
        qv = c.get("title", "")
        if rec:
            qv = _best_question_line(rec, c.get("title", ""))
        qv = (qv or "—")[:300].replace("\n", " ")
        lines.append(f"[{fid}] {qv}")

    titles_blob = "\n".join(lines)
    settings = get_settings()
    user_msg = _PICK_FATWA_USER.format(
        raw=(user_raw or core_question)[:2000],
        q=core_question,
        school_label=_SCHOOL_LABEL.get(school_id, school_id),
        titles=titles_blob,
        top_n=top_n,
    )
    try:
        comp = _client().chat.completions.create(
            model=settings.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
            timeout=25,
            messages=[
                {"role": "system", "content": _PICK_FATWA_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
        )
        parsed = json.loads(comp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("[%s] Step B LLM call failed: %s", school_id, exc)
        parsed = {}

    ranked: list[str] = []
    seen: set[str] = set()
    for fid in (parsed.get("ranked_fatwa_ids") or []):
        if fid in fid_to_cid and fid not in seen:
            ranked.append(fid_to_cid[fid])
            seen.add(fid)
            if len(ranked) >= top_n:
                break

    # Back-compat with older single-pick shape in case it shows up
    if not ranked:
        legacy_primary = parsed.get("primary_fatwa_id")
        if legacy_primary and legacy_primary in fid_to_cid:
            ranked.append(fid_to_cid[legacy_primary])
            seen.add(legacy_primary)
        for alt in (parsed.get("alternate_fatwa_ids") or []):
            if alt in fid_to_cid and alt not in seen:
                ranked.append(fid_to_cid[alt])
                seen.add(alt)
                if len(ranked) >= top_n:
                    break

    if ranked:
        return ranked

    # The model may explicitly say nothing matches — respect that.  The
    # old "keyword fallback" kept returning the top substring hits even
    # when the whole list was the wrong *branch*, which is what the user
    # saw as irrelevant fatawa.
    rids = parsed.get("ranked_fatwa_ids")
    if isinstance(rids, list) and not rids:
        return []

    # Network / shape errors only: salvage first hits if the LLM hiccuped
    if query_keywords and candidates and rids is None:
        logger.warning(
            "[%s] Step B parse missing ranked_fatwa_ids (parsed=%s) — last-resort"
            " keyword top-%d (may be weak if tree branch was wrong)",
            school_id, list(parsed.keys()) if isinstance(parsed, dict) else "?", top_n,
        )
        return [fid_to_cid[c["fid"]] for c in candidates[:top_n]]
    return []


_REFINER_SYS = (
    "You are an Urdu fiqh expert. Return ONLY JSON. No commentary."
)
_REFINER_USER = (
    "User (original phrasing): {raw}\n\n"
    "Standard line: {q}\n\n"
    "Below are {n} fatawa candidates in **current** order. Each has an id"
    " and the masʾala (question) text.\n"
    "{blocks}\n\n"
    "Task: output a JSON object with key \"order\" — a list of the **ids**"
    " 1 to {n} in **best-first** order for answering the user (most relevant"
    " first). Every id 1..{n} must appear exactly once. "
    'Return: {{"order": [1, 3, 2, ...]}}'
)


def _refine_cid_order(
    school_id: str,
    user_q: str,
    cids: list[str],
    top_n: int,
    user_raw: str = "",
) -> list[str]:
    """Second LLM pass: re-order a short list of composite ids by meaning."""
    s = get_settings()
    if not s.pageindex_refiner_enabled or len(cids) <= 1:
        return cids[:top_n]
    n = min(len(cids), s.pageindex_refiner_max)
    cids = cids[:n]
    lookup = _load_lookup()
    blocks: list[str] = []
    for i, cid in enumerate(cids, 1):
        rec = lookup.get(cid) or {}
        qv = (
            (rec.get("question_text") or rec.get("query_text") or "")
            .replace("\n", " ")[:420]
        ) or "—"
        blocks.append(f"#{i}  id={i}\nQ: {qv}")

    try:
        comp = _client().chat.completions.create(
            model=s.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
            timeout=20,
            messages=[
                {"role": "system", "content": _REFINER_SYS},
                {
                    "role": "user",
                    "content": _REFINER_USER.format(
                        raw=(user_raw or user_q)[:2000],
                        q=user_q,
                        n=n,
                        blocks="\n\n".join(blocks),
                    )[:12000],
                },
            ],
        )
        data = json.loads(comp.choices[0].message.content or "{}")
        order = data.get("order")
        if not isinstance(order, list) or len(order) != n:
            raise ValueError("bad order")
        seen: set[int] = set()
        out: list[str] = []
        for x in order:
            j = int(x)
            if j < 1 or j > n or j in seen:
                raise ValueError("invalid perm")
            seen.add(j)
            out.append(cids[j - 1])
        return out[:top_n]
    except Exception as exc:
        logger.warning("[%s] refiner failed (%s); keeping Step B order", school_id, exc)
        return cids[:top_n]


# ──────────────────────────────────────────────────────────────────────────
# Relevance percentage (pure string ops, no LLM)
# ──────────────────────────────────────────────────────────────────────────

_TOK = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]{3,}")


def _relevance_pct(rank: int, total: int, keyword_score: int = 0) -> int:
    """Compute a display-friendly relevance percentage from the LLM's
    ranking position.

    The LLM re-ranker in Step B already judged which fatwa is most
    relevant. We translate rank → percentage using a decay curve:
      rank 1 → 95%, rank 2 → 80%, rank 3 → 65%, rank 4 → 50%

    This is more honest than the old keyword-overlap scorer, which
    penalised paraphrasing (a fatwa about exactly the right topic but
    using different words would score 40% despite being perfect).

    The optional ``keyword_score`` (0-100) provides a small bonus (+5)
    when the title also has strong keyword overlap, giving a tiebreaker
    nudge within the same rank tier.
    """
    if total <= 0:
        return 0
    # Base score from rank (exponential decay)
    base = max(30, 95 - (rank - 1) * 15)
    # Small keyword bonus (capped at +5) so pure rank isn't the only signal
    bonus = min(5, keyword_score // 20) if keyword_score > 0 else 0
    return min(100, base + bonus)


def _keyword_overlap_score(question_words: list[str], record: dict) -> int:
    """Old keyword overlap scorer — kept only as a secondary signal
    for ``_relevance_pct``'s bonus calculation."""
    if not question_words:
        return 0
    haystack = (
        (record.get("query_text") or "")
        + " " + (record.get("question_text") or "")
        + " " + (record.get("answer_text") or "")[:500]
    ).lower()
    if not haystack.strip():
        return 0
    hits = sum(1 for w in question_words if w.lower() in haystack)
    return round(100 * hits / max(1, len(question_words)))


def _question_words(core_question: str) -> list[str]:
    return [t for t in _TOK.findall(core_question or "") if len(t) > 2]


# ──────────────────────────────────────────────────────────────────────────
# Per-school navigation
# ──────────────────────────────────────────────────────────────────────────

def _empty_school_result(school_id: str, reason: str) -> dict:
    return {
        "school_id":     school_id,
        "school_label":  _SCHOOL_LABEL.get(school_id, school_id),
        "maslak":        _SCHOOL_MASLAK.get(school_id, ""),
        "fatwa_no":      "",
        "category":      "",
        "subtopic":      "",
        "query_text":    "",
        "question_text": "متعلقہ فتویٰ نہیں ملا",
        "answer_text":   "",
        "url":           "",
        "fatwa_id":      None,
        "relevance_pct": 0,
        "fatawa":        [],
        "navigation":    {"reason": reason},
    }


def _fatwa_card_from_record(
    cid: str, rec: dict, qwords: list[str],
    rank: int = 1, total: int = 1,
) -> dict:
    """Shape one lookup record into a card-ready fatwa dict."""
    kw_score = _keyword_overlap_score(qwords, rec)
    return {
        "fatwa_id":      cid,
        "fatwa_no":      rec.get("fatwa_no", ""),
        "category":      rec.get("category", ""),
        "subtopic":      rec.get("subtopic", ""),
        "query_text":    rec.get("query_text", ""),
        "question_text": rec.get("question_text", ""),
        "answer_text":   rec.get("answer_text", ""),
        "url":           rec.get("url", ""),
        "relevance_pct": _relevance_pct(rank, total, kw_score),
    }


def navigate_school(
    school_id: str,
    core_question: str,
    category_hint: str | None,
    keywords: list[str] | None = None,
    top_n: int = _DEFAULT_TOP_N_PER_SCHOOL,
    user_raw_query: str = "",
) -> dict:
    """Run the 2-step LLM descent for one school.

    Returns a school result dict containing:
      - ``fatawa``: list of up to ``top_n`` card-ready fatwa dicts
        (ordered by LLM re-ranking), where ``fatawa[0]`` is the primary
      - ``maslak``: short madhab name (Deobandi/Barelvi/Ahle Hadees)
      - Top-level copies of the primary fatwa's fields for backwards
        compatibility with any consumer expecting the old single-fatwa
        response shape.
    """
    school_node = _get_school_subtree(school_id)
    if not school_node:
        return _empty_school_result(school_id, "school subtree missing")

    uq = (user_raw_query or "").strip()
    tight_kw = _tight_scoring_tokens(core_question, keywords, uq)
    kw = keywords or _question_words(core_question)
    nav_trace: dict[str, Any] = {"llm_reranked": True}

    # ── Category-hint shortcut (OFF by default — see settings) ─────────
    # When ON: a single wrong ``category_hint`` used to search *all*
    # fatawa under a huge folder, then lexical scoring picked spurious
    # matches from the wrong branch.
    topic_nodes: list[dict] = []
    s_cfg = get_settings()
    if (
        s_cfg.pageindex_category_hint_shortcut
        and category_hint
        and category_hint.upper() != "OTHER"
    ):
        for cat in school_node.get("nodes", []) or []:
            if _cat_title_matches_hint(
                (cat.get("title") or ""), category_hint
            ):
                topic_nodes = cat.get("nodes") or []
                nav_trace["hint_shortcut"] = category_hint
                logger.info("[%s] Hint shortcut → %s (%d children)",
                            school_id, category_hint, len(topic_nodes))
                break

    # ── Step A: pick categories + topics/super-groups ─────────────────
    if not topic_nodes:
        pick_a = _pick_categories_and_topics(
            school_id, school_node, core_question, category_hint, user_raw=uq
        )
        nav_trace["category_node_ids"] = pick_a["category_node_ids"]
        nav_trace["topic_node_ids"] = pick_a["topic_node_ids"]
        nav_trace["reasoning_brief"] = pick_a["reasoning_brief"]
        topic_ids = pick_a["topic_node_ids"]
        if not topic_ids:
            return _empty_school_result(school_id, "no relevant topics")
        topic_nodes = [pick_a["id_index"][tid] for tid in topic_ids
                       if tid in pick_a["id_index"]]

    # ── Step A.5: drill into super-groups ─────────────────────────────
    # If any of the chosen nodes is a super-group (i.e., has subtopic
    # children rather than fatwa leaves), fire a focused LLM call to
    # narrow it to 2-3 subtopics. This replaces the old problem of
    # showing 260 subtopics in one prompt.
    resolved_topic_nodes: list[dict] = []
    for tn in topic_nodes:
        if _is_supergroup_node(tn):
            subs = _pick_subtopics_within_group(
                school_id, tn, core_question, user_raw=uq
            )
            resolved_topic_nodes.extend(subs)
            nav_trace["step_a5_drilled"] = tn.get("title", "")
        else:
            resolved_topic_nodes.append(tn)

    if not resolved_topic_nodes:
        return _empty_school_result(school_id, "no subtopics resolved")

    # ── Step B: LLM re-rank fatwa candidates ──────────────────────────
    pre_score_kw = tight_kw if tight_kw else kw
    chosen_cids = _pick_fatwas(
        school_id, core_question, resolved_topic_nodes,
        query_keywords=pre_score_kw, top_n=top_n, user_raw=uq,
    )
    if not chosen_cids:
        return _empty_school_result(school_id, "no fatwas picked")

    pre_refine = list(chosen_cids)
    chosen_cids = _refine_cid_order(
        school_id, core_question, chosen_cids, top_n, user_raw=uq
    )
    if (
        get_settings().pageindex_refiner_enabled
        and len(pre_refine) > 1
        and chosen_cids
    ):
        nav_trace["llm_refiner_pass2"] = True

    lookup = _load_lookup()
    qwords = _urdu_scoring_tokens(
        f"{(uq or '')} {(core_question or '')}", cap=24
    ) or _question_words(core_question)
    total = len(chosen_cids)
    # Dedupe by content (same fatwa cross-listed under multiple subtopics
    # gets different composite IDs but identical text).
    _seen_content: set[str] = set()
    fatawa: list[dict] = []
    for rank_idx, cid in enumerate(chosen_cids, 1):
        rec = lookup.get(cid)
        if not rec:
            continue
        _q = (rec.get("question_text") or rec.get("query_text") or "").strip()
        _a = (rec.get("answer_text") or "").strip()[:200]
        _key = " ".join((_q + " " + _a).split())
        if _key and _key in _seen_content:
            continue
        if _key:
            _seen_content.add(_key)
        fatawa.append(_fatwa_card_from_record(
            cid, rec, qwords, rank=len(fatawa) + 1, total=total
        ))

    if not fatawa:
        return _empty_school_result(school_id, "no fatwa records resolved")

    primary = fatawa[0]
    return {
        "school_id":     school_id,
        "school_label":  _SCHOOL_LABEL.get(school_id, school_id),
        "maslak":        _SCHOOL_MASLAK.get(school_id, ""),
        "fatwa_id":      primary["fatwa_id"],
        "fatwa_no":      primary["fatwa_no"],
        "category":      primary["category"],
        "subtopic":      primary["subtopic"],
        "query_text":    primary["query_text"],
        "question_text": primary["question_text"],
        "answer_text":   primary["answer_text"],
        "url":           primary["url"],
        "relevance_pct": primary["relevance_pct"],
        "fatawa":        fatawa,
        "navigation":    nav_trace,
    }


# ──────────────────────────────────────────────────────────────────────────
# Top-level orchestrator (4 schools in parallel)
# ──────────────────────────────────────────────────────────────────────────

_NAV_CACHE: dict[tuple, str] = {}
# Increment when navigation / scoring logic changes (invalidates old cache).
_NAV_CACHE_BUMP: int = 5


def _cached_navigate_school(
    school_id: str,
    core_question: str,
    category_hint: str | None,
    keywords_tuple: tuple[str, ...] = (),
    top_n: int = _DEFAULT_TOP_N_PER_SCHOOL,
    user_raw: str = "",
) -> str:
    """Cached wrapper that only caches SUCCESSFUL results.

    Failed results (empty ``fatawa`` list) are NOT cached, so a
    transient LLM timeout doesn't permanently poison the query.
    """
    key = (
        _NAV_CACHE_BUMP, school_id, core_question, category_hint,
        keywords_tuple, top_n, user_raw,
    )
    if key in _NAV_CACHE:
        return _NAV_CACHE[key]

    result = navigate_school(
        school_id, core_question, category_hint,
        keywords=list(keywords_tuple) or None,
        top_n=top_n,
        user_raw_query=user_raw,
    )
    result_json = json.dumps(result, ensure_ascii=False)

    # Only cache successful results (non-empty fatawa list)
    if result.get("fatawa"):
        if len(_NAV_CACHE) >= _NAV_CACHE_SIZE:
            # Evict oldest entry (FIFO)
            oldest = next(iter(_NAV_CACHE))
            del _NAV_CACHE[oldest]
        _NAV_CACHE[key] = result_json

    return result_json


def pageindex_search(
    core_question: str,
    *,
    category_hint: str | None = None,
    keywords: list[str] | None = None,
    user_raw_query: str = "",
    schools: list[str] | None = None,
    top_n: int = _DEFAULT_TOP_N_PER_SCHOOL,
) -> list[dict]:
    """Run navigate_school for all 4 madhabs in parallel and return the
    list of result cards (one per school, ordered as
    ``[Banuri, fatwaqa, IslamQA, urdufatwa]``).

    Each result card contains a ``fatawa`` list (up to ``top_n`` entries)
    ranked by LLM re-ranking on top of the keyword-overlap pre-filter.
    """
    schools = schools or list(ALL_SCHOOLS)
    # Eagerly preload — first call pays the cost
    _load_tree()
    _load_lookup()

    kw_tuple = tuple(keywords or ())
    urq = (user_raw_query or "").strip()

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(schools))
    ) as ex:
        futures = {
            ex.submit(
                lambda sid=sid: json.loads(
                    _cached_navigate_school(
                        sid, core_question, category_hint, kw_tuple, top_n,
                        urq,
                    )
                )
            ): sid
            for sid in schools
        }
        for fut in concurrent.futures.as_completed(futures):
            sid = futures[fut]
            try:
                results[sid] = fut.result()
            except Exception as exc:
                logger.exception("[%s] navigate failed", sid)
                results[sid] = _empty_school_result(sid, f"error: {exc}")

    return [results.get(sid) or _empty_school_result(sid, "missing")
            for sid in schools]
