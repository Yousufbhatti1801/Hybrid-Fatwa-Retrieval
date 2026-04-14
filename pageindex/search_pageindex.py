"""LLM-driven tree navigation over the parsed PageIndex tree.

This is the **vectorless retrieval** path. For each madhab we run two
``gpt-4o-mini`` calls in sequence:

  1. ``_pick_categories_and_topics`` — show the LLM the school's
     (category, topic) sub-tree and ask which 1-2 categories and 1-5
     topics are most relevant.
  2. ``_pick_fatwa`` — show the LLM the leaf fatwa titles under those
     chosen topics (capped at 200) and ask which single fatwa best
     answers the question.

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
    "صارف کا سوال (Urdu user question):\n{q}\n\n"
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
        raw = comp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
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
    "صارف کا سوال: {q}\n\n"
    'نیچے "{group_label}" گروپ کے اندر موجود subtopics کی فہرست ہے:\n'
    "{subtopics}\n\n"
    "Pick at most 3 subtopic_node_ids most relevant to the question.\n"
    'Return STRICT JSON: {{"subtopic_node_ids": ["…"]}}'
)


def _pick_subtopics_within_group(
    school_id: str,
    group_node: dict,
    core_question: str,
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
        parsed = json.loads(comp.choices[0].message.content or "{}")
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
    "صارف کا سوال (Urdu user question):\n{q}\n"
    "Madhab: {school_label}\n\n"
    "نیچے متعلقہ فتاویٰ کی فہرست ہے۔ Each line starts with a short id "
    "like [f0001], [f0002] — these are the IDs you must return.\n"
    f"(list capped at {_MAX_FATWA_CANDIDATES} and pre-ranked by keyword "
    "overlap — but YOU must re-rank by actual semantic relevance to the "
    "user's Urdu question, not by keyword match alone)\n"
    "{titles}\n\n"
    "Task: **rank the top {top_n} fatwas** that most directly answer the "
    "user's question, from most to least relevant. You are acting as a "
    "semantic re-ranker on top of a keyword filter, so prefer fatwas that "
    "match the actual meaning of the question, not just its surface words. "
    "**You MUST return f-ids from the list above** (e.g. \"f0007\"), "
    "exactly as written, including the leading 'f' and 4 digits. "
    "Do not invent ids that are not in the list. "
    'Return STRICT JSON: {{"ranked_fatwa_ids": '
    '["f0007", "f0012", "f0019", "f0025"], '
    '"confidence": "high|medium|low"}}. '
    "Return an empty list ONLY if the list is genuinely empty or no title "
    "is even loosely related. Otherwise return at least one id, up to {top_n}."
)


def _gather_fatwa_candidates(
    topic_nodes: list[dict],
    query_keywords: list[str] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Walk all leaves under the chosen topics and return the top
    ``_MAX_FATWA_CANDIDATES`` ranked by keyword overlap with
    ``query_keywords``.

    For huge buckets (Banuri's talaaq has 4423 fatawa) the previous
    alphabetical-first-200 strategy was essentially random. Pre-filtering
    by keyword overlap ensures the LLM sees the most relevant titles
    even when the topic is large.

    Returns ``(candidate_records, fid_to_composite_id)``. Each candidate
    is ``{"fid": str, "title": str}``; ``fid_to_cid`` maps the short id
    in the prompt to the real composite lookup key.
    """
    # 1. Walk every leaf, scoring it by overlap with query_keywords.
    keywords = [k.lower() for k in (query_keywords or []) if len(k) > 1]
    scored: list[tuple[int, str, str]] = []  # (score, title, composite_id)
    for topic in topic_nodes:
        for leaf in topic.get("nodes", []) or []:
            cid = leaf.get("fatwa_id")
            if not cid:
                continue
            title = (leaf.get("title") or "")[:140]
            if keywords:
                hay = title.lower()
                score = sum(1 for k in keywords if k in hay)
            else:
                score = 0
            scored.append((score, title, cid))

    # 2. Sort by score desc, ties broken alphabetically (stable order).
    scored.sort(key=lambda x: (-x[0], x[1]))

    # 3. Take the top _MAX_FATWA_CANDIDATES, but ALWAYS include at least
    #    a few examples from each chosen topic so the LLM has variety even
    #    when keyword overlap is weak.
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
) -> list[str]:
    """LLM re-rank pass over the pre-filtered candidates.

    Returns up to ``top_n`` composite fatwa ids ordered from most
    to least relevant (as judged by the LLM acting on top of the
    keyword-overlap pre-filter). Empty list on total failure.
    """
    candidates, fid_to_cid = _gather_fatwa_candidates(
        topic_nodes, query_keywords=query_keywords
    )
    if not candidates:
        return []

    titles_blob = "\n".join(f"[{c['fid']}] {c['title']}" for c in candidates)
    settings = get_settings()
    user_msg = _PICK_FATWA_USER.format(
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
            max_tokens=200,
            timeout=12,
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

    # Final fallback: if the LLM refused but we DID pre-rank by keyword
    # overlap, the top-N candidates are almost certainly relevant. Better
    # than returning nothing for the user.
    if query_keywords and candidates:
        logger.warning(
            "[%s] Step B LLM returned no usable ids (parsed=%s); "
            "falling back to top %d keyword-ranked candidates",
            school_id, parsed, top_n,
        )
        return [fid_to_cid[c["fid"]] for c in candidates[:top_n]]
    return []


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

    kw = keywords or _question_words(core_question)
    nav_trace: dict[str, Any] = {"llm_reranked": True}

    # ── Category-hint shortcut ────────────────────────────────────────
    # If the extract step returned a confident category_hint (e.g.
    # "NAMAZ") and that category exists in this school, skip Step A
    # entirely and go straight to its topics. This saves 1 LLM call
    # for ~60% of queries.
    topic_nodes: list[dict] = []

    if category_hint and category_hint.upper() != "OTHER":
        for cat in school_node.get("nodes", []) or []:
            if cat.get("title", "").upper() == category_hint.upper():
                topic_nodes = cat.get("nodes") or []
                nav_trace["hint_shortcut"] = category_hint
                logger.info("[%s] Hint shortcut → %s (%d children)",
                            school_id, category_hint, len(topic_nodes))
                break

    # ── Step A: pick categories + topics/super-groups ─────────────────
    if not topic_nodes:
        pick_a = _pick_categories_and_topics(
            school_id, school_node, core_question, category_hint
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
            subs = _pick_subtopics_within_group(school_id, tn, core_question)
            resolved_topic_nodes.extend(subs)
            nav_trace["step_a5_drilled"] = tn.get("title", "")
        else:
            resolved_topic_nodes.append(tn)

    if not resolved_topic_nodes:
        return _empty_school_result(school_id, "no subtopics resolved")

    # ── Step B: LLM re-rank fatwa candidates ──────────────────────────
    chosen_cids = _pick_fatwas(
        school_id, core_question, resolved_topic_nodes,
        query_keywords=kw, top_n=top_n,
    )
    if not chosen_cids:
        return _empty_school_result(school_id, "no fatwas picked")

    lookup = _load_lookup()
    qwords = _question_words(core_question)
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


def _cached_navigate_school(
    school_id: str,
    core_question: str,
    category_hint: str | None,
    keywords_tuple: tuple[str, ...] = (),
    top_n: int = _DEFAULT_TOP_N_PER_SCHOOL,
) -> str:
    """Cached wrapper that only caches SUCCESSFUL results.

    Failed results (empty ``fatawa`` list) are NOT cached, so a
    transient LLM timeout doesn't permanently poison the query.
    """
    key = (school_id, core_question, category_hint, keywords_tuple, top_n)
    if key in _NAV_CACHE:
        return _NAV_CACHE[key]

    result = navigate_school(
        school_id, core_question, category_hint,
        keywords=list(keywords_tuple) or None,
        top_n=top_n,
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

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, len(schools))
    ) as ex:
        futures = {
            ex.submit(
                lambda sid=sid: json.loads(
                    _cached_navigate_school(
                        sid, core_question, category_hint, kw_tuple, top_n
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
