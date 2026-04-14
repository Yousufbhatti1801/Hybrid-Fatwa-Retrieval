"""Tree-guided keyword retrieval over the fatawa corpus.

Two-stage architecture:
  1. **Tree navigation** (keyword-based, no LLM) — score tree nodes by
     keyword overlap with the query to narrow the search to the most
     relevant category → subtopic → sub-group branches.
  2. **Inverted index search** — within the narrowed branch, find the
     top candidates via weighted term lookup (title 3× > question 2× >
     answer 1×).
  3. **LLM re-ranking** (optional) — call gpt-4o-mini to re-order the
     candidates by semantic relevance.

Built once at startup from ``fatawa_lookup.json`` + the PageIndex tree.
Sub-second for steps 1-2; ~2-3s with LLM re-ranking.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent
LOOKUP_PATH = _PKG_DIR / "data" / "fatawa_lookup.json"
DOC_ID_PATH = _PKG_DIR / "index" / "doc_id.txt"

# Tokenizer (Urdu + Arabic + Latin, 2+ chars)
_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFFa-zA-Z0-9]{2,}")

_STOP_WORDS = frozenset({
    "کیا", "کی", "کا", "کے", "ہے", "ہیں", "ہو", "ہوتا", "ہوتی",
    "میں", "پر", "سے", "اور", "کب", "کس", "کیوں", "نہیں", "اگر",
    "یہ", "وہ", "جو", "تو", "ایک", "دو", "تین", "بھی", "نے",
    "ان", "اس", "جب", "پھر", "ابھی", "اب", "کوئی", "کچھ",
    "ہوں", "تھا", "تھی", "تھے", "گا", "گی", "گے",
    "the", "is", "of", "and", "in", "to", "for", "it", "on",
})

# Islamic term expansion for better recall
_TERM_EXPANSIONS: dict[str, list[str]] = {
    "namaz": ["نماز", "صلاۃ", "صلوۃ"],
    "salah": ["نماز", "صلاۃ"],
    "wudu": ["وضو", "وضوء"],
    "wazu": ["وضو"],
    "zakat": ["زکوٰۃ", "زکاۃ", "زکوۃ"],
    "roza": ["روزہ", "صوم"],
    "fast": ["روزہ", "صوم"],
    "hajj": ["حج"],
    "nikah": ["نکاح"],
    "talaq": ["طلاق"],
    "divorce": ["طلاق"],
    "talaaq": ["طلاق"],
    "bitcoin": ["بٹ کوائن", "کرپٹو"],
    "crypto": ["کرپٹو", "ڈیجیٹل کرنسی"],
    "interest": ["سود"],
    "loan": ["قرض", "لون"],
    "bank": ["بینک", "بنک"],
    "inheritance": ["وراثت", "میراث"],
    "prayer": ["نماز", "دعا"],
    "quran": ["قرآن"],
    "hadith": ["حدیث"],
    "halal": ["حلال"],
    "haram": ["حرام"],
    "sajda": ["سجدہ"],
    "sahw": ["سہو"],
    "witr": ["وتر"],
    "qaza": ["قضا", "قضاء"],
    "tayammum": ["تیمم"],
    "ghusl": ["غسل"],
    "janaza": ["جنازہ"],
    "iddah": ["عدت"],
    "khula": ["خلع"],
    "mahr": ["مہر"],
}


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]


def _expand_query_terms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for tok in tokens:
        if tok in _TERM_EXPANSIONS:
            expanded.extend(_TERM_EXPANSIONS[tok])
        for eng, urdu_list in _TERM_EXPANSIONS.items():
            if tok in [u.lower() for u in urdu_list]:
                expanded.append(eng)
                expanded.extend(u.lower() for u in urdu_list if u.lower() != tok)
    return list(set(expanded))


# ──────────────────────────────────────────────────────────────────────────
# Tree node for keyword navigation
# ──────────────────────────────────────────────────────────────────────────

class TreeNode:
    """Lightweight tree node for keyword navigation."""

    __slots__ = ("title", "node_id", "keywords", "children", "fatwa_ids")

    def __init__(self, title: str, node_id: str) -> None:
        self.title = title
        self.node_id = node_id
        self.keywords: set[str] = set()  # terms from title + child titles
        self.children: list[TreeNode] = []
        self.fatwa_ids: list[str] = []   # only on leaf-parent nodes

    def keyword_score(self, query_terms: list[str]) -> int:
        """Count how many query terms appear in this node's keywords."""
        return sum(1 for t in query_terms if t in self.keywords)

    def all_fatwa_ids(self) -> list[str]:
        """Recursively collect all fatwa_ids under this node."""
        ids = list(self.fatwa_ids)
        for child in self.children:
            ids.extend(child.all_fatwa_ids())
        return ids


# ──────────────────────────────────────────────────────────────────────────
# Main index class
# ──────────────────────────────────────────────────────────────────────────

class RawFatwasIndex:
    """Tree-guided inverted index over the fatawa corpus."""

    def __init__(self) -> None:
        # Inverted indexes (flat, across all fatwas)
        self._title_index: dict[str, set[str]] = defaultdict(set)
        self._question_index: dict[str, set[str]] = defaultdict(set)
        self._answer_index: dict[str, set[str]] = defaultdict(set)
        self._store: dict[str, dict] = {}

        # Tree structure per school (for keyword navigation)
        self._school_trees: dict[str, TreeNode] = {}

        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        return len(self._store)

    def build(self, lookup_path: Path = LOOKUP_PATH) -> None:
        """Build both the inverted index and the navigation tree."""
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"Lookup not found at {lookup_path}. "
                "Run `python -m pageindex.convert` first."
            )

        logger.info("Building raw fatwas index from %s …", lookup_path)
        lookup = json.loads(lookup_path.read_text(encoding="utf-8"))

        for fid, rec in lookup.items():
            self._store[fid] = rec

            title_text = (rec.get("query_text") or "") + " " + (rec.get("fatwa_no") or "")
            for term in _tokenize(title_text):
                self._title_index[term].add(fid)

            for term in _tokenize(rec.get("question_text") or ""):
                self._question_index[term].add(fid)

            for term in _tokenize((rec.get("answer_text") or "")[:500]):
                self._answer_index[term].add(fid)

        # Build navigation tree from the PageIndex tree JSON
        self._build_nav_tree()

        self._loaded = True
        logger.info(
            "Raw fatwas index built: %d fatwas, %d terms, %d school trees",
            len(self._store),
            len(self._title_index) + len(self._question_index) + len(self._answer_index),
            len(self._school_trees),
        )

    def _build_nav_tree(self) -> None:
        """Build lightweight navigation trees from the PageIndex tree JSON."""
        if not DOC_ID_PATH.exists():
            logger.warning("No PageIndex tree found — navigation disabled")
            return

        doc_id = DOC_ID_PATH.read_text(encoding="utf-8").strip()
        tree_path = _PKG_DIR / "index" / f"{doc_id}.json"
        if not tree_path.exists():
            logger.warning("Tree JSON not found at %s", tree_path)
            return

        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        root_nodes = tree.get("structure", [])
        if not root_nodes:
            return

        # The tree has: root → schools → categories → topics → fatwas
        # We need to find school nodes (depth 1 or 2 depending on whether
        # there's a "Fatawa Index" root)
        from pageindex.search_pageindex import ALL_SCHOOLS

        candidates = list(root_nodes)
        for node in root_nodes:
            candidates.extend(node.get("nodes") or [])

        for raw_node in candidates:
            title = raw_node.get("title", "")
            for sid in ALL_SCHOOLS:
                if sid.lower() in title.lower():
                    school_tree = self._convert_tree_node(raw_node)
                    self._school_trees[sid] = school_tree
                    n_fatwas = len(school_tree.all_fatwa_ids())
                    logger.info("  Nav tree: %s → %d categories, %d total fatwas",
                                sid, len(school_tree.children), n_fatwas)
                    break

    def _convert_tree_node(self, raw: dict) -> TreeNode:
        """Recursively convert a PageIndex tree dict into a TreeNode."""
        node = TreeNode(
            title=raw.get("title", ""),
            node_id=raw.get("node_id", ""),
        )

        # Add title keywords
        node.keywords.update(_tokenize(node.title))

        children = raw.get("nodes") or []
        for child_raw in children:
            if child_raw.get("fatwa_id"):
                # This is a leaf (fatwa)
                node.fatwa_ids.append(child_raw["fatwa_id"])
                # Add leaf title keywords to parent for better matching
                node.keywords.update(_tokenize(child_raw.get("title", "")))
            else:
                # This is an intermediate node
                child = self._convert_tree_node(child_raw)
                node.children.append(child)
                # Propagate child keywords up (so category nodes have
                # keywords from their subtopic titles)
                node.keywords.update(child.keywords)

        return node

    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Keyword tree navigation (no LLM)
    # ──────────────────────────────────────────────────────────────────

    def _navigate_tree_for_school(
        self,
        school_id: str,
        query_terms: list[str],
        max_categories: int = 3,
        max_topics: int = 6,
    ) -> list[str]:
        """Navigate the tree using keyword scoring to find the most
        relevant fatwa_ids for this school.

        Returns a focused set of fatwa_ids (typically 200-2000 instead
        of the full 60k+ per school).
        """
        school_tree = self._school_trees.get(school_id)
        if not school_tree:
            # No tree available — fall back to all fatwas for this school
            return [fid for fid, rec in self._store.items()
                    if rec.get("school_id") == school_id]

        # Level 1: Score categories
        categories = school_tree.children
        if not categories:
            return school_tree.all_fatwa_ids()

        cat_scores = [(cat, cat.keyword_score(query_terms)) for cat in categories]
        cat_scores.sort(key=lambda x: -x[1])
        # Take top categories (at least 1, even if score is 0)
        top_cats = [c for c, s in cat_scores[:max_categories] if s > 0]
        if not top_cats:
            top_cats = [cat_scores[0][0]]  # fallback to highest-scoring

        # Level 2: Score topics/subtopics within chosen categories
        all_topics: list[tuple[TreeNode, int]] = []
        for cat in top_cats:
            for topic in cat.children:
                score = topic.keyword_score(query_terms)
                all_topics.append((topic, score))

        all_topics.sort(key=lambda x: -x[1])
        top_topics = [t for t, s in all_topics[:max_topics] if s > 0]
        if not top_topics:
            # No keyword match at topic level — use all topics from top categories
            for cat in top_cats:
                top_topics.extend(cat.children[:3])

        # Level 3: If any topic has sub-children (super-groups or sub-groups),
        # score those too and pick the best
        fatwa_ids: list[str] = []
        for topic in top_topics:
            if topic.children:
                # Has sub-nodes — score and pick the best
                sub_scores = [(sub, sub.keyword_score(query_terms))
                              for sub in topic.children]
                sub_scores.sort(key=lambda x: -x[1])
                # Take top 3 sub-nodes
                best_subs = [s for s, sc in sub_scores[:3] if sc > 0]
                if not best_subs:
                    best_subs = [sub_scores[0][0]] if sub_scores else []
                for sub in best_subs:
                    fatwa_ids.extend(sub.all_fatwa_ids())
            else:
                # Leaf-parent — get its fatwa_ids directly
                fatwa_ids.extend(topic.fatwa_ids)

        return fatwa_ids

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Inverted index search within narrowed set
    # ──────────────────────────────────────────────────────────────────

    def search_within(
        self,
        query: str,
        allowed_ids: set[str] | None = None,
        *,
        top_n: int = 30,
    ) -> list[dict]:
        """Search the inverted index, optionally restricted to ``allowed_ids``.

        Scoring: 3 × title + 2 × question + 1 × answer hits.
        """
        if not self._loaded:
            raise RuntimeError("Index not built.")

        raw_tokens = _tokenize(query)
        tokens = _expand_query_terms(raw_tokens)
        if not tokens:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in tokens:
            for fid in self._title_index.get(term, ()):
                if allowed_ids is None or fid in allowed_ids:
                    scores[fid] += 3.0
            for fid in self._question_index.get(term, ()):
                if allowed_ids is None or fid in allowed_ids:
                    scores[fid] += 2.0
            for fid in self._answer_index.get(term, ()):
                if allowed_ids is None or fid in allowed_ids:
                    scores[fid] += 1.0

        if not scores:
            return []

        max_possible = len(tokens) * 6.0
        # Sort by score desc — take MORE than top_n so we have headroom
        # after content-based deduplication.
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n * 3]

        # Dedupe by content (same fatwa can exist under multiple composite IDs)
        seen_content: set[str] = set()
        results: list[dict] = []
        for fid, score in ranked:
            if len(results) >= top_n:
                break
            rec = self._store[fid]
            # Content-based dedupe key
            _q = (rec.get("question_text") or rec.get("query_text") or "").strip()
            _a = (rec.get("answer_text") or "").strip()[:200]
            _key = " ".join((_q + " " + _a).split())
            if _key and _key in seen_content:
                continue
            if _key:
                seen_content.add(_key)
            results.append({
                "fatwa_id":      fid,
                "fatwa_no":      rec.get("fatwa_no", ""),
                "school_id":     rec.get("school_id", ""),
                "school_label":  rec.get("school_label", ""),
                "madhab":        rec.get("school", ""),
                "category":      rec.get("category", ""),
                "subtopic":      rec.get("subtopic", ""),
                "query_text":    rec.get("query_text", ""),
                "question_text": rec.get("question_text", ""),
                "answer_text":   rec.get("answer_text", ""),
                "url":           rec.get("url", ""),
                "score":         min(100, round(100 * score / max_possible)),
            })

        return results

    # ──────────────────────────────────────────────────────────────────
    # Combined: tree navigate → keyword search → LLM re-rank
    # ──────────────────────────────────────────────────────────────────

    def search_by_school(
        self,
        query: str,
        *,
        top_n_per_school: int = 4,
        schools: list[str] | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """Full pipeline: tree navigation → inverted index → LLM re-rank.

        1. Keyword-navigate the tree to narrow to ~200-2000 fatwa_ids per school
        2. Run inverted index search within that narrowed set
        3. LLM re-ranks top candidates (1 call per school, parallel)
        """
        from pageindex.search_pageindex import (
            ALL_SCHOOLS, _SCHOOL_LABEL, _SCHOOL_MASLAK
        )

        schools = schools or ALL_SCHOOLS
        raw_tokens = _tokenize(query)
        query_terms = _expand_query_terms(raw_tokens)

        # Stage 1+2: Per school, navigate tree then search
        by_school: dict[str, list[dict]] = {}
        for sid in schools:
            # Tree navigation narrows the search space
            focused_ids = self._navigate_tree_for_school(sid, query_terms)
            focused_set = set(focused_ids) if focused_ids else None
            n_focused = len(focused_set) if focused_set else 0

            # Inverted index search within narrowed set
            candidates = self.search_within(
                query, allowed_ids=focused_set, top_n=30
            )
            school_candidates = [c for c in candidates if c["school_id"] == sid]
            by_school[sid] = school_candidates

            logger.debug("[%s] Tree narrowed to %d fatwas → %d keyword hits",
                         sid, n_focused, len(school_candidates))

        # Stage 3: LLM re-ranking (parallel across schools)
        if rerank:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(schools)
            ) as ex:
                futures = {}
                for sid in schools:
                    cands = by_school.get(sid, [])
                    if len(cands) > 1:
                        futures[ex.submit(
                            _llm_rerank_candidates,
                            query, sid, cands, top_n_per_school,
                        )] = sid

                for fut in concurrent.futures.as_completed(futures):
                    sid = futures[fut]
                    try:
                        by_school[sid] = fut.result()
                    except Exception as exc:
                        logger.warning("[%s] Re-rank failed: %s", sid, exc)
                        by_school[sid] = by_school[sid][:top_n_per_school]
        else:
            for sid in schools:
                by_school[sid] = by_school[sid][:top_n_per_school]

        # Build output (same shape as PageIndex mode)
        output: list[dict] = []
        for sid in schools:
            fatawa = by_school.get(sid, [])
            if not fatawa:
                output.append({
                    "school_id":     sid,
                    "school_label":  _SCHOOL_LABEL.get(sid, sid),
                    "maslak":        _SCHOOL_MASLAK.get(sid, ""),
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
                    "navigation":    {"method": "tree-keyword-index"},
                })
                continue

            for i, f in enumerate(fatawa):
                f["relevance_pct"] = max(30, 95 - i * 15)

            primary = fatawa[0]
            output.append({
                "school_id":     sid,
                "school_label":  _SCHOOL_LABEL.get(sid, sid),
                "maslak":        _SCHOOL_MASLAK.get(sid, ""),
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
                "navigation":    {
                    "method": "tree-keyword-index + llm-rerank" if rerank
                              else "tree-keyword-index",
                },
            })

        return output


# ──────────────────────────────────────────────────────────────────────────
# LLM re-ranker (1 call per school, same as before)
# ──────────────────────────────────────────────────────────────────────────

_RERANK_SYSTEM = (
    "You are an Urdu Islamic fiqh research assistant. "
    "Always reply with strict JSON, no commentary."
)

_RERANK_USER = (
    "صارف کا سوال (Urdu user question):\n{q}\n"
    "Madhab: {school_label}\n\n"
    "نیچے keyword search سے ملنے والے فتاویٰ کی فہرست ہے۔\n"
    "Each line: [id] title\n"
    "{titles}\n\n"
    "Task: **re-rank the top {top_n}** fatwas by SEMANTIC relevance "
    "to the user's question (not just keyword match). "
    "Return STRICT JSON: {{\"ranked_ids\": [\"r001\", \"r002\", ...]}}"
)


def _llm_rerank_candidates(
    query: str,
    school_id: str,
    candidates: list[dict],
    top_n: int,
) -> list[dict]:
    """Call gpt-4o-mini to re-rank keyword-retrieved candidates."""
    from openai import OpenAI
    from src.config import get_settings
    from pageindex.search_pageindex import _SCHOOL_LABEL

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    id_map: dict[str, dict] = {}
    lines: list[str] = []
    for i, c in enumerate(candidates[:50]):
        rid = f"r{i+1:03d}"
        id_map[rid] = c
        title = (c.get("query_text") or c.get("fatwa_no") or "")[:120]
        lines.append(f"[{rid}] {title}")

    try:
        comp = client.chat.completions.create(
            model=settings.chat_model,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
            timeout=10,
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": _RERANK_USER.format(
                    q=query,
                    school_label=_SCHOOL_LABEL.get(school_id, school_id),
                    titles="\n".join(lines),
                    top_n=top_n,
                )},
            ],
        )
        parsed = json.loads(comp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning("[%s] Re-rank LLM failed: %s", school_id, exc)
        return candidates[:top_n]

    ranked_ids = parsed.get("ranked_ids") or []
    result: list[dict] = []
    seen: set[str] = set()
    for rid in ranked_ids:
        if rid in id_map and rid not in seen:
            result.append(id_map[rid])
            seen.add(rid)
            if len(result) >= top_n:
                break

    if len(result) < top_n:
        for c in candidates:
            if c["fatwa_id"] not in {r["fatwa_id"] for r in result}:
                result.append(c)
                if len(result) >= top_n:
                    break

    return result
