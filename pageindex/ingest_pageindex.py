"""Parse fatawa_index.md into a PageIndex tree JSON.

Wraps VectifyAI's ``pageindex.page_index_md.md_to_tree`` with
``if_add_node_summary='no'`` and ``if_add_doc_description='no'`` so the
call is **pure markdown parsing** — no LLM, no cost. The resulting tree
is persisted to ``pageindex/index/{doc_id}.json`` and a tiny ``_meta.json``
+ ``doc_id.txt`` are kept beside it for the runtime loader.

Run once after ``pageindex/convert.py``::

    python -m pageindex.ingest_pageindex

The runtime ``search_pageindex`` only needs the tree to know which fatwa
``id`` (carried in ``<!-- id:... -->`` HTML comments emitted by
``convert.py``) lives under which (school, category, subtopic) path.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from pathlib import Path

# Make src.* importable when running as `python -m pageindex.ingest_pageindex`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_MD_PATH = _PKG_DIR / "data" / "fatawa_index.md"
INDEX_DIR       = _PKG_DIR / "index"
META_PATH       = INDEX_DIR / "_meta.json"
DOC_ID_PATH     = INDEX_DIR / "doc_id.txt"


# ──────────────────────────────────────────────────────────────────────────
# Composite-id recovery from <!-- id:... --> markers
# ──────────────────────────────────────────────────────────────────────────

# convert.py emits a single-line HTML comment immediately after each
# fatwa heading. The VectifyAI parser preserves the body text of leaf
# nodes, so we can recover the composite ID by regex on `node['text']`.
_ID_COMMENT = re.compile(r"<!--\s*id:([^\s>-][^>]*?)\s*-->")


def _extract_composite_id(node_text: str | None) -> str | None:
    if not node_text:
        return None
    m = _ID_COMMENT.search(node_text)
    return m.group(1).strip() if m else None


def _annotate_tree_with_ids(structure: list) -> int:
    """Walk the tree and copy the composite id from each leaf's body
    text into a top-level ``fatwa_id`` field. Returns the count.

    Mutates ``structure`` in place. Also strips the verbose ``text``
    field from leaves to keep the persisted JSON small.
    """
    n = 0

    def _walk(nodes: list) -> None:
        nonlocal n
        for node in nodes:
            children = node.get("nodes")
            if children:
                _walk(children)
            else:
                cid = _extract_composite_id(node.get("text"))
                if cid:
                    node["fatwa_id"] = cid
                    n += 1
                # We don't need the full body text in the persisted tree —
                # the runtime looks fatwas up by id from fatawa_lookup.json.
                node.pop("text", None)
                node.pop("line_num", None)

    _walk(structure)
    return n


# ──────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────

def build_tree_from_md(
    md_path: Path = DEFAULT_MD_PATH,
    index_dir: Path = INDEX_DIR,
) -> str:
    """Parse ``md_path`` into a tree and persist it. Returns the doc_id.

    Uses VectifyAI's ``md_to_tree`` with summaries disabled, so this is
    pure markdown parsing. No LLM call.
    """
    if not md_path.exists():
        raise FileNotFoundError(
            f"Markdown index not found at {md_path}. "
            "Run `python -m pageindex.convert` first."
        )

    # Use our vendored pure-Python parser (~120 LOC, no external deps)
    # instead of VectifyAI's md_to_tree which requires litellm/pymupdf.
    from pageindex._md_parser import md_to_tree  # noqa: PLC0415

    logger.info("Parsing %s …", md_path)
    result = md_to_tree(md_path=str(md_path))

    structure = result.get("structure", [])
    n_with_ids = _annotate_tree_with_ids(structure)
    logger.info("Tree parsed: %d top-level nodes, %d annotated leaves",
                len(structure), n_with_ids)

    # Persist
    index_dir.mkdir(parents=True, exist_ok=True)
    doc_id = str(uuid.uuid4())
    tree_path = index_dir / f"{doc_id}.json"
    payload = {
        "doc_id":     doc_id,
        "doc_name":   result.get("doc_name", "fatawa_index"),
        "line_count": result.get("line_count"),
        "structure":  structure,
    }
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info("Wrote tree → %s", tree_path)

    # Meta + active doc id
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta[doc_id] = {
        "doc_name": payload["doc_name"],
        "path":     str(tree_path),
        "md_path":  str(md_path),
    }
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    DOC_ID_PATH.write_text(doc_id, encoding="utf-8")
    logger.info("Active doc_id: %s", doc_id)

    return doc_id


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="pageindex.ingest_pageindex",
        description="Parse the fatawa markdown index into a PageIndex tree JSON.",
    )
    p.add_argument("--md-path",   type=Path, default=DEFAULT_MD_PATH)
    p.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    doc_id = build_tree_from_md(md_path=args.md_path, index_dir=args.index_dir)
    print(json.dumps({"doc_id": doc_id, "index_dir": str(args.index_dir)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
