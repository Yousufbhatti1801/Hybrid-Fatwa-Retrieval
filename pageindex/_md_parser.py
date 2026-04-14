"""Vendored markdown-to-tree parser from VectifyAI/PageIndex (MIT license).

Source: https://github.com/VectifyAI/PageIndex
Commit: f2dcffc0b79a8ccaddcaa9f51e6a54e3b7e7020b (2026-04-08)
License: MIT

Only the pure-Python markdown parsing functions are included here
(no LLM calls, no litellm, no PyPDF2, no pymupdf). This avoids
pulling ~50 MB of transitive dependencies for a 120-line parser.
"""

from __future__ import annotations

import re


# ── Extract headers from markdown ────────────────────────────────────────

def extract_nodes_from_markdown(markdown_content: str):
    """Find all ``# … ######`` headings and return (node_list, lines).

    Each node in ``node_list`` is ``{"node_title": str, "line_num": int}``.
    ``lines`` is the full list of lines (split on ``\\n``).
    """
    header_pattern = r"^(#{1,6})\s+(.+)$"
    code_block_pattern = r"^```"
    node_list = []
    lines = markdown_content.split("\n")
    in_code_block = False

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(code_block_pattern, stripped):
            in_code_block = not in_code_block
            continue
        if not stripped:
            continue
        if not in_code_block:
            match = re.match(header_pattern, stripped)
            if match:
                node_list.append({
                    "node_title": match.group(2).strip(),
                    "line_num": line_num,
                })

    return node_list, lines


# ── Attach body text to each node ────────────────────────────────────────

def extract_node_text_content(node_list: list[dict], markdown_lines: list[str]):
    all_nodes = []
    for node in node_list:
        line_content = markdown_lines[node["line_num"] - 1]
        header_match = re.match(r"^(#{1,6})", line_content)
        if header_match is None:
            continue
        all_nodes.append({
            "title": node["node_title"],
            "line_num": node["line_num"],
            "level": len(header_match.group(1)),
        })

    for i, nd in enumerate(all_nodes):
        start = nd["line_num"] - 1
        end = all_nodes[i + 1]["line_num"] - 1 if i + 1 < len(all_nodes) else len(markdown_lines)
        nd["text"] = "\n".join(markdown_lines[start:end]).strip()

    return all_nodes


# ── Build hierarchical tree from flat node list ─────────────────────────

def build_tree_from_nodes(node_list: list[dict]) -> list[dict]:
    if not node_list:
        return []
    stack: list[tuple[dict, int]] = []
    root_nodes: list[dict] = []
    counter = 1

    for node in node_list:
        level = node["level"]
        tree_node = {
            "title": node["title"],
            "node_id": str(counter).zfill(4),
            "text": node.get("text", ""),
            "line_num": node["line_num"],
            "nodes": [],
        }
        counter += 1

        while stack and stack[-1][1] >= level:
            stack.pop()

        if not stack:
            root_nodes.append(tree_node)
        else:
            stack[-1][0]["nodes"].append(tree_node)

        stack.append((tree_node, level))

    return root_nodes


# ── Public entry point (replaces VectifyAI's md_to_tree) ────────────────

def md_to_tree(md_path: str) -> dict:
    """Parse a markdown file into a tree structure.

    Equivalent to VectifyAI's ``md_to_tree(md_path,
    if_add_node_summary='no', if_add_doc_description='no',
    if_add_node_text='yes', if_add_node_id='yes')`` — pure parsing,
    no LLM, no cost.
    """
    import os

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    line_count = content.count("\n") + 1

    node_list, lines = extract_nodes_from_markdown(content)
    nodes_with_text = extract_node_text_content(node_list, lines)
    structure = build_tree_from_nodes(nodes_with_text)

    return {
        "doc_name": os.path.splitext(os.path.basename(md_path))[0],
        "line_count": line_count,
        "structure": structure,
    }
