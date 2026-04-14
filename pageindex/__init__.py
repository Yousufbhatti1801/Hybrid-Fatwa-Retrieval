"""Vectorless PageIndex retrieval mode for the Fatawa RAG system.

This package adds a second retrieval mode alongside the existing hybrid
(Pinecone + BM25) retrieval. It uses the VectifyAI/PageIndex approach:

  - Convert the CSV corpus to a hierarchical Markdown index +
    flat lookup JSON (offline, no LLM)
  - Parse the Markdown into a tree via VectifyAI's md_to_tree()
  - At query time, walk the tree with two LLM calls per madhab
    (category+topic pick, then fatwa pick) and return the chosen
    fatwa from the flat lookup

Public API is exposed via :class:`pageindex.client.PageIndexClient`.
"""
