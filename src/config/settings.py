"""Centralized configuration loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ───────────────────────────────────────────────
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    chat_model: str = "gpt-4o-mini"

    # ── Islam360-only RAG (1536-dim index; separate from legacy hybrid index) ─
    islam360_data_dir: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "islam-360-fatwa-data"
    )
    islam360_pinecone_index: str = "islam360-fatwa-1536"
    islam360_embedding_model: str = "text-embedding-3-small"
    islam360_embedding_dimensions: int = 1536
    islam360_bm25_cache_path: Path = (
        Path(__file__).resolve().parent.parent.parent / ".bm25_islam360_cache.pkl"
    )
    islam360_retrieval_candidates: int = 20
    islam360_rerank_top_k: int = 5
    islam360_min_relevance_score: float = 0.55  # raised: stricter Q-to-Q rubric
    islam360_query_rewrite_temperature: float = 0.0  # deterministic JSON rewrite

    # ── Islam360 BM25-first fast path ────────────────────────────────────────
    # When ``islam360_use_fast_path=True`` (default), the Flask UI and the
    # default ``Islam360Retriever`` entry point use a streamlined pipeline
    # that mirrors the original working setup: query → (light canonicalise)
    # → BM25 (full-text + question-field boost) → top-k → synthesise.
    # No LLM rewrite, no dense embedding, no LLM rerank, no strict gate.
    # This was the configuration that consistently produced relevant Urdu
    # results in <2s per query.  The heavy hybrid+rerank pipeline is still
    # available via ``Islam360Retriever.retrieve()`` for ablation/debug.
    islam360_use_fast_path: bool = True
    islam360_fast_question_boost: float = 1.5     # weight for Q-only BM25 score
    islam360_fast_min_bm25_score: float = 1.5     # absolute floor applied per-candidate
    # Relative-to-top floor: drop any candidate scoring below this fraction
    # of the #1 candidate's score.  Stops the top-k slot from being padded
    # with weak filler hits when only 1-2 candidates are actually strong.
    # 0.55 → a candidate must be ≥ 55 %% as strong as the best hit.
    islam360_fast_min_score_ratio: float = 0.55
    # Minimum number of rare-anchor tokens a candidate must contain when
    # anchors exist. 1 = the baseline (rare-anchor filter); raising it
    # forces stronger topical coverage.
    islam360_fast_min_anchor_hits: int = 1
    # Anchor-coverage floor: if the best candidate hits N rare anchors,
    # every other candidate must hit at least `ceil(N * ratio)` anchors.
    # Stops slots from being padded with "1-anchor" candidates when the
    # query has multiple distinctive tokens and the top hits cover 2-3.
    islam360_fast_min_anchor_coverage_ratio: float = 0.5
    # ── LLM question-to-question reranker (precision gate) ──────────
    # After the cheap gates (sect / anchor / score floor) we ask the LLM
    # to score each surviving candidate's QUESTION against the user
    # query on a 0-1 scale.  Anything below the threshold is dropped;
    # "No relevant fatwa found" is returned if nothing clears it.
    # Quality over quantity — prefer returning fewer or zero results.
    # ── LLM query expansion (recall-side) ───────────────────────────
    # Runs AFTER the deterministic fiqh-synonym dictionary.  Asks the
    # LLM for additional Urdu tokens that are genuine synonyms or
    # technical variants of terms in the query — covers the long
    # tail of fiqh vocabulary the hand-curated dictionary can't
    # enumerate (خلع, عدة, ظہار, ایصالِ ثواب, …).  Output is
    # appended to the BM25 query just like the dictionary terms are;
    # the downstream LLM reranker then filters any noise the
    # expansion lets in.  Cached in-process so three parallel sect
    # calls share one LLM invocation per unique query.
    islam360_llm_expand_enabled: bool = True
    # Hard ceiling on expansion token count.  The LLM emits
    # 2-5 fiqh concepts × 0-4 alternates each (see _LLM_EXPAND_SYS)
    # and we concatenate every alternate's tokens into the BM25
    # query.  A realistic multi-concept query (e.g. "دعائے خوف یا
    # خوابِ بد سے بچاؤ کی تدابیر") naturally produces ~10-14
    # useful tokens — the old ceiling of 6 was under-budgeted and
    # silently truncated helpful expansions for long-tail queries.
    islam360_llm_expand_max_tokens: int = 16
    islam360_llm_expand_timeout_s: float = 10.0  # JSON call is slightly slower than plain-text
    islam360_llm_expand_cache_size: int = 256
    islam360_llm_expand_model: str = ""          # empty → reuse chat_model

    islam360_rerank_enabled: bool = True
    islam360_rerank_threshold: float = 0.75
    islam360_rerank_max_candidates: int = 12     # cap on LLM input size
    islam360_rerank_timeout_s: float = 35.0      # hard timeout, fall back on miss
    # Empty string → reuse settings.chat_model for the rerank call.
    islam360_rerank_model: str = ""
    # When the anchor filter fell back to matching the full body (not
    # just the question), we're in a "noisier" regime where the rerank
    # is doing more work.  If the rerank then fails or times out, the
    # pre-rerank BM25 ordering is unreliable — rather than publish
    # potentially irrelevant results, treat it as no_match.
    islam360_strict_when_body_fallback: bool = True
    islam360_fast_canonicalise_roman: bool = True  # quick LLM translate roman→Urdu

    # ── Smart router (/api/query-smart) ──────────────────────────────
    # Escalation order: raw-fatwas fast probe  →  islam360 retrieve_fast
    # (primary accurate path)  →  PageIndex tree descent (fallback).
    # Each stage computes a confidence score; the first stage that clears
    # its bar wins and the rest are skipped.  See src/routing/router.py
    # for the scoring functions and the orchestrator.
    smart_router_enabled: bool = True
    # In-process query→response cache (TTL-bounded, size-capped).  Keyed
    # on a lightly-normalised form of the user's question so minor
    # spelling/casing differences hit the same entry.
    smart_router_cache_ttl_s: int = 3600
    smart_router_cache_max: int = 500
    # Stage-1 probe: raw-fatwas with ``rerank=False`` (~200-500 ms, keyword
    # only).  If confidence clears this bar, we return the raw-fatwa text
    # directly (no LLM synthesis) — users still see the exact fatwa.
    smart_router_raw_fatwas_enabled: bool = True
    smart_router_raw_fatwas_timeout_s: float = 2.0
    smart_router_raw_fatwas_high_bar: float = 0.80
    # Stage-2 primary: islam360 retrieve_fast.  If ANY sect returns a
    # non-empty result (rerank already applied its strict 0.75 gate), we
    # return.  We only escalate when ALL three sects return
    # NO_RELEVANT_FATWA but at least one sect had a healthy pre-rerank
    # pool — that's the "BM25 found candidates but rerank rejected them
    # all" signal where PageIndex's category-tree navigation may still
    # succeed.
    smart_router_islam360_timeout_s: float = 45.0
    smart_router_escalate_if_all_empty: bool = True
    smart_router_escalate_min_pool_size: int = 20
    # Stage-3 fallback: PageIndex.
    smart_router_pageindex_enabled: bool = True
    smart_router_pageindex_timeout_s: float = 25.0
    smart_router_pageindex_min_confidence: float = 0.45

    # PageIndex (vectorless): see pageindex/* for the pipeline
    # ``category_hint_shortcut`` is OFF by default: a wrong hint used to
    # drop the *entire* category into the pool (hundreds of off-topic
    # fatawa) before the tree LLM even ran.
    pageindex_category_hint_shortcut: bool = False
    pageindex_lookup_scoring: bool = True
    # Synonym expansion (namaz → صلاۃ, …) helps recall in raw search but
    # was flooding lexical scoring; keep OFF for PageIndex unless needed.
    pageindex_lexical_synonym_expand: bool = False
    pageindex_refiner_enabled: bool = True
    pageindex_refiner_max: int = 8

    # ── Embedding retry ──────────────────────────────────────
    embed_max_retries: int = 5
    embed_base_delay: float = 1.0   # seconds; doubles each retry
    embed_max_delay: float = 60.0

    # ── Pinecone ─────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_name: str = "fatawa-hybrid"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    # ── Data ─────────────────────────────────────────────────
    # Default resolves to <repo_root>/data.
    # Override at runtime via DATA_ROOT=<path> in your .env file.
    data_root: Path = (
        Path(__file__).resolve().parent  # src/config/
        .parent                          # src/
        .parent                          # repo root
        / "data"                         # <repo_root>/data/
    )
    data_sources: list[str] = [
        "Banuri-ExtractedData-Output",
        "IslamQA-ExtractedData-Output",
        "fatwaqa-ExtractedData-Output",
        "urdufatwa-ExtractedData-Output",
    ]

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 10
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    bm25_cache_path: Path = Path(__file__).resolve().parent.parent.parent / ".bm25_cache.pkl"
    # Quality gates (reduce irrelevant Pinecone / fusion hits)
    retrieval_min_pinecone_score: float = 0.18  # drop weak cosine matches before fusion
    retrieval_question_boost: float = 0.28      # BM25 match on fatwa *question* field (was 0.15)
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_pool: int = 30             # hybrid retrieves this many; then LLM re-ranks to top_k
    retrieval_rerank_min_score: float = 0.42    # min LLM relevance (0–1) to keep a chunk

    # ── Pipeline / token budget ────────────────────────────
    # Total context window for gpt-4o-mini is 128k tokens.
    # We reserve ~2k for the system prompt + instructions overhead,
    # ~1k for the answer, leaving ~3k for context by default.
    context_token_budget: int = 3000   # max tokens for retrieved fatawa
    answer_max_tokens: int = 512       # max tokens in the LLM response

    # ── Ingestion ────────────────────────────────────────────
    batch_size: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
