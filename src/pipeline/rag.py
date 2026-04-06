"""End-to-end RAG query pipeline: retrieve → trim → augment → generate.

Pipeline steps
--------------
1.  Normalise the Urdu query.
2.  Generate a dense embedding (OpenAI).
3.  Run hybrid retrieval (Pinecone dense + BM25 sparse, weighted fusion).
4.  Trim retrieved context to the configured token budget.
5.  Build the structured RAG prompt (system + user turn).
6.  Call the OpenAI chat model.
7.  Return answer + sources + timing breakdown.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from openai import OpenAI

from src.config import get_settings
from src.pipeline.context_trimmer import trim_to_budget
from src.pipeline.prompt_builder import build_messages, build_prompt, NO_ANSWER_SENTINEL
from src.preprocessing.urdu_normalizer import normalize_urdu
from src.retrieval.hybrid_retriever import hybrid_search

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_chat_client() -> OpenAI:
    """Return a cached OpenAI client for chat completions."""
    return OpenAI(api_key=get_settings().openai_api_key)


# ── Public entry point ────────────────────────────────────────────────────────

def query(
    question: str,
    *,
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    context_token_budget: int | None = None,
    stream: bool = False,
) -> dict:
    """Run the full RAG pipeline for an Urdu fatawa query.

    Parameters
    ----------
    question:
        Raw Urdu question from the user.
    top_k:
        Number of candidates to retrieve before trimming.
        Defaults to ``settings.top_k``.
    dense_weight / sparse_weight:
        Override retrieval weights (must sum to ≤ 1.0).
    context_token_budget:
        Maximum tokens to pass as context to the LLM.
        Defaults to ``settings.context_token_budget``.
    stream:
        When True, returns a generator that yields answer tokens.

    Returns
    -------
    Non-streaming::

        {
            "answer":       str,
            "sources":      list[dict],   # one per retrieved chunk
            "num_chunks":   int,          # after trimming
            "prompt":       str,          # exact user-turn sent to LLM
            "timings": {
                "normalise_ms":  float,
                "retrieve_ms":   float,
                "trim_ms":       float,
                "generate_ms":   float,
                "total_ms":      float,
            }
        }

    Streaming: returns a generator of token strings (sources not included).
    """
    settings = get_settings()
    budget = context_token_budget if context_token_budget is not None \
        else settings.context_token_budget
    t_start = time.perf_counter()

    # ── Step 1: Normalise ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    normalised = normalize_urdu(question)
    t_normalise = (time.perf_counter() - t0) * 1000
    logger.debug("Normalised query: %.80s", normalised)

    # ── Step 2+3: Hybrid retrieval (embed + dense + BM25 + fusion) ───────────
    t0 = time.perf_counter()
    retrieved = hybrid_search(
        normalised,
        top_k=top_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    t_retrieve = (time.perf_counter() - t0) * 1000
    logger.info("Retrieved %d chunks in %.0fms", len(retrieved), t_retrieve)

    # ── Step 4: Trim to token budget ─────────────────────────────────────────
    t0 = time.perf_counter()
    trimmed = trim_to_budget(retrieved, budget)
    t_trim = (time.perf_counter() - t0) * 1000
    dropped = len(retrieved) - len(trimmed)
    if dropped:
        logger.info("Trimmed %d chunks to fit token budget (%d tokens)", dropped, budget)

    # ── Step 5: Build prompt ─────────────────────────────────────────────────
    messages = build_messages(question, trimmed)
    user_message = messages[-1]["content"]  # kept for `prompt` in return value

    # ── Step 6+7: Call LLM ───────────────────────────────────────────────────
    client = _get_chat_client()

    if stream:
        return _stream_response(client, settings.chat_model, messages)

    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=0,          # strictly grounded — no creative deviation
        max_tokens=settings.answer_max_tokens,
    )
    t_generate = (time.perf_counter() - t0) * 1000
    t_total = (time.perf_counter() - t_start) * 1000

    answer = completion.choices[0].message.content or NO_ANSWER_SENTINEL
    logger.info("Generated answer in %.0fms (total pipeline: %.0fms)", t_generate, t_total)

    return {
        "answer":     answer,
        "sources":    [r["metadata"] for r in trimmed],
        "num_chunks": len(trimmed),
        "prompt":     user_message,
        "timings": {
            "normalise_ms": round(t_normalise, 1),
            "retrieve_ms":  round(t_retrieve, 1),
            "trim_ms":      round(t_trim, 1),
            "generate_ms":  round(t_generate, 1),
            "total_ms":     round(t_total, 1),
        },
    }


# Keep `ask` as an alias for backwards compatibility
ask = query


# ── Streaming ─────────────────────────────────────────────────────────────────

def _stream_response(client: OpenAI, model: str, messages: list[dict]):
    """Yield answer tokens as they arrive from the streaming API."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=get_settings().answer_max_tokens,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta
        if delta.content:
            yield delta.content
