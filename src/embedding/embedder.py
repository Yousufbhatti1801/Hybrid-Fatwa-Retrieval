"""Generate dense embeddings via the OpenAI API.

Optimisation notes
------------------
* Uses ``text-embedding-3-large`` (3072-dim) for best Urdu retrieval quality.
* Batches requests to maximise throughput (OpenAI accepts up to 2048 inputs
  per call, but we use a conservative default of 100 to stay within the
  token-per-minute limit).
* Exponential back-off with jitter on RateLimitError / APIStatusError 429.
* Deduplicates identical texts within a batch before sending so repeated
  boilerplate doesn't consume extra tokens.
* ``embed_single`` reuses the same retry logic for query-time use.
"""

from __future__ import annotations

import logging
import random
import time
from functools import lru_cache
from itertools import islice
from typing import Iterator

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError

from src.config import get_settings
from src.preprocessing.chunker import Chunk

logger = logging.getLogger(__name__)

# Maximum tokens OpenAI accepts per embedding request (conservative)
_MAX_TOKENS_PER_BATCH = 300_000
# Rough upper bound: text-embedding-3-large, Urdu words avg ~4 chars + space
_AVG_CHARS_PER_TOKEN = 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _batched(iterable, n: int) -> Iterator[list]:
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    # max_retries=1 disables the SDK's silent internal retries.
    # We manage retries via _call_with_retry() in this file.
    return OpenAI(
        api_key=get_settings().openai_api_key,
        max_retries=1,
        timeout=10.0,
    )


def _jitter(delay: float) -> float:
    """Full jitter: random value in [0, delay]."""
    return random.uniform(0, delay)


def _call_with_retry(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call the embeddings endpoint with exponential back-off.

    Retries on RateLimitError (429) and transient 5xx errors.
    Raises immediately on 4xx errors other than 429.
    """
    settings = get_settings()
    delay = settings.embed_base_delay

    for attempt in range(1, settings.embed_max_retries + 1):
        try:
            # Explicit timeout — OpenAI SDK default is 600s and has its
            # own internal retries that can silently add 10+s to cold calls.
            # We manage retries ourselves via this loop.
            response = client.embeddings.create(
                input=texts,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                encoding_format="float",
                timeout=8.0,
            )
            return [item.embedding for item in response.data]

        except RateLimitError:
            wait = min(_jitter(delay), settings.embed_max_delay)
            logger.warning(
                "Rate limit hit (attempt %d/%d). Retrying in %.1fs…",
                attempt, settings.embed_max_retries, wait,
            )
            time.sleep(wait)
            delay = min(delay * 2, settings.embed_max_delay)

        except APIStatusError as exc:
            if exc.status_code >= 500 or exc.status_code == 429:
                wait = min(_jitter(delay), settings.embed_max_delay)
                logger.warning(
                    "API error %d (attempt %d/%d). Retrying in %.1fs…",
                    exc.status_code, attempt, settings.embed_max_retries, wait,
                )
                time.sleep(wait)
                delay = min(delay * 2, settings.embed_max_delay)
            else:
                raise

        except APIConnectionError:
            wait = min(_jitter(delay), settings.embed_max_delay)
            logger.warning(
                "Connection error (attempt %d/%d). Retrying in %.1fs…",
                attempt, settings.embed_max_retries, wait,
            )
            time.sleep(wait)
            delay = min(delay * 2, settings.embed_max_delay)

    raise RuntimeError(
        f"Embedding failed after {settings.embed_max_retries} retries."
    )


def _safe_batches(chunks: list[Chunk], max_items: int) -> Iterator[list[Chunk]]:
    """Yield batches that stay within both item count and token-budget limits."""
    current: list[Chunk] = []
    current_chars = 0
    token_budget = _MAX_TOKENS_PER_BATCH * _AVG_CHARS_PER_TOKEN

    for chunk in chunks:
        chunk_chars = len(chunk.text)
        if current and (len(current) >= max_items or current_chars + chunk_chars > token_budget):
            yield current
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += chunk_chars

    if current:
        yield current


# ── Public API ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4096)
def embed_single(text: str) -> tuple[float, ...]:
    """Embed a single text string with retry logic (used at query time).

    Results are cached (up to 4096 unique queries) so repeated questions
    skip the OpenAI API call entirely (~500-2000ms saved per hit).
    Returns a tuple (hashable) so the LRU cache can store it.
    """
    client = _get_client()
    return tuple(_call_with_retry(client, [text])[0])


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of raw strings and return their vectors.

    Unlike ``generate_embeddings``, this function does *no* batching or
    deduplication – the caller is responsible for grouping inputs.  Intended
    for use by the higher-level :mod:`pipeline` module.
    """
    client = _get_client()
    return _call_with_retry(client, texts)


def generate_embeddings(
    chunks: list[Chunk],
    batch_size: int | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Embed chunks and return (Chunk, vector) pairs — used by the ingest pipeline."""
    settings = get_settings()
    client = _get_client()
    batch_size = batch_size or settings.batch_size
    results: list[tuple[Chunk, list[float]]] = []

    for batch in _safe_batches(chunks, batch_size):
        # Deduplicate texts within this batch to save tokens
        unique_texts: list[str] = []
        seen: dict[str, int] = {}  # text → index in unique_texts
        for c in batch:
            if c.text not in seen:
                seen[c.text] = len(unique_texts)
                unique_texts.append(c.text)

        embeddings = _call_with_retry(client, unique_texts)

        for chunk in batch:
            results.append((chunk, embeddings[seen[chunk.text]]))

        logger.info("Embedded batch of %d chunks (%d unique texts, %d total)",
                    len(batch), len(unique_texts), len(results))

    return results


def generate_embeddings_as_dicts(
    chunks: list[Chunk],
    batch_size: int | None = None,
) -> list[dict]:
    """Embed chunks and return the requested output schema.

    Output per record::

        {
            "id":        "<chunk_id>",
            "embedding": [float, …],   # 3072-dim
            "metadata": {
                "question":  str,
                "category":  str,
                "source":    str,
                "fatwa_no":  str,
                "url":       str,
            }
        }

    The ``question`` field is taken from the chunk's doc-level metadata
    stored in the Chunk object; for sub-chunks the full question may be
    truncated, so the raw ``fatwa_no`` and ``url`` are included for
    look-up.
    """
    pairs = generate_embeddings(chunks, batch_size)
    return [
        {
            "id": chunk.chunk_id,
            "embedding": vector,
            "metadata": {
                "question": "",          # populated by caller if needed
                "category": chunk.category,
                "source":   chunk.source,
                "fatwa_no": chunk.fatwa_no,
                "url":      chunk.url,
            },
        }
        for chunk, vector in pairs
    ]
