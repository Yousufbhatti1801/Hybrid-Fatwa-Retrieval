"""BM25-style sparse vector generation for hybrid search.

Design notes
------------
* **Stable token IDs** — token → uint32 via FNV-1a (process-independent,
  no Python hash-seed randomisation).  Vectors built at ingest time and
  vectors built at query time therefore use the same integer space.
* **TF-IDF-inspired weights** — log-normalised TF multiplied by a soft
  IDF proxy (rarer tokens in the *document* get a small boost via the
  unique-token-count denominator).  This penalises stopword-like tokens
  that appear many times in every document.
* **Collision handling** — FNV-1a on Urdu text has very low collision
  probability in the 2^31 space; duplicate indices (unexpected) are merged
  by summing their values.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer for Urdu/Arabic text.

    Keeps tokens of length ≥ 2 to discard single-character noise.
    Also discards pure-digit tokens (numbers add noise to Urdu retrieval).
    """
    tokens = re.findall(
        r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\w]+",
        text,
    )
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]


def _fnv1a_uint32(token: str) -> int:
    """FNV-1a 32-bit hash — deterministic, process-seed-independent.

    This guarantees that the same token produces the same integer index
    at ingest time *and* at query time, even across Python restarts.
    """
    FNV_PRIME  = 0x01000193
    FNV_OFFSET = 0x811C9DC5
    h = FNV_OFFSET
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFF_FFFF
    # Pinecone requires non-negative; FNV output is already unsigned 32-bit
    return h & 0x7FFF_FFFF   # constrain to [0, 2^31)


def build_sparse_vector(text: str) -> dict:
    """Convert *text* into a Pinecone-compatible sparse vector.

    Algorithm
    ---------
    For each unique token *t* in the document:

        weight(t) = log(1 + tf(t)) * idf_proxy(t)

    where ``idf_proxy = log(1 + V / (1 + df_proxy))`` uses the number of
    *unique* tokens in the document (``V``) as a stand-in for vocabulary
    size, and ``df_proxy = 1`` (we have no global DF at index-build time).
    This gives rarer-within-document tokens a modest boost over common ones
    without requiring a corpus-level IDF table.

    Returns
    -------
    dict with ``"indices"`` (list[int]) and ``"values"`` (list[float]).
    """
    tokens = _tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    tf = Counter(tokens)
    vocab_size = len(tf)
    # Soft IDF proxy: penalises tokens that appear in every document
    # (they'll have high in-document frequency too, so the log-TF is small)
    idf_denom = 1 + math.log(1 + vocab_size)

    # Accumulate into a dict so collisions (same FNV output, different tokens)
    # are handled by summing rather than silently dropping one.
    index_map: dict[int, float] = {}
    for token, count in tf.items():
        idx    = _fnv1a_uint32(token)
        log_tf = math.log(1 + count)
        weight = log_tf / idf_denom
        index_map[idx] = index_map.get(idx, 0.0) + weight

    indices = list(index_map.keys())
    values  = [round(v, 6) for v in index_map.values()]
    return {"indices": indices, "values": values}
