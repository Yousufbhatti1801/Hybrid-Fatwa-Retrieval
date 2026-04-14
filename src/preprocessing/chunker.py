"""Token-aware chunking for fatwa Q&A documents.

Strategy
--------
* Treat each fatwa as a single semantic unit (question + answer).
* If the token count ≤ MAX_TOKENS  → keep as one chunk (no splitting).
* If longer                        → split on sentence boundaries first;
  fall back to word-level sliding window only when a sentence is itself
  too long.  Every chunk except the first carries an overlap of
  OVERLAP_TOKENS words from the previous chunk to preserve context.
* Question and Answer are NEVER split apart at the seam — the combined
  text is split as a whole so the Q/A pairing is preserved inside each
  chunk whenever possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Generator

from src.config import get_settings
from src.ingestion.loader import FatwaDocument
from src.preprocessing.urdu_normalizer import normalize_urdu

# ── Token approximation ───────────────────────────────────────────────────────
# Urdu words are the natural token unit; we approximate 1 word ≈ 1.3 tokens
# (accounts for sub-word tokenisation by OpenAI).  This is conservative so
# we never exceed the real token limit.
_WORDS_PER_TOKEN = 1.3
MAX_TOKENS = 500
OVERLAP_TOKENS = 50


def _token_estimate(text: str) -> int:
    """Fast word-count-based token estimate."""
    return int(len(text.split()) * _WORDS_PER_TOKEN)


# Urdu sentence boundary: ends with ۔ ؟ ! . ? or newline
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[۔؟!.?\n])\s+")


def _split_into_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return text.split()


def _sliding_window(words: list[str], max_w: int, overlap_w: int) -> Generator[str, None, None]:
    """Word-level sliding window — last resort for very long sentences."""
    start = 0
    while start < len(words):
        end = start + max_w
        yield " ".join(words[start:end])
        if end >= len(words):
            break
        start = end - overlap_w


def _chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split *text* into chunks that each fit within *max_tokens*.

    1.  If the full text fits → return as-is.
    2.  Accumulate sentences greedily; when adding the next sentence would
        overflow, flush the buffer as a chunk and start the next chunk with
        *overlap_tokens* worth of words carried over from the previous chunk.
    3.  If a single sentence is itself too long, apply word-level sliding
        window on that sentence only.
    """
    if _token_estimate(text) <= max_tokens:
        return [text]

    max_words = int(max_tokens / _WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens / _WORDS_PER_TOKEN)

    sentences = _split_into_sentences(text)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        sent_tokens = _token_estimate(sentence)

        # Single sentence too big → sub-chunk it with sliding window
        if sent_tokens > max_tokens:
            # Flush current buffer first
            if buffer:
                chunks.append(" ".join(buffer))
                buffer = _words(" ".join(buffer))[-overlap_words:]
                buffer_tokens = _token_estimate(" ".join(buffer))

            for sub in _sliding_window(_words(sentence), max_words, overlap_words):
                chunks.append(sub)
            # Carry overlap from the last sub-chunk
            last_sub_words = _words(chunks[-1])
            buffer = last_sub_words[-overlap_words:]
            buffer_tokens = _token_estimate(" ".join(buffer))
            continue

        if buffer_tokens + sent_tokens > max_tokens and buffer:
            chunks.append(" ".join(buffer))
            # Start next chunk with overlap from previous
            overlap_buf = _words(" ".join(buffer))[-overlap_words:]
            buffer = overlap_buf + [sentence]
            buffer_tokens = _token_estimate(" ".join(buffer))
        else:
            buffer.append(sentence)
            buffer_tokens += sent_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if c.strip()]


# ── Public dataclass kept for pipeline compatibility ──────────────────────────

@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    category: str
    subcategory: str
    fatwa_no: str
    url: str
    chunk_index: int


def chunk_document(doc: FatwaDocument) -> list[Chunk]:
    """Normalise and chunk a single FatwaDocument into Chunk objects."""
    settings = get_settings()
    normalized = normalize_urdu(doc.full_text)
    fragments = _chunk_text(normalized, MAX_TOKENS, OVERLAP_TOKENS)

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}_{i}",
            doc_id=doc.doc_id,
            text=fragment,
            source=doc.source,
            category=doc.category,
            subcategory=doc.subcategory,
            fatwa_no=doc.fatwa_no,
            url=doc.url,
            chunk_index=i,
        )
        for i, fragment in enumerate(fragments)
    ]


def chunk_document_as_dicts(doc: FatwaDocument) -> list[dict]:
    """Return chunks in the requested dict schema.

    Output per chunk::

        {
            "id":       "<doc_id>_<index>",
            "text":     "<cleaned chunk text>",
            "metadata": {
                "doc_id":      str,
                "source":      str,
                "category":    str,
                "subcategory": str,
                "fatwa_no":    str,
                "url":         str,
                "chunk_index": int,
                "total_chunks":int,
            }
        }
    """
    chunks = chunk_document(doc)
    total = len(chunks)
    return [
        {
            "id": c.chunk_id,
            "text": c.text,
            "metadata": {
                "doc_id":       c.doc_id,
                "source":       c.source,
                "category":     c.category,
                "subcategory":  c.subcategory,
                "fatwa_no":     c.fatwa_no,
                "url":          c.url,
                "chunk_index":  c.chunk_index,
                "total_chunks": total,
            },
        }
        for c in chunks
    ]


# ── Dynamic-record API (works on unified dicts from dynamic_loader) ───────────
#
# These functions accept the output of src.ingestion.dynamic_loader and do NOT
# assume any fixed column names.  They operate on the unified record schema:
#   {id, question, answer, text, category, source_file, folder, date, reference}

# Token thresholds
_SHORT_TOKEN_MIN  = 10    # below this → flagged as "too_short", discarded by default
_LONG_TOKEN_MIN   = 500   # at or above this → flagged as "long", will be chunked

# ── Source display name mapping ───────────────────────────────────────────────
# Maps the raw folder name (from the file path) to a human-readable source
# name used in citations, prompts, and the web UI.
SOURCE_DISPLAY_NAMES: dict[str, str] = {
    "Banuri-ExtractedData-Output":    "Banuri Institute (Deobandi)",
    "IslamQA-ExtractedData-Output":   "IslamQA (Ahle Hadees)",
    "fatwaqa-ExtractedData-Output":   "FatwaQA (Ahle Hadees)",
    "urdufatwa-ExtractedData-Output": "UrduFatwa (Barelvi)",
}

# ── School-of-thought (maslak) mapping ───────────────────────────────────────
# Filterable maslak tag stored as Pinecone metadata so callers can request
# answers from a specific school of thought.
SOURCE_MASLAK: dict[str, str] = {
    "Banuri-ExtractedData-Output":    "Deobandi",
    "IslamQA-ExtractedData-Output":   "Ahle Hadees",
    "fatwaqa-ExtractedData-Output":   "Ahle Hadees",
    "urdufatwa-ExtractedData-Output": "Barelvi",
}


def get_source_display_name(folder: str) -> str:
    """Return a short display name for a data-source folder string.

    Falls back to stripping the common ``-ExtractedData-Output`` suffix so
    new sources are handled gracefully even without an explicit mapping entry.

    Examples
    --------
    >>> get_source_display_name("Banuri-ExtractedData-Output")
    'Banuri Institute (Deobandi)'
    >>> get_source_display_name("SomeNewSource-ExtractedData-Output")
    'SomeNewSource'
    """
    if folder in SOURCE_DISPLAY_NAMES:
        return SOURCE_DISPLAY_NAMES[folder]
    return folder.replace("-ExtractedData-Output", "").strip() or folder


def get_source_maslak(folder: str) -> str:
    """Return the school-of-thought (maslak) for a data-source folder.

    Returns an empty string for unknown sources so filters composed on this
    field still behave gracefully.

    Examples
    --------
    >>> get_source_maslak("urdufatwa-ExtractedData-Output")
    'Barelvi'
    >>> get_source_maslak("fatwaqa-ExtractedData-Output")
    'Ahle Hadees'
    """
    return SOURCE_MASLAK.get(folder, "")


def _classify_length(token_count: int) -> str:
    """Return 'too_short', 'normal', or 'long'."""
    if token_count < _SHORT_TOKEN_MIN:
        return "too_short"
    if token_count >= _LONG_TOKEN_MIN:
        return "long"
    return "normal"


def preprocess_record(
    record: dict,
    *,
    strip_diacritics: bool = True,
    discard_short: bool = True,
) -> list[dict]:
    """Normalise and chunk a single unified record dict.

    Parameters
    ----------
    record:
        One record from ``dynamic_loader.stream_corpus()`` / ``load_corpus()``.
    strip_diacritics:
        Pass through to the Urdu normaliser (default True).
    discard_short:
        When True (default), records flagged as ``too_short`` are dropped and
        an empty list is returned.  Set False to keep them (they are returned
        as a single chunk flagged with ``length_flag="too_short"``).

    Returns
    -------
    List of chunk dicts.  Each chunk::

        {
            "id":          "<record_id>_<chunk_index>",
            "text":        "<normalised chunk text>",
            "metadata": {
                "doc_id":        str,   # original record id
                "question":      str,   # original (un-normalised) question
                "answer":        str,   # original (un-normalised) answer
                "category":      str,
                "source_file":   str,
                "folder":        str,
                "date":          str|None,
                "reference":     str|None,
                "chunk_index":   int,
                "total_chunks":  int,
                "length_flag":   "normal" | "long" | "too_short",
                "token_estimate": int,
            }
        }
    """
    doc_id     = record.get("id", "")
    question   = record.get("question", "").strip()
    query_text = record.get("query", "").strip()
    answer     = record.get("answer", "").strip()

    # Embedding target = query/question only (no answer body).
    # If both exist and differ, concatenate both once; otherwise keep one.
    if query_text and question and query_text != question:
        raw_text = f"{query_text}\n{question}"
    else:
        raw_text = query_text or question

    if not raw_text:
        raw_text = question
    folder     = record.get("folder", "")

    # Normalise the combined text
    clean_text = normalize_urdu(raw_text, strip_diacritics=strip_diacritics)
    token_est  = _token_estimate(clean_text)
    flag       = _classify_length(token_est)

    if flag == "too_short" and discard_short:
        return []

    # Split into chunks (returns [clean_text] unchanged if ≤ MAX_TOKENS)
    fragments = _chunk_text(clean_text, MAX_TOKENS, OVERLAP_TOKENS)
    total     = len(fragments)

    base_meta = {
        "doc_id":        doc_id,
        "query":         query_text,
        "question":      question,
        "answer":        answer,
        "category":      record.get("category", ""),
        "source_file":   record.get("source_file", ""),
        "folder":        folder,
        # Human-readable source name for citations and UI display
        "source_name":   get_source_display_name(folder),
        "date":          record.get("date"),
        "reference":     record.get("reference"),
        "length_flag":   flag,
        "token_estimate": token_est,
    }

    return [
        {
            "id":   f"{doc_id}_{i}",
            "text": fragment,
            "metadata": {**base_meta, "chunk_index": i, "total_chunks": total},
        }
        for i, fragment in enumerate(fragments)
    ]


def preprocess_records(
    records: list[dict] | Generator,
    *,
    strip_diacritics: bool = True,
    discard_short: bool = True,
) -> Generator[dict, None, None]:
    """Yield preprocessed chunks from an iterable of unified records.

    Accepts both lists and generators (e.g. from ``stream_corpus``), so the
    full pipeline stays memory-efficient for 100k+ records::

        from src.ingestion.dynamic_loader import stream_corpus
        from src.preprocessing.chunker import preprocess_records

        for chunk in preprocess_records(stream_corpus("data")):
            # chunk ready for embedding
            ...
    """
    for record in records:
        yield from preprocess_record(
            record,
            strip_diacritics=strip_diacritics,
            discard_short=discard_short,
        )
