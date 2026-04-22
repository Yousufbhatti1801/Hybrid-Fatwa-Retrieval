"""Smart router for fatwa retrieval — escalates across three paths.

Stage design (per the routing spec)
-----------------------------------

1.  **Fast probe** (``raw_fatwas`` with ``rerank=False``)

    Pure inverted-index keyword search across all four source schools.
    Returns sub-second.  If the top keyword hits are overwhelmingly
    strong and well-distributed across schools, we treat that as a
    ``clear keyword match`` query and serve the fatwa text directly —
    no synthesis, no further hops.

2.  **Primary accurate path** (``Islam360.retrieve_fast``)

    The existing BM25 + LLM-rerank pipeline.  Each sect is retrieved
    with its own source allow-list; the reranker's 0.75 threshold is
    already enforced internally.  If *any* sect returns a non-empty
    result we commit and return — that's the gold standard for
    precision in this system.

3.  **Category-tree fallback** (``PageIndex``)

    LLM-driven descent through the pre-built category/topic tree.  We
    only reach here when Islam360 rejected every candidate — typically
    because the query's vocabulary didn't overlap well with BM25-able
    fatwa titles but the category hierarchy has the right bucket.

The router never merges partial results across stages.  Whichever
stage first clears its confidence bar wins; the rest are skipped.

Observability
-------------

Every ``route_query`` call emits a ``routing`` dict in its response
containing:

    * ``path_used`` — which of the three stages produced the answer
      (``"cache" | "raw_fatwas" | "islam360_fast" | "pageindex" | "none"``)
    * ``confidence`` — the winning stage's confidence score (0..1)
    * ``stage_latencies_ms`` — wall-clock per stage (``null`` if skipped)
    * ``stage_confidences`` — confidence of every stage that ran
    * ``escalation_reasons`` — short phrases explaining why earlier
      stages were rejected
    * ``cache_hit`` — boolean

These fields are also logged as a single INFO-level JSON line per
query so routing behaviour can be replayed from logs without
instrumentation code.

Caching
-------

The request-level cache keys on a lightly-normalised form of the raw
question (case-folded, whitespace-collapsed, punctuation-stripped).
We do NOT run the Urdu canonicaliser at cache-key time — it is
itself expensive (can fire an LLM call).  The key is deterministic
and sub-millisecond to compute.
"""

from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import logging
import re
import time
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)


# ╔════════════════════════════════════════════════════════════════╗
# ║  Cache                                                          ║
# ╚════════════════════════════════════════════════════════════════╝
#
# Simple bounded TTL cache.  Keys are hashed-normalised queries;
# values are ``(inserted_at, response_dict)``.  Eviction is LRU-ish:
# when we need space we drop the oldest-inserted entry.  That's good
# enough here — the cache is mainly to absorb retry bursts and hot
# canonical queries (e.g. "namaz ka tareeqa" variants).

_ROUTER_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[؟?!،,.;:\-_\"'\(\)\[\]{}]+")


def _normalise_for_cache(raw: str) -> str:
    """Produce a stable, case/whitespace-insensitive cache key.

    Deliberately light — we do NOT call the Islam360 canonicaliser
    here because it can fire an LLM call.  Two callers passing the
    same Urdu/Roman mix with different spacing or punctuation will
    share a cache entry; genuinely-different queries won't collide.
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # Short hash keeps the dict keys small without sacrificing
    # collision safety for our scale (hundreds of entries).
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def _cache_get(key: str) -> dict[str, Any] | None:
    s = get_settings()
    entry = _ROUTER_CACHE.get(key)
    if not entry:
        return None
    inserted_at, payload = entry
    if time.time() - inserted_at > s.smart_router_cache_ttl_s:
        _ROUTER_CACHE.pop(key, None)
        return None
    return payload


def _cache_put(key: str, payload: dict[str, Any]) -> None:
    s = get_settings()
    if len(_ROUTER_CACHE) >= s.smart_router_cache_max:
        # Drop the oldest entry (insertion order, Python 3.7+ dicts).
        try:
            oldest = next(iter(_ROUTER_CACHE))
            _ROUTER_CACHE.pop(oldest, None)
        except StopIteration:
            pass
    _ROUTER_CACHE[key] = (time.time(), payload)


# ╔════════════════════════════════════════════════════════════════╗
# ║  Shape helpers                                                  ║
# ╚════════════════════════════════════════════════════════════════╝
#
# The three retrievers return different shapes.  We pick the
# ``/api/query-all-schools`` shape as the canonical output — three
# sect cards (Deobandi / Barelvi / Ahle Hadees) each with an
# ``answer``, ``sources`` list, and labelling metadata — and convert
# raw-fatwas' and PageIndex's four-school outputs into it.

SECT_DEOBANDI = "deobandi"
SECT_BARELVI = "barelvi"
SECT_AHLE_HADITH = "ahle_hadith"
_ALL_SECTS = (SECT_DEOBANDI, SECT_BARELVI, SECT_AHLE_HADITH)

_SECT_LABELS = {
    SECT_DEOBANDI: "Deobandi",
    SECT_BARELVI: "Barelvi",
    SECT_AHLE_HADITH: "Ahle Hadees",
}
_SECT_SOURCE_DESC = {
    SECT_DEOBANDI: "Banuri Institute",
    SECT_BARELVI: "UrduFatwa",
    SECT_AHLE_HADITH: "IslamQA + FatwaQA",
}

# Raw-fatwas / PageIndex school_id → sect code.
_SCHOOL_TO_SECT = {
    "Banuri": SECT_DEOBANDI,
    "urdufatwa": SECT_BARELVI,
    "IslamQA": SECT_AHLE_HADITH,
    "fatwaqa": SECT_AHLE_HADITH,
}


def _empty_sect_card(sect: str, *, reason: str) -> dict[str, Any]:
    return {
        "maslak": _SECT_LABELS[sect],
        "sect": sect,
        "source_label": _SECT_SOURCE_DESC[sect],
        "answer": "متعلقہ فتویٰ نہیں ملا۔",
        "no_match": True,
        "sources": [],
        "num_chunks": 0,
        "elapsed_ms": 0,
        "reason": reason,
    }


def _source_from_school_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw-fatwas / PageIndex per-fatwa dict to the source
    shape ``/api/query-all-schools`` already uses."""
    return {
        "category": rec.get("category", "—"),
        "source_name": rec.get("school_label", ""),
        "maslak": rec.get("madhab", "") or rec.get("maslak", ""),
        "sect": _SCHOOL_TO_SECT.get(rec.get("school_id", ""), ""),
        "source": rec.get("school_id", ""),
        "source_file": rec.get("school_id", ""),
        "reference": rec.get("url", ""),
        "question": rec.get("question_text") or rec.get("query_text") or "",
        "answer": (rec.get("answer_text") or "")[:1500],
        "fatwa_no": rec.get("fatwa_no", ""),
        "score": float(rec.get("score", 0.0) or 0.0) / 100.0,
        "anchor_hits": 0,
        "rerank_score": None,
        "rerank_reason": "",
    }


def _convert_raw_fatwas_to_sects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert raw-fatwas 4-school output to 3-sect Islam360 shape.

    IslamQA + FatwaQA are both Ahle Hadees sources — we merge their
    top fatwas (score-sorted) into one sect card rather than pick an
    arbitrary winner.
    """
    by_sect: dict[str, list[dict]] = {s: [] for s in _ALL_SECTS}
    for school_card in payload.get("results") or []:
        sid = school_card.get("school_id", "")
        sect = _SCHOOL_TO_SECT.get(sid)
        if not sect:
            continue
        for fatwa in school_card.get("fatawa") or []:
            # Tag with school_id so the source converter can map it back.
            fatwa = dict(fatwa)
            fatwa.setdefault("school_id", sid)
            fatwa.setdefault("school_label", school_card.get("school_label", ""))
            fatwa.setdefault("madhab", school_card.get("maslak", ""))
            by_sect[sect].append(fatwa)

    cards: list[dict[str, Any]] = []
    for sect in _ALL_SECTS:
        fatwas = by_sect[sect]
        # For Ahle Hadees we merged two schools — re-rank by score so
        # the strongest hit (regardless of source) is first.
        fatwas.sort(key=lambda r: -float(r.get("score", 0.0) or 0.0))
        fatwas = fatwas[:5]
        if not fatwas:
            cards.append(_empty_sect_card(sect, reason="no keyword hits"))
            continue
        primary = fatwas[0]
        # We don't synthesize — we cite the top fatwa text directly.
        answer = (primary.get("answer_text") or "").strip()
        if not answer:
            answer = (primary.get("question_text") or "").strip()
        cards.append({
            "maslak": _SECT_LABELS[sect],
            "sect": sect,
            "source_label": _SECT_SOURCE_DESC[sect],
            "answer": answer or "(empty fatwa body)",
            "no_match": False,
            "sources": [_source_from_school_record(r) for r in fatwas],
            "num_chunks": len(fatwas),
            "elapsed_ms": 0,
        })
    return cards


def _convert_pageindex_to_sects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert PageIndex 4-school output to 3-sect Islam360 shape.

    Identical merge-semantics as ``_convert_raw_fatwas_to_sects``
    (IslamQA + FatwaQA → Ahle Hadees).  PageIndex's per-fatwa dict
    already carries ``relevance_pct`` rather than raw keyword score,
    but ``_source_from_school_record`` treats ``score`` as a
    percentage so we normalise upstream.
    """
    by_sect: dict[str, list[dict]] = {s: [] for s in _ALL_SECTS}
    for school_card in payload.get("results") or []:
        sid = school_card.get("school_id", "")
        sect = _SCHOOL_TO_SECT.get(sid)
        if not sect:
            continue
        for rank, fatwa in enumerate(school_card.get("fatawa") or [], 1):
            f = dict(fatwa)
            f.setdefault("school_id", sid)
            f.setdefault("school_label", school_card.get("school_label", ""))
            f.setdefault("madhab", school_card.get("maslak", ""))
            # Map relevance_pct → score so the downstream converter
            # (and the UI) see a uniform 0-100 field.
            if "score" not in f and f.get("relevance_pct"):
                f["score"] = f["relevance_pct"]
            by_sect[sect].append(f)

    cards: list[dict[str, Any]] = []
    for sect in _ALL_SECTS:
        fatwas = by_sect[sect]
        fatwas.sort(key=lambda r: -float(r.get("score", 0.0) or 0.0))
        fatwas = fatwas[:5]
        if not fatwas:
            cards.append(_empty_sect_card(sect, reason="pageindex empty"))
            continue
        primary = fatwas[0]
        answer = (primary.get("answer_text") or "").strip()
        if not answer:
            answer = (primary.get("question_text") or "").strip()
        cards.append({
            "maslak": _SECT_LABELS[sect],
            "sect": sect,
            "source_label": _SECT_SOURCE_DESC[sect],
            "answer": answer or "(empty fatwa body)",
            "no_match": False,
            "sources": [_source_from_school_record(r) for r in fatwas],
            "num_chunks": len(fatwas),
            "elapsed_ms": 0,
        })
    return cards


# ╔════════════════════════════════════════════════════════════════╗
# ║  Confidence scorers                                             ║
# ╚════════════════════════════════════════════════════════════════╝
#
# Each scorer returns a float in [0, 1].  The orchestrator compares
# these against the thresholds in ``settings``.  All scorers degrade
# gracefully — a None/empty input returns 0, never throws.


def score_raw_fatwas(payload: dict[str, Any]) -> float:
    """Score the raw-fatwas fast probe output.

    Signals used
    ------------
    * ``coverage``  — how many of the 4 schools returned at least one
      hit.  A query whose keywords only match a single school is
      almost never a ``clear keyword match'' query; it's usually a
      spelling coincidence.
    * ``top_quality`` — normalised top-1 score across schools
      (max/100; the raw index already caps at 100).  Queries with a
      strong dominant match usually score ≥ 70.
    * ``spread``    — how sharp the score distribution is.  If the
      top-1 dominates top-3, the match is well-localised.

    Returns 0 if the probe returned nothing or if the index is
    unbuilt.  Returns ≥0.8 only when the probe is confident enough
    that serving the raw fatwa text without further work is safe.
    """
    results = payload.get("results") if payload else None
    if not results:
        return 0.0

    top_scores: list[float] = []        # one per school (0 if school had no hits)
    all_hit_scores: list[float] = []
    schools_with_hits = 0
    for school_card in results:
        fatawa = school_card.get("fatawa") or []
        if not fatawa:
            top_scores.append(0.0)
            continue
        schools_with_hits += 1
        scores = [float(f.get("score", 0.0) or 0.0) for f in fatawa]
        top_scores.append(max(scores))
        all_hit_scores.extend(scores)

    if not all_hit_scores:
        return 0.0

    coverage = schools_with_hits / 4.0
    # Top-quality: the best top-1 across all schools, on 0..1.
    top_quality = min(max(top_scores) / 100.0, 1.0)
    # Spread: top-1 / mean-top-3.  Values >1.2 mean the top is
    # strongly dominant; clamp to [1.0, 2.0] and normalise.
    sorted_scores = sorted(all_hit_scores, reverse=True)[:3]
    mean_top3 = sum(sorted_scores) / len(sorted_scores)
    spread_raw = (sorted_scores[0] / mean_top3) if mean_top3 > 0 else 1.0
    spread = min(max((spread_raw - 1.0), 0.0), 1.0)  # 0..1

    # Weighted combination — top-quality dominates, coverage is a
    # multiplier (we penalise single-school hits heavily because the
    # user expects results from multiple sects), spread is a tiebreak.
    conf = (0.55 * top_quality + 0.10 * spread) * (0.35 + 0.65 * coverage)
    # Hard-floor: if fewer than 2 schools had hits, cap at 0.5 — we
    # should never short-circuit to raw-fatwas on sparse coverage.
    if schools_with_hits < 2:
        conf = min(conf, 0.5)
    return round(conf, 3)


def score_islam360(per_sect: dict[str, dict[str, Any]]) -> float:
    """Score the Islam360 retrieve_fast output.

    Per-sect payloads have already passed the reranker's strict 0.75
    gate — every surviving ``source.rerank_score`` is ≥ 0.75.  So our
    job here is to aggregate coverage (how many sects answered) and
    mean rerank quality.

    Returns 0 when all three sects are empty.  Returns close to 1
    when all three sects have hits with rerank_score ≥ 0.9.
    """
    if not per_sect:
        return 0.0
    sect_tops: list[float] = []
    for sect_code, sect_payload in per_sect.items():
        if not sect_payload or sect_payload.get("no_match"):
            continue
        sources = sect_payload.get("sources") or []
        if not sources:
            continue
        rerank_scores = [
            float(s.get("rerank_score"))
            for s in sources
            if isinstance(s, dict) and s.get("rerank_score") is not None
        ]
        if rerank_scores:
            sect_tops.append(max(rerank_scores))
        else:
            # Sources present but no rerank score attached — treat
            # as pass-value (0.75) so we don't under-weight a sect
            # that passed the gate but didn't emit the field.
            sect_tops.append(0.75)

    if not sect_tops:
        return 0.0

    # Coverage = fraction of the three sects with ≥1 hit.
    coverage = len(sect_tops) / 3.0
    mean_top = sum(sect_tops) / len(sect_tops)
    # 0.35 base from coverage, 0.65 from mean rerank quality.
    return round(0.35 * coverage + 0.65 * mean_top, 3)


def _max_pool_size(per_sect: dict[str, dict[str, Any]]) -> int:
    """Extract the largest ``pool_size`` seen across sects.

    ``retrieve_fast`` emits this in its ``log`` sub-dict.  If no sect
    had any pool at all (e.g. the query tokenised to nothing) this is
    the signal to NOT bother escalating to PageIndex — it'd be just
    as blind.
    """
    best = 0
    for sect_payload in per_sect.values():
        if not isinstance(sect_payload, dict):
            continue
        log_blob = sect_payload.get("log") or {}
        if not isinstance(log_blob, dict):
            continue
        ps = log_blob.get("pool_size")
        try:
            ps_int = int(ps) if ps is not None else 0
        except (TypeError, ValueError):
            ps_int = 0
        if ps_int > best:
            best = ps_int
    return best


def score_pageindex(payload: dict[str, Any]) -> float:
    """Score the PageIndex tree-descent output.

    PageIndex doesn't emit a per-fatwa confidence; its primary
    quality signal is ``relevance_pct`` (derived from the Step-B LLM
    ranking) and ``fatawa`` list presence per school.

    Returns 0 when every school was empty.
    """
    results = payload.get("results") if payload else None
    if not results:
        return 0.0

    top_pcts: list[float] = []
    schools_with_hits = 0
    for school_card in results:
        fatawa = school_card.get("fatawa") or []
        if not fatawa:
            continue
        schools_with_hits += 1
        # relevance_pct is 0..100 (set by rank-decay in search_pageindex.py).
        pcts = [float(f.get("relevance_pct", 0.0) or 0.0) for f in fatawa]
        top_pcts.append(max(pcts))

    if not top_pcts:
        return 0.0
    coverage = schools_with_hits / 4.0
    top_quality = min(max(top_pcts) / 100.0, 1.0)
    return round(0.4 * coverage + 0.6 * top_quality, 3)


# ╔════════════════════════════════════════════════════════════════╗
# ║  Stage runners                                                  ║
# ╚════════════════════════════════════════════════════════════════╝
#
# Each runner wraps its retriever in a timeout + try/except and
# returns a tuple ``(payload, confidence, elapsed_ms, error)``.  A
# runner never raises; orchestration logic is all inside
# ``route_query``.


def _run_raw_fatwas_fast(
    raw_client: Any,
    question: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, float, float, str | None]:
    t0 = time.perf_counter()
    if raw_client is None:
        return None, 0.0, 0.0, "raw_client_missing"
    # ``fast_search`` is the rerank=False variant we added to
    # ``RawFatwasClient`` for this router (see pageindex/raw_fatwas_client.py).
    fn = getattr(raw_client, "fast_search", None) or getattr(raw_client, "search", None)
    if fn is None:
        return None, 0.0, 0.0, "raw_client_no_search_fn"
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn, question)
            payload = fut.result(timeout=timeout_s)
    except _cf.TimeoutError:
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), "timeout"
    except FileNotFoundError:
        # Index not built yet — skip this stage silently.
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), "index_not_built"
    except Exception as exc:
        logger.warning("[router] raw_fatwas probe failed: %s", exc)
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), f"error:{exc}"
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    conf = score_raw_fatwas(payload or {})
    return payload, conf, elapsed, None


def _run_islam360(
    retriever: Any,
    question: str,
    top_k: int,
    timeout_s: float,
) -> tuple[dict[str, dict[str, Any]], float, float, str | None]:
    """Run the three sect retrievals in parallel (same orchestration
    pattern as ``/api/query-all-schools``) and score the aggregate.
    """
    t0 = time.perf_counter()
    if retriever is None:
        return {}, 0.0, 0.0, "retriever_missing"

    def _one(sect: str) -> tuple[str, dict[str, Any]]:
        try:
            return sect, retriever.retrieve_by_sect(question, sect, top_k=top_k)
        except Exception as exc:
            logger.warning("[router] islam360 sect=%s failed: %s", sect, exc)
            return sect, {
                "answer": "(retrieval failed)",
                "sources": [],
                "no_match": True,
                "log": {"error": str(exc)},
            }

    per_sect: dict[str, dict[str, Any]] = {}
    deadline = t0 + timeout_s
    try:
        with _cf.ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(_one, s): s for s in _ALL_SECTS}
            for fut in _cf.as_completed(futs, timeout=timeout_s):
                try:
                    sect, result = fut.result(timeout=max(0.1, deadline - time.perf_counter()))
                    per_sect[sect] = result
                except Exception as exc:
                    logger.warning("[router] islam360 future failed: %s", exc)
    except _cf.TimeoutError:
        logger.warning("[router] islam360 hit overall timeout (%.1fs)", timeout_s)
        # Keep whatever we got — partial results are still useful.

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    # Fill in any sect that didn't come back at all so the scorer
    # treats them as empty.
    for s in _ALL_SECTS:
        per_sect.setdefault(s, {"answer": "", "sources": [], "no_match": True, "log": {}})
    conf = score_islam360(per_sect)
    return per_sect, conf, elapsed, None


def _run_pageindex(
    pi_client: Any,
    question: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, float, float, str | None]:
    t0 = time.perf_counter()
    if pi_client is None:
        return None, 0.0, 0.0, "pi_client_missing"
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(pi_client.search, question)
            payload = fut.result(timeout=timeout_s)
    except _cf.TimeoutError:
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), "timeout"
    except FileNotFoundError:
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), "index_not_built"
    except Exception as exc:
        logger.warning("[router] pageindex fallback failed: %s", exc)
        return None, 0.0, round((time.perf_counter() - t0) * 1000, 1), f"error:{exc}"
    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    conf = score_pageindex(payload or {})
    return payload, conf, elapsed, None


# ╔════════════════════════════════════════════════════════════════╗
# ║  Formatters — produce the canonical response shape             ║
# ╚════════════════════════════════════════════════════════════════╝


def _format_islam360_result(
    question: str,
    per_sect: dict[str, dict[str, Any]],
    islam360_elapsed_ms: float,
) -> list[dict[str, Any]]:
    """Shape ``retrieve_by_sect`` outputs into the three-sect card
    list that ``/api/query-all-schools`` already produces — we keep
    the exact same fields so the frontend doesn't have to branch.
    """
    def _format_sources(raw_sources: list[dict]) -> list[dict]:
        out: list[dict] = []
        for s in (raw_sources or [])[:5]:
            meta = s.get("metadata", s) if isinstance(s, dict) else {}
            folder = meta.get("folder", "") or meta.get("source", "")
            reference_url = (
                meta.get("reference")
                or meta.get("url")
                or (meta.get("date") if str(meta.get("date", "")).startswith("http") else "")
            )
            source_name = (
                meta.get("source_name")
                or folder.replace("-ExtractedData-Output", "").strip()
                or meta.get("source_file", "Islam360")
            )
            out.append({
                "category":    meta.get("category", "—"),
                "source_name": source_name,
                "maslak":      meta.get("maslak", ""),
                "sect":        meta.get("sect", ""),
                "source":      meta.get("source", ""),
                "source_file": meta.get("source_file", meta.get("source", "—")),
                "reference":   reference_url,
                "question":    meta.get("question", ""),
                "answer":      str(meta.get("answer", meta.get("text", "")))[:1500],
                "fatwa_no":    meta.get("fatwa_no", ""),
                "score":       round(s.get("score", s.get("final_score", 0.0)), 4)
                                    if isinstance(s, dict) else 0.0,
                "anchor_hits": s.get("anchor_hits", 0) if isinstance(s, dict) else 0,
                "rerank_score": (
                    round(s.get("rerank_score", 0.0), 3)
                    if isinstance(s, dict) and s.get("rerank_score") is not None
                    else None
                ),
                "rerank_reason": (
                    s.get("rerank_reason", "") if isinstance(s, dict) else ""
                ),
            })
        return out

    cards = []
    for sect in _ALL_SECTS:
        payload = per_sect.get(sect, {})
        sources = _format_sources(payload.get("sources"))
        cards.append({
            "maslak":       _SECT_LABELS[sect],
            "sect":         sect,
            "source_label": _SECT_SOURCE_DESC[sect],
            "answer":       payload.get("answer") or "متعلقہ فتویٰ نہیں ملا۔",
            "no_match":     bool(payload.get("no_match")),
            "sources":      sources,
            "num_chunks":   len(sources),
            "elapsed_ms":   islam360_elapsed_ms if sect == _ALL_SECTS[0] else 0,
        })
    return cards


# ╔════════════════════════════════════════════════════════════════╗
# ║  Orchestrator                                                   ║
# ╚════════════════════════════════════════════════════════════════╝


def route_query(
    question: str,
    *,
    top_k: int = 5,
    retriever: Any = None,
    pi_client: Any = None,
    raw_client: Any = None,
    force_path: str | None = None,
) -> dict[str, Any]:
    """Run the three-stage router and return a unified response.

    Parameters
    ----------
    question
        Raw user question (Urdu, Roman-Urdu, or English).
    top_k
        Passed through to Islam360 for per-sect retrieval.
    retriever, pi_client, raw_client
        Dependency injection — the Flask app passes its singletons
        here so the router stays testable in isolation.
    force_path
        Optional override for debugging / ablation.  Accepted values:
        ``"raw_fatwas"``, ``"islam360"``, ``"pageindex"``.  When set,
        the escalation logic is skipped and that stage alone runs.

    Returns
    -------
    dict
        Same top-level shape as ``/api/query-all-schools`` plus a
        ``routing`` field.  Callers can ``jsonify(...)`` it directly.
    """
    s = get_settings()
    question = (question or "").strip()
    if not question:
        return {
            "error": "question_empty",
            "results": [],
            "routing": {
                "path_used": "none",
                "confidence": 0.0,
                "cache_hit": False,
                "stage_latencies_ms": {},
                "stage_confidences": {},
                "escalation_reasons": ["empty question"],
            },
        }

    # ── 0) cache lookup ────────────────────────────────────────────
    # ``force_path`` is an ablation / debugging escape hatch — when
    # set we MUST run the requested stage, not serve a stale cached
    # result that was produced by a different stage.
    cache_key = _normalise_for_cache(question)
    if not force_path:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.info("[router] cache HIT key=%s q=%r", cache_key, question[:80])
            out = dict(cached)  # shallow copy so we don't mutate the cached entry
            routing = dict(out.get("routing", {}))
            routing["cache_hit"] = True
            out["routing"] = routing
            out["query"] = question
            out["original_question"] = question
            return out

    stage_latencies: dict[str, float | None] = {
        "cache": 0.0,
        "raw_fatwas": None,
        "islam360": None,
        "pageindex": None,
    }
    stage_confidences: dict[str, float] = {}
    escalation_reasons: list[str] = []

    # ── 1) raw-fatwas fast probe ──────────────────────────────────
    if (
        force_path in (None, "raw_fatwas")
        and s.smart_router_raw_fatwas_enabled
        and raw_client is not None
    ):
        rf_payload, rf_conf, rf_elapsed, rf_err = _run_raw_fatwas_fast(
            raw_client, question, timeout_s=s.smart_router_raw_fatwas_timeout_s,
        )
        stage_latencies["raw_fatwas"] = rf_elapsed
        stage_confidences["raw_fatwas"] = rf_conf
        if rf_err:
            escalation_reasons.append(f"raw_fatwas:{rf_err}")
        else:
            if rf_conf >= s.smart_router_raw_fatwas_high_bar or force_path == "raw_fatwas":
                cards = _convert_raw_fatwas_to_sects(rf_payload or {})
                out = {
                    "query": question,
                    "original_question": question,
                    "urdu_question": question,
                    "retrieval_query": question,
                    "effective_top_k": top_k,
                    "detected_sect": None,
                    "category": None,
                    "results": cards,
                    "dry_run": False,
                    "routing": {
                        "path_used": "raw_fatwas",
                        "confidence": rf_conf,
                        "cache_hit": False,
                        "stage_latencies_ms": stage_latencies,
                        "stage_confidences": stage_confidences,
                        "escalation_reasons": escalation_reasons,
                    },
                }
                if not force_path:
                    _cache_put(cache_key, out)
                logger.info(
                    "[router] path=raw_fatwas conf=%.3f lat_ms=%.0f forced=%s q=%r",
                    rf_conf, rf_elapsed, bool(force_path), question[:80],
                )
                return out
            escalation_reasons.append(
                f"raw_fatwas_conf={rf_conf:.2f}<bar={s.smart_router_raw_fatwas_high_bar}"
            )

    # ── 2) islam360 retrieve_fast (primary accurate path) ─────────
    if force_path in (None, "islam360") and retriever is not None:
        per_sect, is_conf, is_elapsed, is_err = _run_islam360(
            retriever, question, top_k,
            timeout_s=s.smart_router_islam360_timeout_s,
        )
        stage_latencies["islam360"] = is_elapsed
        stage_confidences["islam360"] = is_conf
        # Accept the result if at least one sect has a non-empty sources
        # list — the rerank's 0.75 gate already enforces per-source
        # quality; we don't want to throw away a valid single-sect answer.
        any_sect_has_sources = any(
            (per_sect.get(sect, {}).get("sources") or [])
            for sect in _ALL_SECTS
        )
        if (any_sect_has_sources and force_path != "pageindex") or force_path == "islam360":
            cards = _format_islam360_result(question, per_sect, is_elapsed)
            out = {
                "query": question,
                "original_question": question,
                "urdu_question": question,
                "retrieval_query": question,
                "effective_top_k": top_k,
                "detected_sect": None,
                "category": None,
                "results": cards,
                "dry_run": False,
                "routing": {
                    "path_used": "islam360_fast",
                    "confidence": is_conf,
                    "cache_hit": False,
                    "stage_latencies_ms": stage_latencies,
                    "stage_confidences": stage_confidences,
                    "escalation_reasons": escalation_reasons,
                },
            }
            if not force_path:
                _cache_put(cache_key, out)
            logger.info(
                "[router] path=islam360_fast conf=%.3f lat_ms=%.0f sects_with_hits=%d forced=%s q=%r",
                is_conf, is_elapsed,
                sum(1 for sect in _ALL_SECTS if per_sect.get(sect, {}).get("sources")),
                bool(force_path), question[:80],
            )
            return out

        # All three sects empty.  Only escalate to PageIndex if the
        # pool was non-trivial for at least one sect — that's the
        # signal that BM25 *had* candidates and the rerank rejected
        # them all, so PageIndex might pick something different.
        max_pool = _max_pool_size(per_sect)
        if (
            s.smart_router_escalate_if_all_empty
            and max_pool >= s.smart_router_escalate_min_pool_size
        ):
            escalation_reasons.append(
                f"islam360_all_empty_but_pool={max_pool}"
            )
        else:
            escalation_reasons.append(
                f"islam360_all_empty_pool={max_pool}_no_escalation"
            )
            # Return the empty Islam360 result — honest "not found"
            # rather than guess.
            cards = _format_islam360_result(question, per_sect, is_elapsed)
            out = {
                "query": question,
                "original_question": question,
                "urdu_question": question,
                "retrieval_query": question,
                "effective_top_k": top_k,
                "detected_sect": None,
                "category": None,
                "results": cards,
                "dry_run": False,
                "routing": {
                    "path_used": "islam360_fast",
                    "confidence": is_conf,
                    "cache_hit": False,
                    "stage_latencies_ms": stage_latencies,
                    "stage_confidences": stage_confidences,
                    "escalation_reasons": escalation_reasons,
                },
            }
            # Don't cache empty Islam360 results either — same reason
            # as the end-of-pipeline empty below.
            logger.info(
                "[router] path=islam360_fast (empty) conf=%.3f lat_ms=%.0f q=%r",
                is_conf, is_elapsed, question[:80],
            )
            return out

    # ── 3) PageIndex fallback ─────────────────────────────────────
    if (
        force_path in (None, "pageindex")
        and s.smart_router_pageindex_enabled
        and pi_client is not None
    ):
        pi_payload, pi_conf, pi_elapsed, pi_err = _run_pageindex(
            pi_client, question, timeout_s=s.smart_router_pageindex_timeout_s,
        )
        stage_latencies["pageindex"] = pi_elapsed
        stage_confidences["pageindex"] = pi_conf
        if pi_err:
            escalation_reasons.append(f"pageindex:{pi_err}")
        elif pi_conf >= s.smart_router_pageindex_min_confidence or force_path == "pageindex":
            cards = _convert_pageindex_to_sects(pi_payload or {})
            out = {
                "query": question,
                "original_question": question,
                "urdu_question": question,
                "retrieval_query": question,
                "effective_top_k": top_k,
                "detected_sect": None,
                "category": None,
                "results": cards,
                "dry_run": False,
                "routing": {
                    "path_used": "pageindex",
                    "confidence": pi_conf,
                    "cache_hit": False,
                    "stage_latencies_ms": stage_latencies,
                    "stage_confidences": stage_confidences,
                    "escalation_reasons": escalation_reasons,
                },
            }
            if not force_path:
                _cache_put(cache_key, out)
            logger.info(
                "[router] path=pageindex conf=%.3f lat_ms=%.0f forced=%s q=%r",
                pi_conf, pi_elapsed, bool(force_path), question[:80],
            )
            return out
        else:
            escalation_reasons.append(
                f"pageindex_conf={pi_conf:.2f}<bar={s.smart_router_pageindex_min_confidence}"
            )

    # ── 4) nothing cleared any bar — honest empty response ────────
    empty_cards = [_empty_sect_card(sect, reason="no_stage_matched") for sect in _ALL_SECTS]
    out = {
        "query": question,
        "original_question": question,
        "urdu_question": question,
        "retrieval_query": question,
        "effective_top_k": top_k,
        "detected_sect": None,
        "category": None,
        "results": empty_cards,
        "dry_run": False,
        "routing": {
            "path_used": "none",
            "confidence": 0.0,
            "cache_hit": False,
            "stage_latencies_ms": stage_latencies,
            "stage_confidences": stage_confidences,
            "escalation_reasons": escalation_reasons,
        },
    }
    # We deliberately do NOT cache "no match" responses — they're
    # often a sign of a query the retrievers haven't been taught yet,
    # and caching would freeze the poor answer for an hour.
    logger.info(
        "[router] path=none reasons=%s q=%r",
        escalation_reasons, question[:80],
    )
    return out
