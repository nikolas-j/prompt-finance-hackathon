"""Hybrid retrieval = dense (OpenAI embeddings) + sparse (BM25), fused with RRF.

Motivation: the FAIL diagnostic on section_v1 (see data/fail_diagnostic_section_v1.json)
showed that recall misses outnumber ranking issues ~3:1 on strong evidence —
i.e. the right chunk often isn't even in the top-k, not merely ranked wrong.
Legal text is full of exact-string anchors (precise euro thresholds like
"30 000", statute markers like "§ 124a", percentages like "34 %") that BM25
nails and dense embeddings smooth over. Fusing dense and BM25 with Reciprocal
Rank Fusion (RRF) lets each retriever surface what it's best at.

Method routing: `hybrid_section_v1` (and any `hybrid_*`) reuses the
SECTION_V1 index — same chunks_section_v1.json, same embeddings_section_v1.bin
— so no second build is needed. Only the retrieval policy differs.
Section_v1 already beat baseline in the classifier comparison, and the
recall-miss diagnostic was run on section_v1, so stacking hybrid on top
of section_v1 compounds both improvements.

Public API:

    build_bm25(chunks)                                 -> BM25Okapi
    tokenize(text)                                     -> list[str]
    retrieve_hybrid(question, query_vec, mat, chunks,
                    bm25, k_dense, k_bm25, k_final,
                    rrf_k)                             -> list[hit_dict]

Each returned hit is a normal chunk dict plus:

    similarity      : float  — cosine of the chunk to the query (always set)
    bm25_score      : float  — raw BM25 score (0.0 if chunk wasn't in BM25 pool)
    dense_rank      : int    — 0-based rank in dense pool, -1 if absent
    bm25_rank       : int    — 0-based rank in BM25 pool, -1 if absent
    rrf_score       : float  — fused RRF score used for the final ordering

The hybrid signal extractor (signals.extract_hybrid_signals) reads these
keys; downstream code that only cares about `similarity` keeps working.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi


# --- Tokenisation -----------------------------------------------------------
# Numbers in Finnish legal text often carry a thin/non-breaking space as a
# thousands separator: "30 000", "1 234 567". BM25 tokenisers split on those
# and lose the numeric anchor. We collapse digit-internal spaces *before*
# tokenising so "30 000" becomes the single token "30000" — the same token a
# query like "30000" or "30,000" (after normalisation) will produce.
_NUM_INTERIOR_SPACE_RE = re.compile(r"(\d)[   ](\d)")
_TOKEN_RE = re.compile(r"§|\w+", re.UNICODE)


def _collapse_digit_spaces(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _NUM_INTERIOR_SPACE_RE.sub(r"\1\2", text)
    return text


def tokenize(text: str) -> list[str]:
    """Lowercase, collapse spaces inside numbers, keep `§` as its own token."""
    if not text:
        return []
    text = _collapse_digit_spaces(text)
    # Also normalise thousands-comma to nothing: "30,000" -> "30000".
    # Keep decimal commas alone — too risky to disambiguate ("0,5" should
    # stay one token but "30,000" should collapse). Conservative rule:
    # only collapse comma when both sides have ≥3 digits.
    text = re.sub(r"(\d)[,](\d{3}\b)", r"\1\2", text)
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --- BM25 index -------------------------------------------------------------

def build_bm25(chunks: list[dict]) -> BM25Okapi:
    """Tokenise every chunk's text and build an in-memory BM25Okapi index.

    ~78 000 chunks at ~500 tokens each fits in a couple hundred MB of RAM.
    Built once per process; the query_rag cache keeps it warm across queries.
    """
    corpus = [tokenize(c.get("text", "")) for c in chunks]
    return BM25Okapi(corpus)


# --- Retrieval --------------------------------------------------------------

def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Argsort top-k descending. Uses argpartition for O(n) when k << n."""
    k = min(k, scores.size)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= scores.size:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def retrieve_hybrid(
    question: str,
    query_vec: np.ndarray,
    mat: np.ndarray,
    chunks: list[dict],
    bm25: BM25Okapi,
    k_dense: int,
    k_bm25: int,
    k_final: int,
    rrf_k: int,
) -> list[dict]:
    """Dense top-k_dense ∪ BM25 top-k_bm25, fused by RRF, truncated to k_final.

    RRF score for a candidate doc d:
        score(d) = sum over retrievers r of 1 / (rrf_k + rank_r(d))
    where rank is the 0-based position in that retriever's list. A doc absent
    from a retriever's top-k contributes 0 from that retriever.
    """
    # Dense pool
    sims = mat @ query_vec
    dense_idx = _topk_indices(sims, k_dense)
    dense_rank = {int(i): r for r, i in enumerate(dense_idx)}

    # BM25 pool
    q_tokens = tokenize(question)
    if q_tokens:
        bm25_scores = bm25.get_scores(q_tokens)
    else:
        bm25_scores = np.zeros(len(chunks), dtype=np.float64)
    bm25_idx = _topk_indices(bm25_scores, k_bm25)
    bm25_rank = {int(i): r for r, i in enumerate(bm25_idx)}

    # Fuse: every candidate in either pool gets an RRF score.
    candidates = set(dense_rank) | set(bm25_rank)
    fused: list[tuple[int, float]] = []
    for ci in candidates:
        s = 0.0
        if ci in dense_rank:
            s += 1.0 / (rrf_k + dense_rank[ci])
        if ci in bm25_rank:
            s += 1.0 / (rrf_k + bm25_rank[ci])
        fused.append((ci, s))
    fused.sort(key=lambda kv: kv[1], reverse=True)
    top = fused[:k_final]

    hits: list[dict] = []
    for ci, rrf_score in top:
        hit = {
            **chunks[ci],
            "similarity": float(sims[ci]),
            "bm25_score": float(bm25_scores[ci]),
            "dense_rank": int(dense_rank.get(ci, -1)),
            "bm25_rank": int(bm25_rank.get(ci, -1)),
            "rrf_score": float(rrf_score),
        }
        hits.append(hit)
    return hits
