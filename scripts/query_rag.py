"""Single-question RAG: embed → retrieve → generate.

Usage:
    uv run scripts/query_rag.py "What is the capital income tax rate above 30000 EUR?"

For full evaluation against the question bank, see scripts/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    DEFAULT_METHOD,
    EMBED_DIM,
    EMBED_MODEL,
    GRAPH_MAX_DEPTH,
    GRAPH_PER_NODE_NEIGHBOUR_CAP,
    GRAPH_SEED_K,
    HYBRID_BM25_K,
    HYBRID_DENSE_K,
    HYBRID_FINAL_K,
    HYBRID_GRAPH_BM25_SEED_K,
    HYBRID_GRAPH_DENSE_SEED_K,
    HYBRID_GRAPH_MAX_DEPTH,
    HYBRID_GRAPH_PER_NODE_NEIGHBOUR_CAP,
    HYBRID_RRF_K,
    LEGACY_CHUNKS_PATH,
    LEGACY_EMBEDDINGS_PATH,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    SYSTEM_PROMPT,
    TOP_K,
    index_method_for,
    is_qfi_method,
    top_k_for,
)
from query_translate import translate_question

_embed_client: OpenAI | None = None


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        if not OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY missing — set it in .env")
        _embed_client = OpenAI(api_key=OPENAI_API_KEY)
    return _embed_client


def chunks_path_for(method: str) -> Path:
    return DATA_DIR / f"chunks_{index_method_for(method)}.json"


def embeddings_path_for(method: str) -> Path:
    return DATA_DIR / f"embeddings_{index_method_for(method)}.bin"


def _migrate_legacy_if_needed(chunks_path: Path, embeddings_path: Path) -> None:
    """Rename the old un-versioned files to the requested method (idempotent)."""
    if LEGACY_CHUNKS_PATH.exists() and not chunks_path.exists():
        LEGACY_CHUNKS_PATH.rename(chunks_path)
        print(f"[migrate] {LEGACY_CHUNKS_PATH.name} → {chunks_path.name}")
    if LEGACY_EMBEDDINGS_PATH.exists() and not embeddings_path.exists():
        LEGACY_EMBEDDINGS_PATH.rename(embeddings_path)
        print(f"[migrate] {LEGACY_EMBEDDINGS_PATH.name} → {embeddings_path.name}")


def load_index(method: str = DEFAULT_METHOD) -> tuple[list[dict], np.ndarray]:
    chunks_path = chunks_path_for(method)
    embeddings_path = embeddings_path_for(method)
    _migrate_legacy_if_needed(chunks_path, embeddings_path)
    if not chunks_path.exists() or not embeddings_path.exists():
        idx_method = index_method_for(method)
        hint = (
            f"Run: uv run scripts/build_index.py --method {idx_method}"
            if idx_method != method
            else f"Run: uv run scripts/build_index.py --method {method}"
        )
        raise SystemExit(
            f"missing index for method={method!r} (reads {chunks_path.name}). Expected:\n"
            f"  {chunks_path}\n  {embeddings_path}\n{hint}"
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    raw = np.fromfile(embeddings_path, dtype=np.float32)
    if raw.size % EMBED_DIM:
        raise RuntimeError(f"{embeddings_path.name} size not divisible by EMBED_DIM={EMBED_DIM}")
    mat = raw.reshape(-1, EMBED_DIM)
    if mat.shape[0] != len(chunks):
        print(f"[index] WARNING: {mat.shape[0]:,} embeddings vs {len(chunks):,} chunks — "
              f"using embedded prefix only")
        chunks = chunks[: mat.shape[0]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    print(f"[index] method={method}  {len(chunks):,} chunks, dim={mat.shape[1]}")
    return chunks, mat


def embed_query(text: str) -> np.ndarray:
    client = _get_embed_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def retrieval_query_for(method: str, question: str) -> str:
    """Return the text that should be fed to the embedder + BM25.

    For methods ending in `_qfi`, this is the Finnish translation of the
    question (cached on disk). For all other methods it's the question
    itself, unchanged.

    Generation and judging should still receive the *original* question
    so the answer comes back in the same language the user asked in.
    """
    if is_qfi_method(method):
        return translate_question(question)
    return question


def retrieve(query_vec: np.ndarray, mat: np.ndarray, chunks: list[dict], k: int) -> list[dict]:
    sims = mat @ query_vec
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [{**chunks[i], "similarity": float(sims[i])} for i in top_idx]


# --- Method-aware retrieval dispatch ----------------------------------------
# Graph-mode adds two columns to every hit:
#   `graph_distance`  — 0 if seed, else BFS hop distance from nearest seed
#   `graph_edge`      — edge-type that reached this node (None for seeds)
# Downstream code that only cares about `similarity` keeps working unchanged.
_GRAPHS: dict[int, object] = {}   # cache: id(chunks_list) -> Graph (skip rebuild on reuse)
_BM25_INDEX: dict[int, object] = {}  # cache: id(chunks_list) -> BM25Okapi


def _get_graph_for(chunks: list[dict]):
    """Lazily build and cache the structural graph for this chunks list."""
    from graph import build_graph  # local import keeps baseline runs fast
    key = id(chunks)
    g = _GRAPHS.get(key)
    if g is None:
        g = build_graph(chunks)
        _GRAPHS[key] = g
    return g


def _get_bm25_for(chunks: list[dict]):
    """Lazily build and cache the BM25 index for this chunks list."""
    from hybrid import build_bm25  # local import keeps non-hybrid runs fast
    key = id(chunks)
    b = _BM25_INDEX.get(key)
    if b is None:
        print(f"[hybrid] building BM25 index over {len(chunks):,} chunks…")
        b = build_bm25(chunks)
        _BM25_INDEX[key] = b
    return b


def retrieve_for_method(
    method: str,
    query_vec: np.ndarray,
    mat: np.ndarray,
    chunks: list[dict],
    k: int,
    question: str | None = None,
) -> list[dict]:
    """Pick the retrieval policy by method name.

    Dispatch order matters: `hybrid_graph_*` is checked before plain `hybrid`
    and plain `graph` so the combined retriever wins. Combined retrieval uses
    BM25 ∪ dense seeds, structural BFS expansion, and RRF rerank over the
    candidate pool. Plain hybrid is dense + BM25 RRF only; plain graph is
    dense-seeded BFS with cosine rerank. All other methods fall through to
    the original cosine-only top-k.
    """
    if method.startswith("hybrid_graph"):
        from graph import retrieve_hybrid_graph  # local import
        if question is None:
            raise ValueError(
                "hybrid_graph retrieval needs the raw question text — callers "
                "must pass question=... to retrieve_for_method"
            )
        graph = _get_graph_for(chunks)
        bm25 = _get_bm25_for(chunks)
        return retrieve_hybrid_graph(
            question=question,
            query_vec=query_vec,
            mat=mat,
            chunks=chunks,
            graph=graph,
            bm25=bm25,
            k_dense_seed=HYBRID_GRAPH_DENSE_SEED_K,
            k_bm25_seed=HYBRID_GRAPH_BM25_SEED_K,
            k_final=k,
            max_depth=HYBRID_GRAPH_MAX_DEPTH,
            per_node_cap=HYBRID_GRAPH_PER_NODE_NEIGHBOUR_CAP,
            rrf_k=HYBRID_RRF_K,
        )
    if method.startswith("graph"):
        from graph import retrieve_graph  # local import
        graph = _get_graph_for(chunks)
        return retrieve_graph(
            query_vec=query_vec,
            mat=mat,
            chunks=chunks,
            graph=graph,
            k_seed=GRAPH_SEED_K,
            k_final=k,
            max_depth=GRAPH_MAX_DEPTH,
            per_node_cap=GRAPH_PER_NODE_NEIGHBOUR_CAP,
        )
    if method.startswith("hybrid"):
        from hybrid import retrieve_hybrid  # local import
        if question is None:
            raise ValueError(
                "hybrid retrieval needs the raw question text — callers must "
                "pass question=... to retrieve_for_method"
            )
        bm25 = _get_bm25_for(chunks)
        return retrieve_hybrid(
            question=question,
            query_vec=query_vec,
            mat=mat,
            chunks=chunks,
            bm25=bm25,
            k_dense=HYBRID_DENSE_K,
            k_bm25=HYBRID_BM25_K,
            k_final=k,
            rrf_k=HYBRID_RRF_K,
        )
    return retrieve(query_vec, mat, chunks, k)


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(
        f"[{h['text']}]\n(source: {h['source_file']})" for h in hits
    )


def generate(question: str, hits: list[dict], client: OpenAI) -> str:
    user_msg = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-question RAG.")
    ap.add_argument("question", help="the question to ask")
    ap.add_argument(
        "--method", default=DEFAULT_METHOD,
        help=f"which built index to query (default: {DEFAULT_METHOD})",
    )
    args = ap.parse_args()

    chunks, mat = load_index(args.method)
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")
    client = OpenAI(api_key=OPENAI_API_KEY)
    retrieval_q = retrieval_query_for(args.method, args.question)
    if retrieval_q != args.question:
        print(f"[qfi] retrieval query (fi): {retrieval_q}")
    qv = embed_query(retrieval_q)
    k = top_k_for(args.method)
    hits = retrieve_for_method(args.method, qv, mat, chunks, k, question=retrieval_q)

    print("\n[retrieved]")
    for h in hits:
        tag = ""
        if "graph_distance" in h:
            d = h["graph_distance"]
            edge = h.get("graph_edge") or "seed"
            tag = f"  d={d} via={edge}"
        elif "rrf_score" in h:
            dr = h.get("dense_rank", -1)
            br = h.get("bm25_rank", -1)
            tag = f"  d_rank={dr} b_rank={br} rrf={h['rrf_score']:.4f} bm25={h['bm25_score']:.2f}"
        print(f"  {h['similarity']:.3f}  {h['source_file']}{tag}")
    print("\n[answer]")
    print(generate(args.question, hits, client))


if __name__ == "__main__":
    main()
