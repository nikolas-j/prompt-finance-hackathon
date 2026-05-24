"""In-memory structural-neighbour graph for section-aware chunks.

The graph is built once at retrieval load time (no extra files on disk) from
the metadata already stored in `chunks_section_*.json`. Edges encode
relationships the *embedder cannot recover from the chunk text itself* —
i.e. document structure. Semantic neighbours are deliberately omitted, since
those are exactly what the seed retrieval step is already best at.

Edges (each undirected unless noted):

    "part"     — chunks that came from the same logical section split into
                 multiple parts by `_split_section_by_paragraphs` (same
                 source_file + section_title + section_number, different
                 part_index). Tightly related: parts 1/3 and 2/3 of one § are
                 effectively one document.

    "sibling"  — chunks that are immediate neighbours in document order
                 inside the *same* source_file (±1 and ±2 hops). Captures the
                 "next § / previous §" relation, which legal text leans on
                 heavily ("see also § 5" implicitly via proximity).

    "chapter"  — same source_file + same `chapter` (finlex "luku"). Capped per
                 node so a 60-§ chapter doesn't dominate every expansion.

    "statute"  — same source_file (any other chunk from the same statute /
                 vero guidance page). Cheapest edge; useful only as a
                 last-resort tiebreaker. Capped tightly.

Cross-references (`cross_refs`) are *external URLs* in this corpus and don't
resolve to local files reliably, so we don't try to use them. If a future
chunker emits internal hrefs that map to `source_file#section_id`, add an
"xref" edge between the two indices.

Public API:

    build_graph(chunks)                    -> Graph
    expand_seeds(seeds, graph, ...)        -> dict[idx -> ExpansionMeta]
    retrieve_graph(query_vec, mat, chunks, graph, k_seed, k_final, ...) -> list[hit_dict]

Hits returned by `retrieve_graph` add two keys on top of the normal chunk
dict + `similarity`:

    graph_distance : int   — 0 if the chunk was an original embedding seed,
                             else the BFS hop distance from the nearest seed.
    graph_edge     : str|None — the edge type that first reached this node
                                (None for seeds).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Edge-type priorities matter when GRAPH_PER_NODE_NEIGHBOUR_CAP truncates a
# node's expansion: earlier edge types are kept first.
DEFAULT_EDGE_PRIORITY = ("part", "sibling", "chapter", "statute")


@dataclass
class Graph:
    """Adjacency representation. Index space is chunk-list positions, so
    callers always work with `chunks[idx]` directly — no separate id table."""

    neighbours: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    # Diagnostic counters, useful when tuning edge generation.
    edge_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def out_edges(self, idx: int) -> list[tuple[int, str]]:
        return self.neighbours.get(idx, [])


@dataclass
class ExpansionMeta:
    distance: int
    edge: str | None     # edge type that first reached this node (None for seeds)
    seed: int            # which original seed reached this node first


# --- Build ----------------------------------------------------------------
def _add(graph: Graph, a: int, b: int, etype: str) -> None:
    if a == b:
        return
    graph.neighbours.setdefault(a, []).append((b, etype))
    graph.neighbours.setdefault(b, []).append((a, etype))
    graph.edge_counts[etype] += 1


def build_graph(chunks: list[dict]) -> Graph:
    """Construct the structural graph from chunk metadata.

    O(n) over chunks for sibling + part edges; O(c²) only inside each chapter
    bucket for chapter edges (chapters in this corpus are small, so this is
    cheap in practice).
    """
    g = Graph()

    # Group chunks by source_file *while preserving document order*.
    # The chunk list as produced by build_index is already in document order
    # per file, so we just walk it once.
    by_file: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_file[c["source_file"]].append(i)

    for indices in by_file.values():
        # --- part edges: same (section_title, section_number) across part_index
        parts_by_key: dict[tuple, list[int]] = defaultdict(list)
        for i in indices:
            c = chunks[i]
            if (c.get("part_total") or 1) <= 1:
                continue
            key = (c.get("section_title", ""), c.get("section_number", ""))
            parts_by_key[key].append(i)
        for group in parts_by_key.values():
            # Fully connect a small group of parts (≤ ~5 in practice).
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    _add(g, group[a], group[b], "part")

        # --- sibling edges: ±1 and ±2 in document order
        for k in range(len(indices)):
            for delta in (1, 2):
                if k + delta < len(indices):
                    _add(g, indices[k], indices[k + delta], "sibling")

        # --- chapter edges: same chapter inside the same file
        by_chapter: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            ch = chunks[i].get("chapter", "")
            if ch:
                by_chapter[ch].append(i)
        # Skip the implicit "no chapter" bucket — that would just be
        # statute-membership again.
        for group in by_chapter.values():
            if len(group) <= 1:
                continue
            # Connect each node to its 4 nearest chapter neighbours (in doc
            # order) — full cliques explode on big chapters and add noise.
            for k, a in enumerate(group):
                for b in group[max(0, k - 2):k] + group[k + 1:k + 3]:
                    _add(g, a, b, "chapter")

        # --- statute edges: weakest. Connect each chunk to up to 2 other
        # chunks in the same file that are NOT already siblings or chapter
        # neighbours. This is a last-resort fallback for files without a
        # populated `chapter` field (most vero pages).
        if len(indices) > 1:
            # Build a quick lookup of who already has a structural edge to whom.
            seen: dict[int, set[int]] = defaultdict(set)
            for src, edges in g.neighbours.items():
                if src in indices:
                    for dst, _ in edges:
                        seen[src].add(dst)
            for k, a in enumerate(indices):
                added = 0
                for b in indices[k + 1:]:
                    if added >= 2:
                        break
                    if b in seen[a]:
                        continue
                    _add(g, a, b, "statute")
                    added += 1

    # Sort each adjacency list by edge priority so that capped expansion
    # consistently prefers stronger relations.
    priority = {e: i for i, e in enumerate(DEFAULT_EDGE_PRIORITY)}
    for idx, edges in g.neighbours.items():
        edges.sort(key=lambda nbr: priority.get(nbr[1], len(priority)))

    return g


# --- Traverse -------------------------------------------------------------
def expand_seeds(
    seeds: Iterable[int],
    graph: Graph,
    max_depth: int,
    per_node_cap: int,
) -> dict[int, ExpansionMeta]:
    """BFS from `seeds` capped to `max_depth` hops and `per_node_cap`
    neighbours considered per expansion step. Returns a mapping from
    *every visited node* (including seeds) to expansion metadata.
    """
    visited: dict[int, ExpansionMeta] = {}
    frontier: list[tuple[int, int]] = []   # (idx, seed_origin)
    for s in seeds:
        if s not in visited:
            visited[s] = ExpansionMeta(distance=0, edge=None, seed=s)
            frontier.append((s, s))

    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier: list[tuple[int, int]] = []
        for node, seed_origin in frontier:
            neighbours = graph.out_edges(node)[:per_node_cap]
            for nbr, etype in neighbours:
                if nbr in visited:
                    continue
                visited[nbr] = ExpansionMeta(
                    distance=depth, edge=etype, seed=seed_origin,
                )
                next_frontier.append((nbr, seed_origin))
        frontier = next_frontier

    return visited


# --- Retrieve -------------------------------------------------------------
def retrieve_graph(
    query_vec: np.ndarray,
    mat: np.ndarray,
    chunks: list[dict],
    graph: Graph,
    k_seed: int,
    k_final: int,
    max_depth: int,
    per_node_cap: int,
) -> list[dict]:
    """Two-stage retrieval: seed by embedding similarity, expand
    structurally, then re-rank the (seed ∪ expanded) candidate set by
    cosine to the query and return the top-`k_final` hits.

    Re-ranking by the *same* similarity keeps the LLM context comparable to
    the baseline section method — what changes is the candidate pool, not
    how relevance is ultimately scored. This is the right knob to A/B:
    "does adding structural neighbours bring in better chunks than the
    next embedding-similar ones would have?"
    """
    sims = mat @ query_vec
    # Seed: top-k_seed by embedding similarity (kept tight; the expansion
    # does the broadening).
    k_seed = max(1, min(k_seed, len(chunks)))
    top_seed = np.argpartition(-sims, k_seed - 1)[:k_seed]
    top_seed = top_seed[np.argsort(-sims[top_seed])]

    visited = expand_seeds(
        seeds=[int(i) for i in top_seed],
        graph=graph,
        max_depth=max_depth,
        per_node_cap=per_node_cap,
    )

    # Re-rank the whole candidate set by similarity and keep top-k_final.
    candidates = list(visited.keys())
    cand_sims = sims[candidates]
    if len(candidates) <= k_final:
        order = np.argsort(-cand_sims)
    else:
        # Partial sort then full sort over the chosen slice.
        part = np.argpartition(-cand_sims, k_final - 1)[:k_final]
        order = part[np.argsort(-cand_sims[part])]

    hits: list[dict] = []
    for k in order:
        idx = candidates[int(k)]
        meta = visited[idx]
        hits.append({
            **chunks[idx],
            "similarity": float(sims[idx]),
            "graph_distance": meta.distance,
            "graph_edge": meta.edge,
            "graph_seed_origin": int(meta.seed),
        })
    return hits


# --- Hybrid + Graph: BM25/dense seed union, structural expansion, RRF rerank
def retrieve_hybrid_graph(
    question: str,
    query_vec: np.ndarray,
    mat: np.ndarray,
    chunks: list[dict],
    graph: Graph,
    bm25,
    k_dense_seed: int,
    k_bm25_seed: int,
    k_final: int,
    max_depth: int,
    per_node_cap: int,
    rrf_k: int,
) -> list[dict]:
    """Combine hybrid retrieval with structural graph expansion.

    Pipeline:
        1. Two parallel seed pools: dense top-k_dense_seed AND BM25
           top-k_bm25_seed. Union (deduped) becomes the BFS starting set.
        2. Structural expansion over the graph (same edges as retrieve_graph).
        3. Fuse-rank the entire candidate pool by RRF of *candidate-pool-
           relative* dense and BM25 ranks, so pure graph-expanded chunks
           (absent from both initial pools) still get a finite score and
           can compete on whichever retriever they're stronger on.
        4. Keep top-k_final.

    Each returned hit carries metadata from all three families so downstream
    feature extraction can decide which signals to read:
        similarity        — dense cosine (always set)
        bm25_score        — raw BM25 score
        dense_rank        — rank in *initial* dense pool, -1 if absent
        bm25_rank         — rank in *initial* BM25 pool, -1 if absent
        rrf_score         — final fusion score (drove the ordering)
        graph_distance    — 0 if seeded directly, else BFS hops from seed
        graph_edge        — edge type that reached this node (None for seeds)
        graph_seed_origin — which seed index first reached this node
    """
    from hybrid import tokenize  # local import keeps non-hybrid runs fast

    # ---- Seeds: union of dense top-K and BM25 top-K -----------------------
    sims = mat @ query_vec
    n = len(chunks)
    k_dense_seed = max(1, min(k_dense_seed, n))
    k_bm25_seed = max(1, min(k_bm25_seed, n))

    dense_seed_idx = np.argpartition(-sims, k_dense_seed - 1)[:k_dense_seed]
    dense_seed_idx = dense_seed_idx[np.argsort(-sims[dense_seed_idx])]
    dense_rank_initial = {int(i): r for r, i in enumerate(dense_seed_idx)}

    q_tokens = tokenize(question)
    if q_tokens:
        bm25_scores = bm25.get_scores(q_tokens)
    else:
        bm25_scores = np.zeros(n, dtype=np.float64)
    bm25_seed_idx = np.argpartition(-bm25_scores, k_bm25_seed - 1)[:k_bm25_seed]
    bm25_seed_idx = bm25_seed_idx[np.argsort(-bm25_scores[bm25_seed_idx])]
    bm25_rank_initial = {int(i): r for r, i in enumerate(bm25_seed_idx)}

    # Stable union preserving dense-first ordering — BM25-only seeds appended
    # at the end. This is purely cosmetic: BFS order through `expand_seeds`
    # depends on iteration order, but the `per_node_cap` makes the result
    # robust to that within a few neighbours.
    seed_set: list[int] = []
    seen: set[int] = set()
    for i in list(dense_seed_idx) + list(bm25_seed_idx):
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        seed_set.append(ii)

    # ---- Structural expansion ---------------------------------------------
    visited = expand_seeds(
        seeds=seed_set,
        graph=graph,
        max_depth=max_depth,
        per_node_cap=per_node_cap,
    )

    # ---- RRF rerank using *candidate-pool-relative* ranks -----------------
    # Re-rank within the candidate pool so graph-only chunks aren't pinned
    # at score 0. Initial pool ranks are still exposed on each hit (as
    # `dense_rank` / `bm25_rank`) for the hybrid signals to read — those
    # signals check membership in the *initial* pools, not the candidate
    # pool ordering.
    candidates = np.array(list(visited.keys()), dtype=np.int64)
    cand_dense = sims[candidates]
    cand_bm25 = bm25_scores[candidates]
    # Order both by descending score; tied scores get adjacent ranks.
    dense_order = np.argsort(-cand_dense)
    bm25_order = np.argsort(-cand_bm25)
    cand_dense_rank = np.empty_like(dense_order)
    cand_bm25_rank = np.empty_like(bm25_order)
    cand_dense_rank[dense_order] = np.arange(len(candidates))
    cand_bm25_rank[bm25_order] = np.arange(len(candidates))

    rrf = 1.0 / (rrf_k + cand_dense_rank) + 1.0 / (rrf_k + cand_bm25_rank)
    if len(candidates) <= k_final:
        order = np.argsort(-rrf)
    else:
        part = np.argpartition(-rrf, k_final - 1)[:k_final]
        order = part[np.argsort(-rrf[part])]

    hits: list[dict] = []
    for k in order:
        idx = int(candidates[k])
        meta = visited[idx]
        hits.append({
            **chunks[idx],
            "similarity": float(sims[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "dense_rank": int(dense_rank_initial.get(idx, -1)),
            "bm25_rank": int(bm25_rank_initial.get(idx, -1)),
            "rrf_score": float(rrf[k]),
            "graph_distance": meta.distance,
            "graph_edge": meta.edge,
            "graph_seed_origin": int(meta.seed),
        })
    return hits


# --- CLI sanity check -----------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import DATA_DIR

    path = DATA_DIR / "chunks_section_v1.json"
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    g = build_graph(chunks)
    n_nodes = len(g.neighbours)
    n_edges = sum(len(v) for v in g.neighbours.values()) // 2
    print(f"chunks: {len(chunks):,}")
    print(f"graph nodes with >=1 edge: {n_nodes:,}")
    print(f"undirected edges: {n_edges:,}")
    print("edges by type:")
    for etype, n in sorted(g.edge_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {etype:<10} {n:,}")
