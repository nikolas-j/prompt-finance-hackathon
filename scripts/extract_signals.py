"""Stage A: extract retrieval + question features for every question in the bank.

Runs the baseline top-k retrieval (no LLM generation, no judge), computes the feature
dict via signals.extract_signals, writes data/signals_{method}.json keyed by question id.

Cheap, deterministic, no chat-LLM cost. Re-run freely whenever features change or the
retrieval method changes.

Usage:
    uv run scripts/extract_signals.py                  # method=baseline
    uv run scripts/extract_signals.py --method graph_v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, DEFAULT_METHOD, QA_PATH, top_k_for
from query_rag import embed_query, load_index, retrieval_query_for, retrieve_for_method
from signals import extract_for, feature_names_for


def signals_path_for(method: str) -> Path:
    return DATA_DIR / f"signals_{method}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method", default=DEFAULT_METHOD,
        help=f"retrieval method name → signals_{{method}}.json (default: {DEFAULT_METHOD})",
    )
    args = ap.parse_args()
    out_path = signals_path_for(args.method)
    feature_names = feature_names_for(args.method)

    with QA_PATH.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    entries = bank["entries"]
    print(f"[signals] method={args.method}  →  {out_path}")
    print(f"[signals] computing {len(feature_names)} features for {len(entries)} questions")

    chunks, mat = load_index(args.method)
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    k = top_k_for(args.method)
    if args.method.startswith("graph"):
        retrieval_mode = "graph-traversal"
    elif args.method.startswith("hybrid"):
        retrieval_mode = "hybrid (dense + BM25, RRF-fused)"
    else:
        retrieval_mode = "embedding"
    print(f"[signals] top_k = {k}  retrieval = {retrieval_mode}")

    out: list[dict] = []
    t0 = time.time()
    for i, e in enumerate(entries, 1):
        # `_qfi` methods translate the question to Finnish before retrieval.
        # The features must reflect what retrieval actually saw, so we pass
        # the translated query to embed + retrieve + extract.
        retrieval_q = retrieval_query_for(args.method, e["question"])
        qv = embed_query(retrieval_q)
        hits = retrieve_for_method(args.method, qv, mat, chunks, k, question=retrieval_q)
        hit_idx = [id_to_idx[h["chunk_id"]] for h in hits]
        hit_vecs = mat[hit_idx]  # already L2-normalised by load_index
        sigs = extract_for(args.method, retrieval_q, hits, hit_vecs)
        out.append({"id": e["id"], "tier": e["tier"], **sigs})
        if i % 10 == 0 or i == len(entries):
            print(f"[signals] {i}/{len(entries)} ({time.time() - t0:.0f}s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_path}  ({len(out)} rows × {len(feature_names)} features)")


if __name__ == "__main__":
    main()
