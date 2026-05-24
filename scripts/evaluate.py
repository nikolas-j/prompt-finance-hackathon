"""Evaluate the RAG pipeline against question_bank.json with LLM-as-judge.

Usage:
    uv run scripts/evaluate.py                                # 100% of the bank, method=baseline
    uv run scripts/evaluate.py 25                             # random 25% sample
    uv run scripts/evaluate.py --method graph_v1              # named run → results_graph_v1.json
    uv run scripts/evaluate.py --method baseline --restart    # wipe & redo this method

Per-method results live in data/results_{method}.json. If the file already exists,
questions already answered are skipped (resume). --restart ignores any existing
file and starts from scratch.

Each question is answered by the RAG pipeline (same retrieval+generation as
query_rag.py) and graded by a fast OpenAI judge (PASS / FAIL only).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    DEFAULT_METHOD,
    EVAL_CONCURRENCY,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PROMPT_TEMPLATE,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    QA_PATH,
    SYSTEM_PROMPT,
    TOP_K,
    top_k_for,
)
from query_rag import (  # type: ignore
    build_context,
    embed_query,
    load_index,
    retrieval_query_for,
    retrieve_for_method,
)


def generate_eval(question: str, hits: list[dict], client: OpenAI) -> str:
    """Chat-completions generation for eval using the configured LLM endpoint.

    Same model and prompt as query_rag.generate, so interactive query results
    match what evaluate.py grades.
    """
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

PRINT_LOCK = Lock()
SAVE_LOCK = Lock()


def save_results(results: list[dict], path: Path) -> None:
    """Atomic-write the results file. Safe to call concurrently (guarded by SAVE_LOCK)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def results_path_for(method: str) -> Path:
    return DATA_DIR / f"results_{method}.json"


def _judge_one_fact(key_fact: str, candidate: str, client: OpenAI) -> dict:
    """Judge a single key_fact against the candidate answer. Always returns a dict."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(key_fact=key_fact, generated_answer=candidate)
    raw = ""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("judge response was not a JSON object")
        verdict = str(parsed.get("verdict", "")).strip().upper()
        return {
            "key_fact": key_fact,
            "core_claim": parsed.get("core_claim", ""),
            "present_in_answer": bool(parsed.get("present_in_answer", False)),
            "reasoning": parsed.get("reasoning", ""),
            "verdict": verdict,
            "passed": verdict == "PASS",
        }
    except Exception as ex:
        return {
            "key_fact": key_fact,
            "core_claim": "",
            "present_in_answer": False,
            "reasoning": f"(judge error: {ex})",
            "verdict": "ERROR",
            "passed": False,
            "raw": raw,
        }


def judge(key_facts: list[str], candidate: str, client: OpenAI) -> tuple[bool, list[dict]]:
    """Per-fact judging. Overall PASS only if every fact PASSES (no partial credit)."""
    if not key_facts or not candidate:
        return False, []
    fact_results = [_judge_one_fact(kf, candidate, client) for kf in key_facts]
    overall = all(f["passed"] for f in fact_results)
    return overall, fact_results


def _process_one(
    idx: int,
    total: int,
    e: dict,
    chunks,
    mat,
    gen_client: OpenAI,
    judge_client: OpenAI,
    k: int,
    method: str,
) -> dict:
    # Defaults so a failure anywhere still produces a sane record with passed=False.
    answer = ""
    hits: list[dict] = []
    gen_err: str | None = None
    judge_err: str | None = None
    fact_judgements: list[dict] = []
    passed = False

    # Key facts: prefer the hand-extracted list; fall back to the full canonical answer
    # for entries where answer_key_facts is empty (some N* entries — see question_bank header).
    key_facts = e.get("answer_key_facts") or [e["answer"]]

    try:
        retrieval_q = retrieval_query_for(method, e["question"])
        qv = embed_query(retrieval_q)
        hits = retrieve_for_method(method, qv, mat, chunks, k, question=retrieval_q)
        try:
            # Generation still sees the ORIGINAL question so the answer comes
            # back in the language the user asked in. Only retrieval was
            # rewritten (e.g. into Finnish for `_qfi` methods).
            answer = generate_eval(e["question"], hits, gen_client)
        except Exception as ex:
            gen_err = str(ex)
        if answer:
            try:
                passed, fact_judgements = judge(key_facts, answer, judge_client)
            except Exception as ex:
                judge_err = str(ex)
    except Exception:
        # last-resort: never let one bad question kill the eval
        gen_err = gen_err or traceback.format_exc(limit=2)

    n_facts_passed = sum(1 for f in fact_judgements if f["passed"])
    n_facts_total = len(fact_judgements)

    def _source_line(h: dict) -> str:
        tag = ""
        if "graph_distance" in h:
            edge = h.get("graph_edge") or "seed"
            tag = f"  d={h['graph_distance']} via={edge}"
        return f"  {h['similarity']:.3f}  {h['source_file']}{tag}"

    block = [
        "=" * 72,
        f"[{idx}/{total}] {e['id']}  ({e['tier']})  →  "
        f"{'PASS' if passed else 'FAIL'}  ({n_facts_passed}/{n_facts_total} facts)",
        f"Q: {e['question']}",
        f"Sources (top-{k}):",
        *[_source_line(h) for h in hits],
        f"A: {answer or '(generation failed: ' + (gen_err or 'no answer') + ')'}",
    ]
    for f in fact_judgements:
        mark = "✓" if f["passed"] else "✗"
        kf_short = (f["key_fact"][:90] + "…") if len(f["key_fact"]) > 90 else f["key_fact"]
        block.append(f"  {mark} {kf_short}")
        if not f["passed"] and f.get("reasoning"):
            block.append(f"      reason: {f['reasoning']}")
    if judge_err:
        block.append(f"(judge failed: {judge_err})")

    with PRINT_LOCK:
        print("\n" + "\n".join(block))

    record = {
        "id": e["id"],
        "tier": e["tier"],
        "question": e["question"],
        "answer_generated": answer,
        "answer_expected": e["answer"],
        "sources": [h["source_file"] for h in hits],
        "top1_similarity": hits[0]["similarity"] if hits else 0.0,
        "passed": passed,
        "n_facts_total": n_facts_total,
        "n_facts_passed": n_facts_passed,
        "fact_judgements": fact_judgements,
        "gen_error": gen_err,
        "judge_error": judge_err,
    }
    # Graph mode adds per-hit traversal metadata. Persist it so downstream
    # signal extraction / debugging doesn't need to re-run retrieval to see
    # which hits were seeds vs. expanded neighbours.
    if hits and "graph_distance" in hits[0]:
        record["graph"] = {
            "distances": [int(h["graph_distance"]) for h in hits],
            "edges": [h.get("graph_edge") for h in hits],
            "seed_origins": [int(h.get("graph_seed_origin", -1)) for h in hits],
        }
    return record


def sample(entries: list[dict], percent: float, seed: int) -> list[dict]:
    if percent >= 100:
        return entries
    n = max(1, round(len(entries) * percent / 100))
    return random.Random(seed).sample(entries, n)


def print_summary(results: list[dict], elapsed: float) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    bar = "=" * 72
    print(f"\n{bar}\nSUMMARY  ({elapsed:.0f}s)\n{bar}")
    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {total - passed}")
    if total:
        print(f"Accuracy: {passed / total * 100:.1f}%")

    by_tier: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        by_tier[r["tier"]]["total"] += 1
        by_tier[r["tier"]]["passed"] += int(r["passed"])
    print(f"\n{'tier':<14}{'pass':>6}{'fail':>6}{'total':>7}{'acc':>9}")
    print("-" * 42)
    for tier in sorted(by_tier):
        s = by_tier[tier]
        acc = s["passed"] / s["total"] * 100
        print(f"{tier:<14}{s['passed']:>6}{s['total'] - s['passed']:>6}{s['total']:>7}{acc:>8.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "percent", type=float, nargs="?", default=100.0,
        help="percentage of question_bank to sample (default: 100)",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    ap.add_argument(
        "--method", default=DEFAULT_METHOD,
        help=f"retrieval method name → results_{{method}}.json (default: {DEFAULT_METHOD})",
    )
    ap.add_argument(
        "--restart", action="store_true",
        help="ignore any existing results file for this method and start from scratch",
    )
    args = ap.parse_args()
    if not (0 < args.percent <= 100):
        ap.error("percent must be in (0, 100]")

    results_path = results_path_for(args.method)
    print(f"[eval] method={args.method}  →  {results_path}")

    # --- Load any prior results for resume ----------------------------------
    results: list[dict] = []
    done_ids: set[str] = set()
    if results_path.exists():
        if args.restart:
            print(f"[restart] discarding existing {results_path.name} "
                  f"({results_path.stat().st_size / 1024:.1f} KB)")
        else:
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
            print(f"[resume] {len(done_ids)} questions already in {results_path.name} — skipping those")

    with QA_PATH.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    sampled = sample(bank["entries"], args.percent, args.seed)
    to_do = [e for e in sampled if e["id"] not in done_ids]
    print(f"[eval] sampled {len(sampled)}/{len(bank['entries'])} "
          f"({args.percent:g}%, seed={args.seed}); {len(to_do)} remaining to answer")

    if not to_do:
        print("[eval] nothing to do — all sampled questions already in results file.")
        print_summary(results, 0.0)
        return

    chunks, mat = load_index(args.method)
    gen_api_key = LLM_API_KEY or OPENAI_API_KEY
    if not gen_api_key:
        raise SystemExit("LLM_API_KEY (or OPENAI_API_KEY fallback) missing — set it in .env")
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — required for judge model")

    gen_client = OpenAI(
        api_key=gen_api_key,
        base_url=LLM_BASE_URL if LLM_BASE_URL else None,
    )
    judge_client = OpenAI(api_key=OPENAI_API_KEY)
    k = top_k_for(args.method)
    gen_provider = LLM_BASE_URL if LLM_BASE_URL else "openai-default"
    print(f"[eval] gen model = {LLM_MODEL}  judge model = {JUDGE_MODEL}  "
          f"gen endpoint = {gen_provider}  "
          f"top_k = {k}  concurrency = {EVAL_CONCURRENCY}")

    t0 = time.time()
    n_target = len(done_ids) + len(to_do)
    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as ex:
        futures = [
            ex.submit(_process_one, len(done_ids) + i, n_target, e,
                      chunks, mat, gen_client, judge_client, k, args.method)
            for i, e in enumerate(to_do, 1)
        ]
        # Block in submission order so the file stays in a stable order.
        # After every completion: append, atomic-save, print progress line.
        for fut in futures:
            r = fut.result()
            with SAVE_LOCK:
                results.append(r)
                save_results(results, results_path)
            n_pass = sum(1 for x in results if x["passed"])
            n_done = len(results)
            with PRINT_LOCK:
                print(f"[progress] {n_done}/{n_target}  "
                      f"PASS={n_pass}  FAIL={n_done - n_pass}  "
                      f"acc={n_pass / n_done * 100:.1f}%  "
                      f"elapsed={time.time() - t0:.0f}s  "
                      f"→ saved to {results_path.name}")

    print_summary(results, time.time() - t0)
    print(f"\n[save] {results_path}")


if __name__ == "__main__":
    main()
