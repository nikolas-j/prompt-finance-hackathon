"""Local server for the Taxxa mock chat UI + classifier-admin demo page.

stdlib http.server. Serves the static front-end from frontend/static/ and
exposes:

    GET  /api/health
    GET  /api/methods                 list selectable retrieval methods
    POST /api/ask                     {question, method?} -> answer, sources,
                                       confidence flag, per-feature breakdown
    GET  /api/admin?method=<m>        classifier-shape data for the admin viz
                                       (top features, tree+gini OR coefs, CV
                                       confusion matrix)

Indexes and classifiers load lazily per method and are cached in memory.
The first request for a new method takes a few seconds; subsequent are
instant. The classifier's CV confusion matrix is computed once per method
on first /api/admin hit and cached.

Run:
    uv run python frontend/server.py
    # then open http://localhost:8000/
"""
from __future__ import annotations

import json
import pickle
import socketserver
import sys
import threading
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import (  # noqa: E402
    DATA_DIR,
    EMBED_MODEL,
    JUDGE_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    QA_PATH,
    SYSTEM_PROMPT,
    TRANSLATE_MODEL,
    is_qfi_method,
    top_k_for,
)
from query_rag import (  # noqa: E402
    build_context,
    embed_query,
    load_index,
    retrieval_query_for,
    retrieve_for_method,
)
from signals import extract_for, to_row  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_METHOD = "section_v1"
DEFAULT_PORT = 8000

# Methods exposed in the selector. Order = ranking by classifier-balanced-
# accuracy delta from the chat history (best first). The `recommended` flag
# becomes the default selection in the UI. All listed methods have a
# trained classifier on the full bank.
_METHOD_CATALOG = [
    {"id": "section_v1",              "label": "section_v1 (recommended)",
     "blurb": "Section-aware chunks + cosine retrieval. Best classifier delta.",
     "recommended": True},
    {"id": "baseline",                "label": "baseline (naive RAG)",
     "blurb": "512-token sliding window + cosine. Reference baseline.",
     "recommended": False},
    {"id": "hybrid_section_v1",       "label": "hybrid_section_v1 (BM25+dense)",
     "blurb": "Section chunks + BM25/dense RRF fusion.",
     "recommended": False},
    {"id": "section_v1_qfi",          "label": "section_v1_qfi (Finnish query rewrite)",
     "blurb": "Section + gpt-4o-translated Finnish query before retrieval.",
     "recommended": False},
    {"id": "hybrid_graph_section_v1", "label": "hybrid_graph_section_v1",
     "blurb": "Hybrid retrieval + structural graph expansion.",
     "recommended": False},
    {"id": "hybrid_section_v1_qfi",   "label": "hybrid_section_v1_qfi (hybrid + Finnish rewrite)",
     "blurb": "Hybrid retrieval + gpt-4o-translated Finnish query.",
     "recommended": False},
]

# Pre-derived calibration headline (5-fold OOF on section_v1 / decision_tree).
# Bigger picture is recomputed per-method by /api/admin from the actual CV.
_CALIBRATION = {
    "raw_accuracy_pct": 35,
    "confident_precision_pct": 56,
    "confident_coverage_pct": 39,
    "flag_catch_rate_pct": 74,
    "flag_wrong_rate_pct": 78,
}


# --- Lazy index / classifier caches ---------------------------------------
_INDEX_CACHE: dict[str, tuple] = {}     # method -> (chunks, mat, id_to_idx, k)
_CLASSIFIER_CACHE: dict[str, dict] = {} # method -> bundle dict
_ADMIN_CACHE: dict[str, dict] = {}      # method -> /api/admin payload
_CACHE_LOCK = threading.Lock()


def _get_index(method: str):
    with _CACHE_LOCK:
        if method in _INDEX_CACHE:
            return _INDEX_CACHE[method]
    print(f"[index] loading index for method={method} ...")
    chunks, mat = load_index(method)
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    k = top_k_for(method)
    entry = (chunks, mat, id_to_idx, k)
    with _CACHE_LOCK:
        _INDEX_CACHE[method] = entry
    return entry


def _get_classifier(method: str):
    with _CACHE_LOCK:
        if method in _CLASSIFIER_CACHE:
            return _CLASSIFIER_CACHE[method]
    path = DATA_DIR / f"classifier_{method}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"classifier pickle not found at {path}. "
            f"Run: uv run scripts/train_classifier.py --method {method}"
        )
    with path.open("rb") as f:
        bundle = pickle.load(f)
    print(f"[classifier] loaded {bundle['model_name']} for method={method}")
    with _CACHE_LOCK:
        _CLASSIFIER_CACHE[method] = bundle
    return bundle


# --- OpenAI generation client (single, reused) ----------------------------
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY missing (needed for embeddings)")
_gen_key = LLM_API_KEY or OPENAI_API_KEY
GEN_CLIENT = OpenAI(api_key=_gen_key, base_url=LLM_BASE_URL or None)
print(f"[startup] gen model = {LLM_MODEL}  endpoint = {LLM_BASE_URL or 'openai-default'}")


# --- Pipeline helpers -----------------------------------------------------
def _generate(question: str, hits: list[dict]) -> str:
    user_msg = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    resp = GEN_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return resp.choices[0].message.content or ""


def _inner_estimator(model):
    """Return the actual classifier inside a Pipeline, or the model itself."""
    if hasattr(model, "named_steps"):
        # train_classifier.py: Pipeline([('scale', StandardScaler), ('lr', LR)])
        # The classifier step is always last.
        return list(model.named_steps.values())[-1]
    return model


def _confidence_payload(label: int, proba_pass: float) -> dict:
    """Soft, calibration-grounded wording. Most "wrong" answers in our
    eval were not factually wrong but missing a piece (a sub-clause, an
    aggregation rule, an exception); we phrase the chip accordingly."""
    if label == 1:
        return {
            "label": "Looks complete",
            "score": proba_pass,
            "headline": "Answer looks complete based on the retrieved sources.",
            "detail": (
                f"Calibrated on our eval: about "
                f"{_CALIBRATION['confident_precision_pct']}% of answers flagged this way "
                f"were complete (the rest were usually missing a small detail). "
                f"A quick read of the sources is still a good idea."
            ),
        }
    return {
        "label": "May be incomplete",
        "score": proba_pass,
        "headline": "This answer might be missing a piece - worth a closer look.",
        "detail": (
            f"Calibrated on our eval: about "
            f"{_CALIBRATION['flag_wrong_rate_pct']}% of answers flagged this way "
            f"left out a sub-rule or threshold. Check the cited sources before "
            f"using this answer."
        ),
    }


def _decision_path_for_tree(model, x_row: np.ndarray, feature_names: list[str]) -> list[dict]:
    """For a DT, walk the path the sample took. Returns one dict per node
    visited, in root->leaf order, ready to render."""
    t = model.tree_
    node = 0
    out: list[dict] = []
    while True:
        feat_idx = int(t.feature[node])
        gini = float(t.impurity[node])
        samples = int(t.n_node_samples[node])
        val = t.value[node][0]
        cls = int(val.argmax())
        if feat_idx == -2:  # leaf
            out.append({
                "node": int(node), "is_leaf": True, "depth": len(out),
                "gini": gini, "samples": samples,
                "class_counts": [float(v) for v in val],
                "predicted_class": cls,
            })
            return out
        thresh = float(t.threshold[node])
        feat_name = feature_names[feat_idx]
        actual = float(x_row[feat_idx])
        went_left = actual <= thresh
        out.append({
            "node": int(node), "is_leaf": False, "depth": len(out),
            "gini": gini, "samples": samples,
            "feature": feat_name, "threshold": thresh,
            "actual_value": actual,
            "decision": "left" if went_left else "right",
            "decision_text": (
                f"{feat_name} = {actual:.3f}  {'≤' if went_left else '>'}  {thresh:.3f}"
            ),
        })
        node = int(t.children_left[node] if went_left else t.children_right[node])


def _feature_breakdown(method: str, question: str, hits: list[dict]) -> dict:
    """Per-question contribution view shown both in the chat (compact) and
    on the admin page (expanded). The shape is the same; the UI picks how
    much to render."""
    bundle = _get_classifier(method)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    _, mat, id_to_idx, _ = _get_index(method)

    hit_idx = [id_to_idx[h["chunk_id"]] for h in hits if h["chunk_id"] in id_to_idx]
    hit_vecs = mat[hit_idx] if hit_idx else np.zeros((0, mat.shape[1]), dtype=np.float32)
    sigs = extract_for(method, question, hits, hit_vecs)
    row = np.asarray(to_row(sigs, feature_names), dtype=np.float32).reshape(1, -1)

    pred = int(model.predict(row)[0])
    proba_pass = float(model.predict_proba(row)[0, 1])
    confidence = _confidence_payload(pred, proba_pass)

    inner = _inner_estimator(model)
    model_kind = "decision_tree" if hasattr(inner, "tree_") else "logreg"

    # Per-feature contribution
    if model_kind == "decision_tree":
        # Importances from the trained tree (Gini-based, already normalised).
        importances = inner.feature_importances_
        # Only the features the sample's decision path actually used count
        # toward an explanation; we surface those as "active" plus their
        # global importance for context.
        decision_path = _decision_path_for_tree(inner, row[0], feature_names)
        active_feature_names = {n["feature"] for n in decision_path if not n["is_leaf"]}
        top = sorted(
            ({"name": n, "importance": float(importances[i]),
              "value": float(row[0, i]),
              "active_in_path": (n in active_feature_names)}
             for i, n in enumerate(feature_names)
             if importances[i] > 0),
            key=lambda d: -d["importance"],
        )[:10]
        return {
            "model_kind": model_kind,
            "predicted_class": pred,
            "pass_probability": proba_pass,
            "confidence": confidence,
            "top_contributions": top,
            "decision_path": decision_path,
        }

    # Logistic regression -- score is logit. Per-feature contribution is
    # standardised_value * coefficient (sign matters: + pushes PASS, -
    # pushes FAIL). The scaler is the prior pipeline step.
    if hasattr(model, "named_steps") and "scale" in model.named_steps:
        scaler = model.named_steps["scale"]
        x_std = (row[0] - scaler.mean_) / scaler.scale_
    else:
        x_std = row[0]
    coefs = inner.coef_[0]
    contribs = x_std * coefs
    intercept = float(inner.intercept_[0])
    top = sorted(
        ({"name": feature_names[i], "coef": float(coefs[i]),
          "value": float(row[0, i]),
          "standardised_value": float(x_std[i]),
          "contribution": float(contribs[i])}
         for i in range(len(feature_names))
         if abs(coefs[i]) > 1e-9),
        key=lambda d: -abs(d["contribution"]),
    )[:10]
    return {
        "model_kind": model_kind,
        "predicted_class": pred,
        "pass_probability": proba_pass,
        "confidence": confidence,
        "intercept": intercept,
        "top_contributions": top,
        "decision_path": None,
    }


def _format_sources(hits: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in hits:
        src = (h.get("source_file") or "").replace("\\", "/")
        pretty = src.rsplit("/", 1)[-1]
        if pretty.endswith(".html"):
            pretty = pretty[:-5]
        text = h.get("text", "")
        preview = text[:280] + ("..." if len(text) > 280 else "")
        out.append({
            "file": src,
            "title": pretty,
            "publisher": h.get("publisher", ""),
            "section": h.get("section_title", "") or h.get("statute_name", ""),
            "similarity": float(h.get("similarity", 0.0)),
            "text_preview": preview,
        })
    return out


# --- Endpoints ------------------------------------------------------------
def _handle_methods() -> dict:
    """List every catalog method. Methods with a classifier pickle report
    its kind; methods without one (e.g. hybrid_section_v1_qfi which we
    haven't fully evaluated) are still listed so the user can pick them
    for retrieval, but they show up as 'retrieval only - no classifier'."""
    available = []
    for m in _METHOD_CATALOG:
        pkl = DATA_DIR / f"classifier_{m['id']}.pkl"
        has_classifier = pkl.exists()
        entry = {**m, "has_classifier": has_classifier}
        if has_classifier:
            try:
                bundle = _get_classifier(m["id"])
                entry["model_name"] = bundle["model_name"]
                entry["n_features"] = len(bundle["feature_names"])
            except Exception as ex:
                print(f"[/api/methods] failed to load {m['id']}: {ex}")
                entry["has_classifier"] = False
        available.append(entry)
    return {"methods": available, "default": DEFAULT_METHOD}


def _handle_ask(payload: dict) -> dict:
    q = (payload.get("question") or "").strip()
    method = (payload.get("method") or DEFAULT_METHOD).strip() or DEFAULT_METHOD
    if not q:
        return {"error": "question is required"}

    out = {
        "answer": "",
        "sources": [],
        "confidence": None,
        "feature_breakdown": None,
        "method": method,
        "calibration": _CALIBRATION,
        "error": None,
    }
    try:
        chunks, mat, _, k = _get_index(method)
        retrieval_q = retrieval_query_for(method, q)
        qv = embed_query(retrieval_q)
        hits = retrieve_for_method(method, qv, mat, chunks, k, question=retrieval_q)
        out["sources"] = _format_sources(hits)
        # Every catalog method now has a trained pickle, so always score.
        if (DATA_DIR / f"classifier_{method}.pkl").exists():
            try:
                fb = _feature_breakdown(method, q, hits)
                out["confidence"] = fb["confidence"]
                out["feature_breakdown"] = fb
            except Exception as ex_clf:
                print(f"[/api/ask] classifier error: {ex_clf!r}")
                traceback.print_exc()
        try:
            out["answer"] = _generate(q, hits)
        except Exception as ex_gen:
            print(f"[/api/ask] generation error: {ex_gen!r}")
            out["error"] = (
                f"Generation failed: {type(ex_gen).__name__}: {ex_gen}\n\n"
                "Retrieved sources and confidence flag are still shown below "
                "so you can verify manually."
            )
    except Exception as ex:
        print(f"[/api/ask] fatal error: {ex!r}")
        traceback.print_exc()
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


# --- Admin endpoint -------------------------------------------------------
def _tree_layout(model, feature_names: list[str]) -> dict:
    """Walk the decision tree, return positioned nodes ready to render in
    SVG. Layout is in-order x positions per depth level; the SVG sets a
    natural width based on the leaf count."""
    t = model.tree_
    nodes: list[dict] = []
    edges: list[dict] = []
    in_order: list[int] = []

    def walk(node: int, depth: int) -> None:
        feat_idx = int(t.feature[node])
        gini = float(t.impurity[node])
        samples = int(t.n_node_samples[node])
        val = t.value[node][0]
        cls = int(val.argmax())
        is_leaf = feat_idx == -2

        if not is_leaf:
            left = int(t.children_left[node])
            right = int(t.children_right[node])
            walk(left, depth + 1)
        in_order.append(node)
        depth_map[node] = depth
        meta_map[node] = {
            "node": int(node), "depth": depth, "is_leaf": is_leaf,
            "gini": gini, "samples": samples,
            "class_counts": [int(v) for v in val], "predicted_class": cls,
        }
        if not is_leaf:
            meta_map[node].update({
                "feature": feature_names[feat_idx],
                "threshold": float(t.threshold[node]),
            })
            walk(right, depth + 1)
            edges.append({"from": int(node), "to": int(t.children_left[node]),  "label": "≤"})
            edges.append({"from": int(node), "to": int(t.children_right[node]), "label": ">"})

    depth_map: dict[int, int] = {}
    meta_map: dict[int, dict] = {}
    walk(0, 0)

    # x = in-order index (0..n-1), y = depth. Front-end will scale them.
    for x_idx, node_id in enumerate(in_order):
        meta_map[node_id]["x"] = x_idx
        meta_map[node_id]["y"] = depth_map[node_id]
        nodes.append(meta_map[node_id])

    return {
        "nodes": nodes,
        "edges": edges,
        "max_depth": max((n["depth"] for n in nodes), default=0),
        "n_leaves": sum(1 for n in nodes if n["is_leaf"]),
        "n_nodes": len(nodes),
    }


def _cv_confusion_for(method: str) -> dict | None:
    """5-fold out-of-fold confusion matrix for this method's classifier.
    Mirrors how the classifier metrics in the report were computed."""
    sigs_path = DATA_DIR / f"signals_{method}.json"
    res_path  = DATA_DIR / f"results_{method}.json"
    if not sigs_path.exists() or not res_path.exists():
        return None
    bundle = _get_classifier(method)
    feature_names = bundle["feature_names"]
    with sigs_path.open("r", encoding="utf-8") as f:
        sigs = {r["id"]: r for r in json.load(f)}
    with res_path.open("r", encoding="utf-8") as f:
        res  = {r["id"]: r for r in json.load(f)}
    ids = sorted(set(sigs) & set(res))
    if not ids:
        return None
    X = np.asarray([[sigs[i].get(n, 0.0) for n in feature_names] for i in ids], dtype=np.float32)
    y = np.asarray([1 if res[i].get("passed") else 0 for i in ids])

    # Refit using the saved best hyperparams from the report (so the CV
    # confusion matches what train_classifier reported). Falls back to the
    # pickled model's own params if the report isn't around.
    rep_path = DATA_DIR / f"classifier_{method}_report.json"
    best_model_name = bundle["model_name"]
    if rep_path.exists():
        with rep_path.open("r", encoding="utf-8") as f:
            rep = json.load(f)
        meta = next((m for m in rep.get("models", []) if m["name"] == best_model_name), {})
        params = meta.get("best_params", {})
    else:
        params = {}
    model = bundle["model"]
    # Build a fresh estimator with those params so cross_val_predict refits
    # cleanly per fold (the pickled one is already fit on all data).
    from sklearn.base import clone
    fresh = clone(model)
    try:
        fresh.set_params(**params)
    except Exception:
        pass

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        y_pred = cross_val_predict(fresh, X, y, cv=cv, method="predict")
    except Exception as ex:
        print(f"[cv-confusion] failed for {method}: {ex}")
        return None

    tp = int(((y_pred == 1) & (y == 1)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    # PASS bucket = predicted_CONFIDENT (tp + fp). Precision of that bucket
    # is the "if it says confident, how often is it actually right" number.
    precision_pass = tp / (tp + fp) if (tp + fp) else 0.0
    # FAIL bucket = predicted_VERIFY (tn + fn). Precision of that bucket is
    # "if it says verify, how often is the answer actually wrong".
    precision_fail = tn / (tn + fn) if (tn + fn) else 0.0
    # Recall over actually-wrong answers = "of all wrong answers, how many
    # did we successfully flag for verification".
    recall_fail    = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "raw_accuracy": float(y.mean()),
        "confident_precision":  precision_pass,
        "confident_coverage":   (tp + fp) / n if n else 0.0,
        "flag_catch_rate":      recall_fail,
        "flag_wrong_rate":      precision_fail,
        "splits": "5-fold StratifiedKFold (seed=42), refit per fold",
    }


_SUMMARY_CACHE: dict | None = None


def _handle_summary() -> dict:
    """Cross-method test-set summary: how many questions in the bank, how
    many each method got right, which classifier wins on calibrated
    balanced-accuracy. Pure JSON, cheap to compute (just reads the JSON
    files written by evaluate.py + train_classifier.py)."""
    global _SUMMARY_CACHE
    if _SUMMARY_CACHE is not None:
        return _SUMMARY_CACHE

    per_method: list[dict] = []
    total_questions: int | None = None
    for m in _METHOD_CATALOG:
        mid = m["id"]
        res_path = DATA_DIR / f"results_{mid}.json"
        if not res_path.exists():
            per_method.append({**m, "evaluated_actual": False})
            continue
        with res_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        n = len(results)
        n_pass = sum(1 for r in results if r.get("passed"))
        basics = [r for r in results if r.get("tier") == "basic"]
        n_basic_pass = sum(1 for r in basics if r.get("passed"))
        total_questions = max(total_questions or 0, n)

        # Pull the trained classifier's best CV stats so the headline can
        # cite "best classifier delta" without the frontend doing the join.
        # delta = balanced_accuracy - 0.5 (majority-class balanced acc) -
        # this is the metric the user targeted (+10pp = 0.10 above 0.5).
        rep_path = DATA_DIR / f"classifier_{mid}_report.json"
        bal_acc = delta = None
        model_name = None
        if rep_path.exists():
            with rep_path.open("r", encoding="utf-8") as f:
                rep = json.load(f)
            best = max(
                (mod for mod in rep.get("models", []) if mod.get("name") != "dummy_majority"),
                key=lambda mod: mod.get("cv_mean_balanced_accuracy", 0) or 0,
                default=None,
            )
            if best:
                bal_acc = best.get("cv_mean_balanced_accuracy")
                delta = (bal_acc - 0.5) if bal_acc is not None else None
                model_name = best.get("name")
        per_method.append({
            **m, "evaluated_actual": True,
            "n_total": n, "n_pass": n_pass,
            "pass_rate": n_pass / n if n else 0.0,
            "n_basic_total": len(basics), "n_basic_pass": n_basic_pass,
            "basic_pass_rate": (n_basic_pass / len(basics)) if basics else 0.0,
            "classifier_model": model_name,
            "classifier_bal_acc": bal_acc,
            "classifier_delta_bal_acc": delta,
        })

    # Pick the best method by classifier delta (positive => beats majority).
    best = max(
        (m for m in per_method if m.get("classifier_delta_bal_acc") is not None),
        key=lambda m: m["classifier_delta_bal_acc"],
        default=None,
    )

    _SUMMARY_CACHE = {
        "n_questions": total_questions or 0,
        "methods": per_method,
        "best_method": best["id"] if best else None,
        "best_classifier": best["classifier_model"] if best else None,
        "best_bal_acc": best["classifier_bal_acc"] if best else None,
        "best_delta_bal_acc": best["classifier_delta_bal_acc"] if best else None,
    }
    return _SUMMARY_CACHE


# --- Example pathways for the /learn page -------------------------------
# Two contrasting questions from the bank, both about lahjavero (gift tax),
# so the comparison isolates "pathway shape" rather than topic. Q6 is a
# straight threshold lookup; Q26 requires three rules (aggregation +
# generation skip + minor recipient) that aren't all in the top-k chunks
# on section_v1, so the classifier flags it.
_EXAMPLE_PATHWAYS_METHOD = "section_v1"
_EXAMPLE_PATHWAY_IDS = ("Q6", "Q26")
# Features we surface in the side-by-side comparison. Picked because they
# appear in the section_v1 / decision_tree's nonzero importance set OR are
# easy to explain intuitively.
_EXAMPLE_FEATURES = (
    ("top1_similarity",          "How relevant the best chunk is"),
    ("top1_gap",                 "How clearly the best chunk wins"),
    ("confidence_x_focus",       "Concentration of strong matches"),
    ("top3_similarity_mass",     "Strength of the top-3 combined"),
    ("frac_section_number_present", "Share of chunks with proper § numbers"),
)
_EXAMPLE_CACHE: dict | None = None


def _handle_example_pathways() -> dict:
    """Two real pre-computed examples from the bank for the /learn page.

    No live LLM call -- pulls signals + label from on-disk JSON and runs
    the section_v1 classifier on them. Cached after first computation.
    """
    global _EXAMPLE_CACHE
    if _EXAMPLE_CACHE is not None:
        return _EXAMPLE_CACHE

    method = _EXAMPLE_PATHWAYS_METHOD
    bundle = _get_classifier(method)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    sigs_path = DATA_DIR / f"signals_{method}.json"
    res_path  = DATA_DIR / f"results_{method}.json"
    sigs = {r["id"]: r for r in json.load(sigs_path.open(encoding="utf-8"))}
    res  = {r["id"]: r for r in json.load(res_path.open(encoding="utf-8"))}
    with QA_PATH.open(encoding="utf-8") as f:
        bank = {e["id"]: e for e in json.load(f)["entries"]}

    examples: list[dict] = []
    for qid in _EXAMPLE_PATHWAY_IDS:
        s = sigs.get(qid)
        r = res.get(qid)
        q = bank.get(qid)
        if not (s and r and q):
            continue
        row = np.asarray(
            [[float(s.get(n, 0.0)) for n in feature_names]], dtype=np.float32,
        )
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0, 1])
        examples.append({
            "id": qid,
            "tier": r.get("tier"),
            "question": q["question"],
            "actually_passed": bool(r.get("passed")),
            "n_facts_total": r.get("n_facts_total"),
            "n_facts_passed": r.get("n_facts_passed"),
            "predicted_class": pred,
            "pass_probability": proba,
            "label": "Looks complete" if pred == 1 else "May be incomplete",
            "features": [
                {"name": name, "label": label, "value": float(s.get(name, 0.0))}
                for name, label in _EXAMPLE_FEATURES
            ],
            # Top-3 retrieved sources (titles only; useful pedagogically)
            "top_sources": [
                Path(p.replace("\\", "/")).name.removesuffix(".html")
                for p in (r.get("sources") or [])[:3]
            ],
        })

    _EXAMPLE_CACHE = {
        "method": method,
        "examples": examples,
        "features_legend": [
            {"name": name, "label": label} for name, label in _EXAMPLE_FEATURES
        ],
    }
    return _EXAMPLE_CACHE


def _handle_admin(method: str) -> dict:
    if method in _ADMIN_CACHE:
        return _ADMIN_CACHE[method]

    bundle = _get_classifier(method)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    inner = _inner_estimator(model)

    rep_path = DATA_DIR / f"classifier_{method}_report.json"
    rep = {}
    if rep_path.exists():
        with rep_path.open("r", encoding="utf-8") as f:
            rep = json.load(f)
    meta = next((m for m in rep.get("models", []) if m["name"] == bundle["model_name"]), {})

    payload: dict = {
        "method": method,
        "model_name": bundle["model_name"],
        "models": {
            "embedder": EMBED_MODEL,
            "answer": LLM_MODEL,
            "judge": JUDGE_MODEL,
            "query_translation": TRANSLATE_MODEL if is_qfi_method(method) else None,
        },
        "baseline_accuracy": rep.get("baseline_accuracy"),
        "n_samples": rep.get("n_samples"),
        "n_pass":    rep.get("n_pass"),
        "n_fail":    rep.get("n_fail"),
        "cv": {
            "balanced_accuracy_mean": meta.get("cv_mean_balanced_accuracy"),
            "balanced_accuracy_std":  meta.get("cv_std_balanced_accuracy"),
            "accuracy_mean":          meta.get("cv_mean_accuracy"),
            "accuracy_std":           meta.get("cv_std_accuracy"),
            "f1_mean":                meta.get("cv_mean_f1"),
            "improvement_over_baseline": meta.get("cv_improvement_over_baseline"),
        },
        "best_params": meta.get("best_params", {}),
        "feature_names": feature_names,
    }

    if hasattr(inner, "tree_"):
        payload["model_kind"] = "decision_tree"
        importances = inner.feature_importances_
        ranked = sorted(
            ({"name": feature_names[i], "importance": float(importances[i])}
             for i in range(len(feature_names))),
            key=lambda d: -d["importance"],
        )
        payload["feature_importances"] = ranked
        payload["nonzero_feature_count"] = int((importances > 0).sum())
        payload["tree"] = _tree_layout(inner, feature_names)
        payload["importance_label"] = "Gini importance"
        payload["importance_explainer"] = (
            "Gini importance = fraction of weighted impurity reduction "
            "this feature is responsible for across the tree. Higher = "
            "the feature does more work separating right answers from wrong ones."
        )
    else:
        payload["model_kind"] = "logreg"
        coefs = inner.coef_[0]
        ranked = sorted(
            ({"name": feature_names[i], "coef": float(coefs[i]),
              "abs_coef": float(abs(coefs[i])),
              "direction": "PASS" if coefs[i] > 0 else "FAIL"}
             for i in range(len(feature_names))),
            key=lambda d: -d["abs_coef"],
        )
        payload["coefficients"] = ranked
        payload["intercept"] = float(inner.intercept_[0])
        payload["nonzero_feature_count"] = int((np.abs(coefs) > 1e-9).sum())
        payload["importance_label"] = "|standardised coefficient|"
        payload["importance_explainer"] = (
            "Coefficients are on standardised features, so |coef| values "
            "are directly comparable. Sign indicates direction: positive "
            "pushes the prediction toward PASS, negative toward FAIL."
        )

    payload["cv_confusion"] = _cv_confusion_for(method)
    payload["calibration_headline"] = (
        f"On 5-fold cross-validation: when this classifier says CONFIDENT, "
        f"the answer was right "
        f"{int((payload['cv_confusion'] or {}).get('confident_precision', 0)*100)}% of the time. "
        f"When it says VERIFY, the answer was wrong "
        f"{int((payload['cv_confusion'] or {}).get('flag_wrong_rate', 0)*100)}% of the time."
        if payload.get("cv_confusion") else
        "Calibration unavailable for this method (signals or results missing)."
    )

    _ADMIN_CACHE[method] = payload
    return payload


# --- HTTP plumbing --------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404, f"missing: {path.name}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # Routing
    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/admin":
            return self._send_file(STATIC_DIR / "admin.html", "text/html; charset=utf-8")
        if path in ("/learn", "/learn.html"):
            return self._send_file(STATIC_DIR / "learn.html", "text/html; charset=utf-8")
        if path in ("/roadmap", "/roadmap.html"):
            return self._send_file(STATIC_DIR / "roadmap.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
        if path == "/admin.js":
            return self._send_file(STATIC_DIR / "admin.js", "application/javascript; charset=utf-8")
        if path == "/learn.js":
            return self._send_file(STATIC_DIR / "learn.js", "application/javascript; charset=utf-8")
        if path == "/api/health":
            return self._send_json(200, {"ok": True, "default_method": DEFAULT_METHOD})
        if path == "/api/methods":
            return self._send_json(200, _handle_methods())
        if path == "/api/summary":
            try:
                return self._send_json(200, _handle_summary())
            except Exception as ex:
                traceback.print_exc()
                return self._send_json(500, {"error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/admin":
            qs = urllib.parse.parse_qs(url.query)
            method = (qs.get("method", [DEFAULT_METHOD])[0] or DEFAULT_METHOD).strip()
            try:
                return self._send_json(200, _handle_admin(method))
            except Exception as ex:
                traceback.print_exc()
                return self._send_json(500, {"error": f"{type(ex).__name__}: {ex}"})
        if path == "/api/example_pathways":
            try:
                return self._send_json(200, _handle_example_pathways())
            except Exception as ex:
                traceback.print_exc()
                return self._send_json(500, {"error": f"{type(ex).__name__}: {ex}"})
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            return self.send_error(404, "unknown endpoint")
        try:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw)
        except Exception as ex:
            return self._send_json(400, {"error": f"bad JSON: {ex}"})
        self._send_json(200, _handle_ask(payload))


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    # Warm the default method so the first user request is fast.
    print(f"[startup] warming default method {DEFAULT_METHOD} ...")
    _get_index(DEFAULT_METHOD)
    _get_classifier(DEFAULT_METHOD)
    with ReusableTCPServer((args.host, args.port), Handler) as httpd:
        print(f"\n[ready] http://{args.host}:{args.port}/")
        print(f"[ready] admin viz at http://{args.host}:{args.port}/admin")
        print(f"[ready] roadmap at http://{args.host}:{args.port}/roadmap")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[stop] bye")


if __name__ == "__main__":
    main()
