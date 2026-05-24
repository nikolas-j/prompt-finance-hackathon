"""Feature extraction library for the confidence classifier.

Pure functions: given a question + its retrieval result + the chunk vectors,
return a flat dict[str, float]. No LLM calls, no IO, deterministic.

This module intentionally keeps a compact feature set tuned for tiny datasets.
The selected families are retrieval confidence, retrieval agreement, source
focus, and question complexity. A few lightweight interaction features are
included to capture legal-QA failure modes (number mismatch, statute citation
alignment).

To add/remove a feature: update FEATURE_NAMES and extract_signals() — the
training and prediction scripts read FEATURE_NAMES so column order stays
consistent. `to_row` defaults missing keys to 0.0 so an older signals JSON
written before a feature was added still loads.
"""
from __future__ import annotations

import math
import re

import numpy as np

# Order matters: rows in signals.json and matrix columns at train time follow this order.
# Grouped by family for readability; the classifier doesn't care about order.
#
# Feature-set per retrieval method:
#   baseline / section / graph  → BASELINE_FEATURE_NAMES
#   hybrid_*                    → BASELINE_FEATURE_NAMES + HYBRID_FEATURE_NAMES
# Use `feature_names_for(method)` to pick the right list at train / inference
# time. `FEATURE_NAMES` remains the alias for the baseline set (back-compat).
BASELINE_FEATURE_NAMES: list[str] = [
    # --- Retrieval confidence ---
    "top1_similarity",
    "top1_gap",                 # top1 - top2; large = lonely best match
    "top1_dominance",           # top1 - mean(top2..topk)
    "std_similarity",
    "sim_decay_slope",          # OLS slope of sims vs rank
    "softmax_entropy_norm",     # H(softmax(sims)) / log(k); 0 = peaked, 1 = uniform
    "top3_similarity_mass",     # softmax mass in top-3 hits
    "sim_gini",                 # Gini concentration of softmax(sims)

    # --- Retrieval agreement ---
    "chunk_cohesion",           # mean off-diagonal pairwise cosine
    "centroid_concentration",   # mean cosine of each hit to the cluster centroid
    "max_pairwise_distance",    # 1 - min off-diagonal cosine; worst-case spread

    # --- Source focus / structure ---
    "max_statute_share",        # max share of any single statute among top-k
    "statute_switch_rate",      # adjacent-rank statute changes / (k-1)
    "n_unique_files",
    "frac_section_number_present",  # share with non-empty section_number

    # --- Question / context complexity ---
    "total_context_chars",      # total material handed to the LLM
    "question_token_count",
    "n_numbers",                # tokens containing a digit
    "n_conditionals",           # if/when/jos/kun/mikäli/ellei…
    "has_statute_ref",          # 1 if question cites a statute (§, TVL, EPL, …)
    "has_year",                 # 1 if question contains a 4-digit year

    # --- Interaction features ---
    "numeric_question",         # question contains any numeric token
    "numeric_hit_coverage",     # share of hits containing numbers
    "numeric_overlap_ratio",    # share of question numeric tokens found in context
    "statute_ref_alignment",    # statute citation in question and section marks in context
    "confidence_x_focus",       # top1_similarity * max_statute_share

    # --- High-signal lexical alignment (any method) ---
    # These directly check whether the retrieved chunks "talk about the same
    # things" as the question. Strong predictive signal on legal QA where the
    # right § typically reuses the question's domain terms.
    "title_token_jaccard_top1",   # Jaccard(question kwords, top-1 section_title)
    "title_token_jaccard_topk",   # max Jaccard over all retrieved section_titles
    "title_token_coverage_topk",  # share of question content words covered by any title
    "body_content_overlap_top1",  # token overlap with top-1 chunk body (content words)
    "body_content_overlap_topk",  # max body overlap across the top-k chunks
    "rare_term_recall",           # share of "rare" question terms (long / capitalised) found in context

    # --- Retrieval redundancy / diversity ---
    "near_dup_share",             # share of hits whose first 80 chars match another hit
    "statute_diversity_norm",     # n_unique_files / k  (1 - max_statute_share variant)
    "superseded_share",           # share of hits flagged is_superseded=True

    # --- Question structural cues ---
    "has_multipart_question",     # multiple sentences / "and"/"as well as"/"sekä"
    "question_char_count",
    "has_currency_amount",        # detected € / EUR / euro(a)
    "n_question_marks",

    # --- Graph traversal signals (zeros when method is not graph_*) ---
    # These attempt to capture *why* graph expansion succeeded or failed:
    # how much of the answer came from structural neighbours vs. raw seeds,
    # whether the seeds clustered around one statute (high seed_focus), and
    # how big a lift the expansion produced over its own seeds.
    "graph_is_graph_mode",         # 1 if hits carry graph metadata, else 0
    "graph_seed_share_in_final",   # share of final hits that were original seeds (distance==0)
    "graph_neighbour_share_in_final",  # 1 - seed share
    "graph_mean_distance",         # mean graph distance of final hits from nearest seed
    "graph_max_distance",          # deepest final hit
    "graph_top1_is_neighbour",     # 1 if top-1 (by sim) is a non-seed neighbour
    "graph_top1_distance",         # graph distance of top-1
    "graph_unique_seeds_used",     # number of distinct seeds that reached a final hit
    "graph_neighbour_sim_uplift",  # max neighbour sim - mean seed sim (positive = expansion helped)
    "graph_neighbour_sim_share",   # share of total similarity mass from neighbours
]

# Hybrid-specific features. Only meaningful when hits carry the keys produced
# by hybrid.retrieve_hybrid: bm25_score, dense_rank, bm25_rank, rrf_score.
# For non-hybrid methods these columns aren't part of the row at all (the
# classifier for those methods uses BASELINE_FEATURE_NAMES only).
HYBRID_EXTRA_FEATURE_NAMES: list[str] = [
    # --- BM25 / RRF score geometry ---
    "bm25_top1_score",             # raw BM25 score of the fused top-1
    "bm25_score_gap",              # bm25 top1 - top2 inside the fused top-k
    "bm25_score_mean_topk",        # mean BM25 score across the fused top-k
    "rrf_top1_score",              # fused RRF score of the top-1 (higher = more agreement)
    "rrf_top1_gap",                # rrf top1 - top2; large = unambiguous winner

    # --- Retriever agreement ---
    # High agreement (both retrievers surface the same chunks) is the single
    # strongest hybrid-only confidence signal. Disagreement says "one of them
    # is wrong" — useful warning for the classifier.
    "frac_both_in_topk",           # share of fused top-k present in BOTH pools
    "frac_bm25_only_in_topk",      # share present in BM25 pool but not dense
    "frac_dense_only_in_topk",     # share present in dense pool but not BM25
    "top1_in_dense_pool",          # 1 if top-1 also surfaced by dense retriever
    "top1_in_bm25_pool",           # 1 if top-1 also surfaced by BM25
    "top1_both_retrievers",        # 1 if top-1 surfaced by BOTH retrievers

    # --- BM25-specific provenance ---
    # BM25's value is recall on exact-string anchors — these features check
    # whether that's actually happening. If the question has numeric anchors
    # and BM25 surfaced chunks that contain them, the classifier should
    # update upward.
    "bm25_rank_mean_topk",         # mean of bm25_rank for hits that were in BM25 pool
    "dense_rank_mean_topk",        # same for dense
    "query_num_in_top1_chunk",     # 1 if a query numeric token appears verbatim in top-1 text
]

# Full hybrid feature list = baseline ⨁ extras. Defined as a property so a
# future edit to BASELINE_FEATURE_NAMES propagates automatically.
HYBRID_FEATURE_NAMES: list[str] = BASELINE_FEATURE_NAMES + HYBRID_EXTRA_FEATURE_NAMES

# Back-compat alias. Older imports `from signals import FEATURE_NAMES` still
# work and resolve to the baseline set; for hybrid use feature_names_for().
FEATURE_NAMES: list[str] = BASELINE_FEATURE_NAMES


def feature_names_for(method: str) -> list[str]:
    """Per-method feature column order. Hybrid (and hybrid_graph) gets extra
    BM25/RRF columns; other methods stick with the baseline set. Graph
    features live inside BASELINE_FEATURE_NAMES — they fire whenever the
    hits carry `graph_distance`, no separate schema needed."""
    if method.startswith("hybrid"):  # also matches "hybrid_graph_*"
        return HYBRID_FEATURE_NAMES
    return BASELINE_FEATURE_NAMES


# --- Regexes ---------------------------------------------------------------
_CONDITIONAL_RE = re.compile(
    r"\b(if|when|whether|depending|provided|unless|in\s+case|jos|kun|mikäli|ellei)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d")
_NUM_TOKEN_RE = re.compile(r"\b\d+[.,]?\d*\b")
_TOKEN_RE = re.compile(r"\S+")
# Section / statute citation patterns. Finnish abbreviations: TVL, EPL, PerVL, AVL, etc.
_STATUTE_REF_RE = re.compile(r"§|\bTVL\b|\bEPL\b|\bPerVL\b|\bAVL\b|\bEVL\b|\blaki\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SECTION_MARK_RE = re.compile(r"§|\b\d+\s*§\b")
_WORD_RE = re.compile(r"[A-Za-zÅÄÖåäö][A-Za-zÅÄÖåäö0-9'\-]+")
_CURRENCY_RE = re.compile(r"€|\bEUR\b|\beuro(?:a|ja|n)?\b", re.IGNORECASE)
_MULTIPART_RE = re.compile(r"\?\s*[A-ZÅÄÖ]|\band\b|\bas well as\b|\bsekä\b|\b;\s", re.IGNORECASE)

# Very small Finnish + English stop list for content-word extraction. The
# point is not perfect linguistic coverage — it's to drop the tokens that
# would otherwise dominate Jaccard / coverage features ("the", "is", "of"
# / "ja", "on", "että"). Keep it short; bigger lists give worse signal on
# tiny eval samples because they over-prune the question.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at", "to",
    "for", "from", "by", "with", "as", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "have", "has", "had", "this",
    "that", "these", "those", "it", "its", "his", "her", "their", "our",
    "your", "my", "what", "when", "where", "which", "who", "whom", "how",
    "why", "than", "then", "so", "such", "into", "about", "between", "over",
    "under", "any", "all", "some", "no", "not",
    # Finnish (high-frequency function words)
    "ja", "on", "ei", "se", "että", "tai", "kuin", "olla", "ovat", "oli",
    "ovatko", "mitä", "mikä", "miten", "miksi", "milloin", "missä", "joka",
    "jolla", "jonka", "jossa", "jotta", "kun", "vai", "vaan", "mutta",
    "mutta", "kuitenkin", "myös", "vielä", "jo", "ne", "ne", "tämä", "tuo",
    "ei", "ovat", "ovatko", "voi", "voiko", "saa", "saako", "tulee", "pitää",
    "olemme", "olen",
})


def _content_words(text: str) -> set[str]:
    """Lowercased non-stopword word tokens length >= 3.

    Pulls the salient lexical surface of a piece of text. Used for Jaccard /
    coverage features between the question and retrieved titles / bodies.
    Length filter drops single letters and noisy two-letter tokens.
    """
    if not text:
        return set()
    out: set[str] = set()
    for m in _WORD_RE.finditer(text):
        t = m.group(0).lower()
        if len(t) < 3 or t in _STOPWORDS:
            continue
        out.add(t)
    return out


def _rare_terms(text: str) -> set[str]:
    """Long (>= 7 chars) or capitalised tokens — proxy for domain-specific
    Finnish words (e.g. 'arvonlisävero', 'lähdevero', 'aggregation') that
    a generic retriever would normally fail on. High predictive value when
    the question contains rare terms and the context echoes them back."""
    if not text:
        return set()
    rare: set[str] = set()
    for m in _WORD_RE.finditer(text):
        t = m.group(0)
        if len(t) >= 7 or (t[:1].isupper() and len(t) >= 4):
            rare.add(t.lower())
    return rare


def _extract_num_tokens(text: str) -> set[str]:
    return {m.group(0).replace(",", ".") for m in _NUM_TOKEN_RE.finditer(text)}


def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    return e / s if s > 0 else np.full_like(e, 1.0 / len(e))


def _gini(x: np.ndarray) -> float:
    """Gini coefficient for non-negative values."""
    if x.size == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * float((i * xs).sum()) / (n * s)) - (n + 1.0) / n
    return float(max(0.0, min(1.0, g)))


def extract_signals(
    question: str,
    hits: list[dict],
    hit_vectors: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute features for a single (question, retrieval) pair.

    hits         — list of chunk dicts. Required keys: 'similarity', 'source_file', 'text'.
                   Optional metadata used when present: 'is_superseded', 'node_type',
                   'statute_name', 'chapter', 'publisher', 'cross_refs', 'part_total',
                   'section_id', 'section_title', 'graph_distance', 'graph_edge',
                   'graph_seed_origin'.
    hit_vectors  — (k, dim) L2-normalised chunk vectors, in hit order. If None,
                   embedding-cluster features fall back to 0.0.

    Graph-aware features fire automatically when at least one hit carries
    `graph_distance` (i.e. the retrieval was graph-mode). Non-graph methods
    get 0.0 for those columns — that's load-bearing: it keeps the feature
    matrix shape identical across methods, so the classifier comparison in
    summary.py is apples-to-apples.
    """
    k = len(hits)
    sims = np.asarray([h["similarity"] for h in hits], dtype=np.float64)

    # ---------- Retrieval confidence -----------------------------------------
    top1 = float(sims[0]) if k else 0.0
    top2 = float(sims[1]) if k > 1 else top1
    gap = top1 - top2
    top1_dominance = top1 - float(sims[1:].mean()) if k > 1 else 0.0
    std_sim = float(sims.std()) if k > 1 else 0.0
    if k > 1:
        ranks = np.arange(k, dtype=np.float64)
        slope = float(np.polyfit(ranks, sims, 1)[0])
    else:
        slope = 0.0
    if k > 1:
        probs = _safe_softmax(sims)
        # Normalise by max possible entropy log(k) so the feature is in [0, 1].
        entropy_norm = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(k))
        top3_mass = float(probs[: min(3, k)].sum())
        sim_gini = _gini(probs)
    else:
        entropy_norm = 0.0
        top3_mass = 1.0 if k == 1 else 0.0
        sim_gini = 0.0

    # ---------- Retrieval agreement -------------------------------------------
    if hit_vectors is not None and len(hit_vectors) > 1:
        V = np.asarray(hit_vectors, dtype=np.float64)
        gram = V @ V.T
        k_v = gram.shape[0]
        off_diag_sum = gram.sum() - np.trace(gram)
        cohesion = float(off_diag_sum / (k_v * (k_v - 1)))
        centroid = V.mean(axis=0)
        cn = np.linalg.norm(centroid)
        centroid_concentration = float((V @ (centroid / cn)).mean()) if cn > 0 else 0.0
        # Off-diagonal cosines — exclude self-similarities.
        mask = ~np.eye(k_v, dtype=bool)
        off_diag = gram[mask]
        max_pair_dist = float(1.0 - off_diag.min())
    else:
        cohesion = 0.0
        centroid_concentration = 0.0
        max_pair_dist = 0.0

    # ---------- Source focus / structure --------------------------------------
    if k:
        statute_seq = [h.get("statute_name", "") for h in hits]
        statute_counts = {}
        for s in statute_seq:
            if s:
                statute_counts[s] = statute_counts.get(s, 0) + 1
        max_statute_share = (max(statute_counts.values()) / k) if statute_counts else 0.0
        if k > 1:
            statute_switch_rate = sum(
                1 for a, b in zip(statute_seq[:-1], statute_seq[1:]) if a != b
            ) / (k - 1)
        else:
            statute_switch_rate = 0.0
    else:
        max_statute_share = 0.0
        statute_switch_rate = 0.0

    # ---------- Context size / structure --------------------------------------
    n_unique_files = float(len({h["source_file"] for h in hits}))
    chunk_chars = np.asarray([len(h.get("text", "")) for h in hits], dtype=np.float64)
    total_context_chars = float(chunk_chars.sum())
    frac_section_number_present = (
        sum(1 for h in hits if h.get("section_number")) / k if k else 0.0
    )

    # ---------- Question features --------------------------------------------
    tokens = _TOKEN_RE.findall(question)
    n_numbers = sum(1 for t in tokens if _NUMBER_RE.search(t))
    n_conditionals = len(_CONDITIONAL_RE.findall(question))
    has_statute_ref = float(bool(_STATUTE_REF_RE.search(question)))
    has_year = float(bool(_YEAR_RE.search(question)))

    # ---------- Targeted interactions -----------------------------------------
    hit_text = " ".join(h.get("text", "") for h in hits)
    q_nums = _extract_num_tokens(question)
    hit_nums = _extract_num_tokens(hit_text)
    numeric_question = float(bool(q_nums))
    numeric_hit_coverage = (
        sum(1 for h in hits if _NUM_TOKEN_RE.search(h.get("text", ""))) / k if k else 0.0
    )
    numeric_overlap_ratio = (len(q_nums & hit_nums) / len(q_nums)) if q_nums else 0.0
    hit_has_section_marks = float(bool(_SECTION_MARK_RE.search(hit_text)))
    statute_ref_alignment = float(has_statute_ref > 0.0 and hit_has_section_marks > 0.0)
    confidence_x_focus = float(top1 * max_statute_share)

    # ---------- High-signal lexical alignment ---------------------------------
    # Direct check: do the retrieved chunks talk about the same domain terms
    # as the question? On legal QA this is the strongest single retrieval-
    # quality signal we have access to without re-grading the answer.
    q_words = _content_words(question)
    q_rare = _rare_terms(question)
    if k and q_words:
        title_words_each = [_content_words(h.get("section_title", "")) for h in hits]
        body_words_each = [_content_words(h.get("text", "")) for h in hits]

        def _jacc(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            u = a | b
            return len(a & b) / len(u) if u else 0.0

        title_jacc_top1 = _jacc(q_words, title_words_each[0])
        title_jacc_topk = max((_jacc(q_words, tw) for tw in title_words_each), default=0.0)
        # Coverage = share of question content words appearing in ANY retrieved title.
        union_titles = set().union(*title_words_each) if title_words_each else set()
        title_coverage = (len(q_words & union_titles) / len(q_words)) if q_words else 0.0
        # Body overlap is overlap_count / |question_words|: how much of the
        # question's vocabulary the chunk actually echoes (not Jaccard,
        # because chunk bodies are much larger than questions).
        body_overlap_top1 = (
            len(q_words & body_words_each[0]) / len(q_words) if q_words else 0.0
        )
        body_overlap_topk = max(
            (len(q_words & bw) / len(q_words) for bw in body_words_each), default=0.0
        ) if q_words else 0.0
        if q_rare:
            union_bodies = set().union(*body_words_each) if body_words_each else set()
            rare_recall = len(q_rare & union_bodies) / len(q_rare)
        else:
            rare_recall = 0.0
    else:
        title_jacc_top1 = 0.0
        title_jacc_topk = 0.0
        title_coverage = 0.0
        body_overlap_top1 = 0.0
        body_overlap_topk = 0.0
        rare_recall = 0.0

    # ---------- Retrieval redundancy / diversity ------------------------------
    if k > 1:
        starts = [(h.get("text", "") or "")[:80] for h in hits]
        seen: dict[str, int] = {}
        dup_count = 0
        for s in starts:
            if s and s in seen:
                dup_count += 1
            seen[s] = seen.get(s, 0) + 1
        near_dup_share = dup_count / k
    else:
        near_dup_share = 0.0
    statute_diversity_norm = float(n_unique_files / k) if k else 0.0
    superseded_share = (
        sum(1 for h in hits if h.get("is_superseded")) / k if k else 0.0
    )

    # ---------- Question structural cues --------------------------------------
    has_multipart_question = float(
        bool(_MULTIPART_RE.search(question)) or question.count("?") > 1
    )
    question_char_count = float(len(question))
    has_currency_amount = float(bool(_CURRENCY_RE.search(question)))
    n_question_marks = float(question.count("?"))

    # ---------- Graph traversal signals (zero if not in graph mode) -----------
    distances = [h["graph_distance"] for h in hits if "graph_distance" in h]
    is_graph_mode = float(bool(distances))
    if distances:
        seeds = [int(h.get("graph_seed_origin", -1)) for h in hits]
        seed_mask = [d == 0 for d in distances]
        n_seeds = sum(seed_mask)
        seed_share = n_seeds / len(distances)
        neighbour_share = 1.0 - seed_share
        mean_distance = float(np.mean(distances))
        max_distance = float(max(distances))
        top1_is_neighbour = float(distances[0] > 0) if distances else 0.0
        top1_distance = float(distances[0]) if distances else 0.0
        # Only seeds whose `graph_seed_origin` actually appears as a final hit
        # are "used"; an unused seed means structural expansion crowded it out.
        seed_idx_set = {seeds[i] for i, is_s in enumerate(seed_mask) if is_s}
        used_seed_set = {seeds[i] for i in range(len(seeds))}
        unique_seeds_used = float(len(used_seed_set | seed_idx_set))
        seed_sims = [hits[i]["similarity"] for i in range(len(hits)) if seed_mask[i]]
        nbr_sims = [hits[i]["similarity"] for i in range(len(hits)) if not seed_mask[i]]
        if seed_sims and nbr_sims:
            uplift = max(nbr_sims) - (sum(seed_sims) / len(seed_sims))
        else:
            uplift = 0.0
        total_sim = sum(h["similarity"] for h in hits) or 1.0
        nbr_sim_share = sum(nbr_sims) / total_sim if nbr_sims else 0.0
    else:
        seed_share = 0.0
        neighbour_share = 0.0
        mean_distance = 0.0
        max_distance = 0.0
        top1_is_neighbour = 0.0
        top1_distance = 0.0
        unique_seeds_used = 0.0
        uplift = 0.0
        nbr_sim_share = 0.0

    return {
        # similarity geometry
        "top1_similarity": top1,
        "top1_gap": gap,
        "top1_dominance": top1_dominance,
        "std_similarity": std_sim,
        "sim_decay_slope": slope,
        "softmax_entropy_norm": entropy_norm,
        "top3_similarity_mass": top3_mass,
        "sim_gini": sim_gini,
        # retrieval agreement
        "chunk_cohesion": cohesion,
        "centroid_concentration": centroid_concentration,
        "max_pairwise_distance": max_pair_dist,
        # source focus / structure
        "max_statute_share": float(max_statute_share),
        "statute_switch_rate": float(statute_switch_rate),
        "n_unique_files": n_unique_files,
        "frac_section_number_present": float(frac_section_number_present),
        # question / context complexity
        "total_context_chars": total_context_chars,
        "question_token_count": float(len(tokens)),
        "n_numbers": float(n_numbers),
        "n_conditionals": float(n_conditionals),
        "has_statute_ref": has_statute_ref,
        "has_year": has_year,
        # interactions
        "numeric_question": numeric_question,
        "numeric_hit_coverage": float(numeric_hit_coverage),
        "numeric_overlap_ratio": float(numeric_overlap_ratio),
        "statute_ref_alignment": statute_ref_alignment,
        "confidence_x_focus": confidence_x_focus,
        # high-signal lexical alignment
        "title_token_jaccard_top1": float(title_jacc_top1),
        "title_token_jaccard_topk": float(title_jacc_topk),
        "title_token_coverage_topk": float(title_coverage),
        "body_content_overlap_top1": float(body_overlap_top1),
        "body_content_overlap_topk": float(body_overlap_topk),
        "rare_term_recall": float(rare_recall),
        # retrieval redundancy / diversity
        "near_dup_share": float(near_dup_share),
        "statute_diversity_norm": float(statute_diversity_norm),
        "superseded_share": float(superseded_share),
        # question structural cues
        "has_multipart_question": has_multipart_question,
        "question_char_count": question_char_count,
        "has_currency_amount": has_currency_amount,
        "n_question_marks": n_question_marks,
        # graph traversal
        "graph_is_graph_mode": is_graph_mode,
        "graph_seed_share_in_final": float(seed_share),
        "graph_neighbour_share_in_final": float(neighbour_share),
        "graph_mean_distance": float(mean_distance),
        "graph_max_distance": float(max_distance),
        "graph_top1_is_neighbour": float(top1_is_neighbour),
        "graph_top1_distance": float(top1_distance),
        "graph_unique_seeds_used": float(unique_seeds_used),
        "graph_neighbour_sim_uplift": float(uplift),
        "graph_neighbour_sim_share": float(nbr_sim_share),
    }


def extract_hybrid_signals(question: str, hits: list[dict]) -> dict[str, float]:
    """Hybrid-only features. Reads keys put on each hit by hybrid.retrieve_hybrid:
    `bm25_score`, `dense_rank`, `bm25_rank`, `rrf_score`. If those keys are
    absent (e.g. a non-hybrid hit list was passed by mistake) the function
    safely returns zeros — useful so an older signals JSON still loads.
    """
    k = len(hits)
    if not k:
        return {n: 0.0 for n in HYBRID_EXTRA_FEATURE_NAMES}

    bm25_scores = np.asarray([float(h.get("bm25_score", 0.0)) for h in hits], dtype=np.float64)
    rrf_scores = np.asarray([float(h.get("rrf_score", 0.0)) for h in hits], dtype=np.float64)
    dense_ranks = np.asarray([int(h.get("dense_rank", -1)) for h in hits], dtype=np.int64)
    bm25_ranks = np.asarray([int(h.get("bm25_rank", -1)) for h in hits], dtype=np.int64)

    in_dense = dense_ranks >= 0
    in_bm25 = bm25_ranks >= 0
    in_both = in_dense & in_bm25

    bm25_top1 = float(bm25_scores[0])
    bm25_top2 = float(bm25_scores[1]) if k > 1 else bm25_top1
    bm25_gap = bm25_top1 - bm25_top2
    bm25_mean = float(bm25_scores.mean())

    rrf_top1 = float(rrf_scores[0])
    rrf_top2 = float(rrf_scores[1]) if k > 1 else rrf_top1
    rrf_gap = rrf_top1 - rrf_top2

    frac_both = float(in_both.sum() / k)
    frac_bm25_only = float((in_bm25 & ~in_dense).sum() / k)
    frac_dense_only = float((in_dense & ~in_bm25).sum() / k)

    top1_in_dense = float(in_dense[0])
    top1_in_bm25 = float(in_bm25[0])
    top1_both = float(in_both[0])

    # Mean rank within whichever pool the hit was in. Ignore -1 absences,
    # otherwise a hit "missing" from a pool would distort the mean toward
    # the wrong end.
    dense_present = dense_ranks[in_dense]
    bm25_present = bm25_ranks[in_bm25]
    dense_rank_mean = float(dense_present.mean()) if dense_present.size else float(0.0)
    bm25_rank_mean = float(bm25_present.mean()) if bm25_present.size else float(0.0)

    # Did the question carry a distinctive numeric token (≥3 chars) and did
    # it appear verbatim in the top-1 chunk? Mirrors what BM25 was *supposed*
    # to fix on the recall-miss diagnostic.
    q_nums = {n for n in _extract_num_tokens(question) if len(n) >= 3}
    top1_nums = _extract_num_tokens(hits[0].get("text", ""))
    query_num_in_top1 = float(bool(q_nums & top1_nums)) if q_nums else 0.0

    return {
        "bm25_top1_score": bm25_top1,
        "bm25_score_gap": bm25_gap,
        "bm25_score_mean_topk": bm25_mean,
        "rrf_top1_score": rrf_top1,
        "rrf_top1_gap": rrf_gap,
        "frac_both_in_topk": frac_both,
        "frac_bm25_only_in_topk": frac_bm25_only,
        "frac_dense_only_in_topk": frac_dense_only,
        "top1_in_dense_pool": top1_in_dense,
        "top1_in_bm25_pool": top1_in_bm25,
        "top1_both_retrievers": top1_both,
        "bm25_rank_mean_topk": bm25_rank_mean,
        "dense_rank_mean_topk": dense_rank_mean,
        "query_num_in_top1_chunk": query_num_in_top1,
    }


def extract_for(
    method: str,
    question: str,
    hits: list[dict],
    hit_vectors: np.ndarray | None = None,
) -> dict[str, float]:
    """Unified per-method feature extractor.

    Always computes the baseline feature set. For `hybrid_*` methods, also
    computes the hybrid-only extras and merges them in. Returns one flat
    dict; missing keys for the *other* method's features just aren't there
    (and `to_row` defaults them to 0.0 if needed).
    """
    out = extract_signals(question, hits, hit_vectors)
    if method.startswith("hybrid"):
        out.update(extract_hybrid_signals(question, hits))
    return out


def to_row(
    signal_dict: dict[str, float],
    feature_names: list[str] | None = None,
) -> list[float]:
    """Flatten a signal dict into a feature row in `feature_names` order.

    If `feature_names` is None, falls back to the baseline list — keeps older
    call sites working unchanged. Missing keys default to 0.0 so an older
    signals_{method}.json (written before a feature was added) still loads.
    """
    names = feature_names if feature_names is not None else BASELINE_FEATURE_NAMES
    return [float(signal_dict.get(name, 0.0)) for name in names]
