"""All tuneable constants for the RAG MVP. Change the pipeline here, not in scripts."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)

# --- Repo paths (absolute, resolved from this file's location) ---------------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"

# Two publisher roots; build_index.py globs *.html recursively under each.
FINLEX_DIR = DATA_RAW / "finlex"
VERO_DIR = DATA_RAW / "vero"  # actual leaves live under "Syventävät vero-ohjeet/..."

# Index artefacts (per chunking method) — written by build_index.py:
#   data/chunks_{method}.json     chunk metadata, write-once
#   data/embeddings_{method}.bin  raw float32, append-only
QA_PATH = DATA_DIR / "question_bank.json"

# Legacy artefacts (pre-versioning) — auto-migrated to {method}-named files on next build.
LEGACY_INDEX_PATH = DATA_DIR / "index.json"          # very old: chunks+embeddings combined
LEGACY_CHUNKS_PATH = DATA_DIR / "chunks.json"        # mid-old: split, but un-versioned
LEGACY_EMBEDDINGS_PATH = DATA_DIR / "embeddings.bin"

# Confidence-classifier pipeline artefacts (per method, keyed by question `id`).
# Each script accepts --method NAME (default "baseline") and writes to:
#   data/results_{method}.json           # eval stage: labels (PASS/FAIL) + answers
#   data/signals_{method}.json           # signals stage: per-question feature dict
#   data/classifier_{method}.pkl         # train stage: pickled model + feature order
#   data/classifier_{method}_report.json # train stage: CV metrics + importances
DEFAULT_METHOD = "baseline"

# --- Chunking ---------------------------------------------------------------
CHUNK_SIZE = 512        # tokens (tiktoken cl100k)
CHUNK_OVERLAP = 50      # tokens

# --- Retrieval --------------------------------------------------------------
TOP_K = 5               # chunks per query (baseline ~512-token chunks)
# Section-aware chunks are ~3× smaller than the baseline window, so we retrieve
# more of them to keep the total context delivered to the LLM comparable.
TOP_K_SECTION = 12      # chunks per query for any method starting with "section"

# --- Graph traversal retrieval ---------------------------------------------
# "graph_*" methods reuse the section index but, after a small embedding-based
# seed retrieval, expand to structural neighbours (same statute/chapter, adjacent
# §, sibling parts of a split section) up to GRAPH_MAX_DEPTH hops, then re-rank
# the expanded candidate pool by query·embedding cosine and keep GRAPH_FINAL_K.
# Semantic neighbours are intentionally NOT included as edges — those are
# already found by the seed retrieval itself, so graph edges only add value
# when they are *structural* (information the embedder can't recover).
GRAPH_SEED_K = 6              # initial embedding-based seeds
GRAPH_MAX_DEPTH = 2           # BFS depth from each seed
GRAPH_PER_NODE_NEIGHBOUR_CAP = 6   # max expanded neighbours kept per node per hop
GRAPH_FINAL_K = 12            # how many chunks the LLM ultimately sees
# Edge-type priority order — neighbours from earlier edge types are kept first
# when GRAPH_PER_NODE_NEIGHBOUR_CAP truncates a node's expansion.
GRAPH_EDGE_PRIORITY = ("part", "sibling", "chapter", "statute")

# --- Hybrid retrieval (BM25 + dense, RRF fusion) ----------------------------
# "hybrid_*" methods reuse the SECTION_V1 index. The recall-miss diagnostic
# (data/fail_diagnostic_section_v1.json) was run on section_v1 and showed
# that recall dominates ranking issues there; BM25 directly attacks recall
# on exact-string anchors — euro thresholds, statute markers, percentages —
# that dense embeddings smooth over. We stack hybrid on top of section_v1
# (rather than baseline) because section_v1's structured chunks already
# outperform baseline, so we compound both improvements.
HYBRID_DENSE_K = 30           # dense pool size before fusion
HYBRID_BM25_K = 30            # BM25 pool size before fusion
HYBRID_FINAL_K = 12           # chunks the LLM ultimately sees (matches TOP_K_SECTION)
HYBRID_RRF_K = 60             # RRF damping constant (60 is the canonical default)

# --- Hybrid + Graph combined retrieval -------------------------------------
# "hybrid_graph_*" methods compose both improvements:
#   1. Seed pool = (dense top-K_seed) ∪ (BM25 top-K_seed)  — high recall on
#      both semantic similarity AND exact-string anchors.
#   2. Structural BFS expansion (same edges as graph_v1: part / sibling /
#      chapter / statute) from every seed up to max_depth hops.
#   3. Final rank = RRF over candidate-pool-relative dense and BM25 ranks,
#      so pure graph-expanded chunks (neither retriever surfaced them) can
#      still compete on whichever signal they're stronger on.
# Smaller seed pools than plain hybrid (10 vs 30 each) because BFS already
# multiplies the candidate set ~4-6× — keeping seeds tight prevents the
# rerank pool from drowning the BM25 / dense agreement signal.
HYBRID_GRAPH_DENSE_SEED_K = 10
HYBRID_GRAPH_BM25_SEED_K = 10
HYBRID_GRAPH_FINAL_K = 12
HYBRID_GRAPH_MAX_DEPTH = 2
HYBRID_GRAPH_PER_NODE_NEIGHBOUR_CAP = 6


def _strip_qfi(method: str) -> str:
    """Drop the `_qfi` (query-translate-to-Finnish) suffix for routing.

    `_qfi` is a retrieval-time wrapper that translates the question to Finnish
    before embedding/BM25. It changes neither the chunks nor the embeddings,
    so any `<base>_qfi` method should route to the same on-disk index and the
    same top-k as `<base>` itself.
    """
    return method[:-len("_qfi")] if method.endswith("_qfi") else method


def top_k_for(method: str) -> int:
    """Per-method top-k handed to the LLM. Section + graph + hybrid methods
    retrieve more, smaller chunks to keep the total context comparable."""
    method = _strip_qfi(method)
    # Order matters: longer prefixes must be checked first so "hybrid_graph_*"
    # doesn't match the plain "hybrid_*" branch.
    if method.startswith("hybrid_graph"):
        return HYBRID_GRAPH_FINAL_K
    if method.startswith("graph"):
        return GRAPH_FINAL_K
    if method.startswith("hybrid"):
        return HYBRID_FINAL_K
    if method.startswith("section"):
        return TOP_K_SECTION
    return TOP_K


def index_method_for(method: str) -> str:
    """Which on-disk chunks_*.json / embeddings_*.bin to load for a given method.

    Graph, hybrid, and hybrid+graph methods all reuse the section_v1 index —
    none of them changes the chunker, they change the retrieval policy.
    Naming is suffix-based: `hybrid_graph_section_v1` reads the section_v1
    chunks; any other method is its own index. The `_qfi` query-translation
    wrapper is stripped first since it doesn't alter the index either.
    """
    method = _strip_qfi(method)
    if method.startswith("hybrid_graph"):
        # hybrid_graph_section_v1 -> section_v1. Strip the leading method
        # prefix and any remaining "section_" duplicate.
        suffix = method[len("hybrid_graph"):].lstrip("_")
        if suffix.startswith("section_"):
            return suffix
        return f"section_{suffix}" if suffix else "section_v1"
    if method.startswith("graph"):
        # graph_v1 -> section_v1, graph_foo -> section_foo. graph alone -> section_v1.
        suffix = method[len("graph"):].lstrip("_")
        return f"section_{suffix}" if suffix else "section_v1"
    if method.startswith("hybrid"):
        return "section_v1"
    return method


def is_qfi_method(method: str) -> bool:
    """True if this method translates the query to Finnish before retrieval."""
    return method.endswith("_qfi")

# --- Embeddings (OpenAI) ----------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"          # 1536-dim
EMBED_DIM = 1536                                # must match EMBED_MODEL output
EMBED_BATCH_SIZE = 256                          # OpenAI accepts up to 2048 per request
EMBED_CONCURRENCY = 3                           # parallel in-flight requests (Tier 1 = 1M TPM; lower if 429ing)
EMBED_MAX_RETRIES = 8                           # SDK rides out 429 backoff
EMBED_PRICE_PER_M_TOKENS = 0.02                 # USD; text-embedding-3-small
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Corpus filter (tax / accounting relevance) -----------------------------
# Applied to finlex filenames; all of vero/ is kept unconditionally.
TAX_KEYWORDS = (
    "vero", "verotus", "verosopim",
    "tulo", "lähde", "arvonlisä",
    "ennakkop",                       # ennakkoperintä / ennakkopidätys
    "kirjanpito", "tilintarkast",
    "elinkein", "avainhenki",
    "perintö", "lahja", "kiinteistö",
    "osinko", "luovutus", "palkka", "eläke",
)

# --- LLM generation (Featherless by default, OpenAI fallback) ----------------
# Generation client is OpenAI-compatible and can target Featherless via base_url.
# If LLM_API_KEY is absent, scripts fall back to OPENAI_API_KEY.
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512     # most QA answers fit in ~150–300 tokens; 512 leaves headroom
EVAL_CONCURRENCY = 1     # parallel chat calls during evaluate.py

# OpenAI-compatible generation endpoint/key.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.featherless.ai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")

# --- Judge (separate, fast, cheap — OpenAI gpt-4o-mini) ---------------------
JUDGE_MODEL = "gpt-4o-mini"
JUDGE_MAX_TOKENS = 256   # need headroom for the structured JSON verdict

# --- Query-translation wrapper (`_qfi` methods) ----------------------------
# Powerful, accurate Finnish translation so terminology lines up with the
# corpus. Used only for the retrieval query (embedding + BM25). Generation
# and judging still see the original-language question + answer.
TRANSLATE_MODEL = "gpt-4o"
TRANSLATE_MAX_TOKENS = 400
TRANSLATE_CACHE_PATH = DATA_DIR / "query_translations.json"

# --- Prompts ----------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a Finnish tax expert. Answer using only the provided context. "
    "Be accurate and concise. Address every part of the question. State exact "
    "numbers, percentages, euro thresholds, time periods, and statute references "
    "(e.g. \"PerVL 18 §\") verbatim from the context. Include aggregation or "
    "look-back rules when the context mentions them. Do not invent or hedge. "
    "Reference the relevant source briefly."
)

# Per-key-fact judge prompt. Used as the user message; no system prompt.
# Variables: {key_fact}, {generated_answer}. The judge is called once per fact;
# overall PASS only if every per-fact verdict is PASS.
JUDGE_PROMPT_TEMPLATE = """You are evaluating whether a generated answer demonstrates \
knowledge of a specific key fact. The generated answer may use different wording, \
include extra details, or omit citations — none of these matter. What matters is \
whether the core factual claim is present and correct.

Key fact to verify:
<key_fact>
{key_fact}
</key_fact>

Generated answer to evaluate:
<generated_answer>
{generated_answer}
</generated_answer>

Evaluation rules:
1. Focus only on whether the factual SUBSTANCE is present — numbers, thresholds, \
mechanisms, time periods must be correct
2. Different wording is fine as long as the meaning is equivalent
3. Extra details or citations in the generated answer are irrelevant
4. Missing citations (e.g. "PerVL 18 §") do NOT count as a failure unless the \
key fact itself is specifically about citing that statute
5. A partially correct fact (e.g. right mechanism, wrong number) is INCORRECT
6. If the generated answer explicitly contradicts the key fact it is INCORRECT

Think step by step:
- What is the core factual claim in the key fact?
- Is that specific claim present in the generated answer?
- Are any critical numbers, percentages, thresholds or time periods correct?

Reply with JSON only, no other text:
{{
  "core_claim": "one sentence extracting the essential claim from the key fact",
  "present_in_answer": true or false,
  "reasoning": "one sentence explaining your decision",
  "verdict": "PASS" or "FAIL"
}}"""
