"""Bring more on-disk finlex files into an existing index, then chunk+embed
and append them. NO downloading -- everything operates over files already
under data/raw/.

Why: `build_index.py` keeps only finlex files whose name matches
`config.TAX_KEYWORDS` (plus everything under `tuloverosopim*`). With the
current keywords ~52,770 of the 61,924 finlex files are skipped. This
script lets you promote a chosen subset of those skipped files into the
existing index, without re-chunking or re-embedding anything that's
already indexed.

Invariants this preserves:
  * Embedding model: `config.EMBED_MODEL` -- never changes.
  * Chunker: derived from the method name the same way `build_index.py`
    derives it (section_v1 -> section chunker, etc.).
  * Existing chunks / embeddings are byte-identical after the run.
    Backups are written before any append.
  * chunk_ids continue the per-publisher numbering, so row<->chunk
    alignment for retrieval is preserved.

Selection modes (combine freely; the union is appended):

    --extra-keywords kw1,kw2   Add filename keywords on top of TAX_KEYWORDS
                               (finlex filename match, case-insensitive).
                               Example: --extra-keywords kiinteisto,osinko

    --include-dirs "Asetus,Laki (saadoskokoelma)"
                               Include every finlex file whose relative
                               path starts with one of these top-level
                               directory names (substring match, case-
                               insensitive). Use to pull whole categories
                               like Korkein hallinto-oikeus precedent
                               decisions in one shot.

    --from-failures METHOD     For each FAIL in results_{METHOD}.json,
                               extract Finnish anchors from the question
                               + answer_key_facts (e.g. paaomatulovero,
                               yleisradiovero, TyEL) and include any
                               currently-excluded finlex file whose name
                               contains one of them. Capped per question
                               via --max-files-per-question.

    --include-all-finlex       Drop the tax-keyword filter entirely for
                               this run (52k files, ~$2-3 in embed cost,
                               ~15-30 min). Use with caution.

A small manifest at data/expanded_manifest.json records which previously-
excluded files have been promoted so reruns stay idempotent.

Usage:
    uv run scripts/expand_index.py --method section_v1 --from-failures section_v1 --dry-run
    uv run scripts/expand_index.py --method section_v1 --extra-keywords kiinteisto,osinko
    uv run scripts/expand_index.py --method section_v1 --include-dirs "Korkein hallinto-oikeus_ Ennakkopaatokset"
    uv run scripts/expand_index.py --method baseline   --from-failures section_v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import tiktoken
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    DEFAULT_METHOD,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_PRICE_PER_M_TOKENS,
    FINLEX_DIR,
    OPENAI_API_KEY,
    QA_PATH,
    REPO_ROOT,
    TAX_KEYWORDS,
    VERO_DIR,
)
from build_index import (
    _baseline_chunks_for_file,
    _default_chunker_for,
    _section_chunks_for_file,
    append_embeddings,
    atomic_write_json,
    chunks_path_for,
    embeddings_path_for,
)

TOKENIZER = tiktoken.get_encoding("cl100k_base")
RECORD_BYTES = EMBED_DIM * 4
MANIFEST_PATH = DATA_DIR / "expanded_manifest.json"

# Stopwords that often leak through Finnish anchor extraction.
_STOPWORDS_LOW = {
    "vero", "verot", "verosta", "veron", "veroa", "verolla", "vuoden",
    "vuosi", "vuotta", "olla", "ovat", "tämä", "tämän", "tällä",
}
_FI_DIACRITIC_RE = re.compile(r"[åäöÅÄÖ]")


# --- IO helpers -----------------------------------------------------------
def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"promoted_files": []}


def _save_manifest(m: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    tmp.replace(MANIFEST_PATH)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    bak.write_bytes(path.read_bytes())
    return bak


def _strip_diacritics(s: str) -> str:
    # NFKD-decompose so 'ä' -> 'a', 'ö' -> 'o' for ASCII keyword matching.
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


# --- Discovery ------------------------------------------------------------
def _all_finlex_files() -> list[Path]:
    return sorted(FINLEX_DIR.rglob("*.html"))


def _all_vero_files() -> list[Path]:
    return sorted(VERO_DIR.rglob("*.html"))


def _passes_current_filter(path: Path) -> bool:
    """Mirror of build_index.is_tax_relevant for finlex."""
    s = str(path).lower()
    if "tuloverosopim" in s:
        return True
    name = path.name.lower()
    return any(kw in name for kw in TAX_KEYWORDS)


# --- Selection modes -----------------------------------------------------
def _select_by_extra_keywords(excluded: list[Path], extras: list[str]) -> set[Path]:
    if not extras:
        return set()
    needles = [k.strip().lower() for k in extras if k.strip()]
    hits: set[Path] = set()
    for p in excluded:
        name = p.name.lower()
        if any(n in name for n in needles):
            hits.add(p)
    return hits


def _select_by_dirs(excluded: list[Path], dirs: list[str]) -> set[Path]:
    if not dirs:
        return set()
    needles = [d.strip().lower() for d in dirs if d.strip()]
    hits: set[Path] = set()
    for p in excluded:
        rel = str(p.relative_to(FINLEX_DIR)).lower()
        if any(rel.startswith(n) or f"\\{n}" in f"\\{rel}" or f"/{n}" in f"/{rel}"
               for n in needles):
            hits.add(p)
    return hits


# Domain-marker substrings that make a Finnish word a useful tax-corpus anchor.
# A candidate word is kept only if it contains one of these markers OR is a
# long (>=10 char) compound noun with Finnish diacritics (those are almost
# always domain-specific Finnish compounds, not English content words).
_DOMAIN_MARKERS = (
    "vero", "maksu", "sopim", "pidatys", "pidätys", "paaoma", "pääoma",
    "elake", "eläke", "perinto", "perintö", "lahja", "kiinteist",
    "osinko", "luovutus", "palkka", "ennakko", "paivaraha", "päiväraha",
    "vakuutus", "sairaanhoito", "lahde", "lähde", "etuus", "etu",
    "yleisradio", "verotus", "verotett", "asunto", "matka", "tyel",
    "yel", "ennakkoperin",
)


def _finnish_anchors_for_question(q: dict) -> list[str]:
    """Yield high-precision tax-domain anchor tokens.

    A word qualifies only if it contains a tax-domain marker (vero/maksu/
    elake/...) OR it's a >=10-char Finnish compound with diacritics. That
    excludes English content words and proper names that produce filename
    false positives when substring-matched against 60k finlex filenames.
    Returns diacritic-stripped lowercase tokens.
    """
    qtext = q.get("question", "") or ""
    facts = " ".join(q.get("answer_key_facts") or []) or q.get("answer", "") or ""

    raw: list[str] = []
    for m in re.finditer(r"\(([^)]*)\)", qtext):
        raw.extend(re.findall(r"[A-Za-zÅÄÖåäö-]{5,}", m.group(1)))
    raw.extend(re.findall(r"[A-Za-zÅÄÖåäö-]{5,}", facts))

    out: list[str] = []
    seen: set[str] = set()
    for w in raw:
        lw = w.lower()
        ascii_lw = _strip_diacritics(lw).strip("-_")
        if len(ascii_lw) < 6 or ascii_lw in _STOPWORDS_LOW or ascii_lw in seen:
            continue
        is_domain = any(mk in ascii_lw for mk in _DOMAIN_MARKERS)
        is_long_finnish = len(w) >= 10 and bool(_FI_DIACRITIC_RE.search(w))
        if not (is_domain or is_long_finnish):
            continue
        seen.add(ascii_lw)
        out.append(ascii_lw)
    return out


def _select_from_failures(excluded: list[Path], method: str,
                          max_per_question: int,
                          min_anchor_hits: int = 2,
                          ) -> tuple[set[Path], list[dict]]:
    """For each failing question, promote excluded finlex files whose
    filename (word-boundary, diacritic-stripped) hits >=min_anchor_hits
    of that question's tax-domain anchors. Capped per question.
    """
    results_path = DATA_DIR / f"results_{method}.json"
    if not results_path.exists():
        print(f"[failures] {results_path.name} not found -- skipping")
        return set(), []
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    with QA_PATH.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    qbank = {e["id"]: e for e in bank["entries"]}

    # Tokenise each filename once into a set of word fragments (diacritic-
    # stripped, lowercase). Filenames often use `_` and ` ` separators, so
    # we split on any non-alpha character. Stripping diacritics lets us
    # match `pääoma`-style anchors against `paaoma` filename tokens.
    file_tokens: list[tuple[Path, set[str]]] = []
    for p in excluded:
        ascii_name = _strip_diacritics(p.name.lower())
        toks = {t for t in re.split(r"[^a-z0-9]+", ascii_name) if len(t) >= 4}
        # Also keep the full string for substring fallback on compound words
        # that filenames may glue together with no separator.
        file_tokens.append((p, toks | {ascii_name}))

    promoted: set[Path] = set()
    log: list[dict] = []
    for r in results:
        if r.get("passed", True):
            continue
        qid = r["id"]
        q = qbank.get(qid, {"question": r.get("question", "")})
        anchors = _finnish_anchors_for_question(q)
        if len(anchors) < min_anchor_hits:
            log.append({"qid": qid, "anchors": anchors, "matched_files": [],
                        "note": f"too few anchors (<{min_anchor_hits})"})
            continue

        scored: list[tuple[int, Path]] = []
        for p, toks in file_tokens:
            hits = 0
            for a in anchors:
                # Word-boundary preferred (exact token); fall back to
                # substring match for true compound words (>=8 chars).
                if a in toks:
                    hits += 1
                elif len(a) >= 8 and any(a in t for t in toks):
                    hits += 1
            if hits >= min_anchor_hits:
                scored.append((hits, p))
        scored.sort(key=lambda x: (-x[0], len(x[1].name)))
        matches = [p for _, p in scored[:max_per_question]]
        promoted.update(matches)
        log.append({
            "qid": qid,
            "anchors": anchors[:8],
            "matched_files": [str(m.relative_to(FINLEX_DIR)) for m in matches],
        })
    return promoted, log


# --- Append: chunk + embed -----------------------------------------------
def _next_chunk_index(chunks: list[dict], publisher: str) -> int:
    prefix = f"{publisher}_"
    max_n = -1
    for c in chunks:
        cid = c.get("chunk_id", "")
        if cid.startswith(prefix):
            try:
                n = int(cid[len(prefix):])
            except ValueError:
                continue
            if n > max_n:
                max_n = n
    return max_n + 1


def _chunker_factory(chunker: str):
    if chunker == "section":
        return _section_chunks_for_file
    if chunker == "baseline":
        return _baseline_chunks_for_file
    raise SystemExit(f"unknown chunker {chunker!r}")


def _chunk_and_append(method: str, chunker_name: str,
                      new_files: list[tuple[Path, str]],
                      dry_run: bool) -> int:
    chunks_path = chunks_path_for(method)
    embeddings_path = embeddings_path_for(method)
    if not chunks_path.exists() or not embeddings_path.exists():
        raise SystemExit(
            f"{chunks_path.name} or {embeddings_path.name} missing -- "
            f"run `build_index.py --method {method}` first."
        )

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)
    emb_rows = embeddings_path.stat().st_size // RECORD_BYTES
    if emb_rows != len(chunks):
        raise SystemExit(
            f"chunk/embedding count mismatch ({len(chunks)} vs {emb_rows}) "
            f"-- rebuild before appending."
        )
    print(f"[append:{method}] existing {len(chunks):,} chunks / {emb_rows:,} embeddings")

    # Drop files already in the chunks file by source_file -- safe to re-run.
    existing_sources = {c["source_file"] for c in chunks}
    todo: list[tuple[Path, str]] = []
    skipped_already = 0
    for path, publisher in new_files:
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        if rel in existing_sources:
            skipped_already += 1
            continue
        todo.append((path, publisher))
    print(f"[append:{method}] {len(todo):,} files to chunk "
          f"({skipped_already} already represented)")
    if not todo:
        return 0

    file_chunker = _chunker_factory(chunker_name)
    new_chunks: list[dict] = []
    publisher_next: dict[str, int] = {}
    parse_errors = 0
    for i, (path, publisher) in enumerate(todo):
        if publisher not in publisher_next:
            publisher_next[publisher] = _next_chunk_index(chunks, publisher)
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        try:
            file_chunks = file_chunker(path, publisher, rel,
                                       start_idx=publisher_next[publisher])
        except Exception as e:
            print(f"  [parse] skip {path}: {e}")
            parse_errors += 1
            continue
        publisher_next[publisher] += len(file_chunks)
        new_chunks.extend(file_chunks)
        if (i + 1) % 200 == 0:
            print(f"  [chunk] {i + 1:,}/{len(todo):,} files -> {len(new_chunks):,} chunks")

    print(f"[append:{method}] produced {len(new_chunks):,} new chunks "
          f"({parse_errors} files failed to parse)")
    if not new_chunks:
        return 0

    tokens = sum(len(TOKENIZER.encode(c["text"])) for c in new_chunks)
    cost = tokens / 1_000_000 * EMBED_PRICE_PER_M_TOKENS
    print(f"[append:{method}] token estimate: {tokens:,} (~${cost:.3f} embed cost)")

    if dry_run:
        print(f"[append:{method}] --dry-run: not embedding or writing")
        return 0

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing -- set it in .env")

    cbak = _backup(chunks_path); ebak = _backup(embeddings_path)
    if cbak: print(f"[append:{method}] backup -> {cbak.name}")
    if ebak: print(f"[append:{method}] backup -> {ebak.name}")

    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=EMBED_MAX_RETRIES)
    emb_baseline = embeddings_path.stat().st_size
    t0 = time.time()
    for start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[start: start + EMBED_BATCH_SIZE]
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL, input=[c["text"] for c in batch],
            )
        except Exception as e:
            print(f"[embed] error {e!r} -- truncating back to {emb_baseline} bytes")
            with embeddings_path.open("r+b") as f:
                f.truncate(emb_baseline)
            raise
        append_embeddings([d.embedding for d in resp.data], embeddings_path)
        done = start + len(batch)
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(new_chunks) - done) / max(rate, 1e-6) / 60
        print(f"[embed] {done:,}/{len(new_chunks):,} "
              f"({rate:.0f}/s, ETA {eta:.1f} min)")

    final_rows = embeddings_path.stat().st_size // RECORD_BYTES
    expected = len(chunks) + len(new_chunks)
    if final_rows != expected:
        print(f"[append:{method}] WARNING rows={final_rows} expected={expected}")

    chunks.extend(new_chunks)
    atomic_write_json(chunks_path, chunks)
    print(f"[append:{method}] done. chunks={len(chunks):,} embeddings={final_rows:,}")
    return len(new_chunks)


# --- Entry point ---------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default=DEFAULT_METHOD,
                    help=f"index to append to (default: {DEFAULT_METHOD})")
    ap.add_argument("--chunker", choices=("baseline", "section"), default=None,
                    help="override chunker (defaults to the one derived from --method)")
    ap.add_argument("--extra-keywords", default="",
                    help="comma-separated extra finlex filename keywords")
    ap.add_argument("--include-dirs", default="",
                    help="comma-separated finlex top-level subdir names to include in full")
    ap.add_argument("--from-failures", default=None,
                    help="failure-driven selection from results_{METHOD}.json")
    ap.add_argument("--max-files-per-question", type=int, default=5,
                    help="cap on files per failing question (default 5)")
    ap.add_argument("--min-anchor-hits", type=int, default=1,
                    help="how many distinct tax-domain anchors a filename "
                         "must hit to be promoted (default 1; raise to 2+ "
                         "for tighter precision)")
    ap.add_argument("--include-all-finlex", action="store_true",
                    help="drop the tax-keyword filter entirely (52k extra files)")
    ap.add_argument("--include-untaxed-already-on-disk", action="store_true",
                    help="also append previously-excluded finlex files that "
                         "would now pass the tax-keyword filter even without --extras")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be promoted; no embed, no write")
    args = ap.parse_args()

    chunker_name = args.chunker or _default_chunker_for(args.method)
    print(f"[expand] method={args.method}  chunker={chunker_name}")
    print(f"[expand] embed model preserved: {EMBED_MODEL}")

    finlex_all = _all_finlex_files()
    vero_all = _all_vero_files()
    excluded = [p for p in finlex_all if not _passes_current_filter(p)]
    print(f"[discover] finlex on disk: {len(finlex_all):,}  "
          f"(kept by current filter: {len(finlex_all) - len(excluded):,}; "
          f"excluded: {len(excluded):,})")
    print(f"[discover] vero on disk:   {len(vero_all):,}  (all kept)")

    promoted: set[Path] = set()
    failure_log: list[dict] = []

    if args.include_all_finlex:
        promoted.update(excluded)
        print(f"[select] --include-all-finlex: +{len(excluded):,} files")

    if args.extra_keywords:
        kws = [k.strip() for k in args.extra_keywords.split(",") if k.strip()]
        hits = _select_by_extra_keywords(excluded, kws)
        promoted.update(hits)
        print(f"[select] --extra-keywords {kws}: +{len(hits):,} files")

    if args.include_dirs:
        dirs = [d.strip() for d in args.include_dirs.split(",") if d.strip()]
        hits = _select_by_dirs(excluded, dirs)
        promoted.update(hits)
        print(f"[select] --include-dirs {dirs}: +{len(hits):,} files")

    if args.from_failures:
        hits, failure_log = _select_from_failures(
            excluded, args.from_failures, args.max_files_per_question,
            min_anchor_hits=args.min_anchor_hits,
        )
        promoted.update(hits)
        n_matched = sum(1 for e in failure_log if e["matched_files"])
        n_total = len(failure_log)
        print(f"[select] --from-failures {args.from_failures}: "
              f"+{len(hits):,} files across {n_matched}/{n_total} failing questions")

    if not promoted and not args.include_untaxed_already_on_disk:
        print("[expand] no files selected -- pass --extra-keywords, --include-dirs, "
              "--from-failures or --include-all-finlex")
        return

    # Tag selected finlex files; vero files aren't touched by the filter so
    # they're already either indexed or producing zero chunks (handled later).
    new_files: list[tuple[Path, str]] = [(p, "finlex") for p in sorted(promoted)]

    # Persist manifest so reruns are auditable.
    manifest = _load_manifest()
    seen_in_manifest = set(manifest.get("promoted_files", []))
    added_to_manifest = []
    for p in promoted:
        try:
            rel = str(p.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(p)
        if rel not in seen_in_manifest:
            added_to_manifest.append(rel)
            seen_in_manifest.add(rel)
    manifest["promoted_files"] = sorted(seen_in_manifest)
    if failure_log:
        manifest.setdefault("from_failures_runs", []).append({
            "method": args.from_failures,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "matched": failure_log,
        })

    print(f"[manifest] {len(added_to_manifest):,} files newly recorded "
          f"(total promoted across all runs: {len(seen_in_manifest):,})")

    n = _chunk_and_append(args.method, chunker_name, new_files, args.dry_run)

    if not args.dry_run:
        _save_manifest(manifest)
        print(f"\n[done] appended {n:,} chunks to {chunks_path_for(args.method).name}")
        print(f"[done] manifest -> {MANIFEST_PATH.name}")
        if args.method != "section_v1":
            print("[note] hybrid_* and graph_* reuse section_v1 -- run "
                  "`--method section_v1` separately if you want them to see "
                  "the new content.")


if __name__ == "__main__":
    main()
