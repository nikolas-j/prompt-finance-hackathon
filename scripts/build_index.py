"""Two-stage index build: parse+chunk → checkpoint → embed → checkpoint.

Stage 1 writes data/chunks_{method}.json (no embeddings). Skipped if file exists.
Stage 2 reads it, embeds via OpenAI, appends to data/embeddings_{method}.bin in
        place after every wave. Resumable: re-running picks up where the last
        wave left off.

The --method flag controls which artefact pair is read/written. Two different
chunking strategies (e.g. "baseline" naive 512-token, vs "section" §-aware)
co-exist on disk and can be queried independently — no expensive rebuilds.

Legacy un-versioned files (chunks.json, embeddings.bin, the older index.json)
are auto-migrated to the default-method names on first run.

Run from repo root:
    uv run scripts/build_index.py                       # method=baseline, smart resume
    uv run scripts/build_index.py --method section_v1   # named alternative chunker
    uv run scripts/build_index.py --rechunk             # force re-parse
    uv run scripts/build_index.py --reembed             # force re-embed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DEFAULT_METHOD,
    EMBED_BATCH_SIZE,
    EMBED_CONCURRENCY,
    EMBED_DIM,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_PRICE_PER_M_TOKENS,
    FINLEX_DIR,
    LEGACY_CHUNKS_PATH,
    LEGACY_EMBEDDINGS_PATH,
    LEGACY_INDEX_PATH,
    OPENAI_API_KEY,
    REPO_ROOT,
    TAX_KEYWORDS,
    VERO_DIR,
)
from section_parser import parse_html_to_sections

TOKENIZER = tiktoken.get_encoding("cl100k_base")
RECORD_BYTES = EMBED_DIM * 4  # float32


# --- Path helpers ----------------------------------------------------------
def chunks_path_for(method: str) -> Path:
    return DATA_DIR / f"chunks_{method}.json"


def embeddings_path_for(method: str) -> Path:
    return DATA_DIR / f"embeddings_{method}.bin"


# --- IO helpers ------------------------------------------------------------
def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def migrate_legacy_artifacts(chunks_path: Path, embeddings_path: Path) -> None:
    """Move pre-versioning files to the method-named locations (idempotent, safe)."""
    # Step 1: un-versioned chunks.json/embeddings.bin → versioned names.
    if LEGACY_CHUNKS_PATH.exists() and not chunks_path.exists():
        LEGACY_CHUNKS_PATH.rename(chunks_path)
        print(f"[migrate] {LEGACY_CHUNKS_PATH.name} → {chunks_path.name}")
    if LEGACY_EMBEDDINGS_PATH.exists() and not embeddings_path.exists():
        LEGACY_EMBEDDINGS_PATH.rename(embeddings_path)
        print(f"[migrate] {LEGACY_EMBEDDINGS_PATH.name} → {embeddings_path.name}")

    # Step 2: very old combined index.json → versioned embeddings.bin.
    if LEGACY_INDEX_PATH.exists() and not embeddings_path.exists():
        print(f"[migrate] converting {LEGACY_INDEX_PATH.name} → {embeddings_path.name}")
        with LEGACY_INDEX_PATH.open("r", encoding="utf-8") as f:
            old = json.load(f)
        if old:
            arr = np.array([e["embedding"] for e in old], dtype=np.float32)
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            with embeddings_path.open("wb") as f:
                f.write(arr.tobytes(order="C"))
            print(f"[migrate] wrote {len(arr):,} embeddings")
        LEGACY_INDEX_PATH.rename(LEGACY_INDEX_PATH.with_suffix(".json.bak"))


# --- Stage 1: discover + parse + chunk -------------------------------------
def is_tax_relevant(path: Path, publisher: str) -> bool:
    if publisher == "vero":
        return True
    s = str(path).lower()
    if "tuloverosopim" in s:           # tax-treaty directory: keep all
        return True
    name = path.name.lower()
    return any(kw in name for kw in TAX_KEYWORDS)


def discover() -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for publisher, root in [("finlex", FINLEX_DIR), ("vero", VERO_DIR)]:
        all_files = sorted(root.rglob("*.html"))
        kept = [p for p in all_files if is_tax_relevant(p, publisher)]
        print(f"[discover] {publisher}: kept {len(kept):,} / {len(all_files):,} (tax filter)")
        out.extend([(p, publisher) for p in kept])
    print(f"[discover] total to parse: {len(out):,}")
    return out


def parse_html(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def chunk_text(text: str) -> list[tuple[int, str]]:
    if not text.strip():
        return []
    tokens = TOKENIZER.encode(text)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    out: list[tuple[int, str]] = []
    cursor = 0
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_SIZE]
        if not window:
            break
        chunk = TOKENIZER.decode(window)
        idx = text.find(chunk[:40], cursor) if len(chunk) >= 40 else cursor
        offset = idx if idx >= 0 else cursor
        out.append((offset, chunk))
        cursor = offset + 1
        if start + CHUNK_SIZE >= len(tokens):
            break
    return out


def _baseline_chunks_for_file(path: Path, publisher: str, rel: str, start_idx: int) -> list[dict]:
    """Naive 512-token sliding window with overlap (the original baseline)."""
    text = parse_html(path)
    out: list[dict] = []
    for offset, ct in chunk_text(text):
        out.append(
            {
                "chunk_id": f"{publisher}_{start_idx + len(out):07d}",
                "source_file": rel,
                "publisher": publisher,
                "char_offset": offset,
                "text": ct,
            }
        )
    return out


def _split_section_by_paragraphs(section: dict, max_tokens: int) -> list[dict]:
    """Split a section into one or more chunks on paragraph boundaries.

    Each piece keeps all section metadata; only `text`, `paragraphs`, and the
    `part_index` / `part_total` markers differ. Title is repeated at the top
    of every piece so each chunk stays self-describing for the embedder.
    """
    paras = section.get("paragraphs", []) or []
    title = section.get("section_title", "") or ""
    title_tokens = len(TOKENIZER.encode(title)) if title else 0

    pieces: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = title_tokens
    for p in paras:
        ptok = len(TOKENIZER.encode(p))
        # If a single paragraph alone exceeds the limit, keep it as its own piece
        # (better an oversized chunk than a paragraph cut mid-sentence).
        if cur and cur_tokens + ptok > max_tokens:
            pieces.append(cur)
            cur = [p]
            cur_tokens = title_tokens + ptok
        else:
            cur.append(p)
            cur_tokens += ptok
    if cur:
        pieces.append(cur)

    total = len(pieces) or 1
    out: list[dict] = []
    for i, piece_paras in enumerate(pieces, start=1):
        text = "\n\n".join([title, *piece_paras]) if title else "\n\n".join(piece_paras)
        out.append(
            {
                "text": text,
                "paragraphs": piece_paras,
                "part_index": i,
                "part_total": total,
            }
        )
    return out


def _section_chunks_for_file(path: Path, publisher: str, rel: str, start_idx: int) -> list[dict]:
    """One chunk per section (split on paragraph boundaries when oversized).

    Adds these metadata fields per chunk on top of the baseline schema:
      statute_name, chapter, section_id, section_title, section_number,
      node_type, amendment_date, is_superseded, superseded_by, cross_refs,
      paragraphs, part_index, part_total.
    """
    sections = parse_html_to_sections(path, publisher)
    out: list[dict] = []
    for sec in sections:
        for piece in _split_section_by_paragraphs(sec, CHUNK_SIZE):
            out.append(
                {
                    "chunk_id": f"{publisher}_{start_idx + len(out):07d}",
                    "source_file": rel,
                    "publisher": publisher,
                    "char_offset": 0,
                    "text": piece["text"],
                    # additive metadata
                    "statute_name": sec.get("statute_name", ""),
                    "chapter": sec.get("chapter", ""),
                    "section_id": sec.get("section_id", ""),
                    "section_title": sec.get("section_title", ""),
                    "section_number": sec.get("section_number", ""),
                    "node_type": sec.get("node_type", "section"),
                    "amendment_date": sec.get("amendment_date", ""),
                    "is_superseded": sec.get("is_superseded", False),
                    "superseded_by": sec.get("superseded_by", []),
                    "cross_refs": sec.get("cross_refs", []),
                    "paragraphs": piece["paragraphs"],
                    "part_index": piece["part_index"],
                    "part_total": piece["part_total"],
                }
            )
    return out


def stage_chunk(chunks_path: Path, chunker: str, force: bool = False) -> list[dict]:
    if chunks_path.exists() and not force:
        with chunks_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[chunks] reusing existing {chunks_path.name} ({len(chunks):,} chunks). "
              f"Pass --rechunk to rebuild.")
        return chunks

    if chunker == "section":
        file_chunker = _section_chunks_for_file
    elif chunker == "baseline":
        file_chunker = _baseline_chunks_for_file
    else:
        raise SystemExit(f"unknown --chunker {chunker!r}; expected 'baseline' or 'section'")

    print(f"[chunk] strategy: {chunker}")
    t0 = time.time()
    files = discover()
    chunks: list[dict] = []
    for i, (path, publisher) in enumerate(files):
        if i % 200 == 0 and i:
            print(f"[parse] {i:,}/{len(files):,} files ({len(chunks):,} chunks)")
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        try:
            file_chunks = file_chunker(path, publisher, rel, start_idx=len(chunks))
        except Exception as e:
            print(f"[parse] skip {path}: {e}")
            continue
        chunks.extend(file_chunks)

    print(f"[chunk] produced {len(chunks):,} chunks in {time.time() - t0:.1f}s")
    atomic_write_json(chunks_path, chunks)
    print(f"[chunks] wrote {chunks_path}")
    return chunks


# --- Stage 2: embed --------------------------------------------------------
def load_resume(chunks: list[dict], embeddings_path: Path) -> int:
    """Return resume position. Truncates any partial trailing record from a crashed run."""
    if not embeddings_path.exists():
        return 0
    size = embeddings_path.stat().st_size
    n = size // RECORD_BYTES
    leftover = size - n * RECORD_BYTES
    if leftover:
        with embeddings_path.open("r+b") as f:
            f.truncate(n * RECORD_BYTES)
        print(f"[embed] discarded {leftover} trailing bytes (partial record)")
    if n > len(chunks):
        raise RuntimeError(
            f"{embeddings_path.name} has {n} records but chunks file has {len(chunks)}. "
            f"Delete {embeddings_path.name} and rerun."
        )
    return n


def append_embeddings(vectors: list[list[float]], embeddings_path: Path) -> None:
    """Append a batch of embeddings as raw float32 bytes; fsync for durability."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape[1] != EMBED_DIM:
        raise RuntimeError(f"expected dim {EMBED_DIM}, got {arr.shape[1]}")
    with embeddings_path.open("ab") as f:
        f.write(arr.tobytes(order="C"))
        f.flush()
        os.fsync(f.fileno())


def preflight(remaining: list[dict], total: int, already: int, embeddings_path: Path) -> None:
    tokens = sum(len(TOKENIZER.encode(c["text"])) for c in remaining)
    cost = tokens / 1_000_000 * EMBED_PRICE_PER_M_TOKENS
    eta_min = len(remaining) / 200 / 60
    bar = "=" * 64
    print(bar)
    print("EMBEDDING PRE-FLIGHT")
    print("-" * 64)
    print(f"  Model:             {EMBED_MODEL}")
    print(f"  Target file:       {embeddings_path.name}")
    print(f"  Total chunks:      {total:,}")
    print(f"  Already embedded:  {already:,}")
    print(f"  Remaining:         {len(remaining):,}")
    print(f"  Estimated tokens:  {tokens:,}")
    print(f"  Estimated cost:    ${cost:.3f}  (@ ${EMBED_PRICE_PER_M_TOKENS}/1M tokens)")
    print(f"  Estimated time:    ~{eta_min:.1f} min")
    print(f"  Concurrency:       {EMBED_CONCURRENCY} parallel batches × {EMBED_BATCH_SIZE} chunks")
    print(f"  Checkpoint:        every wave (~{EMBED_BATCH_SIZE * EMBED_CONCURRENCY} chunks)")
    print(bar)


def openai_embed(texts: list[str], client: OpenAI) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def stage_embed(chunks: list[dict], embeddings_path: Path, force: bool = False) -> None:
    if force and embeddings_path.exists():
        embeddings_path.unlink()
        print(f"[embed] --reembed: removed existing {embeddings_path.name}")

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY missing — set it in .env")

    start = load_resume(chunks, embeddings_path)
    if start:
        print(f"[embed] resuming: {start:,} chunks already embedded")
    remaining = chunks[start:]
    preflight(remaining, len(chunks), start, embeddings_path)
    if not remaining:
        print("[embed] nothing to do.")
        return

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=EMBED_MAX_RETRIES)
    t0 = time.time()
    i = start
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as ex:
        while i < len(chunks):
            wave_start = i
            futures: list[object] = []
            for _ in range(EMBED_CONCURRENCY):
                if i >= len(chunks):
                    break
                batch = chunks[i : i + EMBED_BATCH_SIZE]
                futures.append(
                    ex.submit(openai_embed, [c["text"] for c in batch], client)
                )
                i += len(batch)

            wave_vectors: list[list[float]] = []
            for fut in futures:
                try:
                    wave_vectors.extend(fut.result())
                except Exception as e:
                    print(f"[embed] wave starting at {wave_start} failed: {e}")
                    print(f"[embed] progress preserved in {embeddings_path.name}. Rerun to resume.")
                    raise

            append_embeddings(wave_vectors, embeddings_path)
            done = i - start
            rate = done / max(time.time() - t0, 1e-6)
            eta_min = (len(chunks) - i) / max(rate, 1e-6) / 60
            print(
                f"[embed] {i:,}/{len(chunks):,}  "
                f"({rate:.0f} chunks/s, ETA {eta_min:.1f} min) → checkpointed"
            )

    elapsed = time.time() - t0
    print(f"[embed] done: {len(remaining):,} chunks in {elapsed:.0f}s")
    print(f"[save] {embeddings_path} ({embeddings_path.stat().st_size / 1e6:.1f} MB)")


# --- Entry point -----------------------------------------------------------
def _default_chunker_for(method: str) -> str:
    """If the method name starts with 'section' or 'graph' (which reuses
    section chunks), use the section-aware chunker; otherwise fall back to
    the original sliding window."""
    return "section" if method.startswith(("section", "graph")) else "baseline"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--method", default=DEFAULT_METHOD,
        help=f"chunking method name → chunks_{{method}}.json + embeddings_{{method}}.bin "
             f"(default: {DEFAULT_METHOD})",
    )
    ap.add_argument(
        "--chunker", choices=("baseline", "section"), default=None,
        help="chunking strategy. Defaults to 'section' if --method starts with "
             "'section', otherwise 'baseline'.",
    )
    ap.add_argument("--rechunk", action="store_true", help="force re-parse (ignores chunks file)")
    ap.add_argument("--reembed", action="store_true", help="force re-embed (ignores embeddings file)")
    args = ap.parse_args()

    chunker = args.chunker or _default_chunker_for(args.method)
    chunks_path = chunks_path_for(args.method)
    embeddings_path = embeddings_path_for(args.method)
    print(f"[build] method={args.method}  chunker={chunker}")
    print(f"[build] chunks     ↔ {chunks_path.name}")
    print(f"[build] embeddings ↔ {embeddings_path.name}")

    migrate_legacy_artifacts(chunks_path, embeddings_path)

    chunks = stage_chunk(chunks_path, chunker=chunker, force=args.rechunk)
    stage_embed(chunks, embeddings_path, force=args.reembed)


if __name__ == "__main__":
    main()
