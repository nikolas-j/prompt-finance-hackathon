"""Section-aware HTML parser for finlex statutes and vero guidance documents.

Returns a list of *section dicts* per file. A section is one logical unit
(one § for finlex, one numbered subsection for vero) rather than a fixed
token window. Each section carries enough metadata for downstream filtering
(e.g. drop `is_superseded=True` at retrieval time).

Public entry point:
    parse_html_to_sections(path: Path, publisher: str) -> list[dict]

Section dict schema (additive to the baseline chunk schema):
    publisher        "finlex" | "vero"
    source_file      relative path (filled in by caller)
    statute_name     text of the document's main h1
    chapter          nearest preceding h2 ("luku") for finlex; "" for vero
    section_id       id attribute of the section heading, if any
    section_title    raw heading text
    section_number   normalised § number for finlex ("1", "102a"); numeric prefix
                     ("2.1") for vero
    paragraphs       list[str] of paragraph / list contents
    raw_text         "<title>\n\n<para>\n\n<para>..." for embedding
    is_superseded    True if vero attention-box flags "kumottu"
    superseded_by    list of replacement-doc hrefs (vero only)
    amendment_date   "dd.mm.yyyy" if this is a finlex h4 amendment block, else ""
    cross_refs       list of unique hrefs found anywhere in the section body
    node_type        "section" | "amendment_provision"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

# --- Regexes ---------------------------------------------------------------
# Finlex § numbers — "1§", "1 §", "102 a §" (compound with letter suffix).
SECTION_NUMBER_RE = re.compile(r"^\s*(\d+)\s*([a-zåäö])?\s*§", re.IGNORECASE)
# Vero numeric prefix — "1", "2.1", "3.4.2".
VERO_NUMERIC_PREFIX_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")
# Finlex amendment header e.g. "28.12.2012/1081:".
AMENDMENT_DATE_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})")

# Heuristic: trailing author lines on vero documents look like "johtava veroasiantuntija X".
_VERO_AUTHOR_PREFIXES = (
    "johtava veroasiantuntija",
    "ylitarkastaja",
    "veroasiantuntija",
    "ylijohtaja",
    "pääjohtaja",
)


# --- Helpers ---------------------------------------------------------------
def _text(tag: Tag | None) -> str:
    """Get readable text from a tag, joining <li> items with '; ' so lists stay legible."""
    if tag is None:
        return ""
    if tag.name in ("ul", "ol"):
        items = [li.get_text(" ", strip=True) for li in tag.find_all("li")]
        return "; ".join(it for it in items if it)
    return tag.get_text(" ", strip=True)


def _normalise_section_number(num: str, letter: str | None) -> str:
    return f"{num}{letter.lower()}" if letter else num


def _build_raw_text(sec: dict) -> str:
    """Concatenate title + paragraphs for embedding. Title first gives the embedder
    a strong topical anchor even when the body is short."""
    parts: list[str] = []
    if sec.get("section_title"):
        parts.append(sec["section_title"])
    parts.extend(sec.get("paragraphs", []))
    return "\n\n".join(p for p in parts if p)


def _new_section(
    publisher: str,
    statute: str,
    chapter: str = "",
    section_id: str = "",
    section_title: str = "",
    section_number: str = "",
    node_type: str = "section",
    amendment_date: str = "",
    is_superseded: bool = False,
    superseded_by: list[str] | None = None,
) -> dict:
    return {
        "publisher": publisher,
        "statute_name": statute,
        "chapter": chapter,
        "section_id": section_id,
        "section_title": section_title,
        "section_number": section_number,
        "node_type": node_type,
        "amendment_date": amendment_date,
        "is_superseded": is_superseded,
        "superseded_by": list(superseded_by) if superseded_by else [],
        "paragraphs": [],
        "cross_refs": [],
        "raw_text": "",
    }


def _looks_like_author(text: str) -> bool:
    t = text.strip().lower()
    if len(t) > 80:
        return False
    return any(t.startswith(p) for p in _VERO_AUTHOR_PREFIXES)


def _count_toc_items(items: list) -> int:
    n = 0
    for it in items:
        n += 1
        children = it.get("Children") or []
        n += _count_toc_items(children)
    return n


# Sections below this raw-text length get merged forward into the next section.
# ~200 chars ≈ 50 tokens for Finnish, matching the spec's "very short" cutoff.
SHORT_SECTION_MERGE_CHARS = 200


def _merge_short_sections(sections: list[dict]) -> list[dict]:
    """Forward-merge tiny sections into the next sibling so every chunk carries
    enough context for the embedder. Only merges across siblings that share the
    same chapter and node_type — we never collapse a regular section into an
    amendment_provision block (different semantic class)."""
    out: list[dict] = []
    for sec in sections:
        if (
            out
            and len(out[-1]["raw_text"]) < SHORT_SECTION_MERGE_CHARS
            and out[-1].get("chapter") == sec.get("chapter")
            and out[-1].get("node_type") == sec.get("node_type")
            and out[-1].get("is_superseded") == sec.get("is_superseded")
        ):
            prev = out.pop()
            merged = dict(sec)
            merged["paragraphs"] = prev["paragraphs"] + sec["paragraphs"]
            merged["cross_refs"] = list(dict.fromkeys(prev["cross_refs"] + sec["cross_refs"]))
            # Keep the earliest section_id/number for traceability.
            merged["section_id"] = prev.get("section_id") or sec.get("section_id", "")
            if prev.get("section_number") and sec.get("section_number"):
                merged["section_number"] = f"{prev['section_number']}+{sec['section_number']}"
            else:
                merged["section_number"] = prev.get("section_number") or sec.get("section_number", "")
            merged["section_title"] = (
                f"{prev['section_title']} + {sec['section_title']}"
                if prev.get("section_title") and sec.get("section_title")
                else (prev.get("section_title") or sec.get("section_title", ""))
            )
            merged["raw_text"] = _build_raw_text(merged)
            out.append(merged)
        else:
            out.append(sec)
    return out


# --- Finlex ----------------------------------------------------------------
def _parse_finlex(soup: BeautifulSoup) -> list[dict]:
    body = soup.body or soup
    h1 = body.find("h1")
    statute = _text(h1)

    sections: list[dict] = []
    current: dict | None = None
    current_chapter = ""

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current["paragraphs"]:
            current["raw_text"] = _build_raw_text(current)
            sections.append(current)
        current = None

    # find_all with explicit tag list returns descendants in document order.
    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
        name = el.name
        if name == "h1":
            continue
        if name == "h2":
            flush()
            current_chapter = _text(el)
        elif name == "h3":
            flush()
            title = _text(el)
            m = SECTION_NUMBER_RE.match(title)
            section_number = _normalise_section_number(m.group(1), m.group(2)) if m else ""
            current = _new_section(
                publisher="finlex",
                statute=statute,
                chapter=current_chapter,
                section_id=el.get("id", "") or "",
                section_title=title,
                section_number=section_number,
                node_type="section",
            )
        elif name == "h4":
            flush()
            title = _text(el)
            m = AMENDMENT_DATE_RE.search(title)
            current = _new_section(
                publisher="finlex",
                statute=statute,
                chapter=current_chapter,
                section_id=el.get("id", "") or "",
                section_title=title,
                section_number="",
                node_type="amendment_provision",
                amendment_date=m.group(1) if m else "",
            )
        elif name in ("p", "ul", "ol"):
            if current is None:
                continue
            text = _text(el)
            if text:
                current["paragraphs"].append(text)
            for a in el.find_all("a", href=True):
                href = a["href"]
                if href and href not in current["cross_refs"]:
                    current["cross_refs"].append(href)

    flush()
    return _merge_short_sections(sections)


# --- Vero ------------------------------------------------------------------
def _vero_superseded_info(body: Tag) -> tuple[bool, list[str]]:
    """A document is superseded if any attention-box paragraph mentions 'kumottu'."""
    is_superseded = False
    superseded_by: list[str] = []
    for box in body.find_all(["p", "div"], class_="attention-box"):
        if "kumottu" in box.get_text(" ", strip=True).lower():
            is_superseded = True
            for a in box.find_all("a", href=True):
                if a["href"] not in superseded_by:
                    superseded_by.append(a["href"])
    return is_superseded, superseded_by


def _vero_toc_count(body: Tag) -> int:
    toc_el = body.find("taxfi-table-of-contents-mobile")
    if toc_el is None:
        return 0
    toc_attr = toc_el.get(":toc-data") or toc_el.get("toc-data")
    if not toc_attr:
        return 0
    try:
        toc = json.loads(toc_attr)
    except Exception:
        return 0
    return _count_toc_items(toc.get("IndexItems", []))


def _parse_vero(soup: BeautifulSoup) -> list[dict]:
    body = soup.body or soup

    title_tag = body.find("h1", class_="taxxa-title") or body.find("h1")
    statute = _text(title_tag)

    is_superseded, superseded_by = _vero_superseded_info(body)

    sections: list[dict] = []
    current: dict | None = None
    title_h1_skipped = False

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current["paragraphs"]:
            current["raw_text"] = _build_raw_text(current)
            sections.append(current)
        current = None

    for el in body.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol"]):
        name = el.name
        cls = el.get("class") or []

        if name in ("h1", "h2", "h3", "h4"):
            # The page-title h1 (class=taxxa-title) is not a content section.
            if not title_h1_skipped and "taxxa-title" in cls:
                title_h1_skipped = True
                continue
            heading_text = _text(el)
            if not heading_text:
                continue
            flush()
            m = VERO_NUMERIC_PREFIX_RE.match(heading_text)
            section_number = m.group(1) if m else ""
            current = _new_section(
                publisher="vero",
                statute=statute,
                chapter="",
                section_id=el.get("id", "") or "",
                section_title=heading_text,
                section_number=section_number,
                node_type="section",
                is_superseded=is_superseded,
                superseded_by=superseded_by,
            )
        elif name in ("p", "ul", "ol"):
            if "attention-box" in cls:
                continue  # handled separately as superseded metadata
            text = _text(el)
            if not text:
                continue
            if _looks_like_author(text):
                continue
            if current is None:
                # paragraphs before any section heading — typically the lead-in.
                # Stash them into an implicit "intro" section so we don't lose them.
                current = _new_section(
                    publisher="vero",
                    statute=statute,
                    section_title=statute or "intro",
                    section_number="",
                    node_type="section",
                    is_superseded=is_superseded,
                    superseded_by=superseded_by,
                )
            current["paragraphs"].append(text)
            for a in el.find_all("a", href=True):
                href = a["href"]
                if href and href not in current["cross_refs"]:
                    current["cross_refs"].append(href)

    flush()
    return _merge_short_sections(sections)


# --- Public entry point ----------------------------------------------------
def parse_html_to_sections(path: Path, publisher: str) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for t in soup(["script", "style"]):
        t.decompose()
    if publisher == "finlex":
        return _parse_finlex(soup)
    if publisher == "vero":
        return _parse_vero(soup)
    raise ValueError(f"unknown publisher: {publisher!r}")


# --- CLI sanity check ------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) < 3:
        print("usage: python section_parser.py <publisher> <html_path>")
        raise SystemExit(2)
    pub = sys.argv[1]
    p = Path(sys.argv[2])
    secs = parse_html_to_sections(p, pub)
    print(f"{len(secs)} sections from {p.name}")
    for s in secs[:5]:
        print(
            f"  [{s['section_number'] or '-'}] {s['section_title'][:70]!r} "
            f"({len(s['paragraphs'])} paras, superseded={s['is_superseded']})"
        )
