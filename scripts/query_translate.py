"""English -> Finnish question rewriter for retrieval-only use.

Used by methods whose name ends in `_qfi` (e.g. `section_v1_qfi`,
`hybrid_section_v1_qfi`). The translated text replaces the query text that
goes into the embedder and BM25 -- generation and judging still receive
the *original* question + answer.

Why a powerful model: the corpus is in Finnish legal/administrative
register. A weak translator routinely picks the wrong domain term
(e.g. "tax" -> "vero" vs "verotus" vs "verot") which is precisely the
mismatch BM25 punishes. `gpt-4o` is accurate at this and the cost is
~$0.001 per question, billed once because translations are cached on
disk forever (cache key = the raw question string).

Public API:
    translate_question(question: str) -> str
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    OPENAI_API_KEY,
    TRANSLATE_CACHE_PATH,
    TRANSLATE_MAX_TOKENS,
    TRANSLATE_MODEL,
)

_TRANSLATE_SYSTEM = (
    "You translate user questions about Finnish tax and accounting law from "
    "English (or mixed English/Finnish) into NATURAL FINNISH for searching "
    "Finnish legal text (Finlex statutes, Verohallinto guidance). Rules:\n"
    "1. Output ONLY the Finnish translation. No quotes, no preface, no "
    "   explanation.\n"
    "2. Preserve every number, percentage, euro amount, year, and statute "
    "   reference VERBATIM (do not localise '30,000 EUR' or '34 %' formatting).\n"
    "3. Use canonical Finnish tax terminology that matches Finlex/Verohallinto "
    "   wording (e.g. 'pääomatulovero', 'lähdevero', 'avainhenkilölaki', "
    "   'ennakonpidätys', 'TyEL-maksu', 'yleisradiovero', 'lahjavero', "
    "   'perintövero', 'kiinteistövero'). When the question already names a "
    "   Finnish term in parentheses, prefer that term in the translation.\n"
    "4. Keep the translation as a SINGLE QUESTION SENTENCE. Do not add facts, "
    "   examples, or follow-ups.\n"
    "5. If the input is already Finnish, return it unchanged."
)

_lock = threading.Lock()
_cache: dict[str, str] | None = None
_client: OpenAI | None = None


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    if TRANSLATE_CACHE_PATH.exists():
        with TRANSLATE_CACHE_PATH.open("r", encoding="utf-8") as f:
            try:
                _cache = json.load(f)
            except json.JSONDecodeError:
                _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache(cache: dict[str, str]) -> None:
    TRANSLATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TRANSLATE_CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(TRANSLATE_CACHE_PATH)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY missing -- needed for query translation")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def translate_question(question: str) -> str:
    """Translate `question` to Finnish (or return cached translation).

    Thread-safe: many evaluate.py workers can call this in parallel; only
    one OpenAI call ever fires per unique question.
    """
    q = (question or "").strip()
    if not q:
        return q
    with _lock:
        cache = _load_cache()
        if q in cache:
            return cache[q]

    # Outside the lock: hit the API. Worst case two threads translate the
    # same question simultaneously -- harmless, and the second write wins.
    client = _get_client()
    resp = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=[
            {"role": "system", "content": _TRANSLATE_SYSTEM},
            {"role": "user", "content": q},
        ],
        temperature=0.0,
        max_tokens=TRANSLATE_MAX_TOKENS,
    )
    translated = (resp.choices[0].message.content or "").strip()
    # Strip wrapping quotes if the model added them despite instructions.
    if translated.startswith(("'", '"')) and translated.endswith(("'", '"')):
        translated = translated[1:-1].strip()
    if not translated:
        translated = q  # never return empty -- fall back to original

    with _lock:
        cache = _load_cache()
        cache[q] = translated
        _save_cache(cache)
    return translated


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Translate one question for testing.")
    ap.add_argument("question")
    args = ap.parse_args()
    print(translate_question(args.question))
