"""Download the pre-scraped Finnish regulation corpus from GitHub Releases.

Usage:
    python scripts/fetch_data.py

No third-party dependencies — uses only the stdlib so you can run this
before you've set up any environment.
"""

from __future__ import annotations

import tarfile
import os
import urllib.error
import urllib.request
from pathlib import Path

ASSET = "finland_kb.tar.gz"
DEFAULT_BASE = "https://github.com/Taxxa-AI/aalto-hackaton-2026/releases/download/data-v1"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def fetch(base: str) -> None:
    url = f"{base}/{ASSET}"
    dest = RAW / ASSET
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> {url}")
    try:
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            read = 0
            while chunk := r.read(1 << 16):
                f.write(chunk)
                read += len(chunk)
                if total:
                    pct = 100 * read / total
                    print(f"   {read / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)", end="\r")
        print()
    except urllib.error.HTTPError as e:
        print(f"   !! {e.code} {e.reason} — is the release tag published yet?")
        print(f"   !! ask in Slack if you think it should be live.")
        dest.unlink(missing_ok=True)
        raise SystemExit(1)

    print(f"   extracting -> {RAW}/")
    with tarfile.open(dest) as tar:
        # `filter="data"` rejects absolute paths, symlinks pointing outside, etc.
        # Required on Python 3.12+ to avoid a DeprecationWarning; mandatory on 3.14.
        tar.extractall(RAW, filter="data")
    dest.unlink()


def main() -> int:
    base = os.environ.get("TAXXA_DATA_RELEASE", DEFAULT_BASE)
    fetch(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
