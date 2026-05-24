import os, tarfile
from collections import defaultdict

root = os.path.abspath(r"aalto-hackaton-2026\data\raw")
archive = os.path.join(root, "finland_kb.tar.gz")

WIN_RESERVED = ':?*"<>|'

def sanitize(name: str) -> str:
    parts = name.replace("\\", "/").split("/")
    clean = []
    for p in parts:
        for ch in WIN_RESERVED:
            p = p.replace(ch, "_")
        p = p.strip(" .")
        if p:
            clean.append(p)
    return "/".join(clean)

with tarfile.open(archive) as tf:
    files = [m for m in tf.getmembers() if m.isfile()]

    # Group by case-insensitive sanitized name (NTFS rule).
    groups = defaultdict(list)
    for m in files:
        groups[sanitize(m.name).lower()].append(m)

    rescued = 0
    for _, members in groups.items():
        if len(members) <= 1:
            continue
        # First member is already on disk; rename the rest.
        for i, m in enumerate(members[1:], start=1):
            sname = sanitize(m.name)
            base, ext = os.path.splitext(sname)
            m.name = f"{base}__case{i}{ext}"
            tf.extract(m, root, filter="fully_trusted")
            rescued += 1
    print(f"rescued: {rescued}")
