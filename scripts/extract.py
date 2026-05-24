import os, tarfile

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

skipped = []
ok = 0
with tarfile.open(archive) as tf:
    members = tf.getmembers()
    print(f"members in archive: {len(members)}")
    for m in members:
        original = m.name
        m.name = sanitize(m.name)
        if not m.name:
            skipped.append((original, "empty after sanitize"))
            continue
        try:
            tf.extract(m, root, filter="fully_trusted")
            ok += 1
        except Exception as e:
            skipped.append((m.name, repr(e)))

print(f"extracted: {ok}")
print(f"skipped:   {len(skipped)}")
for n, e in skipped[:20]:
    print(" -", n, "->", e)
