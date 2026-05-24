# Prompt Finance Hackathon · 2026

**Agentic GraphRAG for accounting & tax research.**
Hosted by Aalto University · Challenge by [Taxxa AI Oy](https://taxxa.ai).

A weekend on retrieval, graphs, and agents — over real Finnish regulation.

---

## What you're solving

Finnish accountants spend a lot of their day answering questions like:

> *"What withholding-tax rate applies to a foreign specialist with key-personnel
> status, and how long is the tax card valid?"*

The answer touches the **avainhenkilölaki** (Finlex), a Verohallinto
bulletin, and a recent amendment that changed both the rate and the
validity window. Naïve RAG — embed everything, retrieve top-k, stuff
into a prompt — runs into the usual pitfalls: it loses the document's
structure, can't follow cross-references, can't tell a current rule
from a superseded one, and confabulates when the right passage isn't
in the top-k.

**Your job:** build a system that does better.
How you get there is up to you — graph schema, retrieval strategy,
agent topology, eval loop. We have opinions (below) but no required stack.

---

## What's in this repo

```
.
├── README.md                  # you're here
├── docs/challenge.pdf         # the full briefing deck
├── data/
│   └── question_bank.json     # 83 graded QA pairs — your eval set
│                              # (raw corpus lands in data/raw/ via fetch_data.py)
├── scripts/fetch_data.py      # downloads the corpus from GitHub Releases
├── docker-compose.yml         # optional Neo4j (if you want a graph DB)
└── .env.example               # API keys + Neo4j config template
```

That's it. **There is no starter code** — bring whatever language, framework,
and libraries you like. We grade the output, not the stack.

---

## Getting started

### 1. Clone

```bash
git clone https://github.com/Taxxa-AI/aalto-hackaton-2026.git
cd aalto-hackaton-2026
```

### 2. Get the corpus

```bash
python scripts/fetch_data.py
```

One archive, `finland_kb.tar.gz`, containing both publishers. Extracts
to `data/raw/`. The script is stdlib-only — no `pip install` needed
before you have data.

> The corpus release (`data-v1`) is published on this repo's
> [GitHub Releases](https://github.com/Taxxa-AI/aalto-hackaton-2026/releases)
> page. If the download 404s, the tag isn't live yet — check Slack.

### 3. Set up your LLM

Two paths, pick one:

**OpenRouter** — single API key, hundreds of models, OpenAI-compatible API.
```bash
cp .env.example .env
# fill in OPENROUTER_API_KEY
```

**Ollama** — fully local, free, OpenAI-compatible on `localhost:11434`.

On macOS (Apple Silicon, 16GB+ recommended):
```bash
brew install ollama
ollama serve &                   # background process
ollama pull qwen2.5:14b          # main LLM (~9GB, ~35 tok/s on M3)
ollama pull bge-m3               # multilingual embeddings (handles Finnish)
```

On Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama   # if you want it to auto-start
ollama pull qwen2.5:14b
ollama pull bge-m3
```

If you have less than 16 GB of RAM, use `qwen2.5:7b` instead — same
family, half the footprint, noticeably weaker on multi-hop questions.

Use whichever path your team prefers — or both.

### 4. (Optional) Spin up Neo4j

If you want a managed graph store instead of `networkx` / DuckDB / Kuzu:

```bash
docker compose up -d
# Browser:  http://localhost:7474   (neo4j / password)
# Bolt:     bolt://localhost:7687
```

APOC is enabled, `./data` is mounted at `/import` so you can `LOAD CSV`
directly from the host. `docker compose down -v` wipes the volume.

---

## The corpus

Two Finnish publishers — small on purpose, so you can spend the weekend
on the **graph and the agent**, not on writing scrapers.

### Finlex — [finlex.fi](https://finlex.fi)

The official Finnish legal database, maintained by the Ministry of
Justice. It contains the consolidated text of every Finnish statute
(e.g. **AVL** — Arvonlisäverolaki, the VAT Act; **TVL** —
Tuloverolaki, the Income Tax Act; **EPL** — Ennakkoperintälaki, the
Prepayment Act), the full amendment history of each, and case law
from the Supreme Administrative Court (**KHO**) and Supreme Court
(**KKO**). Statutes are structured as books → chapters → sections →
paragraphs, with explicit numbered cross-references between them
(e.g. `§ 102 a momentti 2`). Available in Finnish and Swedish, with
selected statutes in English.

### Verohallinto — [vero.fi](https://vero.fi)

The Finnish Tax Administration. Publishes **syvennetyt ohjeet**
(in-depth guidance) and shorter bulletins explaining how it applies
the statutes in practice — covering income tax, VAT, payroll, social
contributions, international tax, and tax procedure. Each guidance
document cites the Finlex sections it interprets and is dated &
versioned, so amendments produce a new version. Available in Finnish,
Swedish, and partial English.

### Why these two

Finlex gives you the *law*. Vero gives you the *operational
interpretation* of that law. The interesting graph edges run between
them — a Vero bulletin **interprets** a Finlex section, a Finlex
amendment **supersedes** an earlier bulletin, a KHO ruling
**clarifies** how a section applies. That's the multi-hop pattern the
question bank tests.

Documents have **structure**. Throwing that away with 512-token
chunks is leaving signal on the floor.

---

## The question bank

`data/question_bank.json` — 83 graded QA pairs, all answerable from the
shipped Finlex + Vero corpus. Two sets:

- **Set 1 (Q-prefix)** — graded `basic / medium / hard`.
- **Set 2 (N-prefix)** — graded `difficulty_1` (easiest) through `difficulty_5` (hardest).

Top-level shape:

```jsonc
{
  "name": "Taxxa Finland QA Bank V1",
  "version": "1.0",
  "publisher_legend": { "finlex": "...", "vero": "..." },
  "tiers":            { "basic": "...", "difficulty_5": "...", ... },
  "entries":          [ /* 83 question objects */ ]
}
```

Each entry:

```jsonc
{
  "id":         "Q1",                      // stable identifier
  "tier":       "basic",                   // see `tiers` legend
  "difficulty": 5,                         // present on N1–N60 only
  "question":   "What is the capital income tax rate ...",
  "answer":     "The capital income tax rate ... is 34%. ...",
  "answer_key_facts": [                    // atomic claims the answer must contain
    "The capital income tax rate ... is 34%.",
    "Capital income up to 30,000 euros is taxed at the lower rate of 30%."
  ],
  "citations": [
    { "publisher": "vero", "publisher_full": "Verohallinto ...",
      "title": null, "file_path": null, "excerpt": null }
  ]
}
```

The key facts are the strict floor — if your answer doesn't mention
"the rate is 34%" on a rate question, it's wrong, regardless of how
elegant the prose is.

> **Caveats** documented in the JSON's `missing_fields_note`:
> citation `title` / `file_path` / `excerpt` are `null` — the bank gives
> you the publisher, not the exact passage (resolving back to a URL on
> `finlex.fi` / `vero.fi` is part of your job). For Set 2 (N-prefix),
> `answer_key_facts` are auto-extracted from the answer text and may be
> a strict subset of the full answer; a small number of N entries have
> an empty list — fall back to the full `answer` for those.

Use this bank however you like during the weekend — as test cases,
as worked examples, as a way to sanity-check your retrieval. We'll
evaluate submissions in a separate harness with our own held-out
questions, so don't over-fit to this exact set.

---

## Suggested approach (not required)

A path that's worked for us:

1. **Parse, don't chunk.** Respect each document's own structure
   (book → title → article → paragraph). A 50-line Finlex statute
   is not "three 512-token windows" — it's a tree.
2. **Build a typed graph.** Nodes for statutes, articles, clauses,
   case law, guidance, concepts. Edges for `amends`, `repeals`,
   `transposes`, `references`, `interpreted_by`. Cross-references
   in Finnish legal text are explicit (`§ 102 a momentti 2`) and
   highly extractable.
3. **Hybrid retrieve.** Embeddings find entry points; typed graph
   traversal walks to the relevant neighbourhood. Reranker on the
   resulting passages.
4. **Agent loop.** Decompose ("rate → exceptions → effective date"),
   retrieve per sub-question, draft with citations, verify each
   claim against its source node, ask for clarification if confidence
   is low.
5. **Ship.** A working bad pipeline beats a beautiful unfinished one.

If you can beat naïve RAG on the **hard** tier with something simpler —
do it. We're more impressed by clear design than by elaborate plumbing.

---

## Judging

In order of weight:

1. **Correctness** — does your system answer questions right on our
   held-out set?
2. **Groundedness** — does every claim cite a specific source
   (document + section)? An answer with no citations is unverifiable
   and we treat it as wrong.
3. **Approach** — how well does your design model the domain?
   How do you handle the hard cases (multi-hop, amendments, exceptions)?
   We want to see your reasoning.

Show your work. A README in your submission explaining
"we tried X, here's why we ended up with Y" is worth a lot more than
clever code we have to reverse-engineer.

---

## Help

- Slack: see the QR in `docs/challenge.pdf`.
- Email: stephane@taxxa.ai
- We're around all weekend to pair on code or argue about schemas.

Good luck.
