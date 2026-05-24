# Taxxa local demo GUI

Black/orange chat UI mocking the Taxxa.ai assistant, wired to the local
RAG pipeline + trained confidence classifier. Two pages:

- `/` — **chat**: ask a question, see the answer + retrieved sources +
  a calibrated confidence flag + a compact per-question feature
  breakdown. A retrieval-method dropdown lets you switch between the
  trained pipelines.
- `/admin` — **model details**: pick a method, see the headline
  calibration story, full feature importances (bars), confusion matrix
  from 5-fold CV, and either an SVG decision-tree diagram with Gini
  per node OR the ranked logistic-regression coefficients. A "probe"
  input at the bottom runs a question through the classifier and shows
  the decision path live.

## Run

```powershell
uv run python frontend/server.py             # 127.0.0.1:8000
uv run python frontend/server.py --port 8765 # custom port
```

Open <http://localhost:8000/> for the chat, or
<http://localhost:8000/admin> for the model-details view.

Indexes and classifiers load lazily per method and are cached in memory
after the first hit. The default method (`section_v1`) is warmed at
startup.

## Endpoints

| route | description |
|---|---|
| `GET /` | chat page |
| `GET /admin` | classifier-admin page (?method=...) |
| `GET /api/health` | liveness probe |
| `GET /api/methods` | list of selectable methods + their trained classifier |
| `POST /api/ask` | `{question, method?}` → answer, sources, confidence, feature_breakdown |
| `GET /api/admin?method=<m>` | classifier internals: importances, tree+gini OR coefs, CV confusion matrix |

## Confidence semantics

Numbers below come from 5-fold out-of-fold predictions of the best
classifier per method. The admin page recomputes them per method;
chat-page chips show the universal storyline (calibrated on `section_v1`).

| label              | what the model is saying          | section_v1 hit rate                  |
|--------------------|------------------------------------|--------------------------------------|
| 🟢 **Confident**       | "answer is likely correct"        | 56% precision (vs 35% raw RAG)      |
| 🟠 **Verify carefully** | "I'm not sure - check citations" | 78% of these answers are wrong; 74% of all wrong answers are caught here |

The flag is the actual value-add: it lets a tax accountant prioritise
where to spend manual verification effort.

## Method catalog (driven by the trained pickles)

The dropdown lists every method that has a `classifier_{method}.pkl` on
disk. Currently:

- `section_v1` (recommended) — section-aware chunker + cosine
  retrieval, decision tree
- `baseline` — naive 512-token sliding window + cosine, L1 logreg
- `hybrid_section_v1` — section chunks + BM25/dense RRF, decision tree
- `hybrid_graph_section_v1` — hybrid retrieval + structural graph
  expansion, decision tree
- `section_v1_qfi` — section + gpt-4o-translated Finnish query,
  decision tree

To add a method, train its classifier:
```powershell
uv run scripts/train_classifier.py --method <name>
```
then add an entry to `_METHOD_CATALOG` at the top of
[server.py](server.py) so it shows up in the dropdown with a label.

## Files

```
frontend/
  server.py          # stdlib http.server, /api/methods, /api/ask, /api/admin, static
  static/
    index.html       # chat page
    admin.html       # model-details page
    style.css        # shared Taxxa-style theme (chat + admin)
    app.js           # chat client (method picker, feature-breakdown rendering)
    admin.js         # admin client (importance bars, tree SVG, CV CM, probe)
  README.md
```

## Notes for the demo

- The decision tree drawn on `/admin` is the actual trained tree from
  the pickle — node Gini, sample count, and class distribution are read
  from the model's `.tree_` arrays. Splits are not stylised.
- The CV confusion matrix is recomputed from `signals_{method}.json`
  + `results_{method}.json` using 5-fold StratifiedKFold (seed=42),
  refitting per fold with the saved best hyperparams. This mirrors how
  `train_classifier.py` computes the reported metrics.
- The probe input on the admin page sends the question through the
  same pipeline as the chat page and surfaces (1) the active features
  in the prediction and (2) the literal root-to-leaf decision path
  through the tree, with the path highlighted on the SVG.
- Generation failures (e.g. Featherless 429) are soft: the UI still
  renders the sources, confidence, and breakdown so a judge can see
  the classifier story even if the LLM is unavailable.
```
