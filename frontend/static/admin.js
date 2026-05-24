// Admin page: cross-method summary at the top, one tab per trained method
// underneath. Tabs are built dynamically from /api/methods.

const summaryEl   = document.getElementById("card-summary");
const headlineEl  = document.getElementById("card-headline");
const modelsEl    = document.getElementById("card-models");
const confusionEl = document.getElementById("card-confusion");
const importanceEl = document.getElementById("card-importance");
const featuresEl  = document.getElementById("card-features");
const treeEl      = document.getElementById("card-tree");
const probeForm   = document.getElementById("probe-form");
const probeInput  = document.getElementById("probe-input");
const probeSend   = document.getElementById("probe-send");
const probeResult = document.getElementById("probe-result");
const tabbar      = document.getElementById("tabbar");

// Populated by loadMethods() before the first /api/admin call.
let METHODS = [];
let CURRENT_METHOD = null;
let CURRENT_ADMIN  = null;

// Short, readable label for the tab's small sub-line.
function modelKindLabel(modelName) {
  if (!modelName) return "";
  if (modelName.startsWith("decision_tree")) return "Decision Tree";
  if (modelName.startsWith("logreg"))        return "Logistic Regression";
  if (modelName === "dummy_majority")        return "Majority baseline";
  return modelName;
}

// --- DOM + SVG helpers ----------------------------------------------------
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) {
    if (kid == null) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
function svg(tag, attrs = {}, ...kids) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const kid of kids) {
    if (kid == null) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
const pct = (x, d = 0) => (x == null ? "-" : (x * 100).toFixed(d) + "%");
const fmt = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));

// --- Top: test-set summary -----------------------------------------------
async function loadSummary() {
  try {
    const r = await fetch("/api/summary");
    const data = await r.json();
    renderSummary(data);
  } catch (err) {
    summaryEl.textContent = "Failed to load summary: " + err.message;
  }
}

function renderSummary(d) {
  summaryEl.innerHTML = "";

  // Headline row: total questions, best raw RAG, best classifier delta.
  const best = d.methods.find((m) => m.id === d.best_method);
  const bestRaw = d.methods
    .filter((m) => m.evaluated_actual)
    .reduce((a, b) => (a && a.pass_rate >= b.pass_rate ? a : b), null);

  summaryEl.append(
    el("div", { class: "headline-row" },
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, String(d.n_questions)),
        el("div", { class: "headline-label" }, "Expert-written questions"),
        el("div", { class: "headline-sub" }, "Hand-curated tax-law bank, scored by an LLM judge"),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, bestRaw ? pct(bestRaw.pass_rate) : "-"),
        el("div", { class: "headline-label" }, "Best raw RAG accuracy"),
        el("div", { class: "headline-sub" }, bestRaw ? "method: " + bestRaw.id : ""),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, best ? "+" + fmt(best.classifier_delta_bal_acc, 2) : "-"),
        el("div", { class: "headline-label" }, "Best classifier Δ bal_acc"),
        el("div", { class: "headline-sub" },
          best ? `${best.classifier_model} on ${best.id}` : "no classifier trained yet"),
      ),
    ),
  );

  // Per-method comparison strip.
  const maxPass = Math.max(0.01, ...d.methods.filter((m) => m.evaluated_actual).map((m) => m.pass_rate));
  summaryEl.append(
    el("div", { class: "summary-strip-title" }, "Per-method pass rate on the bank"),
    el("div", { class: "summary-strip" },
      ...d.methods.map((m) => {
        if (!m.evaluated_actual) {
          return el("div", { class: "summary-row dim" },
            el("div", { class: "summary-name" }, m.id),
            el("div", { class: "summary-track" }),
            el("div", { class: "summary-val" }, "not evaluated"),
          );
        }
        const w = Math.round((m.pass_rate / maxPass) * 100);
        const bar = el("div", { class: "summary-bar" });
        bar.style.width = w + "%";
        const note = m.classifier_model
          ? ` · clf ${m.classifier_model} (Δ ${m.classifier_delta_bal_acc >= 0 ? "+" : ""}${fmt(m.classifier_delta_bal_acc, 2)})`
          : "";
        return el("div", { class: "summary-row" + (m.id === d.best_method ? " star" : "") },
          el("div", { class: "summary-name" },
            (m.id === d.best_method ? "★ " : "") + m.id,
            el("span", { class: "summary-meta" }, note),
          ),
          el("div", { class: "summary-track" }, bar),
          el("div", { class: "summary-val" },
            `${m.n_pass}/${m.n_total}  ·  ${pct(m.pass_rate)}` +
            (m.n_basic_total ? `  ·  basic ${m.n_basic_pass}/${m.n_basic_total}` : ""),
          ),
        );
      }),
    ),
  );
}

// --- Tabs (one per trained method) ---------------------------------------
async function loadMethods() {
  try {
    const r = await fetch("/api/methods");
    const data = await r.json();
    METHODS = data.methods || [];
    // Resolve which tab to start on: ?method=... in the URL, else default.
    const urlMethod = new URLSearchParams(location.search).get("method")
                   || new URLSearchParams(location.search).get("model");  // legacy
    const hasMethod = (id) => METHODS.some((m) => m.id === id);
    const initial =
      (urlMethod && hasMethod(urlMethod)) ? urlMethod :
      (data.default && hasMethod(data.default)) ? data.default :
      (METHODS[0] && METHODS[0].id);
    buildTabs();
    if (initial) selectTab(initial, { skipUrl: !!urlMethod });
  } catch (err) {
    tabbar.textContent = "Failed to load methods: " + err.message;
  }
}

function buildTabs() {
  tabbar.innerHTML = "";
  for (const m of METHODS) {
    const btn = el("button", {
      class: "tab", "data-method": m.id, type: "button",
      title: m.blurb || m.id,
    },
      el("span", { class: "tab-main" }, m.id),
      el("span", { class: "tab-sub" }, modelKindLabel(m.model_name)),
    );
    btn.addEventListener("click", () => selectTab(m.id));
    tabbar.append(btn);
  }
}

function selectTab(methodId, opts = {}) {
  if (methodId === CURRENT_METHOD) return;
  CURRENT_METHOD = methodId;
  tabbar.querySelectorAll(".tab").forEach((b) =>
    b.classList.toggle("tab-active", b.dataset.method === methodId));
  if (!opts.skipUrl) {
    const u = new URL(location.href);
    u.searchParams.set("method", methodId);
    u.searchParams.delete("model");  // drop the old key
    history.replaceState(null, "", u.toString());
  }
  probeResult.innerHTML = "";
  loadAdmin(methodId);
}

// --- Per-tab classifier details ------------------------------------------
async function loadAdmin(method) {
  setLoading();
  try {
    const r = await fetch("/api/admin?method=" + encodeURIComponent(method));
    if (!r.ok) throw new Error("HTTP " + r.status);
    CURRENT_ADMIN = await r.json();
    render(CURRENT_ADMIN);
  } catch (err) {
    headlineEl.textContent = "Failed to load: " + err.message;
  }
}

function setLoading() {
  for (const c of [headlineEl, modelsEl, confusionEl, importanceEl, featuresEl, treeEl]) c.innerHTML = "";
  headlineEl.append(el("span", { class: "loading" },
    el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }),
    " loading model details..."));
}

function renderModels(d) {
  modelsEl.innerHTML = "";
  const ms = d.models || {};

  const rows = [
    ["Embedder", ms.embedder || "-"],
    ["Answer model", ms.answer || "-"],
    ["Judge model", ms.judge || "-"],
  ];
  if (ms.query_translation) {
    rows.push(["Query translation model", ms.query_translation]);
  }

  modelsEl.append(
    el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, "Models used (transparency)"),
      el("p", { class: "card-sub" },
        "Exact model stack for this method: retrieval embedding, answer generation, and LLM judge."),
    ),
    el("dl", { class: "kv-grid" },
      ...rows.flatMap(([k, v]) => [
        el("dt", {}, k),
        el("dd", { class: "kv-mono" }, v),
      ]),
    ),
  );
}

function renderHeadline(d) {
  headlineEl.innerHTML = "";
  const cm = d.cv_confusion || {};
  headlineEl.append(
    el("div", { class: "headline-row" },
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, pct(cm.confident_precision)),
        el("div", { class: "headline-label" }, "When marked complete, answer holds up"),
        el("div", { class: "headline-sub" },
          cm.raw_accuracy != null ? "baseline pass rate without the flag: " + pct(cm.raw_accuracy) : ""),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num warn" }, pct(cm.flag_wrong_rate)),
        el("div", { class: "headline-label" }, "When flagged, answer was missing a piece"),
        el("div", { class: "headline-sub" }, "Worth a closer look at the cited sources"),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, pct(cm.flag_catch_rate)),
        el("div", { class: "headline-label" }, "Share of incomplete answers caught"),
        el("div", { class: "headline-sub" }, "Verify-bucket share: " + pct(1 - (cm.confident_coverage || 0))),
      ),
    ),
  );
}

function renderConfusion(d) {
  confusionEl.innerHTML = "";
  const cm = d.cv_confusion;
  if (!cm) {
    confusionEl.append(el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, "5-fold confusion matrix"),
      el("p",  { class: "card-sub" }, "No signals/results file for this method.")));
    return;
  }
  const total = cm.n || 1;
  const cell = (n, cls, label) =>
    el("div", { class: "cm-cell " + (cls || "") },
      el("div", { class: "cm-n" }, String(n)),
      el("div", { class: "cm-pct" }, Math.round(n / total * 100) + "%"),
      el("div", { class: "cm-lbl" }, label),
    );

  confusionEl.append(
    el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, "Where the flag agrees with reality"),
      el("p",  { class: "card-sub" },
        "5-fold cross-validation on " + cm.n + " expert-labelled questions. " +
        "Most \"incomplete\" answers were only missing a sub-clause - not strictly wrong."),
    ),
    el("div", { class: "cm-grid" },
      el("div", { class: "cm-corner" }, ""),
      el("div", { class: "cm-h" }, "marked complete"),
      el("div", { class: "cm-h" }, "marked verify"),
      el("div", { class: "cm-v" }, "answer was complete"),
      cell(cm.tp, "tp", "matched - trusted"),
      cell(cm.fn, "fn", "flagged anyway"),
      el("div", { class: "cm-v" }, "answer was incomplete"),
      cell(cm.fp, "fp", "missed by the flag"),
      cell(cm.tn, "tn", "matched - flagged"),
    ),
  );
}

function renderImportances(d) {
  importanceEl.innerHTML = "";
  const isTree = d.model_kind === "decision_tree";
  const list = isTree ? d.feature_importances : d.coefficients;
  const top = (list || []).filter((x) =>
    isTree ? x.importance > 0 : Math.abs(x.coef) > 0).slice(0, 6);
  const max = Math.max(1e-6, ...top.map((x) => isTree ? x.importance : Math.abs(x.coef)));

  importanceEl.append(
    el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, isTree ? "What the tree pays attention to" : "What the regression weighs"),
      el("p",  { class: "card-sub" },
        isTree
          ? "Bigger bar = the feature does more of the work separating complete from incomplete answers."
          : "Each bar is a feature's weight in the score. Orange pushes toward 'complete'; red pushes toward 'verify'."),
    ),
    el("div", { class: "imp-list" },
      ...top.map((f) => {
        const v = isTree ? f.importance : f.coef;
        const w = Math.round((Math.abs(v) / max) * 100);
        const positive = isTree ? true : v >= 0;
        const bar = el("div", { class: "imp-bar " + (positive ? "pos" : "neg") });
        bar.style.width = w + "%";
        return el("div", { class: "imp-row" },
          el("div", { class: "imp-name" }, f.name),
          el("div", { class: "imp-track" }, bar),
          el("div", { class: "imp-val" },
            isTree
              ? fmt(f.importance)
              : (v >= 0 ? "+" : "") + fmt(f.coef) + "  · pushes " + (positive ? "complete" : "verify"),
          ),
        );
      }),
    ),
  );
}

function renderFeatureList(d) {
  featuresEl.innerHTML = "";
  const names = Array.isArray(d.feature_names) ? d.feature_names : [];

  featuresEl.append(
    el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, "Features used by this model"),
      el("p", { class: "card-sub" },
        "Complete trained feature set for this method (names only)."),
    ),
  );

  if (!names.length) {
    featuresEl.append(el("p", { class: "card-sub" }, "No feature list available."));
    return;
  }

  featuresEl.append(
    el("ul", { class: "feature-list" },
      ...names.map((name) => el("li", { class: "feature-list-item" }, name)),
    ),
  );
}

function renderTreeOrCoefs(d) {
  treeEl.innerHTML = "";
  if (d.model_kind === "decision_tree" && d.tree) {
    treeEl.append(
      el("div", { class: "card-head" },
        el("h2", { class: "card-title" }, "The decision rule"),
        el("p",  { class: "card-sub" },
          "Read top to bottom. Each node asks one yes/no question; left = yes, right = no. " +
          "Gini is the mix of classes in a node (0 = clean, 0.5 = 50/50)."),
      ),
      treeSVG(d.tree),
      el("p",  { class: "card-sub", style: "margin-top: 10px;" },
        d.tree.n_nodes + " nodes &middot; " + d.tree.n_leaves + " leaves &middot; depth " + d.tree.max_depth),
    );
    return;
  }
  // Logistic regression: short formula card.
  treeEl.append(
    el("div", { class: "card-head" },
      el("h2", { class: "card-title" }, "The decision rule" ),
      el("p",  { class: "card-sub" },
        "Each feature contributes a small amount to a single score; the sigmoid of that score is the probability the answer is complete."),
    ),
    el("div", { class: "formula" },
      el("div", { class: "formula-line" },
        "score  =  intercept  +  Σ (weight × feature)"),
      el("div", { class: "formula-line" },
        "p(complete)  =  sigmoid(score)"),
      el("div", { class: "formula-vals" },
        "intercept = " + fmt(d.intercept) +
        " · " + (d.nonzero_feature_count ?? "?") + " features in use"),
    ),
  );
}

function treeSVG(tree, opts = {}) {
  const W = Math.max(720, tree.nodes.length * 96);
  const H = (tree.max_depth + 1) * 130 + 60;
  const pad = 40;
  const nx = (n) => pad + (n.x + 0.5) * ((W - 2 * pad) / Math.max(1, tree.nodes.length));
  const ny = (n) => pad + n.y * 130;
  const byId = new Map(tree.nodes.map((n) => [n.node, n]));
  const highlightNodes = new Set(opts.highlightNodes || []);
  const highlightEdges = (opts.highlightEdges || [])
    .map((e) => e.from + "->" + e.to);

  const edges = tree.edges.map((e) => {
    const a = byId.get(e.from), b = byId.get(e.to);
    if (!a || !b) return null;
    const x1 = nx(a), y1 = ny(a) + 36;
    const x2 = nx(b), y2 = ny(b) - 4;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const hot = highlightEdges.includes(e.from + "->" + e.to);
    return svg("g", {},
      svg("path", {
        d: `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`,
        stroke: hot ? "#f25922" : "#2d2d2d",
        "stroke-width": hot ? "2.5" : "1.5",
        fill: "none",
      }),
      svg("text", { x: mx + 6, y: my - 2,
                    fill: hot ? "#f25922" : "#6f6f6f",
                    "font-size": "11", "font-family": "monospace" }, e.label),
    );
  });

  const nodes = tree.nodes.map((n) => {
    const cx = nx(n), cy = ny(n);
    const total = (n.class_counts || []).reduce((a, b) => a + b, 0) || 1;
    const passShare = (n.class_counts && n.class_counts[1]) ? n.class_counts[1] / total : 0;
    const hot = highlightNodes.has(n.node);
    const fill = n.predicted_class === 1
      ? "rgba(74, 222, 128, 0.18)" : "rgba(248, 113, 113, 0.14)";
    const stroke = hot ? "#f25922" : "#2a2a2a";
    const strokeW = hot ? "2.5" : "1";
    const labelLine = n.is_leaf
      ? `leaf · ${n.predicted_class === 1 ? "CONFIDENT" : "VERIFY"}`
      : `${n.feature} ≤ ${fmt(n.threshold)}`;
    return svg("g", {},
      svg("rect", { x: cx - 100, y: cy, width: "200", height: "60",
                    rx: "10", ry: "10", fill, stroke, "stroke-width": strokeW }),
      svg("text", { x: cx, y: cy + 24, fill: "#f5f5f5", "text-anchor": "middle",
                    "font-size": "12", "font-family": "monospace", "font-weight": "600" },
        labelLine),
      svg("text", { x: cx, y: cy + 42, fill: "#a8a8a8", "text-anchor": "middle",
                    "font-size": "11", "font-family": "monospace" },
        `n=${n.samples} · gini=${fmt(n.gini, 3)} · ${Math.round(passShare*100)}% pass`),
    );
  });

  return svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "tree-svg" }, ...edges, ...nodes);
}

// --- Probe ---------------------------------------------------------------
probeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = probeInput.value.trim();
  if (!q) return;
  probeSend.disabled = true;
  probeResult.innerHTML = "";
  probeResult.append(el("div", { class: "loading" },
    el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }),
    " probing the model..."));
  try {
    const r = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, method: CURRENT_METHOD }),
    });
    renderProbe(await r.json());
  } catch (err) {
    probeResult.innerHTML = "";
    probeResult.append(el("div", { class: "error-banner" }, String(err)));
  } finally { probeSend.disabled = false; }
});

function renderProbe(data) {
  probeResult.innerHTML = "";
  const fb = data.feature_breakdown, conf = data.confidence;
  if (!fb || !conf) {
    if (data.error) probeResult.append(el("div", { class: "error-banner" }, data.error));
    else probeResult.append(el("div", { class: "card-sub" }, "(no breakdown returned)"));
    return;
  }
  const isTree = fb.model_kind === "decision_tree";
  // The chip is "good" when the predicted class is 1 (complete) -- robust
  // to whatever soft wording the server uses for `label`.
  const goodish = fb.predicted_class === 1;
  const chipCls = "conf-chip " + (goodish ? "good" : "warn");
  const scoreRingCls = "score-ring " + (goodish ? "good" : "warn");
  const scorePct = isFinite(conf.score) ? Math.round(conf.score * 100) : null;

  probeResult.append(
    el("div", { class: "probe-row" },
      el("div", { class: "probe-side" },
        el("div", { class: scoreRingCls, title: "model-estimated probability the answer is complete" },
          el("div", { class: "score-num" }, scorePct == null ? "-" : scorePct + "%"),
          el("div", { class: "score-label" }, "complete"),
        ),
      ),
      el("div", { class: "probe-side wide" },
        el("div", { class: "conf-row-top" },
          el("span", { class: chipCls }, (goodish ? "● " : "⚠ ") + conf.label),
        ),
        el("div", { class: "conf-headline" }, conf.headline),
      ),
    ),
  );

  // Top 4 contributing features for this question
  const top = (fb.top_contributions || []).slice(0, 4);
  const max = Math.max(1e-6, ...top.map((c) => Math.abs(isTree ? c.importance : c.contribution)));
  probeResult.append(
    el("div", { class: "imp-section-title" }, "What drove this score"),
    el("div", { class: "imp-list" },
      ...top.map((c) => {
        const v = isTree ? c.importance : c.contribution;
        const w = Math.round((Math.abs(v) / max) * 100);
        const positive = isTree ? true : v >= 0;
        const bar = el("div", { class: "imp-bar " + (positive ? "pos" : "neg") });
        bar.style.width = w + "%";
        return el("div", { class: "imp-row" },
          el("div", { class: "imp-name" }, c.name,
            isTree && c.active_in_path ? el("span", { class: "fb-tag" }, "in path") : null,
          ),
          el("div", { class: "imp-track" }, bar),
          el("div", { class: "imp-val" }, "value " + fmt(c.value)),
        );
      }),
    ),
  );

  // Decision path + highlighted tree (DT only)
  if (isTree && fb.decision_path && CURRENT_ADMIN?.tree) {
    const pathNodes = fb.decision_path.map((n) => n.node);
    const pathEdges = [];
    for (let i = 0; i < fb.decision_path.length - 1; i++) {
      pathEdges.push({ from: fb.decision_path[i].node, to: fb.decision_path[i+1].node });
    }
    probeResult.append(
      el("div", { class: "imp-section-title" }, "Path through the tree for this question"),
      treeSVG(CURRENT_ADMIN.tree, { highlightNodes: pathNodes, highlightEdges: pathEdges }),
    );
  }
}

// --- Render orchestration ------------------------------------------------
function render(d) {
  renderHeadline(d);
  renderModels(d);
  renderConfusion(d);
  renderImportances(d);
  renderFeatureList(d);
  renderTreeOrCoefs(d);
}

loadSummary();
loadMethods();
