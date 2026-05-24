// Single-page chat client for the Taxxa mock UI. /api/methods + /api/ask.
const stream = document.getElementById("chat-stream");
const form   = document.getElementById("composer");
const input  = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");
const methodSelect = document.getElementById("method-select");
const methodBlurb  = document.getElementById("method-blurb");

let METHODS = [];

async function loadMethods() {
  try {
    const r = await fetch("/api/methods");
    const data = await r.json();
    METHODS = data.methods || [];
    const def = data.default;
    methodSelect.innerHTML = "";
    for (const m of METHODS) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === def) opt.selected = true;
      methodSelect.append(opt);
    }
    updateMethodBlurb();
  } catch (err) {
    methodSelect.innerHTML = '<option value="section_v1">section_v1</option>';
    console.error("loadMethods failed", err);
  }
}

function currentMethod() {
  return methodSelect.value || "section_v1";
}

function updateMethodBlurb() {
  const m = METHODS.find((x) => x.id === currentMethod());
  if (!m) { methodBlurb.textContent = ""; return; }
  methodBlurb.textContent = m.blurb + "  -  classifier: " + (m.model_name || "?");
}

methodSelect?.addEventListener("change", updateMethodBlurb);
loadMethods();

// --- DOM helpers ----------------------------------------------------------
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

function scrollToBottom() {
  // Defer one frame so freshly-added DOM is measured before scrolling.
  requestAnimationFrame(() => { stream.scrollTop = stream.scrollHeight; });
}

// --- Message blocks -------------------------------------------------------
function userMsg(text) {
  return el("article", { class: "msg user" },
    el("header", { class: "msg-head" },
      el("span", { class: "msg-icon", "aria-hidden": "true" }, "Y"),
      el("span", { class: "msg-author" }, "You"),
    ),
    el("div", { class: "msg-body" }, text),
  );
}

function loadingMsg() {
  return el("article", { class: "msg assistant", "data-loading": "1" },
    el("header", { class: "msg-head" },
      el("span", { class: "msg-icon", "aria-hidden": "true" }, "✱"),
      el("span", { class: "msg-author" }, "Taxxa Assistant"),
    ),
    el("div", { class: "msg-body" },
      el("span", { class: "loading" },
        el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }), el("span", { class: "loading-dot" }),
        " retrieving and drafting...",
      ),
    ),
  );
}

function errorMsg(text) {
  return el("article", { class: "msg error" },
    el("header", { class: "msg-head" },
      el("span", { class: "msg-icon", "aria-hidden": "true" }, "!"),
      el("span", { class: "msg-author" }, "Error"),
    ),
    el("div", { class: "msg-body" }, text),
  );
}

function confidenceBlock(conf) {
  if (!conf) return null;
  const goodish = (conf.label || "").toLowerCase().startsWith("look");
  const chipCls = "conf-chip " + (goodish ? "good" : "warn");
  const dot = goodish ? "● " : "⚠ ";
  const scorePct = isFinite(conf.score) ? Math.round(conf.score * 100) : null;

  // Prominent score: big circular meter on the left so the user sees the
  // number first. Smaller chip + explanatory text on the right.
  const scoreCls = "score-ring " + (goodish ? "good" : "warn");
  const meter = el("div", { class: scoreCls, title: "model-estimated probability of a complete answer" },
    el("div", { class: "score-num" }, scorePct == null ? "-" : scorePct + "%"),
    el("div", { class: "score-label" }, "complete"),
  );

  return el("div", { class: "confidence" },
    el("div", { class: "conf-head-grid" },
      meter,
      el("div", { class: "conf-text" },
        el("div", { class: "conf-row-top" },
          el("span", { class: chipCls }, dot + conf.label),
        ),
        el("div", { class: "conf-headline" }, conf.headline),
        el("div", { class: "conf-detail" }, conf.detail),
      ),
    ),
  );
}

function sourcesBlock(sources) {
  if (!sources || !sources.length) return null;
  const items = sources.map((s, i) =>
    el("li", { class: "source-row" },
      el("span", { class: "source-rank" }, String(i + 1)),
      el("span", { class: "source-title-only" }, s.title || s.file),
    ),
  );
  return el("div", { class: "sources" },
    el("div", { class: "sources-head" },
      el("span", { class: "sources-title" }, "Sources"),
      el("span", { class: "sources-count" }, sources.length + " retrieved"),
    ),
    el("ol", { class: "sources-list-simple" }, ...items),
  );
}

function breakdownBlock(payload) {
  const fb = payload.feature_breakdown;
  if (!fb || !fb.top_contributions) return null;
  const method = payload.method || "section_v1";
  const isTree = fb.model_kind === "decision_tree";

  // Compact top-3 visualisation. Full view lives on /admin.
  const top = fb.top_contributions.slice(0, 4);
  const maxAbs = Math.max(1e-6, ...top.map((c) =>
    Math.abs(isTree ? c.importance : c.contribution)));

  const rows = top.map((c) => {
    const v = isTree ? c.importance : c.contribution;
    const widthPct = Math.round((Math.abs(v) / maxAbs) * 100);
    const positive = isTree ? true : v >= 0;
    const bar = el("div", { class: "fb-bar " + (positive ? "pos" : "neg") });
    bar.style.width = widthPct + "%";
    const valTxt = isTree
      ? "imp=" + v.toFixed(3) + "   val=" + (c.value ?? 0).toFixed(3)
      : (v >= 0 ? "+" : "") + v.toFixed(3) + "   val=" + (c.value ?? 0).toFixed(3);
    const tag = isTree && c.active_in_path
      ? el("span", { class: "fb-tag" }, "in path")
      : null;
    return el("div", { class: "fb-row" },
      el("div", { class: "fb-name" }, c.name, tag),
      el("div", { class: "fb-track" }, bar),
      el("div", { class: "fb-val" }, valTxt),
    );
  });

  const explanation = isTree
    ? "Top features the decision tree relied on. 'in path' = the feature was used to decide this specific question."
    : "Top features for logistic regression. Positive = pushed toward CONFIDENT, negative = pushed toward VERIFY.";

  return el("div", { class: "breakdown" },
    el("div", { class: "breakdown-head" },
      el("span", { class: "breakdown-title" }, "Why this confidence score?"),
      el("a", { class: "breakdown-link", href: "/admin?method=" + encodeURIComponent(method) },
        "see full model details →"),
    ),
    el("div", { class: "breakdown-explainer" }, explanation),
    el("div", { class: "fb-list" }, ...rows),
  );
}

function assistantMsg(payload) {
  const body = el("div", { class: "msg-body" });
  if (payload.error) {
    body.append(el("div", { class: "error-banner" }, payload.error));
  }
  body.append(payload.answer || "(no answer returned)");
  // Confidence chip stays (small + actionable). The deep feature breakdown
  // and tree path live on /admin -- they're not part of an end-user answer.
  const conf = confidenceBlock(payload.confidence);
  if (conf) body.append(conf);
  const src = sourcesBlock(payload.sources);
  if (src) body.append(src);
  return el("article", { class: "msg assistant" },
    el("header", { class: "msg-head" },
      el("span", { class: "msg-icon", "aria-hidden": "true" }, "✱"),
      el("span", { class: "msg-author" }, "Taxxa Assistant"),
    ),
    body,
  );
}

// --- Network --------------------------------------------------------------
async function ask(question) {
  const method = currentMethod();
  const r = await fetch("/api/ask", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question, method }),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return await r.json();
}

// --- Submit flow ----------------------------------------------------------
async function submitQuestion(text) {
  const q = (text || "").trim();
  if (!q) return;
  input.value = "";
  sendBtn.disabled = true;
  // Drop the intro card once the user sends their first message.
  const intro = stream.querySelector(".msg.intro");
  if (intro) intro.remove();

  stream.append(userMsg(q));
  const loader = loadingMsg();
  stream.append(loader);
  scrollToBottom();

  try {
    const result = await ask(q);
    loader.remove();
    // Always render as an assistant message -- the soft-error path inside
    // assistantMsg() shows the error banner together with whatever the
    // pipeline produced (retrieved sources, confidence flag). Only fall
    // back to errorMsg if there's literally nothing to display.
    if (result.error && !result.sources?.length && !result.answer) {
      stream.append(errorMsg(result.error));
    } else {
      stream.append(assistantMsg(result));
    }
  } catch (err) {
    loader.remove();
    stream.append(errorMsg(String(err)));
  } finally {
    sendBtn.disabled = false;
    scrollToBottom();
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  submitQuestion(input.value);
});

input.addEventListener("keydown", (e) => {
  // Enter to send; Shift+Enter inserts a newline (default textarea behaviour).
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitQuestion(input.value);
  }
});

// Wire the example chips in the intro card.
document.querySelectorAll(".example-q").forEach((btn) => {
  btn.addEventListener("click", () => submitQuestion(btn.dataset.q || btn.textContent));
});

input.focus();
