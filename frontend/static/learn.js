// /learn page: fetches /api/summary and /api/example_pathways and
// renders the test-bank stats + the two-question pathway comparison.

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const kid of kids) {
    if (kid == null) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
}
const pct = (x, d = 0) => (x == null ? "-" : (x * 100).toFixed(d) + "%");
const fmt = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));

// --- Test-bank summary card ----------------------------------------------
async function loadBank() {
  const host = document.getElementById("bank-content");
  try {
    const r = await fetch("/api/summary");
    const d = await r.json();
    renderBank(host, d);
  } catch (err) {
    host.textContent = "Failed to load summary: " + err.message;
  }
}

function renderBank(host, d) {
  host.innerHTML = "";
  host.classList.remove("loading");

  // Three big numbers
  const evaluated = d.methods.filter((m) => m.evaluated_actual);
  const bestRaw = evaluated.reduce(
    (a, b) => (a && a.pass_rate >= b.pass_rate ? a : b), null);
  host.append(
    el("div", { class: "headline-row" },
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, String(d.n_questions)),
        el("div", { class: "headline-label" }, "Expert questions in the bank"),
        el("div", { class: "headline-sub" }, "covers basic, medium and hard tiers"),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, String(evaluated.length)),
        el("div", { class: "headline-label" }, "Retrieval methods compared"),
        el("div", { class: "headline-sub" }, "each has its own trained classifier"),
      ),
      el("div", { class: "headline-block" },
        el("div", { class: "headline-num" }, "5"),
        el("div", { class: "headline-label" }, "Cross-validation folds"),
        el("div", { class: "headline-sub" }, "stratified, seed = 42"),
      ),
    ),
  );

  // Per-method pass rate strip (compact)
  const maxPass = Math.max(0.01, ...evaluated.map((m) => m.pass_rate));
  host.append(
    el("div", { class: "summary-strip-title" },
      "Pass rate per method on the bank (★ = best classifier delta)"),
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
        return el("div",
          { class: "summary-row" + (m.id === d.best_method ? " star" : "") },
          el("div", { class: "summary-name" },
            (m.id === d.best_method ? "★ " : "") + m.id),
          el("div", { class: "summary-track" }, bar),
          el("div", { class: "summary-val" },
            `${m.n_pass}/${m.n_total} · ${pct(m.pass_rate)}`),
        );
      }),
    ),
  );
}

// --- Example pathways card -----------------------------------------------
async function loadExamples() {
  const host = document.getElementById("examples-content");
  try {
    const r = await fetch("/api/example_pathways");
    if (!r.ok) throw new Error("HTTP " + r.status);
    const d = await r.json();
    renderExamples(host, d);
  } catch (err) {
    host.textContent = "Failed to load examples: " + err.message;
  }
}

function renderExamples(host, d) {
  host.innerHTML = "";
  host.classList.remove("loading");
  const examples = d.examples || [];
  if (!examples.length) {
    host.textContent = "(no examples returned)";
    return;
  }

  // Build cards (one per example), then put them side by side.
  // Find feature max per feature across examples so bars are comparable.
  const features = d.features_legend || [];
  const maxByFeature = new Map();
  for (const f of features) {
    let max = 0;
    for (const ex of examples) {
      const fv = (ex.features || []).find((x) => x.name === f.name);
      if (fv) max = Math.max(max, Math.abs(fv.value));
    }
    maxByFeature.set(f.name, Math.max(max, 1e-6));
  }

  const cards = examples.map((ex) => {
    const goodish = ex.predicted_class === 1;
    const scorePct = Math.round((ex.pass_probability || 0) * 100);
    const chipCls = "conf-chip " + (goodish ? "good" : "warn");
    const ringCls = "score-ring " + (goodish ? "good" : "warn");
    const factsLine = (ex.n_facts_total != null)
      ? `LLM judge: ${ex.n_facts_passed}/${ex.n_facts_total} facts present`
      : "";
    const labelLine = ex.actually_passed ? "actually complete" : "actually missed a piece";

    const featureRows = (ex.features || []).map((fv) => {
      const max = maxByFeature.get(fv.name) || 1;
      const w = Math.round(Math.min(1, Math.abs(fv.value) / max) * 100);
      const bar = el("div", { class: "imp-bar pos" });
      bar.style.width = w + "%";
      return el("div", { class: "imp-row" },
        el("div", { class: "imp-name" }, fv.label || fv.name,
          el("span", { class: "feature-name-muted" }, " (" + fv.name + ")")),
        el("div", { class: "imp-track" }, bar),
        el("div", { class: "imp-val" }, fmt(fv.value, 3)),
      );
    });

    const sourcesList = (ex.top_sources || []).length
      ? el("div", { class: "example-sources" },
          el("div", { class: "imp-section-title" }, "Top retrieved chunks"),
          ...(ex.top_sources || []).map((s, i) =>
            el("div", { class: "example-source" },
              el("span", { class: "source-rank" }, String(i + 1)),
              el("span", { class: "source-title-only" }, s))),
        )
      : null;

    return el("div", { class: "example-card" },
      el("div", { class: "example-head" },
        el("div", { class: "example-id" }, ex.id + " · " + (ex.tier || "")),
        el("div", { class: ringCls, title: "model-estimated probability the answer is complete" },
          el("div", { class: "score-num" }, scorePct + "%"),
          el("div", { class: "score-label" }, "complete"),
        ),
      ),
      el("div", { class: "example-question" }, ex.question),
      el("div", { class: "example-meta" },
        el("span", { class: chipCls }, (goodish ? "● " : "⚠ ") + ex.label),
        el("span", { class: "example-truth" }, " · " + labelLine),
      ),
      factsLine ? el("div", { class: "example-facts" }, factsLine) : null,
      el("div", { class: "imp-section-title" }, "Pathway fingerprint"),
      el("div", { class: "imp-list" }, ...featureRows),
      sourcesList,
    );
  });

  host.append(el("div", { class: "example-grid" }, ...cards));
  host.append(
    el("p", { class: "card-sub", style: "margin-top: 14px;" },
      "Notice: the questions are both about gift tax, but the second one needs "
    + "three sub-rules at once (aggregation, generation skip, minor recipient). "
    + "Those rules aren't all in the top-3 chunks, so the pathway has lower "
    + "concentration - and the classifier picks that up before the LLM even runs."),
  );
}

loadBank();
loadExamples();
