"use strict";
/* ReproFix UI. No build step. All untrusted text (repo files, logs, web snippets) is rendered with
   textContent via h(); nothing from a run is ever assigned to innerHTML. */

const STEPS = [
  ["inspect", "Inspect repository"], ["deps", "Parse dependencies"], ["entry", "Identify entry points"],
  ["env", "Build environment"], ["baseline", "Run baseline"], ["analyze", "Analyze failures"],
  ["hypothesize", "Generate hypotheses"], ["test", "Test fixes"], ["verify", "Verify results"],
];
const STATUS = {
  verified: ["Reproduced", "ok"], executes: ["Runs, result not checked", "warn"], already_passing: ["Already passing", "ok"],
  partial: ["Partly fixed", "warn"], failed: ["Not fixed", "bad"], cancelled: ["Cancelled", ""], error: ["Error", "bad"],
  interrupted: ["Interrupted", "bad"], running: ["Investigating", "live"], queued: ["Queued", "live"],
};
const STAGE_TEXT = { install_failure: "install fails", crash: "crashes", wrong_result: "wrong result", verified: "verified" };
const EVENT_KINDS = ["run.started", "step", "acquired", "analysis", "claims", "known_issues", "plan", "exec.start", "exec", "verdict", "diagnosis",
  "hypothesis.selected", "edit.proposed", "experiment", "attempt.failed", "graph", "usage", "report", "run.finished", "error"];

/* ---------------------------------------------------------------- helpers */
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  add(el, kids);
  return el;
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}
const SVGNS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  add(el, kids);
  return el;
}
const fmt = (v, d = 4) => (v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
const money = (v) => (v == null ? "n/a" : "$" + Number(v).toFixed(4));
const token = () => sessionStorage.getItem("reprofix_token") || "";
const authQ = (sep) => (token() ? `${sep}token=${encodeURIComponent(token())}` : "");

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(token() ? { Authorization: "Bearer " + token() } : {}) };
  const res = await fetch(path, { ...opts, headers });
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const d = body && (body.error || body.detail);
    throw new Error((Array.isArray(d) ? d.map((x) => x.msg || JSON.stringify(x)).join("; ") : d) || `HTTP ${res.status}`);
  }
  return body;
}
function chip(statusKey) {
  const [text, cls] = STATUS[statusKey] || [statusKey, ""];
  return h("span", { class: "chip " + cls, text });
}
function repoName(src) { return (src || "").replace(/\/+$/, "").split("/").slice(-2).join("/").replace(/\.git$/, "") || "repository"; }
function when(ts) { return new Date(ts * 1000).toLocaleString(); }
function safeUrl(u) { return /^https?:\/\//i.test(u || "") ? u : null; }

/* ---------------------------------------------------------------- routing */
const view = document.getElementById("view");
let cleanup = null;
function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  const [, page, arg] = (location.hash || "#/").split("/");
  document.querySelectorAll("[data-nav]").forEach((a) => a.removeAttribute("aria-current"));
  const nav = { "": "start", runs: "runs", bench: "bench", run: "runs" }[page || ""];
  const cur = document.querySelector(`[data-nav="${nav}"]`);
  if (cur) cur.setAttribute("aria-current", "page");
  // Every navigation gets its own container. A render that is still awaiting the network when the user
  // navigates away then writes into a detached node instead of overwriting the page that replaced it.
  const slot = h("div");
  view.replaceChildren(slot);
  slot.append(h("p", { class: "loading", text: "Loading…" }));
  const r = page === "run" && arg ? renderRun(arg, slot) : page === "runs" ? renderRuns(slot) : page === "bench" ? renderBench(slot) : renderStart(slot);
  Promise.resolve(r).catch((e) => slot.replaceChildren(h("p", { class: "banner bad", text: "Could not load this page: " + e.message })));
}
window.addEventListener("hashchange", route);

/* ---------------------------------------------------------------- start */
async function renderStart(slot) {
  const [health, runs] = await Promise.all([api("/api/health"), api("/api/runs").catch(() => [])]);
  document.getElementById("token-wrap").hidden = !health.auth_required;
  const err = h("p", { class: "form-error", role: "alert" });
  const f = {};
  const field = (key, label, el, hint) => { f[key] = el; return h("label", { class: "field" }, label, el, hint ? h("small", { text: hint }) : null); };
  const form = h("form", { class: "form", novalidate: true },
    field("repo", "Repository", h("input", { type: "url", placeholder: "https://github.com/owner/repo", required: true, autocomplete: "off" }),
      `Public https URL. Allowed hosts: ${health.limits.allowed_git_hosts.join(", ")}.`),
    field("goal", "What should reproduce?", h("textarea", { rows: 3 }, "Reproduce the documented experiment and find out why it fails or why the result differs.")),
    h("div", { class: "row" },
      field("metric", "Metric name", h("input", { value: "val_accuracy" }), "Printed by the program as name: value."),
      field("expected", "Documented value", h("input", { inputmode: "decimal", placeholder: "0.87" }), "Leave empty to only check that it runs."),
      field("tol", "Tolerance (±)", h("input", { inputmode: "decimal", value: "0.02" }))),
    h("div", { class: "row" },
      field("cmd", "Command", h("input", { placeholder: "detected from the README" }), "Must start with python or pytest."),
      field("mode", "Model routing", h("select", {},
        h("option", { value: "router" }, "Router (all tiers)"), h("option", { value: "super-only" }, "Super only"), h("option", { value: "ultra-only" }, "Ultra only")),
        "Router sends cheap work to Nano, most work to Super, and hard diagnosis to Ultra."),
      field("attempts", "Max experiments", h("input", { type: "number", min: 1, max: health.limits.max_attempts, value: Math.min(6, health.limits.max_attempts) }))),
    h("details", { class: "claimbox" },
      h("summary", { text: "Check a paper's headline result (optional)" }),
      field("claims", "Claims to check, one per line", h("textarea", { rows: 2, placeholder: "accuracy=76.4%\nbleu=27.3±0.3", spellcheck: "false" }),
        "name=value. Add % for a percentage, ±tolerance if the paper gives one. The first line is the number ReproFix repairs toward."),
      field("paper", "Paper or README text", h("textarea", { rows: 4, placeholder: "Paste the abstract, results paragraph or results table." }),
        "Numbers like 'accuracy of 76.4%' are read from this text by fixed patterns. It is never sent to a model, so confirm what was read in the report.")),
    h("div", { class: "actions" },
      h("button", { class: "primary", type: "submit" }, "Start investigation"),
      health.demo_available ? h("button", { type: "button", onclick: startDemo }, "Run the offline demo") : null, err),
  );
  async function go(path, body) {
    err.textContent = "";
    try { const { id } = await api(path, { method: "POST", body: JSON.stringify(body || {}) }); location.hash = "#/run/" + id; }
    catch (e) { err.textContent = e.message; }
  }
  function startDemo() { go("/api/demo"); }
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const exp = f.expected.value.trim();
    if (exp && Number.isNaN(Number(exp))) { err.textContent = "Documented value must be a number."; return; }
    let claims;
    try { claims = f.claims.value.split("\n").map((l) => l.trim()).filter(Boolean).map(parseClaimLine); }
    catch (e) { err.textContent = e.message; return; }
    const paper = f.paper.value.trim();
    const body = {
      repo_url: f.repo.value.trim(), goal: f.goal.value.trim(), router_mode: f.mode.value, max_attempts: Number(f.attempts.value) || 6,
      ...(f.cmd.value.trim() ? { command: f.cmd.value.trim() } : {}),
      ...(claims.length ? { claims } : {}), ...(paper ? { paper_text: paper } : {}),
      ...(exp ? { metric: { name: f.metric.value.trim() || "val_accuracy", expected: Number(exp), tolerance: Number(f.tol.value) || 0.02 } } : {}),
    };
    if (!body.repo_url) { err.textContent = "Enter a repository URL."; return; }
    go("/api/runs", body);
  });
  const llmOk = health.llm.configured, sb = health.sandbox;
  const row = (dt, dd) => h("div", {}, h("dt", { text: dt }), h("dd", {}, dd));
  const dot = (cls) => h("span", { class: "dot " + cls });
  slot.replaceChildren(h("div", { class: "start" },
    h("section", {},
      h("h1", { text: "Reproduce the result. Prove the fix." }),
      h("p", { class: "lede", text: "Give ReproFix a broken machine-learning repository and the result it should reproduce. It runs the code in a sandbox, works out what is wrong, patches it, and reruns until the number matches or tells you it could not." }),
      form),
    h("aside", { class: "aside" },
      h("section", {}, h("h2", { text: "This server" }), h("dl", { class: "facts" },
        row("Model", [dot(llmOk ? "ok" : "bad"), llmOk ? "Nemotron on Nebius Token Factory" : "NEBIUS_API_KEY not set. Real runs will fail; the offline demo still works."]),
        row("Sandbox", sb.available ? [dot(sb.filesystem_isolated ? "ok" : "warn"), sb.kind + (sb.filesystem_isolated ? ", isolated" : ", development only, not isolated")] : [dot("bad"), "Unavailable: " + sb.error]),
        row("Web search", [dot(health.tavily.configured ? "ok" : "warn"), health.tavily.configured ? "Tavily enabled" : "Tavily not configured. No external evidence."]),
        row("Limits", `${health.limits.max_attempts} experiments, ${Math.round(health.limits.max_run_seconds / 60)} minutes per run`))),
      h("section", {}, h("h2", { text: "Recent runs" }), runList(runs.slice(0, 6)))),
  ));
}
/* "accuracy=76.4%", "val_acc: 0.912", "bleu=27.3±0.3"  ->  {metric, value, unit?, tolerance?}  (same syntax as `reprofix run --claim`) */
function parseClaimLine(line) {
  const m = /^\s*([^=:]*[A-Za-z][^=:]*?)\s*[=:]\s*(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(%)?\s*(?:(?:±|\+-|\+\/-)\s*((?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*%?)?\s*$/.exec(line);
  if (!m) throw new Error(`Cannot read the claim "${line.slice(0, 60)}". Write it like accuracy=76.4% or val_acc=0.912 or bleu=27.3±0.3.`);
  return { metric: m[1].trim(), value: Number(m[2]), ...(m[3] ? { unit: "percent" } : {}), ...(m[4] ? { tolerance: Number(m[4]) } : {}) };
}
function runList(runs) {
  if (!runs.length) return h("p", { class: "empty", text: "No runs yet. Start one on the left." });
  return h("ul", { class: "runlist" }, runs.map((r) => h("li", {}, h("a", { href: "#/run/" + r.id },
    h("span", { class: "repo", text: repoName(r.repo) }), chip(r.status),
    h("span", { class: "sub", text: `${when(r.created)}${r.label ? " — " + r.label : ""}` })))));
}
async function renderRuns(slot) {
  const runs = await api("/api/runs");
  slot.replaceChildren(h("h1", { text: "Past runs" }), h("div", { style: "max-width:760px;margin-top:14px" }, runList(runs)));
}

/* ---------------------------------------------------------------- benchmark */
async function renderBench(slot) {
  const b = await api("/api/benchmark");
  const kids = [h("h1", { text: "ReproBench" }),
    h("p", { class: "lede", style: "color:var(--muted);max-width:70ch;margin:8px 0 18px", text: `${b.n_tasks} broken repositories, each with a reference fix. Scores are measured by running code, never judged by a model.` })];
  if (!b.results.length) {
    kids.push(h("p", { class: "empty", text: "No results saved yet. Run `reprofix bench run` with a Nebius key; numbers appear here only after a real run." }));
  }
  for (const r of b.results) kids.push(benchSection(r));
  slot.replaceChildren(...kids);
}
function benchSection(r) {
  const m = r.meta, n = r.n_tasks, t = r.totals;
  const stages = ["understood", "reproduced", "root_cause", "patch_generated", "patch_verified"];
  return h("section", { class: "sheet", style: "padding:18px;margin-bottom:20px;display:grid;gap:14px" },
    h("div", { class: "bench-head" }, h("h2", { text: m.label || "Run" }), h("span", { class: "chip", text: m.date }),
      h("span", { class: "chip", text: "routing: " + m.router_mode }), h("span", { class: "chip", text: "sandbox: " + m.sandbox })),
    m.llm_backend !== "nebius" ? h("p", { class: "banner" }, `Model backend "${m.llm_backend}" is not Nemotron. These numbers validate the pipeline and the benchmark only. They say nothing about model capability.`) : null,
    h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", { text: "Stage" }), h("th", { text: "Passed" }), h("th", { text: "" }))),
      h("tbody", {}, stages.map((st) => h("tr", {}, h("td", { text: st.replace("_", " ") }), h("td", { class: "num", text: `${r.stages[st]}/${n}` }),
        h("td", {}, h("div", { class: "bar" }, h("i", { style: `width:${(100 * r.stages[st]) / Math.max(n, 1)}%` }))))))),
    h("p", { class: "empty", text: `${t.tokens.toLocaleString()} tokens, ${t.llm_calls} model calls, mean ${t.mean_attempts} experiments per task, cost ${t.cost_usd_priced_tiers == null ? "n/a" : money(t.cost_usd_priced_tiers) + (t.any_unpriced_tiers ? " (priced tiers only)" : "")}` }),
    h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", { text: "Task" }), h("th", { text: "Category" }), h("th", { text: "Status" }),
      stages.map((st) => h("th", { text: st.split("_")[0] })))),
      h("tbody", {}, r.results.map((x) => h("tr", {}, h("td", { text: x.id }), h("td", { text: x.category }), h("td", {}, chip(x.status)),
        stages.map((st) => h("td", {}, h("span", { class: "mark " + (x.stages[st] ? "y" : "n"), text: x.stages[st] ? "✓" : "✗" }))))))));
}

/* ---------------------------------------------------------------- run view */
function newState() {
  return { steps: {}, execs: [], verdicts: [], diagnoses: [], proposals: [], experiments: [], failures: [], graph: { nodes: [], edges: [] },
    analysis: null, metric: null, metricSource: "", plan: null, usage: null, report: null, started: null, finished: null,
    error: null, running: null, claims: null, issues: [], tab: "graph", sel: null, execSel: null, autoTab: true, pr: null };
}
const HANDLERS = {
  "run.started": (st, d) => { st.started = d; },
  step: (st, d) => { st.steps[d.id] = d; },
  analysis: (st, d) => { st.analysis = d.analysis; st.metric = d.metric; st.metricSource = d.metric_source; },
  claims: (st, d) => { st.claims = d; },
  known_issues: (st, d) => { st.issues.push(d); },
  plan: (st, d) => { st.plan = d; },
  "exec.start": (st, d) => { st.running = d; },
  exec: (st, d) => { st.execs.push(d); st.running = null; },
  verdict: (st, d) => { st.verdicts.push(d); },
  diagnosis: (st, d) => { st.diagnoses.push(d); },
  "edit.proposed": (st, d) => { st.proposals.push(d); },
  experiment: (st, d) => { st.experiments.push(d); },
  "attempt.failed": (st, d) => { st.failures.push(d); },
  graph: (st, d) => { st.graph = d; },
  usage: (st, d) => { st.usage = d; },
  report: (st, d) => { st.report = d; st.graph = d.graph || st.graph; },
  "run.finished": (st, d) => { st.finished = d.status; st.running = null; },
  error: (st, d) => { st.error = d.message; },
};

async function renderRun(id, slot) {
  const meta = await api("/api/runs/" + id);
  const st = newState();
  if (!slot.isConnected) return;            // the user already navigated elsewhere: do not open a stream
  const root = h("div");
  slot.replaceChildren(root);
  let queued = false;
  const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } };
  function draw() {
    const keep = [...root.querySelectorAll(".graph-wrap,.logbox")].map((e) => [e.scrollTop, e.scrollLeft]);
    root.replaceChildren(runView(st, meta, id, schedule));
    root.querySelectorAll(".graph-wrap,.logbox").forEach((e, i) => { if (keep[i]) { e.scrollTop = keep[i][0]; e.scrollLeft = keep[i][1]; } });
  }
  draw();
  const es = new EventSource(`/api/runs/${id}/events${authQ("?")}`);
  for (const kind of EVENT_KINDS) {
    es.addEventListener(kind, (ev) => {
      try { (HANDLERS[kind] || (() => {}))(st, JSON.parse(ev.data)); } catch (e) { console.error(kind, e); }
      if (kind === "report" && st.autoTab) st.tab = "report";
      schedule();
    });
  }
  es.addEventListener("end", () => { es.close(); schedule(); });
  cleanup = () => es.close();
}

function runView(st, meta, id, schedule) {
  const status = st.report ? st.report.status : st.finished || (st.started ? "running" : meta.status);
  const live = !st.finished && !st.report && !["verified", "executes", "already_passing", "partial", "failed", "cancelled", "error", "interrupted"].includes(meta.status);
  const backend = (st.report && st.report.llm_backend) || (st.started && st.started.llm_backend);
  const req = meta.request || {};
  const head = h("div", { class: "run-head" },
    h("h1", { text: repoName(req.repo_url || req.local_path) }), chip(status),
    h("span", { class: "spacer" }),
    live ? h("button", { class: "quiet", onclick: () => api(`/api/runs/${id}/cancel`, { method: "POST" }) }, "Stop") : null,
    h("p", { class: "goal", text: req.goal || "" }));
  const banners = [];
  if (backend && backend !== "nebius") banners.push(h("p", { class: "banner" }, h("strong", {}, "Offline demo. "), `This run used a ${backend} model, not Nemotron. It shows the interface and the pipeline; it says nothing about model quality.`));
  if (st.started && st.started.sandbox && !st.started.sandbox.filesystem_isolated) banners.push(h("p", { class: "banner" }, h("strong", {}, "Development sandbox. "), "It does not isolate the filesystem. Do not run untrusted repositories with it."));
  if (st.error) banners.push(h("p", { class: "banner bad", role: "alert" }, st.error));
  if (st.report && st.report.error) banners.push(h("p", { class: "banner bad", role: "alert" }, st.report.error));

  const tabs = [["graph", "Evidence graph"], ["changes", "Changes"], ["output", "Output"], ["report", "Report"]];
  const body = { graph: () => graphPane(st, schedule), changes: () => changesPane(st, id, schedule), output: () => outputPane(st, schedule), report: () => reportPane(st, id) }[st.tab]();
  return h("div", {},
    head, ...banners, instrument(st),
    h("div", { class: "layout" },
      h("aside", { class: "sheet rail" }, h("section", {}, h("h2", { text: "Progress" }), stepList(st)), experimentList(st)),
      h("section", { class: "sheet" },
        h("div", { class: "tabs", role: "tablist" }, tabs.map(([k, label]) => h("button", { role: "tab", "aria-selected": String(st.tab === k),
          onclick: () => { st.tab = k; st.autoTab = false; schedule(); } }, label,
          k === "output" && st.execs.length ? h("span", { class: "count", text: st.execs.length }) : null,
          k === "changes" && st.experiments.length ? h("span", { class: "count", text: st.experiments.length }) : null))),
        h("div", { class: "pane", role: "tabpanel" }, body))));
}

function stepList(st) {
  return h("ol", { class: "steps" }, STEPS.map(([sid, label]) => {
    const s0 = st.steps[sid]; const state = s0 ? s0.status : "pending";
    return h("li", { class: state }, h("span", { class: "si " + state, "aria-hidden": "true" }),
      h("div", {}, h("div", { class: "lbl" }, label, h("span", { class: "sr", style: "position:absolute;left:-999px", text: ` (${state})` })),
        s0 && s0.detail ? h("div", { class: "det", text: s0.detail }) : null));
  }));
}
function experimentList(st) {
  if (!st.experiments.length && !st.failures.length) return null;
  return h("section", {}, h("h2", { text: "Experiments" }), h("ol", { class: "exps" }, [
    ...st.experiments.map((e) => h("li", {}, h("div", { class: "t" }, `Experiment ${e.n}`, h("span", { class: e.kept ? "kept" : "reverted", style: `color:var(--${e.kept ? "ok" : "bad"})`, text: e.kept ? "kept" : "reverted" })),
      h("div", { class: "d", text: e.hypothesis.statement.slice(0, 110) }), h("div", { class: "d", text: e.reason }))),
    ...st.failures.map((f) => h("li", {}, h("div", { class: "t", text: `Attempt ${f.attempt} stopped` }), h("div", { class: "d", text: `${f.stage}: ${String(f.error).slice(0, 140)}` }))),
  ]));
}

/* ---------------------------------------------------------------- the instrument */
function instrument(st) {
  const m = st.metric;
  const wrap = h("section", { class: "sheet instrument" });
  if (!m || m.expected == null) {
    if (st.analysis) wrap.append(h("header", {}, h("h2", { text: "No documented value to measure against" }), h("span", { text: "ReproFix can only check that the program runs." })));
    else return h("div");
    return wrap;
  }
  const pts = st.verdicts.map((v) => ({ label: v.label, stage: v.verdict.stage, value: v.verdict.metric_value, ok: v.verdict.verified }));
  const nums = pts.filter((p) => p.value != null);
  const vals = nums.map((p) => p.value);
  let lo = Math.min(m.expected - 3 * m.tolerance, ...vals), hi = Math.max(m.expected + 3 * m.tolerance, ...vals);
  if (lo >= 0 && hi <= 1) { lo = 0; hi = 1; } else { const pad = (hi - lo) * 0.05; lo -= pad; hi += pad; }
  const W = 900, H = 176, x0 = 140, x1 = 870, y = 74;
  const xs = (v) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
  const bx = xs(m.expected - m.tolerance), bw = Math.max(xs(m.expected + m.tolerance) - bx, 4);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${m.name}: documented ${m.expected} plus or minus ${m.tolerance}. Runs: ` + (pts.map((p) => `${p.label} ${p.value != null ? fmt(p.value) : STAGE_TEXT[p.stage]}`).join("; ") || "none yet") });
  svg.append(s("line", { class: "ax", x1: x0, x2: x1, y1: y, y2: y }));
  for (let i = 0; i <= 5; i++) {
    const v = lo + ((hi - lo) * i) / 5;
    svg.append(s("line", { class: "ax", x1: xs(v), x2: xs(v), y1: y - 5, y2: y + 5 }), s("text", { class: "ax-text", x: xs(v), y: y + 20, "text-anchor": "middle" }, v.toFixed(2)));
  }
  svg.append(s("rect", { class: "band-rect", x: bx, y: y - 24, width: bw, height: 48 }),
    s("text", { class: "band-label", x: Math.min(Math.max(bx + bw / 2, x0 + 70), x1 - 70), y: y - 32, "text-anchor": "middle" }, `documented ${m.expected} ± ${m.tolerance}`));
  const pathPts = nums.map((p) => `${xs(p.value)},${y}`).join(" ");
  if (nums.length > 1) svg.append(s("polyline", { class: "pt-line", points: pathPts }));
  // Runs that measured the same value share one point (and one label) instead of stacking on top of each other.
  const groups = [];
  nums.forEach((p) => {
    const x = xs(p.value); const g = groups.find((q) => Math.abs(q.x - x) < 8);
    if (g) g.items.push(p); else groups.push({ x, items: [p] });
  });
  groups.forEach((g, i) => {
    const low = i % 2 === 0; const last = g.items[g.items.length - 1];
    svg.append(s("circle", { class: "pt" + (g.items.some((p) => p.ok) ? " ok" : "") + (g.items.includes(pts[pts.length - 1]) ? " latest" : ""), cx: g.x, cy: y, r: 7 }),
      s("text", { class: "pt-text", x: g.x, y: y + (low ? 46 : 70), "text-anchor": "middle" }, fmt(last.value)),
      s("text", { class: "pt-sub", x: g.x, y: y + (low ? 60 : 84), "text-anchor": "middle" }, g.items.map((p) => p.label).join(" · ")));
  });
  const none = pts.filter((p) => p.value == null);
  if (none.length) {
    svg.append(s("text", { class: "pt-sub", x: 12, y: y - 32 }, "did not produce a result"));
    none.slice(0, 4).forEach((p, i) => {          // two columns, two rows; the Report tab lists every run
      const x = 36 + (i % 2) * 70, oy = Math.floor(i / 2) * 52;
      svg.append(s("path", { class: "nores", d: `M${x - 6},${y - 6 + oy} L${x + 6},${y + 6 + oy} M${x + 6},${y - 6 + oy} L${x - 6},${y + 6 + oy}` }),
        s("text", { class: "pt-sub", x, y: y + 24 + oy, "text-anchor": "middle" }, p.label.replace("experiment ", "exp ")),
        s("text", { class: "pt-sub", x, y: y + 37 + oy, "text-anchor": "middle" }, STAGE_TEXT[p.stage] || p.stage));
    });
    if (none.length > 4) svg.append(s("text", { class: "pt-sub", x: 12, y: y + 112 }, `+${none.length - 4} more (see Report)`));
  }
  wrap.append(h("header", {}, h("h2", { text: `Measured ${m.name} against the documented value` }),
    h("span", { text: st.metricSource === "user" ? "documented value entered by you" : st.metricSource })),
    svg, h("p", { class: "note", text: "Each point is a real run in the sandbox; runs with the same result share a point. The shaded band is the tolerance: a run reproduces the result only when its point lands inside it." }));
  return wrap;
}

/* ---------------------------------------------------------------- evidence graph */
const KIND = { symptom: "Symptom", hypothesis: "Hypothesis", evidence: "Evidence", experiment: "Experiment", result: "Result" };
function wrapText(text, max, lines) {
  const words = String(text).split(/\s+/); const out = []; let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > max && cur) { out.push(cur); cur = w; } else cur = (cur + " " + w).trim();
  }
  if (cur) out.push(cur);
  if (out.length > lines) { out.length = lines; out[lines - 1] = out[lines - 1].replace(/.{0,2}$/, "…"); }
  return out;
}
function layoutGraph(g) {
  const W = 230, GAP = 26, VGAP = 56;
  const ids = g.nodes.map((n) => n.id);
  const depth = Object.fromEntries(ids.map((i) => [i, 0]));
  for (let pass = 0; pass < ids.length; pass++) {
    let changed = false;
    for (const e of g.edges) if (depth[e.to] < depth[e.from] + 1) { depth[e.to] = depth[e.from] + 1; changed = true; }
    if (!changed) break;
  }
  const rows = {};
  g.nodes.forEach((n) => { (rows[depth[n.id]] ||= []).push(n); });
  const pos = {};
  let y = 16, maxW = 0;
  const levels = Object.keys(rows).map(Number).sort((a, b) => a - b);
  for (const d of levels) {
    const row = rows[d].sort((a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1)));
    const items = row.map((n) => { const lines = wrapText(n.label, 30, 4); return { n, lines, h: 26 + lines.length * 15 }; });
    const rh = Math.max(...items.map((i) => i.h));
    items.forEach((it, i) => { pos[it.n.id] = { x: 16 + i * (W + GAP), y, w: W, h: it.h, lines: it.lines, n: it.n }; });
    maxW = Math.max(maxW, 16 + items.length * (W + GAP));
    y += rh + VGAP;
  }
  // center each level horizontally
  for (const d of levels) {
    const row = rows[d]; const rowW = row.length * (W + GAP) - GAP;
    const off = (maxW - 16 - rowW) / 2;
    row.forEach((n) => { pos[n.id].x += off - 0; });
  }
  const edges = g.edges.filter((e) => pos[e.from] && pos[e.to]).map((e) => {
    const a = pos[e.from], b = pos[e.to]; const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y;
    const my = (y1 + y2) / 2;
    return { rel: e.relation, d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}` };
  });
  return { pos, edges, width: Math.max(maxW, 320), height: y };
}
const STATUS_WORD = { confirmed: "confirmed by experiment", verified: "verified", rejected: "rejected by experiment", untested: "not tested",
  unverified: "could not be verified", info: "", open: "open" };
function graphPane(st, schedule) {
  const g = st.graph;
  if (!g.nodes.length) return h("p", { class: "empty", text: "The investigation graph builds as soon as the baseline run finishes." });
  const L = layoutGraph(g);
  const svg = s("svg", { width: L.width, height: L.height + 8, viewBox: `0 0 ${L.width} ${L.height + 8}`, role: "group", "aria-label": "Evidence graph" });
  L.edges.forEach((e) => svg.append(s("path", { class: "g-edge", d: e.d }, s("title", {}, e.rel.replace(/_/g, " ")))));
  Object.values(L.pos).forEach((p) => {
    const n = p.n; const kind = KIND[n.type] + (n.type === "evidence" && n.data.kind ? ` (${n.data.kind})` : "");
    const node = s("g", { class: `g-node ${n.status}`, tabindex: 0, role: "button", "aria-pressed": String(st.sel === n.id),
      "aria-label": `${kind}: ${n.label}. ${STATUS_WORD[n.status] || ""}`,
      onclick: () => { st.sel = st.sel === n.id ? null : n.id; schedule(); },
      onkeydown: (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); st.sel = st.sel === n.id ? null : n.id; schedule(); } } },
      s("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: 3 }),
      s("text", { class: "kind", x: p.x + 10, y: p.y + 16 }, kind + (STATUS_WORD[n.status] ? ` — ${STATUS_WORD[n.status]}` : "")));
    p.lines.forEach((ln, i) => node.append(s("text", { x: p.x + 10, y: p.y + 33 + i * 15 }, ln)));
    svg.append(node);
  });
  const sel = st.sel && g.nodes.find((n) => n.id === st.sel);
  return h("div", {},
    h("div", { class: "legend" }, ...[["confirmed", "confirmed or verified"], ["rejected", "rejected by experiment"], ["untested", "proposed, not tested"], ["unverified", "evidence not found in the files"]].map(([c, t]) => h("span", {}, h("i", { class: c }), t))),
    h("div", { class: "graph-wrap", style: "margin-top:14px" }, svg),
    sel ? nodeDetail(sel) : h("p", { class: "empty", style: "margin-top:12px", text: "Select a node to see its evidence. Statuses come from experiments and file checks, not from the model's own confidence." }));
}
function nodeDetail(n) {
  const d = n.data || {}; const kv = (k, v) => (v == null || v === "" || (Array.isArray(v) && !v.length) ? null : h("div", { class: "kv" }, h("strong", { text: k + ": " }), Array.isArray(v) ? v.join(", ") : String(v)));
  const url = safeUrl(d.url);
  return h("div", { class: "detail", style: "margin-top:14px" },
    h("h3", { text: `${KIND[n.type]}${STATUS_WORD[n.status] ? " — " + STATUS_WORD[n.status] : ""}` }), h("p", { text: n.label }),
    kv("Category", d.category), kv("Model-stated confidence (not verified)", d.model_confidence), kv("Files", d.files), kv("How to test", d.test),
    kv("Outcome", d.result || d.why), kv("Stage", d.stage), kv("File", d.file ? d.file + (d.line ? ":" + d.line : "") : null),
    d.quote ? h("pre", { text: d.quote }) : null, d.note ? kv("Note", d.note) : null,
    url ? h("div", { class: "kv" }, h("strong", { text: "Source: " }), h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url })) : null,
    d.reasons && d.reasons.length ? kv("Details", d.reasons.join("; ")) : null,
    d.edits ? h("pre", { text: d.edits.map((e) => (e.create != null ? `+ new file ${e.path}` : `${e.path}\n- ${e.search}\n+ ${e.replace}`)).join("\n\n") }) : null);
}

/* ---------------------------------------------------------------- changes (diffs) */
function parseDiff(text) {
  const files = []; let cur = null, a = 0, b = 0;
  for (const raw of text.split("\n")) {
    if (raw.startsWith("diff --git ")) { cur = { path: raw.split(" b/")[1] || raw, lines: [], add: 0, del: 0 }; files.push(cur); continue; }
    if (!cur || /^(new file|deleted file|index |--- |\+\+\+ )/.test(raw)) continue;
    const hm = raw.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)/);
    if (hm) { a = +hm[1]; b = +hm[2]; cur.lines.push({ t: "hunk", c: raw }); continue; }
    if (raw.startsWith("\\")) continue;
    if (raw.startsWith("+")) { cur.lines.push({ t: "add", b: b++, c: raw.slice(1) }); cur.add++; }
    else if (raw.startsWith("-")) { cur.lines.push({ t: "del", a: a++, c: raw.slice(1) }); cur.del++; }
    else if (raw.length || cur.lines.length) cur.lines.push({ t: "ctx", a: a++, b: b++, c: raw.slice(1) });
  }
  return files;
}
function diffView(text) {
  return parseDiff(text).map((f) => h("div", { class: "diff-file" },
    h("header", {}, h("strong", { class: "mono", text: f.path }), h("span", { class: "stat" }, h("span", { class: "a", text: "+" + f.add }), " ", h("span", { class: "d", text: "−" + f.del }))),
    h("div", { class: "diff-body" }, f.lines.map((l) => h("div", { class: "dl " + l.t },
      h("span", { class: "n", text: l.a ?? "" }), h("span", { class: "n", text: l.b ?? "" }), h("span", { class: "c", text: (l.t === "add" ? "+" : l.t === "del" ? "-" : l.t === "ctx" ? " " : "") + l.c }))))));
}
function changesPane(st, id, schedule) {
  const kids = [];
  if (st.report && st.report.diff) {
    kids.push(h("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap" }, h("h2", { text: "Final patch" }),
      h("span", { class: "chip", text: `${st.report.files_modified.length} file${st.report.files_modified.length === 1 ? "" : "s"} changed` }),
      h("button", { onclick: () => downloadPatch(id) }, "Download patch"),
      h("button", { onclick: () => navigator.clipboard && navigator.clipboard.writeText(st.report.diff) }, "Copy patch")),
      h("p", { class: "empty", text: "Apply with git apply or patch -p1 from the repository root. The report confirms it applies cleanly to the original." }),
      ...diffView(st.report.diff), prPanel(st, id, schedule));
  }
  if (st.experiments.length) {
    kids.push(h("h2", { text: "Every experiment" }));
    st.experiments.forEach((e) => kids.push(h("div", {}, h("p", {}, h("strong", { text: `Experiment ${e.n}: ` }), e.hypothesis.statement,
      " ", h("span", { class: "chip " + (e.kept ? "ok" : "bad"), text: e.kept ? "kept" : "reverted" })),
      h("p", { class: "empty", text: e.reason }), ...(e.diff ? diffView(e.diff) : []))));
  }
  if (!kids.length) kids.push(h("p", { class: "empty", text: "No code has been changed yet." }));
  return h("div", {}, ...kids);
}
/* ---------------------------------------------------------------- pull request */
const PR_URL = /^https:\/\/github\.com\/[\w.\-]+\/[\w.\-]+\/pull\/\d+$/;
function prLink(url) {
  return PR_URL.test(url || "") ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url }) : h("span", { class: "mono", text: String(url || "") });
}
/* A GitHub token typed into this page goes to the server in the request body: only allow that over HTTPS or loopback. */
function secureForToken() {
  const { protocol, hostname } = window.location;
  return protocol === "https:" || ["localhost", "127.0.0.1", "[::1]", "::1"].includes(hostname);
}
function prPanel(st, id, schedule) {
  const pr = (st.pr = st.pr || { state: "idle", data: null, error: null, title: "", body: "", token: "", confirmed: false, busy: false, result: null });
  const sec = h("section", { class: "pr" }, h("h2", { text: "Open a pull request" }));
  if (pr.state === "idle") {
    pr.state = "loading";
    api(`/api/runs/${id}/pull-request`).then((d) => { pr.data = d; pr.title = d.title || ""; pr.body = d.body || ""; })
      .catch((e) => { pr.error = e.message; }).finally(() => { pr.state = "ready"; schedule(); });
  }
  if (pr.state === "loading") return sec.append(h("p", { class: "empty", text: "Checking whether a pull request is possible…" })), sec;
  const done = pr.result || (pr.data && pr.data.existing);
  if (done) return sec.append(h("p", {}, h("span", { class: "chip ok", text: "opened" }), " ", prLink(done.url)),
    h("p", { class: "empty", text: `Branch ${done.branch} in ${done.head_repo}${done.mode === "fork" ? " (a fork, because the token cannot push to the original)" : ""}. One pull request is opened per run.` })), sec;
  if (!pr.data) return sec.append(h("p", { class: "form-error", role: "alert", text: pr.error || "Could not check." })), sec;
  if (!pr.data.available) return sec.append(h("p", { class: "empty", text: "Not available: " + pr.data.reason })), sec;
  const d = pr.data;
  const insecure = !secureForToken();
  const btn = h("button", { class: "primary", type: "button" }, "Open pull request");
  const sync = () => { btn.disabled = insecure || pr.busy || !pr.token.trim() || !pr.confirmed; };
  btn.addEventListener("click", async () => {
    pr.busy = true; pr.error = null; schedule();
    try {
      pr.result = await api(`/api/runs/${id}/pull-request`, { method: "POST",
        body: JSON.stringify({ token: pr.token, confirm: true, title: pr.title, body: pr.body }) });
      pr.token = "";
    } catch (e) { pr.error = e.message; }
    pr.busy = false; schedule();
  });
  sync();
  sec.append(
    h("p", { class: "empty", text: `Opens a pull request against ${d.target}, based on commit ${d.base_commit.slice(0, 10)}, with ${d.files.length} changed file${d.files.length === 1 ? "" : "s"}: ` + d.files.map((f) => `${f.path} (${f.status})`).join(", ") + ". The files are rebuilt from the patch above, not from the run workspace." }),
    insecure ? h("p", { class: "form-error", role: "alert", text: "This page is not served over HTTPS, so a token typed here would cross the network unencrypted. Open the site over https:// (or from localhost) to use this button." }) : null,
    h("div", { class: "form" },
      h("label", { class: "field" }, "Title", h("input", { value: pr.title, maxlength: 120, oninput: (e) => { pr.title = e.target.value; } })),
      h("label", { class: "field" }, "Description", h("textarea", { rows: 10, oninput: (e) => { pr.body = e.target.value; } }, pr.body),
        h("small", { text: "Drafted from the report's computed checks. The diagnosis lines are the model's words and are labelled that way." })),
      h("label", { class: "field" }, "GitHub token", h("input", { type: "password", autocomplete: "off", spellcheck: "false", value: pr.token, disabled: insecure || null,
        oninput: (e) => { pr.token = e.target.value; sync(); } }),
        h("small", { text: "Used for this one request and not stored. A fine-grained token needs Contents and Pull requests write access on the repository; to propose changes to someone else's repository a classic token with the public_repo scope is needed, because ReproFix then forks it." })),
      h("label", { class: "check" }, h("input", { type: "checkbox", checked: pr.confirmed || null, onchange: (e) => { pr.confirmed = e.target.checked; sync(); } }),
        h("span", { text: "I understand this creates a branch (in a fork of the repository if my token cannot push) and opens a public pull request on GitHub as me." })),
      h("div", { class: "actions" }, btn, pr.busy ? h("span", { class: "empty", text: "Working… creating a fork can take up to a couple of minutes." }) : null),
      pr.error ? h("p", { class: "form-error", role: "alert", text: pr.error }) : null));
  return sec;
}
async function downloadPatch(id) {
  const res = await fetch(`/api/runs/${id}/patch`, { headers: token() ? { Authorization: "Bearer " + token() } : {} });
  const url = URL.createObjectURL(await res.blob());
  const a = h("a", { href: url, download: `reprofix-${id}.patch` }); document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(url);
}

/* ---------------------------------------------------------------- output */
function outputPane(st, schedule) {
  if (!st.execs.length && !st.running) return h("p", { class: "empty", text: "Command output appears here as soon as the first run starts." });
  const sel = st.execs.find((e) => e.id === st.execSel) || st.execs[st.execs.length - 1];
  const list = h("ul", {}, st.execs.map((e) => h("li", {}, h("button", { "aria-current": String(sel && sel.id === e.id),
    onclick: () => { st.execSel = e.id; schedule(); } }, `${e.phase}`, h("span", { class: "sub", text: `${e.argv.slice(-2).join(" ").slice(-34)}  ·  exit ${e.exit_code ?? "timeout"}` })))),
    st.running ? h("li", {}, h("button", { disabled: true }, st.running.phase, h("span", { class: "sub", text: "running…" }))) : null);
  const box = !sel ? h("p", { class: "empty", text: "Waiting for output…" }) : h("div", { class: "logbox" },
    h("h3", { class: "mono", text: sel.argv.join(" ") }),
    h("div", { class: "meta-row" }, h("span", { text: `exit code ${sel.exit_code ?? "none (timed out)"}` }), h("span", { text: `${sel.duration_s}s` }),
      h("span", { text: sel.network ? "network on (package install)" : "no network" }), sel.truncated ? h("span", { text: "output truncated" }) : null,
      sel.metric_value != null ? h("span", { text: `metric ${fmt(sel.metric_value)}` }) : null),
    sel.stdout ? h("pre", { text: sel.stdout }) : null, sel.stderr ? h("pre", { class: sel.exit_code ? "err" : "", text: sel.stderr }) : null,
    !sel.stdout && !sel.stderr ? h("pre", { text: "(no output)" }) : null);
  return h("div", { class: "out" }, list, box);
}

/* ---------------------------------------------------------------- report */
function reportPane(st, id) {
  const r = st.report;
  if (!r) return h("div", {}, h("p", { class: "empty", text: "The report is written when the run finishes." }), claimsRead(st.claims), issuesSection(st.issues.flatMap((x) => x.results || [])), st.usage ? usageTable(st.usage) : null);
  const kids = [h("p", { class: "verdict-line", text: r.status_text })];
  if (r.stop_reason) kids.push(h("p", { class: "empty", text: "Stopped because: " + r.stop_reason }));
  r.caveats.forEach((c) => kids.push(h("p", { class: "banner", text: c })));
  kids.push(h("section", {}, h("h2", { text: "Verification" }), h("ul", { class: "checks" }, r.verification.map((c) => {
    const k = c.passed === true ? ["✓", "ok"] : c.passed === false ? ["✗", "bad"] : ["–", "na"];
    return h("li", {}, h("span", { class: "mk " + k[1], text: k[0], "aria-label": c.passed === true ? "passed" : c.passed === false ? "failed" : "not applicable" }),
      h("div", {}, h("div", { text: c.check }), c.detail ? h("div", { class: "det", text: c.detail }) : null));
  }))));
  const cs = claimsSection(r.claims);
  if (cs) kids.push(cs);
  const hp = hardwareProof(r.hardware);
  if (hp) kids.push(hp);
  const ks = issuesSection(r.known_issues ? r.known_issues.results : []);
  if (ks) kids.push(ks);
  if (r.issues.length) kids.push(h("section", {}, h("h2", { text: `Issues found (${r.issues.length})` }), h("div", { style: "display:grid;gap:10px;margin-top:8px" },
    r.issues.map((i) => h("div", { class: "issue" }, h("div", { class: "cat", text: `${i.category} — ${i.files.join(", ")}` }), h("div", { text: i.statement }))))));
  if (r.results.progression.length || r.results.baseline) kids.push(h("section", {}, h("h2", { text: "Results by run" }), h("table", { class: "t" },
    h("thead", {}, h("tr", {}, h("th", { text: "Run" }), h("th", { text: "State" }), h("th", { class: "num", text: r.metric ? r.metric.name : "metric" }), h("th", { text: "Outcome" }))),
    h("tbody", {}, [
      r.results.baseline ? h("tr", {}, h("td", { text: "baseline" }), h("td", { text: STAGE_TEXT[r.results.baseline.stage] }), h("td", { class: "num", text: fmt(r.results.baseline.metric_value) }), h("td", { text: "starting point" })) : null,
      ...r.results.progression.map((p) => h("tr", {}, h("td", { text: "experiment " + p.experiment }), h("td", { text: STAGE_TEXT[p.stage] }), h("td", { class: "num", text: fmt(p.metric_value) }), h("td", {}, h("span", { class: p.kept ? "kept" : "reverted", style: `color:var(--${p.kept ? "ok" : "bad"});font-weight:600`, text: p.kept ? "kept" : "reverted" }), " " + p.reason))),
      r.results.final && r.results.baseline ? h("tr", {}, h("td", { text: "final re-run" }), h("td", { text: STAGE_TEXT[r.results.final.stage] }), h("td", { class: "num", text: fmt(r.results.final.metric_value) }), h("td", { text: r.metric && r.metric.expected != null ? `documented ${r.metric.expected} ± ${r.metric.tolerance}` : "" })) : null]))));
  const sandbox = r.sandbox, env = r.environment;
  kids.push(h("div", { class: "cols" },
    h("section", {}, h("h2", { text: "Model usage" }), usageTable(r.usage), h("p", { class: "empty", style: "margin-top:8px", text: `Routing: ${r.routing.mode}${r.routing.unavailable_tiers.length ? "; unavailable tiers: " + r.routing.unavailable_tiers.join(", ") : ""}. Backend: ${r.llm_backend}.` })),
    h("section", {}, h("h2", { text: "Sandbox and environment" }), h("ul", { class: "caveats" },
      h("li", { text: `${sandbox.kind}; filesystem ${sandbox.filesystem_isolated ? "isolated" : "NOT isolated"}` }),
      h("li", { text: `Network during install: ${sandbox.network_during_install ? "on" : "off"}; while running the repository: ${sandbox.network_during_run ? "ON" : "off"}` }),
      h("li", { text: `Framework: ${env.framework || "unknown"}; command: ${env.run_command || "unknown"} (${env.run_command_source || "?"})` }),
      r.hardware && r.hardware.summary ? h("li", { text: `Hardware: ${r.hardware.summary}${r.hardware.probed_in ? " (nvidia-smi " + r.hardware.probed_in + ")" : ""}` }) : null,
      r.metric ? h("li", { text: `Metric: ${r.metric.name} ≈ ${r.metric.expected} ± ${r.metric.tolerance} (${r.metric.source})` }) : null,
      h("li", { text: `Tavily: ${r.tavily.enabled ? r.tavily.calls + " search call(s)" : "not configured"}` })))));
  const web = st.diagnoses.flatMap((d) => d.web_results || []);
  if (web.length) kids.push(h("section", {}, h("h2", { text: "External evidence" }), h("ul", { class: "caveats" }, web.slice(0, 8).map((w) => {
    const u = safeUrl(w.url); return h("li", {}, u ? h("a", { href: u, target: "_blank", rel: "noopener noreferrer", text: w.title || u }) : w.title, h("div", { class: "empty", text: w.snippet.slice(0, 200) }));
  }))));
  if (r.review) kids.push(h("section", {}, h("h2", { text: "Patch review (model opinion, advisory)" }), h("p", { text: r.review.summary }),
    r.review.concerns.length ? h("ul", { class: "caveats" }, r.review.concerns.map((c) => h("li", { text: c }))) : null));
  return h("div", { class: "rep" }, ...kids);
}
const VERDICT = { reproduced: ["reproduced", "ok"], not_reproduced: ["not reproduced", "bad"], not_measured: ["not measured", "warn"] };
const claimNum = (v, unit) => (v == null ? "—" : unit === "percent" ? `${v}%` : String(v));
function verdictChip(v) { const [t, c] = VERDICT[v] || [v, ""]; return h("span", { class: "chip " + c, text: t }); }
function hardwareProof(hw) {      // only when a GPU was actually seen: the verbatim nvidia-smi output is the proof
  if (!hw || hw.status !== "detected") return null;
  return h("section", {}, h("h2", { text: "Hardware record" }),
    h("p", { text: hw.summary }),
    h("details", { class: "claimbox" }, h("summary", { text: "nvidia-smi output, verbatim" }), h("pre", { class: "raw", text: hw.raw })),
    h("p", { class: "empty", text: hw.note }));
}
function issuesSection(list) {
  const seen = new Set();
  const items = (list || []).filter((x) => !seen.has(x.url) && seen.add(x.url));
  if (!items.length) return null;
  return h("section", {}, h("h2", { text: `Possibly related issues (${items.length}, not verified)` }),
    h("ul", { class: "caveats issues" }, items.map((x) => {
      const u = safeUrl(x.url);
      return h("li", {}, u ? h("a", { href: u, target: "_blank", rel: "noopener noreferrer", text: x.title || u }) : h("span", { text: x.title }),
        h("div", { class: "empty", text: [x.source === "github-issues" ? "this repository's issues" : "web", x.state, x.comments != null ? x.comments + " comments" : null,
          x.updated, "matched: " + (x.matched_terms || []).join(", ")].filter(Boolean).join(" · ") }),
        x.snippet ? h("div", { class: "empty", text: x.snippet.slice(0, 200) }) : null);
    })),
    h("p", { class: "empty", text: "Found by searching for words from the failure. ReproFix does not check that any of these describes it; the model saw them as untrusted text." }));
}
function claimsRead(c) {          // shown while the run is still going: what the patterns read from the pasted text
  if (!c || !c.claims || !c.claims.length) return null;
  return h("section", {}, h("h2", { text: "Claims read from the paper text" }), h("ul", { class: "caveats" }, c.claims.map((x) =>
    h("li", { text: `${x.id}${x.id === c.headline ? " (repair target)" : ""}: ${x.metric}${x.qualifier ? " (" + x.qualifier + ")" : ""} = ${claimNum(x.value, x.unit)}   “${(x.quote || "").slice(0, 140)}”` }))),
    h("p", { class: "empty", text: "Read by fixed patterns, not understood by a model. Stop the run if one of these is wrong." }));
}
function claimsSection(card) {
  if (!card) return null;
  const skipped = (card.skipped || []).map((t) => h("p", { class: "empty", text: "Note: " + t }));
  if (!card.checked) {
    if (!skipped.length && !/no claim could be read/.test(card.reason || "")) return null;
    return h("section", {}, h("h2", { text: "Paper claims" }), h("p", { class: "empty", text: "Not checked: " + card.reason }), ...skipped);
  }
  const s = card.summary;
  const tolText = (i) => `±${i.tolerance}${i.unit === "percent" ? " pt" : ""}`;
  const cell = (r, unit, label) => h("td", { "data-label": label }, h("div", {}, h("span", { class: "num", text: claimNum(r.measured, unit) + " " }), verdictChip(r.verdict)),
    r.verdict === "not_measured" ? h("div", { class: "det", text: r.reason }) : h("div", { class: "det", text: `${r.output} (${r.exec})` }));
  return h("section", {}, h("h2", { text: "Paper claims: claimed vs measured" }),
    h("p", { class: "verdict-line", text: `${s.reproduced_after} of ${s.total} claim${s.total === 1 ? "" : "s"} reproduced after the run (${s.reproduced_before} before). ${s.fixed} fixed, ${s.regressed} regressed, ${s.not_measured} not measured.` }),
    h("table", { class: "t claims" },
      h("thead", {}, h("tr", {}, h("th", { text: "Claim" }), h("th", { class: "num", text: "Paper says" }), h("th", { text: "Before (as cloned)" }), h("th", { text: "After" }))),
      h("tbody", {}, card.items.map((i) => h("tr", {},
        h("td", {}, h("div", {}, h("strong", { text: `${i.headline ? "★ " : ""}${i.id} ${i.metric}` }), i.qualifier ? ` (${i.qualifier})` : ""),
          i.quote ? h("div", { class: "det", text: "“" + i.quote.slice(0, 160) + "”" }) : null,
          i.note ? h("div", { class: "det warn", text: i.note }) : null),
        h("td", { class: "num", "data-label": "Paper says", text: `${claimNum(i.claimed, i.unit)} ${tolText(i)}` }),
        cell(i.baseline, i.unit, "Before (as cloned)"), cell(i.final, i.unit, "After"))))),
    h("p", { class: "empty", text: (card.items.some((i) => i.headline) ? "★ is the claim the repair loop optimised. " : "") + "Each measurement is a line the program printed in the run shown in brackets (see the Output tab)." }),
    card.measured_on ? h("p", { class: "empty", text: "Hardware the sandbox could see: " + card.measured_on + "." }) : null,
    ...skipped);
}
function usageTable(u) {
  const rows = Object.entries(u.by_tier);
  if (!rows.length) return h("p", { class: "empty", text: "No model calls yet." });
  return h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", { text: "Tier" }), h("th", { class: "num", text: "Calls" }), h("th", { class: "num", text: "Tokens in" }), h("th", { class: "num", text: "Tokens out" }), h("th", { class: "num", text: "Cost" }))),
    h("tbody", {}, rows.map(([t, v]) => h("tr", {}, h("td", { text: t }), h("td", { class: "num", text: v.calls }), h("td", { class: "num", text: v.prompt_tokens.toLocaleString() }),
      h("td", { class: "num", text: v.completion_tokens.toLocaleString() }), h("td", { class: "num", text: v.priced ? money(v.cost_usd) : "n/a" })))));
}

/* ---------------------------------------------------------------- boot */
const tokenInput = document.getElementById("token");
tokenInput.value = token();
tokenInput.addEventListener("change", () => { sessionStorage.setItem("reprofix_token", tokenInput.value.trim()); route(); });
if (typeof module !== "undefined") module.exports = { layoutGraph, parseDiff, wrapText };
route();
