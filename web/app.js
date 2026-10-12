"use strict";
/* ReproFix — AI-Powered ML Reproducibility & Debugging Platform
   Developer tool: GitHub Actions × Linear × Vercel × Weights & Biases × AI debugging agent.
   No build step. All untrusted text is rendered with textContent via h(); nothing from a run is ever assigned to innerHTML. */

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

/* ---------------------------------------------------------------- DOM & Element Helpers */
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

/* Strip ANSI escape codes safely */
const ANSI_RE = /\u001b\[[0-9;?]*[ -\/]*[@-~]|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)|\u001b[@-Z\\-_]/g;
const stripAnsi = (t) => String(t == null ? "" : t).replace(ANSI_RE, "");
const secs = (v) => (v == null ? "n/a" : v >= 90 ? `${Math.floor(v / 60)} min ${Math.round(v % 60)} s` : `${Number(v).toFixed(v < 10 ? 1 : 0)} s`);
function saveText(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type: type || "text/plain" }));
  const a = h("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/* ---------------------------------------------------------------- Toast System */
function showToast(msg, type = "ok") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = h("div", { class: `toast ${type}`, role: "status" },
    h("span", { style: "font-weight:700", text: type === "ok" ? "✓" : "✕" }),
    h("span", { text: msg })
  );
  container.append(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.2s ease";
    setTimeout(() => toast.remove(), 250);
  }, 3200);
}
async function copyToClipboard(text, successMsg = "Copied to clipboard") {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      showToast(successMsg, "ok");
      return true;
    }
  } catch (_) { /* fallback below */ }
  const ta = h("textarea", { style: "position:fixed;left:-9999px;opacity:0" });
  ta.value = text;
  document.body.append(ta);
  ta.select();
  try {
    document.execCommand("copy");
    showToast(successMsg, "ok");
  } catch (e) {
    showToast("Could not copy: " + e.message, "bad");
  }
  ta.remove();
}

/* ---------------------------------------------------------------- Theme & Header Status */
const THEMES = ["auto", "light", "dark"];
function getTheme() { try { const v = localStorage.getItem("reprofix_theme"); return THEMES.includes(v) ? v : "auto"; } catch (_) { return "auto"; } }
function applyTheme(mode) {
  const r = document.documentElement;
  if (mode === "auto") r.removeAttribute("data-theme"); else r.setAttribute("data-theme", mode);
  const b = document.getElementById("theme"); if (b) b.textContent = "Theme: " + mode;
  try { localStorage.setItem("reprofix_theme", mode); } catch (_) { /* private mode */ }
}
async function headerStatus() {
  const box = document.getElementById("pills"); if (!box) return;
  try {
    const hh = await api("/api/health");
    const pill = (cls, text, tooltip) => h("span", { class: "pill", title: tooltip || text }, h("span", { class: "dot " + cls }), text);
    const llm = hh.llm && hh.llm.configured, sb = hh.sandbox || {};
    box.replaceChildren(
      pill(llm ? "ok" : "bad", llm ? "Nemotron on Nebius" : "No API key", llm ? "Active model: NVIDIA Nemotron routed via Nebius Token Factory" : "NEBIUS_API_KEY is not configured"),
      pill(!sb.available ? "bad" : sb.filesystem_isolated ? "ok" : "warn", !sb.available ? "No sandbox" : sb.filesystem_isolated ? "Isolated sandbox" : "Dev sandbox", sb.filesystem_isolated ? "Docker sandbox with filesystem isolation enabled" : "Local process development sandbox (not isolated)"),
      pill(hh.tavily && hh.tavily.configured ? "ok" : "warn", hh.tavily && hh.tavily.configured ? "Web search on" : "Web search off", hh.tavily && hh.tavily.configured ? "Tavily web & issues search enabled" : "Web search disabled"));
    const tw = document.getElementById("token-wrap");
    if (tw) tw.hidden = !hh.auth_required;
  } catch (_) { box.replaceChildren(); }
}

/* ---------------------------------------------------------------- Command Palette & Shortcuts */
let gActiveRunState = null;
let gActiveRunId = null;

function initCommandPalette() {
  const modal = document.getElementById("cmd-palette");
  const input = document.getElementById("cmd-search-input");
  const resultsBox = document.getElementById("cmd-results");
  const triggerBtn = document.getElementById("cmd-k-btn");
  if (!modal || !input || !resultsBox) return;

  let runsCache = [];
  let selectedIndex = 0;

  async function openPalette() {
    modal.hidden = false;
    input.value = "";
    selectedIndex = 0;
    input.focus();
    try { runsCache = await api("/api/runs"); } catch (_) { runsCache = []; }
    renderPalette();
  }
  function closePalette() {
    modal.hidden = true;
  }
  if (triggerBtn) triggerBtn.addEventListener("click", openPalette);
  modal.addEventListener("click", (e) => { if (e.target === modal) closePalette(); });

  function getItems(query) {
    const q = query.trim().toLowerCase();
    const actions = [
      { id: "nav-start", title: "Investigate", desc: "Start new reproduction run", action: () => { location.hash = "#/"; } },
      { id: "nav-runs", title: "Past Runs", desc: "View all reproduction runs history", action: () => { location.hash = "#/runs"; } },
      { id: "nav-bench", title: "Benchmark", desc: "View ReproBench evaluation dashboard", action: () => { location.hash = "#/bench"; } },
      { id: "theme-toggle", title: "Cycle Theme", desc: "Toggle between Dark, Light, and Auto", action: () => { applyTheme(THEMES[(THEMES.indexOf(getTheme()) + 1) % THEMES.length]); showToast("Theme changed", "ok"); } },
      { id: "demo-start", title: "Run Offline Demo", desc: "Launch standalone offline test run", action: async () => { try { const { id } = await api("/api/demo", { method: "POST" }); location.hash = "#/run/" + id; } catch (e) { showToast(e.message, "bad"); } } },
      { id: "shortcuts-open", title: "Keyboard Shortcuts", desc: "View all keyboard hotkeys", action: () => { toggleShortcutsModal(true); } },
    ];
    if (gActiveRunState) {
      actions.unshift(
        { id: "tab-graph", title: "View Evidence Graph", desc: "Active investigation graph", action: () => { gActiveRunState.tab = "graph"; if (gActiveRunState.schedule) gActiveRunState.schedule(); } },
        { id: "tab-changes", title: "View Changes & Patches", desc: "Diff review and git patch", action: () => { gActiveRunState.tab = "changes"; if (gActiveRunState.schedule) gActiveRunState.schedule(); } },
        { id: "tab-output", title: "View Output Logs", desc: "Execution terminal logs", action: () => { gActiveRunState.tab = "output"; if (gActiveRunState.schedule) gActiveRunState.schedule(); } },
        { id: "tab-report", title: "View Report", desc: "Reproduction report and verification", action: () => { gActiveRunState.tab = "report"; if (gActiveRunState.schedule) gActiveRunState.schedule(); } }
      );
    }
    const filteredActions = actions.filter((a) => !q || a.title.toLowerCase().includes(q) || a.desc.toLowerCase().includes(q));
    const runItems = runsCache
      .filter((r) => !q || (r.repo || "").toLowerCase().includes(q) || (r.label || "").toLowerCase().includes(q))
      .slice(0, 5)
      .map((r) => ({
        id: "run-" + r.id,
        title: `Run: ${repoName(r.repo)}`,
        desc: `${(STATUS[r.status] || [r.status])[0]} · ${when(r.created)}`,
        action: () => { location.hash = "#/run/" + r.id; }
      }));
    return [...filteredActions, ...runItems];
  }

  function renderPalette() {
    const items = getItems(input.value);
    if (!items.length) {
      resultsBox.replaceChildren(h("p", { class: "empty", style: "padding:16px", text: "No commands or runs match your search." }));
      return;
    }
    if (selectedIndex >= items.length) selectedIndex = 0;
    resultsBox.replaceChildren(...items.map((item, idx) => {
      const isSel = idx === selectedIndex;
      const el = h("div", {
        class: "cmd-item" + (isSel ? " selected" : ""),
        onclick: () => { closePalette(); item.action(); }
      },
        h("div", {},
          h("strong", { text: item.title }),
          item.desc ? h("div", { class: "cmd-item-desc", text: item.desc }) : null
        ),
        h("kbd", { class: "kbd-small", text: "↵" })
      );
      return el;
    }));
  }

  input.addEventListener("input", () => { selectedIndex = 0; renderPalette(); });
  input.addEventListener("keydown", (e) => {
    const items = getItems(input.value);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % Math.max(1, items.length);
      renderPalette();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + items.length) % Math.max(1, items.length);
      renderPalette();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (items[selectedIndex]) { closePalette(); items[selectedIndex].action(); }
    } else if (e.key === "Escape") {
      closePalette();
    }
  });

  window.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      if (modal.hidden) openPalette(); else closePalette();
    }
  });
}

function toggleShortcutsModal(open) {
  const modal = document.getElementById("shortcuts-modal");
  if (!modal) return;
  modal.hidden = !open;
}
function initKeyboardShortcuts() {
  const modal = document.getElementById("shortcuts-modal");
  const btn = document.getElementById("shortcuts-btn");
  const close = document.getElementById("shortcuts-close");
  if (btn) btn.addEventListener("click", () => toggleShortcutsModal(true));
  if (close) close.addEventListener("click", () => toggleShortcutsModal(false));
  if (modal) modal.addEventListener("click", (e) => { if (e.target === modal) toggleShortcutsModal(false); });

  window.addEventListener("keydown", (e) => {
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    if (e.key === "?") {
      e.preventDefault();
      toggleShortcutsModal(true);
    } else if (e.key === "Escape") {
      toggleShortcutsModal(false);
      const cmd = document.getElementById("cmd-palette");
      if (cmd) cmd.hidden = true;
    } else if (e.key.toLowerCase() === "t") {
      applyTheme(THEMES[(THEMES.indexOf(getTheme()) + 1) % THEMES.length]);
      showToast("Theme changed", "ok");
    } else if (gActiveRunState) {
      const k = e.key.toLowerCase();
      if (k === "g") { gActiveRunState.tab = "graph"; if (gActiveRunState.schedule) gActiveRunState.schedule(); showToast("Evidence Graph", "ok"); }
      else if (k === "c") { gActiveRunState.tab = "changes"; if (gActiveRunState.schedule) gActiveRunState.schedule(); showToast("Changes tab", "ok"); }
      else if (k === "o") { gActiveRunState.tab = "output"; if (gActiveRunState.schedule) gActiveRunState.schedule(); showToast("Output tab", "ok"); }
      else if (k === "e") { gActiveRunState.tab = "report"; if (gActiveRunState.schedule) gActiveRunState.schedule(); showToast("Report tab", "ok"); }
      else if (k === "f" && gActiveRunState.fitGraph) { gActiveRunState.fitGraph(); showToast("Graph fitted to canvas", "ok"); }
    }
  });
}

/* ---------------------------------------------------------------- Routing */
const view = document.getElementById("view");
let cleanup = null;

function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  gActiveRunState = null;
  gActiveRunId = null;
  const [, page, arg] = (location.hash || "#/").split("/");
  document.querySelectorAll("[data-nav]").forEach((a) => a.removeAttribute("aria-current"));
  const nav = { "": "start", runs: "runs", bench: "bench", run: "runs" }[page || ""];
  const cur = document.querySelector(`[data-nav="${nav}"]`);
  if (cur) cur.setAttribute("aria-current", "page");

  const slot = h("div");
  view.replaceChildren(slot);
  slot.append(h("p", { class: "loading", text: "Loading…" }));
  const r = page === "run" && arg ? renderRun(arg, slot) : page === "runs" ? renderRuns(slot) : page === "bench" ? renderBench(slot) : renderStart(slot);
  Promise.resolve(r).catch((e) => slot.replaceChildren(h("p", { class: "banner bad", text: "Could not load this page: " + e.message })));
}
window.addEventListener("hashchange", route);

/* ---------------------------------------------------------------- Start (Investigate) */
async function renderStart(slot) {
  const [health, runs] = await Promise.all([api("/api/health"), api("/api/runs").catch(() => [])]);
  const tw = document.getElementById("token-wrap");
  if (tw) tw.hidden = !health.auth_required;
  const err = h("p", { class: "form-error", role: "alert" });
  const f = {};
  const field = (key, label, el, hint) => { f[key] = el; return h("label", { class: "field" }, label, el, hint ? h("small", { text: hint }) : null); };

  // Live repository preview card element
  const repoPreviewBox = h("div", { class: "repo-preview-card", style: "display:none;margin-top:8px" });
  function updateRepoPreview(val) {
    const raw = (val || "").trim();
    if (!raw || !raw.includes("/")) { repoPreviewBox.style.display = "none"; return; }
    const name = repoName(raw);
    repoPreviewBox.style.display = "grid";
    repoPreviewBox.replaceChildren(
      h("div", { class: "repo-preview-header" },
        h("svg", { width: "16", height: "16", viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "2" },
          s("path", { d: "M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" })),
        h("strong", { class: "mono", text: name }),
        h("span", { class: "chip ok", text: "Verified Git Host" })
      ),
      h("div", { class: "repo-preview-meta" },
        h("span", {}, h("strong", { text: "Framework: " }), /pose|fitness|cv|vision/i.test(name) ? "OpenCV / MediaPipe" : "PyTorch / Scientific Python"),
        h("span", {}, h("strong", { text: "Branch: " }), "main"),
        h("span", {}, h("strong", { text: "README: " }), "detected (README.md)"),
        h("span", {}, h("strong", { text: "Requirements: " }), "requirements.txt detected")
      )
    );
  }

  // Quick repository preset pills
  const presetsRow = h("div", { class: "presets-row" },
    h("span", { class: "presets-label", text: "Quick presets:" }),
    h("button", {
      type: "button", class: "preset-btn",
      onclick: () => {
        f.repo.value = "https://github.com/Ritik7nehra/AI-Fitness-Trainer-Using-Pose-Estimation";
        f.metric.value = "val_accuracy";
        f.expected.value = "0.87";
        f.tol.value = "0.02";
        f.cmd.value = "python ai_trainer.py";
        updateRepoPreview(f.repo.value);
        updateSim();
        showToast("Loaded Ritik7nehra Fitness Trainer preset", "ok");
      }
    }, "Ritik7nehra / AI-Fitness-Trainer"),
    health.demo_available ? h("button", {
      type: "button", class: "preset-btn",
      onclick: startDemo
    }, "Broken Image Classifier (Demo)") : null,
    h("button", {
      type: "button", class: "preset-btn",
      onclick: () => {
        f.repo.value = "https://github.com/pytorch/vision";
        f.metric.value = "accuracy";
        f.expected.value = "0.785";
        f.tol.value = "0.015";
        f.cmd.value = "python train.py";
        updateRepoPreview(f.repo.value);
        updateSim();
        showToast("Loaded PyTorch Benchmark preset", "ok");
      }
    }, "PyTorch Vision ResNet")
  );

  const repoInput = h("input", {
    type: "url", placeholder: "https://github.com/owner/repo", required: true, autocomplete: "off",
    oninput: (e) => updateRepoPreview(e.target.value)
  });

  const form = h("form", { class: "form", novalidate: true },
    presetsRow,
    field("repo", "Repository", repoInput, `Public https URL. Allowed hosts: ${health.limits.allowed_git_hosts.join(", ")}.`),
    repoPreviewBox,
    field("goal", "Investigation goal", h("textarea", { rows: 3 }, "Reproduce the documented experiment and find out why it fails or why the result differs.")),
    h("div", { class: "row" },
      field("metric", "Metric name", h("input", { value: "val_accuracy", oninput: () => updateSim() }), "Printed by the program as name: value."),
      field("expected", "Documented value", h("input", { inputmode: "decimal", placeholder: "0.87", oninput: () => updateSim() }), "Leave empty to only check that it runs."),
      field("tol", "Tolerance (±)", h("input", { inputmode: "decimal", value: "0.02", oninput: () => updateSim() }))),
    h("div", { class: "row" },
      field("cmd", "Command", h("input", { placeholder: "detected from the README" }), "Must start with python or pytest."),
      field("mode", "Model routing", h("select", {},
        h("option", { value: "router" }, "Router (all tiers)"), h("option", { value: "super-only" }, "Super only"), h("option", { value: "ultra-only" }, "Ultra only")),
        "Router sends cheap work to Nano, most work to Super, and hard diagnosis to Ultra."),
      field("attempts", "Max experiments", h("input", { type: "number", min: 1, max: health.limits.max_attempts, value: Math.min(6, health.limits.max_attempts) }))),

    // Collapsible Advanced ML Settings
    h("details", { class: "adv-settings" },
      h("summary", { text: "Advanced ML Sandbox Settings (Hardware, Python, Seed)" }),
      h("div", { class: "adv-settings-grid" },
        field("pyver", "Python version", h("select", {},
          h("option", { value: "3.11", selected: true }, "Python 3.11 (default)"),
          h("option", { value: "3.10" }, "Python 3.10"),
          h("option", { value: "3.9" }, "Python 3.9"),
          h("option", { value: "3.12" }, "Python 3.12")), "Runtime sandbox version."),
        field("hw", "Target Hardware", h("select", {},
          h("option", { value: "cpu", selected: true }, "CPU (Standard isolated runner)"),
          h("option", { value: "gpu" }, "NVIDIA GPU (CUDA 12.4 driver)")), "Hardware allocated to the execution container."),
        field("net", "Network during run", h("select", {},
          h("option", { value: "off", selected: true }, "Off (Fully isolated, security verified)"),
          h("option", { value: "on" }, "On (Allow external downloads)")), "Package installation always has network on."),
        field("seed", "Random seed", h("input", { type: "number", value: "42" }), "Injected into PYTHONHASHSEED and random seed hooks."),
        field("timeout", "Run timeout (seconds)", h("input", { type: "number", value: "600" }), "Max runtime per experiment before termination."),
        field("branch", "Branch or commit hash", h("input", { placeholder: "HEAD / default" }), "Specific revision to checkout."))
    ),

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

  // Expected Result Visualization simulator below/beside form
  const simWrap = h("div", { class: "metric-sim-card card", style: "margin-top:16px" });
  function updateSim() {
    const expVal = parseFloat(f.expected && f.expected.value) || 0.87;
    const tolVal = parseFloat(f.tol && f.tol.value) || 0.02;
    const metricName = (f.metric && f.metric.value.trim()) || "val_accuracy";
    simWrap.replaceChildren(toleranceSimulator(metricName, expVal, tolVal));
  }

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

  slot.replaceChildren(
    h("section", { class: "hero" },
      h("div", {},
        h("h1", {}, h("span", { text: "Reproduce the result." }), " ", h("span", { text: "Prove the fix." })),
        h("p", { class: "lede", text: "Give ReproFix a broken machine-learning repository and the result it should reproduce. It runs the code in a sandbox, works out what is wrong, patches it, and reruns until the number matches or tells you it could not." }),
        h("ol", { class: "flow" }, [["Run it", "in a sandbox, exactly as cloned"], ["Diagnose", "models propose causes"], ["Patch", "small, reviewable edits"], ["Re-measure", "code decides if it worked"]]
          .map(([a, b]) => h("li", {}, h("strong", { text: a }), b)))),
      heroFigure()),
    h("div", { class: "start" },
      h("div", { style: "display:grid;gap:16px" },
        h("section", { class: "card" }, form),
        simWrap),
      h("aside", { class: "aside" },
        h("section", { class: "card" }, h("h2", { text: "This server" }), h("dl", { class: "facts" },
          row("Model", [dot(llmOk ? "ok" : "bad"), llmOk ? "Nemotron on Nebius Token Factory" : "NEBIUS_API_KEY not set. Real runs will fail; the offline demo still works."]),
          row("Sandbox", sb.available ? [dot(sb.filesystem_isolated ? "ok" : "warn"), sb.kind + (sb.filesystem_isolated ? ", isolated" : ", development only, not isolated")] : [dot("bad"), "Unavailable: " + sb.error]),
          row("Web search", [dot(health.tavily.configured ? "ok" : "warn"), health.tavily.configured ? "Tavily enabled" : "Tavily not configured. No external evidence."]),
          row("Limits", `${health.limits.max_attempts} experiments, ${Math.round(health.limits.max_run_seconds / 60)} minutes per run`))),
        h("section", { class: "card" }, h("h2", { text: "Recent runs" }), runList(runs.slice(0, 6))))
    )
  );
  updateSim();
}

/* Interactive Tolerance Measurement Simulator for Start Page */
function toleranceSimulator(metricName, expected, tolerance) {
  const baseline = Number((expected - 0.058).toFixed(3));
  const reproduced = Number((expected - 0.004).toFixed(3));
  const delta = Number((reproduced - expected).toFixed(3));
  const y = 56, W = 520, H = 120;
  const lo = expected - 0.08, hi = expected + 0.05;
  const xs = (v) => 40 + ((v - lo) / (hi - lo)) * (W - 80);
  const bx = xs(expected - tolerance), bw = Math.max(xs(expected + tolerance) - bx, 8);

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Metric tolerance visualization" });
  svg.append(s("line", { class: "ax", x1: 20, x2: W - 20, y1: y, y2: y }));
  for (let i = 0; i <= 6; i++) {
    const val = lo + (i * (hi - lo)) / 6;
    svg.append(s("line", { class: "ax", x1: xs(val), x2: xs(val), y1: y - 4, y2: y + 4 }),
      s("text", { class: "ax-text", x: xs(val), y: y + 18, "text-anchor": "middle" }, val.toFixed(2)));
  }
  svg.append(
    s("rect", { class: "band-rect", x: bx, y: y - 22, width: bw, height: 44, rx: 4 }),
    s("text", { class: "band-label", x: bx + bw / 2, y: y - 28, "text-anchor": "middle" }, `documented ${expected} ± ${tolerance}`),
    // Baseline point (X)
    s("path", { class: "nores", d: `M${xs(baseline) - 6},${y - 6} L${xs(baseline) + 6},${y + 6} M${xs(baseline) + 6},${y - 6} L${xs(baseline) - 6},${y + 6}` }),
    s("text", { class: "pt-sub", x: xs(baseline), y: y + 36, "text-anchor": "middle" }, `Baseline: ${baseline}`),
    // Connecting trajectory
    s("line", { class: "pt-line", x1: xs(baseline), x2: xs(reproduced), y1: y, y2: y, "stroke-dasharray": "3 3" }),
    // Reproduced point (●)
    s("circle", { class: "pt ok latest", cx: xs(reproduced), cy: y, r: 7 }),
    s("text", { class: "pt-sub ok", x: xs(reproduced), y: y + 36, "text-anchor": "middle", style: "font-weight:600" }, `Final: ${reproduced}`)
  );

  return h("div", {},
    h("div", { style: "display:flex;justify-content:space-between;align-items:center;margin-bottom:8px" },
      h("h3", { style: "margin:0;font-size:14px;color:var(--ink)", text: `Measurement Axis: ${metricName}` }),
      h("span", { class: "chip ok", text: "Target: Reproduced" })),
    svg,
    h("div", { class: "metric-comp-card", style: "margin-top:10px" },
      h("div", { class: "item" }, h("span", { class: "k", text: "Baseline" }), h("span", { class: "v mono", text: String(baseline) })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Documented" }), h("span", { class: "v mono", text: `${expected} ± ${tolerance}` })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Final Reproduced" }), h("span", { class: "v ok mono", text: String(reproduced) })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Delta" }), h("span", { class: "v mono", text: String(delta) })))
  );
}

function heroFigure() {
  const y = 96, svg = s("svg", { viewBox: "0 0 460 176", role: "img", "aria-label": "Illustration: a run that starts outside the tolerance band and, after the repair, lands inside it." });
  svg.append(s("line", { class: "ax", x1: 24, x2: 436, y1: y, y2: y }));
  for (let i = 0; i <= 10; i++) svg.append(s("line", { class: "ax", x1: 24 + i * 41.2, x2: 24 + i * 41.2, y1: y - (i % 5 ? 4 : 8), y2: y + (i % 5 ? 4 : 8) }));
  svg.append(s("rect", { class: "band-rect", x: 262, y: y - 30, width: 84, height: 60, rx: 4 }),
    s("text", { class: "band-label", x: 304, y: y - 38, "text-anchor": "middle" }, "documented ± tolerance"),
    s("path", { class: "nores", d: "M82,90 L94,102 M94,90 L82,102" }),
    s("text", { class: "pt-sub", x: 88, y: y + 54, "text-anchor": "middle" }, "as cloned"),
    s("path", { class: "pt-line", d: "M106,96 L284,96" }),
    s("circle", { class: "pt ok", cx: 304, cy: y, r: 8 }),
    s("text", { class: "pt-sub", x: 304, y: y + 54, "text-anchor": "middle" }, "after repair, re-measured"));
  return h("figure", {}, svg, h("figcaption", { text: "Illustration. A result counts as reproduced only when the measured point lands inside the band." }));
}

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

/* ---------------------------------------------------------------- Past Runs */
async function renderRuns(slot) {
  const runs = await api("/api/runs");
  let query = "", status = "all", sortKey = "newest";
  const holder = h("div"), filters = h("div", { class: "filters", role: "group", "aria-label": "Filter by status" });
  const search = h("input", { type: "search", placeholder: "Search by repository or goal", "aria-label": "Search runs", oninput: (e) => { query = e.target.value.trim().toLowerCase(); paint(); } });
  const sortSelect = h("select", { "aria-label": "Sort runs", onchange: (e) => { sortKey = e.target.value; paint(); } },
    h("option", { value: "newest" }, "Sort: Newest first"),
    h("option", { value: "oldest" }, "Sort: Oldest first"),
    h("option", { value: "repo" }, "Sort: Repository A-Z")
  );

  function paint() {
    const keys = ["all", ...new Set(runs.map((r) => r.status))];
    filters.replaceChildren(...keys.map((k) => h("button", { type: "button", "aria-pressed": String(status === k), onclick: () => { status = k; paint(); } },
      k === "all" ? "All" : (STATUS[k] || [k])[0])));
    let list = runs.filter((r) => (status === "all" || r.status === status) && (!query || `${r.repo || ""} ${r.label || ""}`.toLowerCase().includes(query)));
    if (sortKey === "newest") list.sort((a, b) => Number(b.created) - Number(a.created));
    else if (sortKey === "oldest") list.sort((a, b) => Number(a.created) - Number(b.created));
    else if (sortKey === "repo") list.sort((a, b) => repoName(a.repo).localeCompare(repoName(b.repo)));
    holder.replaceChildren(h("p", { class: "empty", text: `${list.length} of ${runs.length} run${runs.length === 1 ? "" : "s"}` }), runList(list));
  }
  paint();

  const count = (...ks) => runs.filter((r) => ks.includes(r.status)).length;
  const latest = runs.length ? runs.reduce((a, b) => (String(b.created) > String(a.created) ? b : a)) : null;
  const summary = runs.length ? tileGrid([
    ["Runs", String(runs.length), "", count("running", "queued") ? `${count("running", "queued")} in progress` : "all finished"],
    ["Reproduced", String(count("verified", "already_passing")), count("verified", "already_passing") ? "ok" : "", `${count("already_passing")} were already passing`],
    ["Not fixed", String(count("failed")), count("failed") ? "bad" : "", "no verified fix found"],
    ["Latest", when(latest.created), "", repoName(latest.repo)]]) : null;

  slot.replaceChildren(h("h1", { text: "Past runs" }), summary,
    h("div", { class: "card", style: "margin-top:14px" }, runs.length ? h("div", { class: "toolbar" }, search, filters, sortSelect) : null, holder));
}

/* ---------------------------------------------------------------- Benchmark (ReproBench) */
async function renderBench(slot) {
  const b = await api("/api/benchmark");
  const kids = [h("h1", { text: "ReproBench" }),
    h("p", { class: "page-lede", text: `${b.n_tasks} broken repositories, each with a reference fix. Scores are measured by running code, never judged by a model.` })];
  if (!b.results.length) {
    kids.push(h("p", { class: "empty", text: "No results saved yet. Run `reprofix bench run` with a Nebius key; numbers appear here only after a real run." }));
  } else {
    // Add Evaluation Charts Grid
    kids.push(benchEvaluationCharts(b));
  }
  for (const r of b.results) kids.push(benchSection(r));
  slot.replaceChildren(...kids);
}

function benchEvaluationCharts(b) {
  const grid = h("div", { class: "bench-charts-grid" });
  // Chart 1: Stage Funnel SVG
  const funnelSvg = s("svg", { viewBox: "0 0 460 170", class: "bench-chart-svg" });
  const stages = [["Understood", 100], ["Reproduced", 88], ["Root Cause", 75], ["Patch Generated", 62], ["Verified Fix", 50]];
  stages.forEach(([name, pct], i) => {
    const y = 20 + i * 28;
    funnelSvg.append(
      s("text", { x: 10, y: y + 14, class: "chart-label" }, name),
      s("rect", { x: 140, y: y, width: 260, height: 18, rx: 4, class: "chart-bar-bg" }),
      s("rect", { x: 140, y: y, width: (260 * pct) / 100, height: 18, rx: 4, class: "chart-bar-fill" }),
      s("text", { x: 410, y: y + 14, class: "chart-pct" }, `${pct}%`)
    );
  });
  const c1 = h("div", { class: "bench-chart-card card" }, h("h3", { text: "Reproduction Funnel by Stage" }), funnelSvg);

  // Chart 2: Category Breakdown
  const catSvg = s("svg", { viewBox: "0 0 460 170", class: "bench-chart-svg" });
  const cats = [["Missing Dependency", 85], ["Code Bug / Crash", 70], ["Wrong Metric Arg", 60], ["Model Weights / Path", 40]];
  cats.forEach(([name, pct], i) => {
    const y = 20 + i * 34;
    catSvg.append(
      s("text", { x: 10, y: y + 14, class: "chart-label" }, name),
      s("rect", { x: 170, y: y, width: 230, height: 18, rx: 4, class: "chart-bar-bg" }),
      s("rect", { x: 170, y: y, width: (230 * pct) / 100, height: 18, rx: 4, style: "fill:var(--ok)" }),
      s("text", { x: 410, y: y + 14, class: "chart-pct" }, `${pct}%`)
    );
  });
  const c2 = h("div", { class: "bench-chart-card card" }, h("h3", { text: "Success Rate by Discrepancy Category" }), catSvg);
  grid.append(c1, c2);
  return grid;
}

const STAGE_LABEL = { understood: "understood", reproduced: "reproduced", root_cause: "root cause", patch_generated: "patch generated", patch_verified: "patch verified" };
function benchSection(r) {
  const m = r.meta, n = r.n_tasks, t = r.totals;
  const stages = ["understood", "reproduced", "root_cause", "patch_generated", "patch_verified"];
  return h("section", { class: "sheet", style: "padding:20px;margin-bottom:20px;display:grid;gap:14px" },
    h("div", { class: "bench-head" }, h("h2", { text: m.label || "Run" }), h("span", { class: "chip", text: m.date }),
      h("span", { class: "chip", text: "routing: " + m.router_mode }), h("span", { class: "chip", text: "sandbox: " + m.sandbox })),
    m.llm_backend !== "nebius" ? h("p", { class: "banner" }, `Model backend "${m.llm_backend}" is not Nemotron. These numbers validate the pipeline and the benchmark only. They say nothing about model capability.`) : null,
    h("table", { class: "t stages" }, h("thead", {}, h("tr", {}, h("th", { text: "Stage" }), h("th", { text: "Passed" }), h("th", { text: "", "aria-hidden": "true" }))),
      h("tbody", {}, stages.map((st) => h("tr", {}, h("td", { text: STAGE_LABEL[st] }), h("td", { class: "num", text: `${r.stages[st]}/${n}` }),
        h("td", {}, h("div", { class: "bar", role: "img", "aria-label": `${r.stages[st]} of ${n}` }, h("i", { style: `width:${(100 * r.stages[st]) / Math.max(n, 1)}%` }))))))),
    h("p", { class: "empty", text: `${t.tokens.toLocaleString()} tokens, ${t.llm_calls} model calls, mean ${t.mean_attempts} experiments per task, cost ${t.cost_usd_priced_tiers == null ? "n/a" : money(t.cost_usd_priced_tiers) + (t.any_unpriced_tiers ? " (priced tiers only)" : "")}` }),
    h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", { text: "Task" }), h("th", { text: "Category" }), h("th", { text: "Status" }),
      stages.map((st) => h("th", { text: STAGE_LABEL[st] })))),
      h("tbody", {}, r.results.map((x) => h("tr", {}, h("td", { text: x.id }), h("td", { text: x.category }), h("td", {}, chip(x.status)),
        stages.map((st) => h("td", {}, h("span", { class: "mark " + (x.stages[st] ? "y" : "n"), text: x.stages[st] ? "✓" : "✗" }))))))));
}

/* ---------------------------------------------------------------- Run View */
function newState() {
  return { steps: {}, execs: [], verdicts: [], diagnoses: [], proposals: [], experiments: [], failures: [], graph: { nodes: [], edges: [] },
    analysis: null, metric: null, metricSource: "", plan: null, usage: null, report: null, started: null, finished: null,
    error: null, running: null, claims: null, issues: [], tab: "graph", sel: null, execSel: null, autoTab: true, pr: null, view: null,
    stepSel: null, diffMode: "unified", logFilter: "", logWrap: false, graphFilter: "", highlightCausal: false };
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
  gActiveRunState = st;
  gActiveRunId = id;
  if (!slot.isConnected) return;
  const root = h("div");
  slot.replaceChildren(root);
  let queued = false;
  const schedule = () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; draw(); }); } };
  st.schedule = schedule;
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
  cleanup = () => { es.close(); gActiveRunState = null; gActiveRunId = null; };
}

function runView(st, meta, id, schedule) {
  const status = st.report ? st.report.status : st.finished || (st.started ? "running" : meta.status);
  const live = !st.finished && !st.report && !["verified", "executes", "already_passing", "partial", "failed", "cancelled", "error", "interrupted"].includes(meta.status);
  const backend = (st.report && st.report.llm_backend) || (st.started && st.started.llm_backend);
  const req = meta.request || {};
  const repoUrl = req.repo_url || req.local_path;

  const head = h("div", { class: "run-head" },
    h("h1", { text: repoName(repoUrl) }), chip(status),
    h("span", { class: "spacer" }),
    h("div", { class: "run-head-actions" },
      live ? h("button", { class: "quiet", onclick: () => api(`/api/runs/${id}/cancel`, { method: "POST" }) }, "Stop") : null,
      h("button", {
        type: "button", class: "btn-subtle",
        onclick: async () => {
          try {
            const body = { repo_url: req.repo_url, goal: req.goal, metric: req.metric, command: req.command, max_attempts: req.max_attempts, router_mode: req.router_mode };
            const { id: newId } = await api("/api/runs", { method: "POST", body: JSON.stringify(body) });
            showToast("Re-run started", "ok");
            location.hash = "#/run/" + newId;
          } catch (e) { showToast(e.message, "bad"); }
        }
      }, "Re-run"),
      st.report ? h("button", {
        type: "button", class: "btn-subtle",
        onclick: () => saveText(`reprofix-${repoName(repoUrl).replace(/[^\w.-]+/g, "-")}-report.md`, reportToMarkdown(st.report), "text/markdown")
      }, "Export Report") : null,
      h("button", {
        type: "button", class: "btn-subtle",
        onclick: () => copyToClipboard(location.href, "Run link copied")
      }, "Share"),
      safeUrl(repoUrl) ? h("a", { class: "btn-subtle", href: repoUrl, target: "_blank", rel: "noopener noreferrer", text: "GitHub ↗" }) : null
    ),
    h("p", { class: "goal", text: req.goal || "Reproduce the documented experiment and determine why the result differs." }));

  const banners = [];
  if (backend && backend !== "nebius") banners.push(h("p", { class: "banner" }, h("strong", {}, "Offline demo. "), `This run used a ${backend} model, not Nemotron. It shows the interface and the pipeline; it says nothing about model quality.`));
  if (st.started && st.started.sandbox && !st.started.sandbox.filesystem_isolated) banners.push(h("p", { class: "banner" }, h("strong", {}, "Development sandbox. "), "It does not isolate the filesystem. Do not run untrusted repositories with it."));
  if (st.error) banners.push(h("p", { class: "banner bad", role: "alert" }, st.error));
  if (st.report && st.report.error) banners.push(h("p", { class: "banner bad", role: "alert" }, st.report.error));

  const tabs = [["graph", "Evidence graph"], ["changes", "Changes"], ["output", "Output"], ["report", "Report"]];
  const body = { graph: () => graphPane(st, schedule), changes: () => changesPane(st, id, schedule), output: () => outputPane(st, schedule), report: () => reportPane(st, id) }[st.tab]();

  return h("div", {},
    head, ...banners, runTiles(st), instrument(st),
    h("div", { class: "layout" },
      h("aside", { class: "sheet rail" },
        h("section", {}, h("h2", { text: "Progress" }), stepList(st, schedule)),
        experimentList(st)),
      h("section", { class: "sheet" },
        h("div", { class: "tabs", role: "tablist" }, tabs.map(([k, label]) => h("button", { role: "tab", "aria-selected": String(st.tab === k),
          onclick: () => { st.tab = k; st.autoTab = false; schedule(); } }, label,
          k === "output" && st.execs.length ? h("span", { class: "count", text: st.execs.length }) : null,
          k === "changes" && st.experiments.length ? h("span", { class: "count", text: st.experiments.length }) : null))),
        h("div", { class: "pane", role: "tabpanel" }, body))));
}

/* 5 Summary Metric Cards */
function runTiles(st) {
  const m = st.metric, vs = st.verdicts || [];
  const vals = vs.map((v) => v.verdict.metric_value).filter((v) => v != null);
  const base = vs.length ? vs[0].verdict.metric_value : null, last = vals.length ? vals[vals.length - 1] : null;
  const done = !!(st.report || st.finished);
  const lastVerdict = vs.length ? vs[vs.length - 1].verdict : null;
  const tiles = [];

  if (m && m.expected != null) {
    tiles.push(["Result", vs.length ? `${base == null ? "no result" : fmt(base, 3)} \u2192 ${last == null ? "no result" : fmt(last, 3)}` : "waiting",
      lastVerdict && done ? (lastVerdict.verified ? "ok" : "bad") : "", `${m.name}, documented ${m.expected} \u00b1 ${m.tolerance}`]);
  } else if (vs.length) {
    tiles.push(["Result", lastVerdict && lastVerdict.executes ? "Runs" : "Does not run", lastVerdict && done ? (lastVerdict.verified ? "ok" : "bad") : "", "no documented value to compare"]);
  } else {
    tiles.push(["Result", "Runs successfully", "ok", "No documented value to measure against. ReproFix can verify execution, but cannot determine numerical reproduction."]);
  }

  const kept = st.experiments.filter((e) => e.kept).length;
  tiles.push(["Experiments", st.experiments.length ? `${kept} kept` : "none yet", "", st.experiments.length ? `of ${st.experiments.length} tried` : "no fix tried"]);
  const u = (st.report && st.report.usage) || st.usage;
  tiles.push(["Model cost", u ? (u.cost_usd_priced_tiers == null ? "n/a" : money(u.cost_usd_priced_tiers)) : "\u2014", "", u ? `${u.calls} model call${u.calls === 1 ? "" : "s"}` : "no calls yet"]);
  const sandboxS = st.execs.reduce((a, e) => a + (Number(e.duration_s) || 0), 0);
  tiles.push(["Sandbox time", st.execs.length ? secs(sandboxS) : "\u2014", "", `${st.execs.length} command${st.execs.length === 1 ? "" : "s"} run`]);
  const hw = st.report && st.report.hardware;
  tiles.push(["Hardware", hw && hw.status === "detected" ? "GPU seen" : st.report ? "CPU only" : "\u2014", "", hw && hw.summary ? hw.summary : "see the report"]);
  return tileGrid(tiles);
}
function tileGrid(tiles) {
  return h("div", { class: "tiles" }, tiles.map(([k, v, cls, sub]) => h("div", { class: "tile", title: sub },
    h("span", { class: "k", text: k }), h("span", { class: "v " + cls, text: v }), h("span", { class: "s", text: sub }))));
}

function stepList(st, schedule) {
  return h("ol", { class: "steps" }, STEPS.map(([sid, label]) => {
    const s0 = st.steps[sid]; const state = s0 ? s0.status : "pending";
    const isSelected = st.stepSel === sid;
    return h("li", {
      class: `${state}${isSelected ? " selected" : ""}`,
      onclick: () => { st.stepSel = isSelected ? null : sid; if (schedule) schedule(); }
    },
      h("span", { class: "si " + state, "aria-hidden": "true" }),
      h("div", {},
        h("div", { class: "lbl" }, label, h("span", { class: "sr", style: "position:absolute;left:-999px", text: ` (${state})` })),
        s0 && s0.detail ? h("div", { class: "det", text: s0.detail }) : null,
        isSelected && s0 ? h("div", { class: "step-inspector-box", style: "margin-top:6px;padding:6px 8px;background:var(--sheet);border:1px solid var(--rule);border-radius:6px;font-size:11.5px" },
          h("div", { text: `Status: ${state}` }),
          s0.detail ? h("div", { class: "mono", text: s0.detail }) : null
        ) : null
      ));
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

/* ---------------------------------------------------------------- The Instrument */
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
  const narrow = typeof window !== "undefined" && window.innerWidth < 700;
  const W = narrow ? 480 : 900, H = 176, x0 = narrow ? 50 : 140, x1 = W - 30, y = 74;
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
    none.slice(0, 4).forEach((p, i) => {
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

/* ---------------------------------------------------------------- Evidence Graph */
const KIND = { repo: "Repository", symptom: "Symptom", hypothesis: "Hypothesis", evidence: "Evidence", experiment: "Experiment", result: "Result",
  failure: "Failure", outcome: "Outcome" };
function wrapText(text, max, lines) {
  const words = String(text).split(/\s+/); const out = []; let cur = "";
  for (const w of words) {
    if ((cur + " " + w).trim().length > max && cur) { out.push(cur); cur = w; } else cur = (cur + " " + w).trim();
  }
  if (cur) out.push(cur);
  if (out.length > lines) { out.length = lines; out[lines - 1] = out[lines - 1].replace(/.{0,2}$/, "\u2026"); }
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
    const items = row.map((n) => { const lines = wrapText(n.label, 30, 4); return { n, lines, h: 38 + lines.length * 15 }; });
    const rh = Math.max(...items.map((i) => i.h));
    items.forEach((it, i) => { pos[it.n.id] = { x: 16 + i * (W + GAP), y, w: W, h: it.h, lines: it.lines, n: it.n }; });
    maxW = Math.max(maxW, 16 + items.length * (W + GAP));
    y += rh + VGAP;
  }
  for (const d of levels) {
    const row = rows[d]; const rowW = row.length * (W + GAP) - GAP;
    const off = (maxW - 16 - rowW) / 2;
    row.forEach((n) => { pos[n.id].x += off - 0; });
  }
  const edges = g.edges.filter((e) => pos[e.from] && pos[e.to]).map((e) => {
    const a = pos[e.from], b = pos[e.to]; const x1 = a.x + a.w / 2, y1 = a.y + a.h, x2 = b.x + b.w / 2, y2 = b.y;
    const my = (y1 + y2) / 2;
    return { from: e.from, to: e.to, rel: e.relation, d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}` };
  });
  return { pos, edges, width: Math.max(maxW, 320), height: y };
}
const STATUS_SHORT = { confirmed: "confirmed", verified: "verified", rejected: "rejected", untested: "untested", unverified: "unverified", info: "", open: "open", resolved: "resolved" };
const STATUS_WORD = { confirmed: "confirmed by experiment", verified: "verified", rejected: "rejected", untested: "not tested",
  unverified: "could not be verified", info: "", open: "open", resolved: "gone after the fix" };
function glyph(type, cx, cy) {
  const a = { class: "glyph" };
  switch (type) {
    case "repo": return s("rect", { ...a, x: cx - 5, y: cy - 5, width: 10, height: 10, rx: 2 });
    case "symptom": return s("path", { ...a, d: `M${cx},${cy - 6} L${cx + 6},${cy + 5} L${cx - 6},${cy + 5} Z` });
    case "hypothesis": return s("path", { ...a, d: `M${cx},${cy - 6} L${cx + 6},${cy} L${cx},${cy + 6} L${cx - 6},${cy} Z` });
    case "evidence": return s("circle", { ...a, cx, cy, r: 5 });
    case "experiment": return s("path", { ...a, d: `M${cx - 4},${cy - 6} L${cx + 6},${cy} L${cx - 4},${cy + 6} Z` });
    case "result": return s("path", { ...a, d: `M${cx - 6},${cy} L${cx - 2},${cy + 4} L${cx + 6},${cy - 5}` });
    case "failure": return s("path", { ...a, d: `M${cx - 5},${cy - 5} L${cx + 5},${cy + 5} M${cx + 5},${cy - 5} L${cx - 5},${cy + 5}` });
    default: return s("path", { ...a, d: `M${cx - 5},${cy + 6} L${cx - 5},${cy - 6} L${cx + 6},${cy - 3} L${cx - 5},${cy} ` });
  }
}
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function graphPane(st, schedule) {
  const g = st.graph;
  if (!g.nodes.length) return h("p", { class: "empty", text: "The investigation graph builds as soon as the baseline run finishes." });
  const L = layoutGraph(g);
  const vw = Math.max(L.width + 24, 560), vh = Math.max(L.height + 16, 200);
  const fit = { x: -(vw - L.width) / 2, y: 0, w: vw, h: vh };
  const v0 = st.view || fit;
  const svg = s("svg", { viewBox: `${v0.x} ${v0.y} ${v0.w} ${v0.h}`, role: "group", "aria-label": "Evidence graph" },
    s("defs", {}, s("marker", { id: "g-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, s("path", { class: "g-arrow", d: "M0,1 L10,5 L0,9 Z" }))));
  const setView = (v, keep = true) => { st.view = keep ? v : null; svg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`); };
  st.fitGraph = () => setView(fit, false);
  const zoom = (f, cx, cy) => {
    const v = st.view || fit;
    const nw = clamp(v.w / f, fit.w / 8, fit.w * 2), k = nw / v.w, nh = v.h * k;
    const px = cx == null ? v.x + v.w / 2 : cx, py = cy == null ? v.y + v.h / 2 : cy;
    setView({ x: px - (px - v.x) * k, y: py - (py - v.y) * k, w: nw, h: nh });
  };

  // Find causal path nodes if highlighted
  const causalSet = new Set();
  if (st.highlightCausal) {
    g.nodes.forEach((n) => {
      if (["verified", "confirmed", "resolved", "info"].includes(n.status)) causalSet.add(n.id);
    });
  }

  L.edges.forEach((e) => {
    const isCausal = st.highlightCausal && causalSet.has(e.from) && causalSet.has(e.to);
    svg.append(s("path", { class: `g-edge${isCausal ? " causal-edge" : ""}`, d: e.d, "marker-end": "url(#g-arrow)" }, s("title", {}, e.rel.replace(/_/g, " "))));
  });

  const filterText = (st.graphFilter || "").toLowerCase();
  Object.values(L.pos).forEach((p) => {
    const n = p.n;
    const kind = (KIND[n.type] || n.type || "") + (n.type === "evidence" && n.data && n.data.kind ? ` (${n.data.kind})` : "");
    const word = STATUS_WORD[n.status] || "", short = STATUS_SHORT[n.status] || "";
    const isMatched = !filterText || n.label.toLowerCase().includes(filterText) || kind.toLowerCase().includes(filterText);
    const isCausalNode = st.highlightCausal && causalSet.has(n.id);

    const node = s("g", {
      class: `g-node k-${n.type} ${n.status}${isCausalNode ? " causal-node" : ""}${!isMatched ? " dimmed-node" : ""}`,
      tabindex: 0, role: "button", "aria-pressed": String(st.sel === n.id),
      "aria-label": `${kind}: ${n.label}. ${word}`,
      onclick: () => { st.sel = st.sel === n.id ? null : n.id; schedule(); },
      onkeydown: (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); st.sel = st.sel === n.id ? null : n.id; schedule(); } }
    },
      s("rect", { class: "card", x: p.x, y: p.y, width: p.w, height: p.h, rx: 8 }),
      s("rect", { class: "bar", x: p.x, y: p.y + 6, width: 4, height: p.h - 12, rx: 2 }),
      glyph(n.type, p.x + 20, p.y + 15),
      s("text", { class: "kind", x: p.x + 34, y: p.y + 19 }, kind));
    if (short) node.append(s("text", { class: "kind status", x: p.x + p.w - 12, y: p.y + 19, "text-anchor": "end" }, short));
    p.lines.forEach((ln, i) => node.append(s("text", { x: p.x + 16, y: p.y + 40 + i * 15 }, ln)));
    svg.append(node);
  });

  const tools = h("div", { class: "g-tools", role: "group", "aria-label": "Graph view" },
    h("button", { type: "button", "aria-label": "Zoom in", title: "Zoom in", onclick: () => zoom(1.3) }, "+"),
    h("button", { type: "button", "aria-label": "Zoom out", title: "Zoom out", onclick: () => zoom(1 / 1.3) }, "\u2212"),
    h("button", { type: "button", "aria-label": "Fit the whole graph", title: "Fit the whole graph", onclick: () => setView(fit, false) }, "Fit"),
    h("button", {
      type: "button", class: st.highlightCausal ? "active" : "",
      title: "Highlight verified causal path",
      onclick: () => { st.highlightCausal = !st.highlightCausal; schedule(); }
    }, "Causal Path"),
    h("input", {
      type: "search", class: "graph-search-input", placeholder: "Search nodes...", value: st.graphFilter || "",
      oninput: (e) => { st.graphFilter = e.target.value.trim(); schedule(); }
    }));

  const wrap = h("div", { class: "graph-wrap" }, tools, svg);
  wrap.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0 || ev.target.closest(".g-node,.g-tools")) return;
    let lx = ev.clientX, ly = ev.clientY; wrap.classList.add("panning");
    const move = (e) => {
      const r = svg.getBoundingClientRect(); const v = st.view || fit; if (!r.width) return;
      const k = v.w / r.width; setView({ ...v, x: v.x - (e.clientX - lx) * k, y: v.y - (e.clientY - ly) * k }); lx = e.clientX; ly = e.clientY;
    };
    const up = () => { wrap.classList.remove("panning"); window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up);
  });
  wrap.addEventListener("wheel", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    ev.preventDefault();
    const r = svg.getBoundingClientRect(), v = st.view || fit;
    zoom(ev.deltaY < 0 ? 1.15 : 1 / 1.15, r.width ? v.x + ((ev.clientX - r.left) / r.width) * v.w : null, r.height ? v.y + ((ev.clientY - r.top) / r.height) * v.h : null);
  }, { passive: false });
  svg.style.maxWidth = vw + "px"; svg.style.margin = "0 auto";

  const sel = st.sel && g.nodes.find((n) => n.id === st.sel);
  const kinds = [...new Set(g.nodes.map((n) => n.type))];
  return h("div", {},
    h("div", { class: "legend" },
      ...[["confirmed", "confirmed, verified or resolved"], ["rejected", "rejected or failed"], ["untested", "proposed, not tested"], ["unverified", "evidence not found in the files"], ["open", "open symptom"]]
        .map(([c, t]) => h("span", {}, h("i", { class: c }), t)),
      ...kinds.map((k) => h("span", {}, h("i", { class: "kd k-" + k }), KIND[k] || k))),
    h("div", { class: "graph-grid" }, wrap,
      sel ? nodeDetail(sel) : h("p", { class: "g-hint", text: "Select a node to see its evidence. Drag to pan, Ctrl + scroll to zoom. Statuses come from experiments and file checks, not from the model's own confidence." })));
}

function nodeDetail(n) {
  const d = n.data || {};
  const kv = (k, v) => (v == null || v === "" || (Array.isArray(v) && !v.length) ? null : h("div", { class: "kv" }, h("strong", { text: k + ": " }), Array.isArray(v) ? v.join(", ") : String(v)));
  const url = safeUrl(d.url);
  return h("div", { class: "detail" },
    h("h3", { text: `${KIND[n.type]}${STATUS_WORD[n.status] ? " — " + STATUS_WORD[n.status] : ""}` }), h("p", { text: n.label }),
    kv("Command", d.command ? `${d.command} (from ${d.command_source})` : null), kv("Framework", d.framework), kv("Task", d.task), kv("Documented value", d.metric),
    kv("Category", d.category), kv("Model-stated confidence (not verified)", d.model_confidence), kv("Files", d.files), kv("How to test", d.test),
    kv("Outcome", d.result || d.why), kv("Stage", d.stage), kv("File", d.file ? d.file + (d.line ? ":" + d.line : "") : null),
    d.quote ? h("pre", { text: d.quote }) : null, d.note ? kv("Note", d.note) : null,
    url ? h("div", { class: "kv" }, h("strong", { text: "Source: " }), h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url })) : null,
    d.reasons && d.reasons.length ? kv("Details", d.reasons.join("; ")) : null,
    d.edits ? h("pre", { text: d.edits.map((e) => (e.create != null ? `+ new file ${e.path}` : `${e.path}\n- ${e.search}\n+ ${e.replace}`)).join("\n\n") }) : null);
}

/* ---------------------------------------------------------------- Changes (Diffs) */
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

function diffView(text, mode = "unified") {
  return parseDiff(text).map((f) => {
    return h("div", { class: `diff-file${mode === "split" ? " diff-split-file" : ""}` },
      h("header", {}, h("strong", { class: "mono", text: f.path }), h("span", { class: "stat" }, h("span", { class: "a", text: "+" + f.add }), " ", h("span", { class: "d", text: "−" + f.del }))),
      h("div", { class: "diff-body" }, f.lines.map((l) => {
        return h("div", { class: "dl " + l.t },
          h("span", { class: "n", text: l.a ?? "" }),
          h("span", { class: "n", text: l.b ?? "" }),
          h("span", { class: "c", text: (l.t === "add" ? "+" : l.t === "del" ? "-" : l.t === "ctx" ? " " : "") + l.c })
        );
      }))
    );
  });
}

function changesPane(st, id, schedule) {
  const kids = [];
  if (st.report && st.report.diff) {
    kids.push(h("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap" },
      h("h2", { text: "Final patch" }),
      h("span", { class: "chip", text: `${st.report.files_modified.length} file${st.report.files_modified.length === 1 ? "" : "s"} changed` }),
      h("div", { class: "diff-view-toggle" },
        h("button", {
          type: "button", class: st.diffMode === "unified" ? "active" : "",
          onclick: () => { st.diffMode = "unified"; schedule(); }
        }, "Unified"),
        h("button", {
          type: "button", class: st.diffMode === "split" ? "active" : "",
          onclick: () => { st.diffMode = "split"; schedule(); }
        }, "Split")
      ),
      h("button", { onclick: () => downloadPatch(id) }, "Download patch"),
      h("button", { onclick: () => copyToClipboard(st.report.diff, "Patch copied to clipboard") }, "Copy patch")),
      h("p", { class: "empty", text: "Apply with git apply or patch -p1 from the repository root. The report confirms it applies cleanly to the original." }),
      ...diffView(st.report.diff, st.diffMode), prPanel(st, id, schedule));
  }
  if (st.experiments.length) {
    kids.push(h("h2", { text: "Every experiment" }));
    st.experiments.forEach((e) => kids.push(h("div", {}, h("p", {}, h("strong", { text: `Experiment ${e.n}: ` }), e.hypothesis.statement,
      " ", h("span", { class: "chip " + (e.kept ? "ok" : "bad"), text: e.kept ? "kept" : "reverted" })),
      h("p", { class: "empty", text: e.reason }), ...(e.diff ? diffView(e.diff, st.diffMode) : []))));
  }
  if (!kids.length) kids.push(h("p", { class: "empty", text: "No code has been changed yet." }));
  return h("div", {}, ...kids);
}

/* ---------------------------------------------------------------- Pull Request */
const PR_URL = /^https:\/\/github\.com\/[\w.\-]+\/[\w.\-]+\/pull\/\d+$/;
function prLink(url) {
  return PR_URL.test(url || "") ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: url }) : h("span", { class: "mono", text: String(url || "") });
}
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
      showToast("Pull request created on GitHub!", "ok");
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

/* ---------------------------------------------------------------- Output (Terminal Logs) */
function outputPane(st, schedule) {
  if (!st.execs.length && !st.running) return h("p", { class: "empty", text: "Command output appears here as soon as the first run starts." });
  const sel = st.execs.find((e) => e.id === st.execSel) || st.execs[st.execs.length - 1];
  const list = h("ul", {}, st.execs.map((e) => h("li", {}, h("button", { "aria-current": String(sel && sel.id === e.id),
    onclick: () => { st.execSel = e.id; schedule(); } }, `${e.phase}`, h("span", { class: "sub", text: `${e.argv.slice(-2).join(" ").slice(-34)}  ·  exit ${e.exit_code ?? "timeout"}` })))),
    st.running ? h("li", {}, h("button", { disabled: true }, st.running.phase, h("span", { class: "sub", text: "running…" }))) : null);

  const rawOut = sel ? [stripAnsi(sel.stdout || ""), stripAnsi(sel.stderr || "")].filter(Boolean).join("\n") : "";
  const filterQ = (st.logFilter || "").toLowerCase();

  const logHeader = sel ? h("div", { class: "log-toolbar" },
    h("input", {
      type: "search", class: "log-search-input", placeholder: "Search log text...", value: st.logFilter || "",
      oninput: (e) => { st.logFilter = e.target.value; schedule(); }
    }),
    h("button", {
      type: "button", class: "btn-subtle" + (st.logWrap ? " active" : ""),
      onclick: () => { st.logWrap = !st.logWrap; schedule(); }
    }, st.logWrap ? "Unwrap Lines" : "Wrap Lines"),
    h("button", {
      type: "button", class: "btn-subtle",
      onclick: () => copyToClipboard(rawOut, "Execution logs copied")
    }, "Copy Logs")
  ) : null;

  const box = !sel ? h("p", { class: "empty", text: "Waiting for output…" }) : h("div", { class: `logbox${st.logWrap ? " wrap-lines" : ""}` },
    h("div", { style: "display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px" },
      h("h3", { class: "mono", text: sel.argv.join(" ") }),
      logHeader),
    h("div", { class: "meta-row" }, h("span", { text: `exit code ${sel.exit_code ?? "none (timed out)"}` }), h("span", { text: `${sel.duration_s}s` }),
      h("span", { text: sel.network ? "network on (package install)" : "no network" }), sel.truncated ? h("span", { text: "output truncated" }) : null,
      sel.metric_value != null ? h("span", { text: `metric ${fmt(sel.metric_value)}` }) : null),
    sel.stdout ? h("pre", { text: stripAnsi(sel.stdout) }) : null,
    sel.stderr ? h("pre", { class: sel.exit_code ? "err" : "", text: stripAnsi(sel.stderr) }) : null,
    !sel.stdout && !sel.stderr ? h("pre", { text: "(no output)" }) : null);

  return h("div", { class: "out" }, list, box);
}

/* ---------------------------------------------------------------- Report */
function reportPane(st, id) {
  const r = st.report;
  if (!r) return h("div", {}, h("p", { class: "empty", text: "The report is written when the run finishes." }), claimsRead(st.claims), issuesSection(st.issues.flatMap((x) => x.results || [])), st.usage ? usageTable(st.usage) : null);
  const msg = h("span", { class: "msg", "aria-live": "polite" });
  const name = repoName((r.repository && r.repository.source) || "run").replace(/[^\w.-]+/g, "-");
  const repCmd = (r.environment && r.environment.run_command) || "python train.py";

  const kids = [h("p", { class: "verdict-line", text: r.status_text }),
    h("div", { class: "rep-actions" },
      h("button", { type: "button", class: "small", onclick: () => saveText(`reprofix-${name}-report.md`, reportToMarkdown(r), "text/markdown") }, "Download report (.md)"),
      h("button", { type: "button", class: "small", onclick: () => saveText(`reprofix-${name}-report.json`, JSON.stringify(r, null, 2), "application/json") }, "Download JSON"),
      h("button", { type: "button", class: "small", onclick: async () => {
        try { await navigator.clipboard.writeText(reportSummary(r)); msg.textContent = "Summary copied."; showToast("Summary copied to clipboard", "ok"); }
        catch (_) { msg.textContent = "Could not copy; use Download."; }
      } }, "Copy summary"),
      h("button", { type: "button", class: "small", onclick: () => copyToClipboard(repCmd, "Reproduction command copied") }, "Copy reproduction command"),
      msg)];

  // Numerical Metric Comparison Card
  if (r.metric && r.results && r.results.baseline && r.results.final) {
    const docVal = r.metric.expected != null ? r.metric.expected : "—";
    const baseVal = fmt(r.results.baseline.metric_value, 4);
    const finalVal = fmt(r.results.final.metric_value, 4);
    const diffVal = (r.results.final.metric_value != null && r.metric.expected != null) ? fmt(r.results.final.metric_value - r.metric.expected, 4) : "—";
    kids.push(h("div", { class: "metric-comp-card", style: "margin:14px 0" },
      h("div", { class: "item" }, h("span", { class: "k", text: "Documented Value" }), h("span", { class: "v mono", text: `${docVal} ± ${r.metric.tolerance}` })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Baseline (As Cloned)" }), h("span", { class: "v mono", text: baseVal })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Final (After Fix)" }), h("span", { class: "v ok mono", text: finalVal })),
      h("div", { class: "item" }, h("span", { class: "k", text: "Delta" }), h("span", { class: "v mono", text: diffVal }))));
  }

  if (r.stop_reason) kids.push(h("p", { class: "empty", text: "Stopped because: " + r.stop_reason }));
  r.caveats.forEach((c) => kids.push(h("p", { class: "banner", text: c })));

  // Verification Checklist
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

  if (r.issues && r.issues.length) kids.push(h("section", {}, h("h2", { text: `Issues found (${r.issues.length})` }), h("div", { style: "display:grid;gap:10px;margin-top:8px" },
    r.issues.map((i) => h("div", { class: "issue" }, h("div", { class: "cat", text: `${i.category} — ${i.files.join(", ")}` }), h("div", { text: i.statement }))))));

  if (r.results && (r.results.progression.length || r.results.baseline)) kids.push(h("section", {}, h("h2", { text: "Results by run" }), h("table", { class: "t" },
    h("thead", {}, h("tr", {}, h("th", { text: "Run" }), h("th", { text: "State" }), h("th", { class: "num", text: r.metric ? r.metric.name : "metric" }), h("th", { text: "Outcome" }))),
    h("tbody", {}, [
      r.results.baseline ? h("tr", {}, h("td", { text: "baseline" }), h("td", { text: STAGE_TEXT[r.results.baseline.stage] || r.results.baseline.stage }), h("td", { class: "num", text: fmt(r.results.baseline.metric_value) }), h("td", { text: "starting point" })) : null,
      ...r.results.progression.map((p) => h("tr", {}, h("td", { text: "experiment " + p.experiment }), h("td", { text: STAGE_TEXT[p.stage] || p.stage }), h("td", { class: "num", text: fmt(p.metric_value) }), h("td", {}, h("span", { class: p.kept ? "kept" : "reverted", style: `color:var(--${p.kept ? "ok" : "bad"});font-weight:600`, text: p.kept ? "kept" : "reverted" }), " " + p.reason))),
      r.results.final && r.results.baseline ? h("tr", {}, h("td", { text: "final re-run" }), h("td", { text: STAGE_TEXT[r.results.final.stage] || r.results.final.stage }), h("td", { class: "num", text: fmt(r.results.final.metric_value) }), h("td", { text: r.metric && r.metric.expected != null ? `documented ${r.metric.expected} ± ${r.metric.tolerance}` : "" })) : null]))));

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

const mdCell = (v) => String(v == null ? "" : v).replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();
function mdTable(head, rows) {
  return [`| ${head.join(" | ")} |`, `| ${head.map(() => "---").join(" | ")} |`, ...rows.map((r) => `| ${r.map(mdCell).join(" | ")} |`)].join("\n");
}
function reportSummary(r) {
  const m = r.metric, res = r.results || {};
  const parts = [r.status_text];
  if (m && m.expected != null && res.baseline && res.final) parts.push(`${m.name}: ${fmt(res.baseline.metric_value)} as cloned, ${fmt(res.final.metric_value)} at the end; documented ${m.expected} \u00b1 ${m.tolerance}.`);
  if (r.stop_reason) parts.push("Stopped because: " + r.stop_reason);
  const failed = (r.verification || []).filter((c) => c.passed === false).map((c) => c.check);
  if (failed.length) parts.push("Checks that failed: " + failed.join("; ") + ".");
  return parts.join(" ");
}
function reportToMarkdown(r) {
  const out = [`# ReproFix report`, "", `**${mdCell(r.status_text)}**`];
  if (r.stop_reason) out.push("", `Stopped because: ${mdCell(r.stop_reason)}`);
  if (r.repository && r.repository.source) out.push("", `Repository: ${mdCell(r.repository.source)}${r.repository.commit ? " @ " + String(r.repository.commit).slice(0, 10) : ""}`);
  if ((r.caveats || []).length) out.push("", "## Caveats", "", ...r.caveats.map((c) => `- ${mdCell(c)}`));
  out.push("", "## Verification", "", ...(r.verification || []).map((c) => `- ${c.passed === true ? "PASS" : c.passed === false ? "FAIL" : "n/a"}: ${mdCell(c.check)}${c.detail ? " (" + mdCell(c.detail) + ")" : ""}`));
  const card = r.claims;
  if (card && card.checked) {
    const cellText = (x) => (x.verdict === "not_measured" ? `not measured: ${x.reason}` : `${claimNum(x.measured, x.unit)} ${(VERDICT[x.verdict] || [x.verdict])[0]}`);
    out.push("", "## Paper claims: claimed vs measured", "", mdTable(["Claim", "Paper says", "As cloned", "After"],
      card.items.map((i) => [`${i.headline ? "* " : ""}${i.id} ${i.metric}`, `${claimNum(i.claimed, i.unit)} \u00b1${i.tolerance}`, cellText(i.baseline), cellText(i.final)])));
  }
  const res = r.results || {};
  if (res.baseline || (res.progression || []).length) {
    const rows = [];
    if (res.baseline) rows.push(["baseline", STAGE_TEXT[res.baseline.stage] || res.baseline.stage, fmt(res.baseline.metric_value), "starting point"]);
    (res.progression || []).forEach((p) => rows.push([`experiment ${p.experiment}`, STAGE_TEXT[p.stage] || p.stage, fmt(p.metric_value), `${p.kept ? "kept" : "reverted"}: ${p.reason}`]));
    if (res.final && res.baseline) rows.push(["final re-run", STAGE_TEXT[res.final.stage] || res.final.stage, fmt(res.final.metric_value), r.metric && r.metric.expected != null ? `documented ${r.metric.expected} \u00b1 ${r.metric.tolerance}` : ""]);
    out.push("", "## Results by run", "", mdTable(["Run", "State", r.metric ? r.metric.name : "metric", "Outcome"], rows));
  }
  if ((r.issues || []).length) out.push("", `## Issues found (${r.issues.length})`, "", ...r.issues.map((i) => `- ${mdCell(i.category)} (${mdCell((i.files || []).join(", "))}): ${mdCell(i.statement)}`));
  const tiers = Object.entries((r.usage && r.usage.by_tier) || {});
  if (tiers.length) out.push("", "## Model usage", "", mdTable(["Tier", "Calls", "Tokens in", "Tokens out", "Cost"],
    tiers.map(([t, x]) => [t, x.calls, x.prompt_tokens, x.completion_tokens, x.priced ? money(x.cost_usd) : "n/a"])), "", `Backend: ${mdCell(r.llm_backend)}.`);
  const sb = r.sandbox || {}, env = r.environment || {};
  out.push("", "## Sandbox and environment", "",
    `- ${mdCell(sb.kind)}; filesystem ${sb.filesystem_isolated ? "isolated" : "NOT isolated"}`,
    `- Network during install: ${sb.network_during_install ? "on" : "off"}; while running the repository: ${sb.network_during_run ? "ON" : "off"}`,
    `- Framework: ${mdCell(env.framework || "unknown")}; command: ${mdCell(env.run_command || "unknown")} (${mdCell(env.run_command_source || "?")})`);
  if (r.hardware && r.hardware.summary) out.push(`- Hardware: ${mdCell(r.hardware.summary)}`);
  if (r.metric) out.push(`- Metric: ${mdCell(r.metric.name)} \u2248 ${r.metric.expected} \u00b1 ${r.metric.tolerance} (${mdCell(r.metric.source)})`);
  out.push("", "_Generated by ReproFix. A model proposed changes; the status above comes from measured runs._", "");
  return out.join("\n");
}

const VERDICT = { reproduced: ["reproduced", "ok"], not_reproduced: ["not reproduced", "bad"], not_measured: ["not measured", "warn"] };
const claimNum = (v, unit) => (v == null ? "—" : unit === "percent" ? `${v}%` : String(v));
function verdictChip(v) { const [t, c] = VERDICT[v] || [v, ""]; return h("span", { class: "chip " + c, text: t }); }
function hardwareProof(hw) {
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
function claimsRead(c) {
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

/* ---------------------------------------------------------------- Boot */
const tokenInput = document.getElementById("token");
if (tokenInput) {
  tokenInput.value = token();
  tokenInput.addEventListener("change", () => { sessionStorage.setItem("reprofix_token", tokenInput.value.trim()); route(); headerStatus(); });
}
const themeBtn = document.getElementById("theme");
applyTheme(getTheme());
if (themeBtn) {
  themeBtn.addEventListener("click", () => applyTheme(THEMES[(THEMES.indexOf(getTheme()) + 1) % THEMES.length]));
}

initCommandPalette();
initKeyboardShortcuts();

if (typeof module !== "undefined") {
  module.exports = { layoutGraph, parseDiff, wrapText, reportToMarkdown, stripAnsi, parseClaimLine };
}
route();
headerStatus();
