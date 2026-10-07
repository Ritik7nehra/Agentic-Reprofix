// UI smoke test: loads the real index.html + app.js in jsdom against a running ReproFix server,
// feeds it the SSE stream of a real run, and checks what actually ends up in the DOM.
//   Start a server with the offline demo available, then:  UI_TEST_BASE=http://localhost:8765 npm test
import { JSDOM } from "jsdom";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const BASE = process.env.UI_TEST_BASE || "http://localhost:8765";
const WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../web");
const results = [];
const check = (name, ok, extra = "") => { results.push([ok, name, extra]); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : "  " + extra}`); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms = 8000) { const t = Date.now(); while (Date.now() - t < ms) { try { const v = fn(); if (v) return v; } catch (_) {} await sleep(25); } return false; }

function makeWindow(overrides = {}) {
  const html = fs.readFileSync(path.join(WEB, "index.html"), "utf8").replace(/<script[^>]*src="app.js"[^>]*><\/script>/, "");
  const dom = new JSDOM(html, { url: overrides.url || BASE + "/", runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  w.module = { exports: {} };
  w.fetch = overrides.fetch || ((u, o) => fetch(BASE + u, o));
  w.EventSource = class {
    constructor(url) {
      this.l = {}; this.closed = false;
      (async () => {
        await sleep(0); // like a real EventSource: events arrive after the caller has attached its listeners
        const text = overrides.sse ? overrides.sse(url) : await (await fetch(BASE + url)).text();
        for (const block of text.split("\n\n")) {
          if (this.closed) return;
          let ev = "message", data = "";
          for (const ln of block.split("\n")) { if (ln.startsWith("event:")) ev = ln.slice(6).trim(); else if (ln.startsWith("data:")) data += ln.slice(5).trim(); }
          if (data || ev === "end") (this.l[ev] || []).forEach((f) => f({ data }));
        }
        w.__sseDone = true;
      })();
    }
    addEventListener(k, f) { (this.l[k] ||= []).push(f); }
    close() { this.closed = true; }
  };
  w.eval(fs.readFileSync(path.join(WEB, "app.js"), "utf8"));
  return w;
}
// The status chip is drawn from the run's stored status before the event stream is replayed, so assertions about
// what the events produce must wait for the replay and for the next animation frame.
async function replayed(w) { await until(() => w.__sseDone, 15000); await sleep(120); }
const q = (w, sel) => w.document.querySelectorAll(sel);
const text = (w, sel) => [...q(w, sel)].map((e) => e.textContent).join(" | ");

// ------------------------------------------------------------------ 1. start page
{
  const w = makeWindow();
  const ok = await until(() => q(w, "form.form").length);
  check("start page renders the form", !!ok);
  check("start page headline", /Reproduce the result/.test(text(w, "h1")));
  check("offline demo button offered", /offline demo/i.test(text(w, "form button")));
  check("server facts shown (sandbox)", /Sandbox/.test(text(w, ".facts")) && /not isolated/.test(text(w, ".facts")), text(w, ".facts"));
  check("missing API key is called out", /NEBIUS_API_KEY not set/.test(text(w, ".facts")));
}

// ------------------------------------------------------------------ 2. a finished real run
const demo = await (await fetch(BASE + "/api/demo", { method: "POST" })).json();
let meta;
for (let i = 0; i < 400; i++) { meta = await (await fetch(`${BASE}/api/runs/${demo.id}`)).json(); if (meta.report) break; await sleep(500); }
check("demo run finished with a report", !!meta.report && meta.report.status === "verified", meta.status);
{
  const w = makeWindow();
  w.location.hash = "#/run/" + demo.id;
  const ok = await until(() => q(w, ".chip").length && /Reproduced/.test(text(w, ".run-head .chip")), 10000);
  await replayed(w);
  check("status chip says Reproduced", !!ok, text(w, ".run-head .chip"));
  check("offline-demo banner shown (not Nemotron)", /Offline demo/.test(text(w, ".banner")) && /not Nemotron/.test(text(w, ".banner")));
  check("dev-sandbox banner shown", /Development sandbox/.test(text(w, ".banner")));
  check("instrument: 3 measured runs share 2 points (two runs measured the same value)", q(w, "svg circle.pt").length === 2, String(q(w, "svg circle.pt").length));
  check("instrument: the shared point is labelled with both runs", /experiment 3 · final re-run/.test(text(w, ".instrument svg text")), text(w, ".instrument svg text"));
  check("instrument: 2 runs marked as no result", q(w, "svg .nores").length === 2, String(q(w, "svg .nores").length));
  check("instrument: tolerance band labelled", /documented 0\.881 ± 0\.03/.test(text(w, ".band-label")), text(w, ".band-label"));
  check("instrument: values are the measured ones", /0\.1025/.test(text(w, ".instrument")) && /0\.8815/.test(text(w, ".instrument")));
  check("progress rail lists all 9 steps", q(w, ".steps li").length === 9);
  check("no step left pending", ![...q(w, ".steps li")].some((li) => li.classList.contains("pending")));
  const nNodes = meta.report.graph.nodes.length;
  // live runs open on the graph; a finished run switches to the report tab, so go back to the graph
  [...q(w, '[role="tab"]')].find((b) => /Evidence graph/.test(b.textContent)).click();
  await until(() => q(w, ".g-node").length);
  check(`evidence graph draws every node (${nNodes})`, q(w, ".g-node").length === nNodes, String(q(w, ".g-node").length));
  check("graph has confirmed hypotheses", q(w, ".g-node.confirmed").length >= 3);
  const hyp = [...q(w, ".g-node")].find((g) => /Hypothesis/.test(g.textContent));
  hyp.dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await until(() => q(w, ".detail").length);
  check("node detail labels model confidence as unverified", /Model-stated confidence \(not verified\)/.test(text(w, ".detail")));
  [...q(w, '[role="tab"]')].find((b) => /Changes/.test(b.textContent)).click();
  await until(() => q(w, ".diff-file").length);
  check("changes tab shows the final patch files", q(w, ".diff-file").length >= 3, String(q(w, ".diff-file").length));
  check("diff has added and removed lines", q(w, ".dl.add").length >= 3 && q(w, ".dl.del").length >= 3);
  check("download patch button present", /Download patch/.test(text(w, ".pane button")));
  await until(() => q(w, ".pr").length && !/Checking/.test(text(w, ".pr")));
  check("pull request panel explains that a local-path run cannot open one", /Not available/.test(text(w, ".pr")) && /github\.com/.test(text(w, ".pr")) && q(w, ".pr input").length === 0, text(w, ".pr"));
  [...q(w, '[role="tab"]')].find((b) => /Output/.test(b.textContent)).click();
  await until(() => q(w, ".logbox").length);
  check("output tab lists every execution", q(w, ".out li button").length === meta.report.executions.length, `${q(w, ".out li button").length} vs ${meta.report.executions.length}`);
  check("log box shows the command and exit code", /exit code/.test(text(w, ".logbox .meta-row")));
  [...q(w, '[role="tab"]')].find((b) => /Report/.test(b.textContent)).click();
  await until(() => q(w, ".checks li").length);
  check("report lists the 7 computed checks", q(w, ".checks li").length === 7, String(q(w, ".checks li").length));
  check("all checks passed for this run", q(w, ".checks .mk.ok").length === 7 && q(w, ".checks .mk.bad").length === 0);
  check("report states the backend is scripted", /scripted/.test(text(w, ".pane")));
  check("report shows issues found", /Issues found \(3\)/.test(text(w, ".pane")));
  check("usage cost is n/a when the backend reports no usage", /n\/a/.test(text(w, "table.t")));
}

// ------------------------------------------------------------------ 3. hostile content must stay inert
{
  const evil = '<img src=x onerror="window.__pwned=1"><script>window.__pwned=1</script>';
  const events = [
    ["run.started", { llm_backend: "nebius", sandbox: { filesystem_isolated: true }, request: {} }],
    ["analysis", { analysis: { framework: "x" }, metric: { name: "acc", expected: 0.9, tolerance: 0.02 }, metric_source: "user", scan: {} }],
    ["exec", { id: "e1", phase: "baseline", argv: ["python", "train.py"], exit_code: 1, duration_s: 1, network: false, truncated: false, stdout: evil, stderr: evil, metric_value: null }],
    ["graph", { nodes: [{ id: "n1", type: "symptom", label: evil, status: "open", data: {} },
      { id: "n2", type: "evidence", label: evil, status: "unverified", data: { kind: "external", url: "javascript:window.__pwned=1", quote: evil } }], edges: [{ from: "n1", to: "n2", relation: "supported_by" }] }],
    ["run.finished", { status: "failed" }],
  ];
  const sse = () => events.map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n";
  const w = makeWindow({ sse, fetch: (u, o) => (u.includes("/api/runs/evil") ? Promise.resolve({ ok: true, json: async () => ({ id: "evil", status: "failed", request: { repo_url: evil, goal: evil }, report: null }) }) : fetch(BASE + u, o)) });
  w.location.hash = "#/run/evil";
  await until(() => q(w, ".g-node").length === 2);
  await replayed(w);
  const g = [...q(w, '[role="tab"]')]; g.find((b) => /Output/.test(b.textContent)).click();
  await until(() => q(w, ".logbox pre").length);
  check("hostile log text is shown literally", /<script>/.test(text(w, ".logbox pre")), `pre=${text(w, ".logbox pre").slice(0, 80)} | view=${w.document.querySelector("#view").textContent.slice(0, 120)}`);
  check("no injected <img> or <script> elements exist anywhere in the page", q(w, "#view img").length === 0 && q(w, "#view script").length === 0, `${q(w, "#view img").length} img, ${q(w, "#view script").length} script`);
  check("no script ran", w.__pwned === undefined);
  g.find((b) => /Evidence graph/.test(b.textContent)).click();
  await until(() => q(w, ".g-node").length === 2);
  [...q(w, ".g-node")][1].dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  await until(() => q(w, ".detail").length);
  check("javascript: URLs are never rendered as links", q(w, ".detail a").length === 0);
}

// ------------------------------------------------------------------ 3b. pull request panel (the GitHub calls are mocked)
{
  const report = meta.report;
  const json = (data, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => data });
  const events = [["report", report], ["run.finished", { status: "verified" }]];
  const sse = () => events.map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n";
  const TOKEN = "ghp_ui_test_secret_token";
  const preview = { available: true, target: "acme/widget", base_commit: "a".repeat(40), status: "verified", title: "ReproFix: fix the thing",
    body: "## Summary\nbody text", files: [{ path: "x.py", status: "modified", bytes: 10 }, { path: "y.py", status: "added", bytes: 5 }], existing: null };
  async function open(handler, url) {
    const posts = [];
    const fetchPr = (u, o) => {
      if (u === "/api/runs/prx") return json({ id: "prx", status: "verified", request: { repo_url: "https://github.com/acme/widget", goal: "g" }, report });
      if (u === "/api/runs/prx/pull-request") return handler(o || {}, posts);
      return fetch(BASE + u, o);
    };
    const w = makeWindow({ fetch: fetchPr, sse, url });
    w.location.hash = "#/run/prx";
    await until(() => q(w, ".run-head .chip").length);
    await replayed(w);
    [...q(w, '[role="tab"]')].find((b) => /Changes/.test(b.textContent)).click();
    await until(() => q(w, ".pr").length && !/Checking/.test(text(w, ".pr")));
    return { w, posts };
  }
  const type = (w, el, value) => { el.value = value; el.dispatchEvent(new w.Event("input", { bubbles: true })); };

  // happy path
  const ok = await open((o, posts) => {
    if (o.method === "POST") { posts.push(JSON.parse(o.body)); return json({ url: "https://github.com/acme/widget/pull/7", number: 7, branch: "reprofix/prx", head_repo: "acme/widget", base_repo: "acme/widget", base_branch: "main", mode: "branch", commits: 2 }, 201); }
    return json(preview);
  });
  let w = ok.w;
  check("pr panel: shows target repository, base commit and changed files", /acme\/widget/.test(text(w, ".pr")) && /aaaaaaaaaa/.test(text(w, ".pr")) && /x\.py \(modified\)/.test(text(w, ".pr")) && /y\.py \(added\)/.test(text(w, ".pr")), text(w, ".pr"));
  check("pr panel: title and description are editable and prefilled", q(w, ".pr input")[0].value === "ReproFix: fix the thing" && /body text/.test(q(w, ".pr textarea")[0].value));
  const btn = () => q(w, ".pr button.primary")[0];
  const [tokenEl] = [...q(w, '.pr input[type="password"]')];
  check("pr panel: the token field is a password input", !!tokenEl);
  check("pr panel: button is disabled until a token and the confirmation are given", btn().disabled);
  type(w, tokenEl, TOKEN);
  check("pr panel: a token alone is not enough", btn().disabled);
  const cb = q(w, '.pr input[type="checkbox"]')[0];
  cb.checked = true; cb.dispatchEvent(new w.Event("change", { bubbles: true }));
  check("pr panel: token plus confirmation enables the button", !btn().disabled);
  type(w, q(w, ".pr input")[0], "My edited title");
  btn().click();
  await until(() => q(w, ".pr a").length);
  check("pr panel: the request carries token, explicit confirm and the edited title", ok.posts.length === 1 && ok.posts[0].token === TOKEN && ok.posts[0].confirm === true && ok.posts[0].title === "My edited title" && /body text/.test(ok.posts[0].body), JSON.stringify(ok.posts));
  check("pr panel: success shows a link to the pull request", q(w, ".pr a")[0].href === "https://github.com/acme/widget/pull/7" && /opened/.test(text(w, ".pr")));
  check("pr panel: the token is gone from the page afterwards", !w.document.documentElement.outerHTML.includes(TOKEN) && q(w, '.pr input[type="password"]').length === 0);

  // a token must not be typed into a page served over plain HTTP from a non-loopback address
  const fill = (w2) => {
    type(w2, q(w2, '.pr input[type="password"]')[0], "ghp_x");
    const c = q(w2, '.pr input[type="checkbox"]')[0]; c.checked = true; c.dispatchEvent(new w2.Event("change", { bubbles: true }));
  };
  const plain = await open(() => json(preview), "http://203.0.113.7:8000/");
  fill(plain.w);
  check("pr panel: over plain http from a non-loopback host it warns and cannot be submitted",
    /not served over HTTPS/.test(text(plain.w, ".pr")) && q(plain.w, ".pr button.primary")[0].disabled && q(plain.w, '.pr input[type="password"]')[0].disabled, text(plain.w, ".pr"));
  const secure = await open(() => json(preview), "https://reprofix.example.com/");
  fill(secure.w);
  check("pr panel: over https from any host the form works", !/not served over HTTPS/.test(text(secure.w, ".pr")) && !q(secure.w, ".pr button.primary")[0].disabled);
  const loop = await open(() => json(preview), "http://localhost:8000/");
  fill(loop.w);
  check("pr panel: over http on localhost the form works (no network crossing)", !/not served over HTTPS/.test(text(loop.w, ".pr")) && !q(loop.w, ".pr button.primary")[0].disabled);

  // failure keeps the form usable and shows GitHub's reason
  const bad = await open((o) => (o.method === "POST" ? json({ error: "opening the pull request: GitHub rejected the token (expired, revoked or mistyped)" }, 502) : json(preview)));
  w = bad.w;
  type(w, q(w, '.pr input[type="password"]')[0], "ghp_wrong");
  const cb2 = q(w, '.pr input[type="checkbox"]')[0]; cb2.checked = true; cb2.dispatchEvent(new w.Event("change", { bubbles: true }));
  q(w, ".pr button.primary")[0].click();
  await until(() => /rejected the token/.test(text(w, ".pr")));
  check("pr panel: a failed attempt shows the reason and leaves the form for a retry", /rejected the token/.test(text(w, ".pr")) && q(w, ".pr button.primary").length === 1 && q(w, '.pr [role="alert"]').length === 1, text(w, ".pr"));

  // not available
  const na = await open(() => json({ available: false, reason: "a pull request is only offered for runs that ended verified, executes or partial (this one: failed)", existing: null }));
  check("pr panel: an unavailable run says why and offers no form", /only offered for runs/.test(text(na.w, ".pr")) && q(na.w, ".pr input").length === 0);

  // an existing pull request is shown, and a hostile URL is never turned into a link
  const ex = await open(() => json({ available: true, ...preview, existing: { url: "javascript:window.__pwned=1", branch: "b", head_repo: "me/widget", mode: "fork" } }));
  check("pr panel: an existing pull request replaces the form", /opened/.test(text(ex.w, ".pr")) && q(ex.w, ".pr input").length === 0 && /fork/.test(text(ex.w, ".pr")));
  check("pr panel: a javascript: URL is never rendered as a link", q(ex.w, ".pr a").length === 0 && ex.w.__pwned === undefined);
}

// ------------------------------------------------------------------ 3c. paper claims: start form and report card
{
  const json = (data, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => data });
  const evil = '<img src=x onerror="window.__pwned=1">';
  const side = (measured, verdict, extra = {}) => ({ verdict, measured, delta: null, output: measured == null ? null : "val_accuracy: " + measured, exec: "e2", read_as: "fraction", reason: "", ...extra });
  const card = {
    checked: true, method: "m", headline: "c1", skipped: ["table at line 12 has 2 rows and none is labelled 'ours'"], sources: { explicit: 1, extracted: 2 },
    summary: { total: 3, reproduced_before: 0, reproduced_after: 1, fixed: 1, regressed: 0, not_reproduced: 1, not_measured: 1 },
    items: [
      { id: "c1", metric: "accuracy", qualifier: "val", claimed: 90, unit: "percent", tolerance: 1, quote: "We reach 90% " + evil, source: "explicit", headline: true, note: "",
        baseline: side(0, "not_reproduced"), final: side(90, "reproduced"), verdict: "reproduced", change: "fixed" },
      { id: "c2", metric: "top-1", qualifier: null, claimed: 99, unit: "percent", tolerance: 1, quote: "q", source: "paper text, line 3", headline: false,
        note: "comes after a comparison with other methods: check it", baseline: side(0, "not_reproduced"), final: side(90, "not_reproduced"), verdict: "not_reproduced", change: "not_reproduced" },
      { id: "c3", metric: "bleu", qualifier: null, claimed: 27.3, unit: "percent", tolerance: 1, quote: "", source: "explicit", headline: false, note: "",
        baseline: side(null, "not_measured", { reason: "the command exited with code 1, so its output is not a finished result" }),
        final: side(null, "not_measured", { reason: "no output line is named like 'bleu'" }), verdict: "not_measured", change: "not_measured" }],
  };
  const report = { ...meta.report, claims: card };
  const sse = () => [["report", report], ["run.finished", { status: "verified" }]].map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n";
  const w = makeWindow({ sse, fetch: (u, o) => (u === "/api/runs/cl" ? json({ id: "cl", status: "verified", request: { repo_url: "https://github.com/a/b", goal: "g" }, report }) : fetch(BASE + u, o)) });
  w.location.hash = "#/run/cl";
  await until(() => q(w, ".run-head .chip").length);
  await replayed(w);
  await until(() => q(w, "table.claims").length);
  check("claims: the report has a claimed-vs-measured table", q(w, "table.claims tbody tr").length === 3, String(q(w, "table.claims tbody tr").length));
  check("claims: the summary says how many were reproduced before and after", /1 of 3 claims reproduced after the run \(0 before\)\. 1 fixed, 0 regressed, 1 not measured/.test(text(w, ".pane")), text(w, ".pane").slice(0, 300));
  const rows = [...q(w, "table.claims tbody tr")];
  check("claims: the repair target is starred", /★ c1 accuracy/.test(rows[0].textContent) && !/★/.test(rows[1].textContent));
  check("claims: before and after show the measured value with a verdict", /0% not reproduced/.test(rows[0].children[2].textContent) && /90% reproduced/.test(rows[0].children[3].textContent), rows[0].textContent);
  check("claims: the matched output line and its run are shown as proof", /val_accuracy: 90 \(e2\)/.test(rows[0].children[3].textContent));
  check("claims: every cell carries a label so the table can stack on a phone", [...rows[0].children].slice(1).map((c) => c.getAttribute("data-label")).join("|") === "Paper says|Before (as cloned)|After");
  check("claims: a claim that was not measured says why", /exited with code 1/.test(rows[2].children[2].textContent) && /no output line is named like 'bleu'/.test(rows[2].children[3].textContent) && /not measured/.test(rows[2].textContent));
  check("claims: a possible baseline number carries its warning", /comes after a comparison/.test(rows[1].textContent));
  check("claims: skipped tables are mentioned", /none is labelled 'ours'/.test(text(w, ".pane")));
  check("claims: hostile quote text stays inert", /<img/.test(rows[0].textContent) && q(w, ".pane img").length === 0 && w.__pwned === undefined);
  check("claims: verdict chips use the ok/bad/warn colours", q(w, "table.claims .chip.ok").length === 1 && q(w, "table.claims .chip.bad").length === 3 && q(w, "table.claims .chip.warn").length === 2,
    `${q(w, "table.claims .chip.ok").length}/${q(w, "table.claims .chip.bad").length}/${q(w, "table.claims .chip.warn").length}`);

  // a run without paper claims shows no claims section at all
  check("claims: a report without claims has no claims section", q(makeWindow(), "table.claims").length === 0);
  const plain = { ...meta.report, claims: { checked: false, reason: "no paper text or claims were given, so no paper result was checked", skipped: [], method: "m" } };
  const w2 = makeWindow({ sse: () => [["report", plain], ["run.finished", { status: "verified" }]].map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n",
    fetch: (u, o) => (u === "/api/runs/nc" ? json({ id: "nc", status: "verified", request: { repo_url: "https://github.com/a/b", goal: "g" }, report: plain }) : fetch(BASE + u, o)) });
  w2.location.hash = "#/run/nc";
  await until(() => q(w2, ".run-head .chip").length);
  await replayed(w2);
  check("claims: no paper text means no 'Paper claims' heading", !/Paper claims/.test(text(w2, ".pane")));

  // while the run is going: what the patterns read from the text
  const live = [["claims", { claims: [{ id: "c1", metric: "accuracy", value: 90, unit: "percent", qualifier: "val", quote: "We reach 90%" }], headline: "c1", skipped: [] }]];
  const w3 = makeWindow({ sse: () => live.map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join(""),
    fetch: (u, o) => (u === "/api/runs/lv" ? json({ id: "lv", status: "running", request: { repo_url: "https://github.com/a/b", goal: "g" }, report: null }) : fetch(BASE + u, o)) });
  w3.location.hash = "#/run/lv";
  await until(() => q(w3, '[role="tab"]').length);
  await replayed(w3);
  [...q(w3, '[role="tab"]')].find((b) => /Report/.test(b.textContent)).click();
  await until(() => /Claims read from the paper text/.test(text(w3, ".pane")));
  check("claims: while running, the claims read from the text are listed with the repair target", /c1 \(repair target\): accuracy \(val\) = 90%/.test(text(w3, ".pane")) && /not understood by a model/.test(text(w3, ".pane")), text(w3, ".pane"));

  // start form
  let posted = null;
  const w4 = makeWindow({ fetch: (u, o) => (u === "/api/runs" && o && o.method === "POST" ? (posted = JSON.parse(o.body), json({ id: "zz" }, 202)) : fetch(BASE + u, o)) });
  await until(() => q(w4, "form.form").length);
  check("claims: the start form has a collapsed claims box", q(w4, "details.claimbox").length === 1 && !q(w4, "details.claimbox")[0].open && q(w4, "details.claimbox textarea").length === 2);
  const [claimsEl, paperEl] = q(w4, "details.claimbox textarea");
  const typeIn = (el, v) => { el.value = v; el.dispatchEvent(new w4.Event("input", { bubbles: true })); };
  typeIn(q(w4, 'form input[type="url"]')[0], "https://github.com/a/b");
  typeIn(claimsEl, "accuracy=76.4%\nbleu = 27.3 ± 0.3\nval_acc: 0.912");
  typeIn(paperEl, "We reach an accuracy of 76.4%.");
  q(w4, "form.form")[0].dispatchEvent(new w4.Event("submit", { bubbles: true, cancelable: true }));
  await until(() => posted);
  check("claims: the form sends parsed claims and the paper text", posted && JSON.stringify(posted.claims) === JSON.stringify([
    { metric: "accuracy", value: 76.4, unit: "percent" }, { metric: "bleu", value: 27.3, tolerance: 0.3 }, { metric: "val_acc", value: 0.912 }]) && posted.paper_text === "We reach an accuracy of 76.4%.", JSON.stringify(posted));
  check("claims: no claims fields are sent when the box is empty", await (async () => {
    const w5 = makeWindow({ fetch: (u, o) => (u === "/api/runs" && o && o.method === "POST" ? (posted = JSON.parse(o.body), json({ id: "zz" }, 202)) : fetch(BASE + u, o)) });
    await until(() => q(w5, "form.form").length);
    posted = null;
    q(w5, 'form input[type="url"]')[0].value = "https://github.com/a/b";
    q(w5, "form.form")[0].dispatchEvent(new w5.Event("submit", { bubbles: true, cancelable: true }));
    await until(() => posted);
    return posted && !("claims" in posted) && !("paper_text" in posted);
  })());
  posted = null;
  const w6 = makeWindow({ fetch: (u, o) => (u === "/api/runs" && o && o.method === "POST" ? (posted = JSON.parse(o.body), json({ id: "zz" }, 202)) : fetch(BASE + u, o)) });
  await until(() => q(w6, "form.form").length);
  q(w6, 'form input[type="url"]')[0].value = "https://github.com/a/b";
  q(w6, "details.claimbox textarea")[0].value = "accuracy";
  q(w6, "form.form")[0].dispatchEvent(new w6.Event("submit", { bubbles: true, cancelable: true }));
  await sleep(100);
  check("claims: an unreadable claim line is reported and nothing is sent", posted === null && /Cannot read the claim "accuracy"/.test(text(w6, ".form-error")), text(w6, ".form-error"));
}

// ------------------------------------------------------------------ 3d. known issues
{
  const json = (data, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => data });
  const evil = '<img src=x onerror="window.__pwned=1">';
  const ki = { enabled: true, rounds: [], errors: [], note: "n", results: [
    { title: "val accuracy is zero " + evil, url: "https://github.com/acme/widget/issues/9", source: "github-issues", state: "closed", comments: 5, updated: "2026-05-01", score: 0.8, matched_terms: ["val", "accuracy"], snippet: "snippet " + evil },
    { title: "bad link", url: "javascript:window.__pwned=1", source: "web", state: null, comments: null, updated: null, score: 0.5, matched_terms: ["a", "b"], snippet: "" }] };
  const report = { ...meta.report, known_issues: ki };
  const sse = () => [["report", report], ["run.finished", { status: "verified" }]].map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n";
  const w = makeWindow({ sse, fetch: (u, o) => (u === "/api/runs/ki" ? json({ id: "ki", status: "verified", request: { repo_url: "https://github.com/a/b", goal: "g" }, report }) : fetch(BASE + u, o)) });
  w.location.hash = "#/run/ki";
  await until(() => q(w, ".run-head .chip").length);
  await replayed(w);
  await until(() => q(w, "ul.issues").length);
  check("issues: the report lists possibly related issues and says they are not verified", /Possibly related issues \(2, not verified\)/.test(text(w, ".pane")) && /does not check that any of these describes it/.test(text(w, ".pane")), text(w, ".pane").slice(0, 200));
  const a = q(w, "ul.issues a");
  check("issues: a GitHub issue is a link that opens safely", a.length === 1 && a[0].href === "https://github.com/acme/widget/issues/9" && a[0].rel === "noopener noreferrer" && a[0].target === "_blank");
  check("issues: state, comments and matched words are shown", /this repository's issues · closed · 5 comments · 2026-05-01 · matched: val, accuracy/.test(text(w, "ul.issues")), text(w, "ul.issues"));
  check("issues: hostile titles stay inert and a javascript: URL is never a link", q(w, "ul.issues img").length === 0 && /<img/.test(text(w, "ul.issues")) && w.__pwned === undefined && !/javascript/.test([...q(w, "ul.issues a")].map((x) => x.href).join()));
  const none = { ...meta.report, known_issues: { enabled: true, rounds: [], errors: [], note: "n", results: [] } };
  const w2 = makeWindow({ sse: () => [["report", none], ["run.finished", { status: "verified" }]].map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n",
    fetch: (u, o) => (u === "/api/runs/k0" ? json({ id: "k0", status: "verified", request: { repo_url: "https://github.com/a/b", goal: "g" }, report: none }) : fetch(BASE + u, o)) });
  w2.location.hash = "#/run/k0";
  await until(() => q(w2, ".run-head .chip").length);
  await replayed(w2);
  check("issues: no results, no section", q(w2, "ul.issues").length === 0);
  const live = [["known_issues", { attempt: 1, query: "q", searched: ["web"], results: ki.results, errors: [] }]];
  const w3 = makeWindow({ sse: () => live.map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join(""),
    fetch: (u, o) => (u === "/api/runs/k1" ? json({ id: "k1", status: "running", request: { repo_url: "https://github.com/a/b", goal: "g" }, report: null }) : fetch(BASE + u, o)) });
  w3.location.hash = "#/run/k1";
  await until(() => q(w3, '[role="tab"]').length);
  await replayed(w3);
  [...q(w3, '[role="tab"]')].find((b) => /Report/.test(b.textContent)).click();
  await until(() => q(w3, "ul.issues").length);
  check("issues: while the run is going they already show in the Report tab", q(w3, "ul.issues li").length === 2);
}

// ------------------------------------------------------------------ 3e. hardware record
{
  const json = (data, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => data });
  const detected = { requested: true, status: "detected", probed_in: "inside the sandbox container (docker run --gpus all, no network)", cuda_driver: "12.4",
    gpus: [{ index: 0, name: "NVIDIA H100 80GB HBM3", memory_mib: 81559, driver: "550.54.15" }], summary: "NVIDIA H100 80GB HBM3, 81559 MiB each, driver 550.54.15, CUDA 12.4 (driver's maximum)",
    detail: "d", raw: "0, NVIDIA H100 80GB HBM3, 81559, 550.54.15 <img src=x onerror=\"window.__pwned=1\">", note: "This shows which GPU the sandbox can see. It does not show that the repository's code ran on it." };
  async function open(hwRec, id) {
    const report = { ...meta.report, hardware: hwRec };
    const w = makeWindow({ sse: () => [["report", report], ["run.finished", { status: "verified" }]].map(([k, d]) => `event: ${k}\ndata: ${JSON.stringify(d)}\n\n`).join("") + "event: end\ndata: {}\n\n",
      fetch: (u, o) => (u === "/api/runs/" + id ? json({ id, status: "verified", request: { repo_url: "https://github.com/a/b", goal: "g" }, report }) : fetch(BASE + u, o)) });
    w.location.hash = "#/run/" + id;
    await until(() => q(w, ".run-head .chip").length);
    await replayed(w);
    return w;
  }
  const w = await open(detected, "hw1");
  check("hardware: a detected GPU is named in the report with its probe location", /Hardware: NVIDIA H100 80GB HBM3.*nvidia-smi inside the sandbox container/.test(text(w, ".pane")), text(w, ".pane").slice(0, 120));
  check("hardware: the verbatim nvidia-smi output is available and inert", /550\.54\.15/.test(text(w, "pre.raw")) && q(w, "pre.raw img").length === 0 && w.__pwned === undefined);
  check("hardware: the limit of what it proves is stated", /does not show that the repository's code ran on it/.test(text(w, ".pane")));
  const cpu = await open({ requested: false, status: "not_requested", probed_in: null, gpus: [], summary: "CPU only (GPU not enabled)", detail: "d", raw: "", note: "n" }, "hw2");
  check("hardware: a CPU-only run says CPU only and shows no proof box", /Hardware: CPU only \(GPU not enabled\)/.test(text(cpu, ".pane")) && q(cpu, "pre.raw").length === 0);
}

// ------------------------------------------------------------------ 4. pure layout / diff logic
{
  const w = makeWindow();
  const { layoutGraph, parseDiff } = w.module.exports;
  const g = { nodes: [], edges: [] };
  for (let i = 1; i <= 7; i++) g.nodes.push({ id: "n" + i, type: i === 1 ? "symptom" : "hypothesis", label: "x ".repeat(40), status: "open", data: {} });
  for (let i = 2; i <= 7; i++) g.edges.push({ from: "n1", to: "n" + i, relation: "explained_by" });
  const L = layoutGraph(g);
  const row = Object.values(L.pos).filter((p) => p.y === Object.values(L.pos).find((q2) => q2.n.id === "n2").y).sort((a, b) => a.x - b.x);
  check("layout: siblings on one row never overlap", row.every((p, i) => i === 0 || p.x >= row[i - 1].x + row[i - 1].w), JSON.stringify(row.map((r) => r.x)));
  check("layout: children sit below parents", Object.values(L.pos).filter((p) => p.n.id !== "n1").every((p) => p.y > L.pos.n1.y));
  check("layout: canvas is wide enough for every node", Object.values(L.pos).every((p) => p.x >= 0 && p.x + p.w <= L.width + 1), `width ${L.width}`);
  const files = parseDiff("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n");
  check("diff parser counts added/removed lines", files.length === 1 && files[0].add === 1 && files[0].del === 1 && files[0].lines.find((l) => l.t === "add").b === 2);
}

const failed = results.filter((r) => !r[0]).length;
console.log(`\n${results.length - failed}/${results.length} UI checks passed`);
process.exit(failed ? 1 : 0);
