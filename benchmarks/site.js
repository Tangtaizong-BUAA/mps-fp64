"use strict";

const reportBase = "https://github.com/Tangtaizong-BUAA/mps-fp64/blob/v0.3.0.post1/results/";
const pageSize = 15;
const suites = new Map([
  ["public-ordering.json", "Ordering · CPU thread sweep"],
  ["public-tensor-ops.json", "Public tensor operators"],
  ["v3-large-matrices.json", "Matrix workloads"],
  ["v3-matrix-optimized-cpu.json", "Prepacked CPU matrices"],
  ["v3-matrix-second-seed.json", "Matrices · second seed"],
  ["v2-workloads.json", "0.2.0 named workloads"],
  ["v2-bandwidth.json", "0.2.0 streaming and GEMM"],
  ["speed-20261118.json", "Earlier public workloads"],
  ["benchmark.json", "First-release stream archive"],
  ["benchmark-initial.json", "Initial stream archive"],
]);
const featureSpecs = [
  ["public-ordering.json", "argmax_4096x2048", "Global argmax", "CPU sweep · 1, 4, 8, 12, 16 threads"],
  ["public-ordering.json", "sort_2048x512", "Stable row sort", "CPU sweep · 1, 4, 8, 12, 16 threads"],
  ["v3-matrix-optimized-cpu.json", "dyadic_12_4096", "Exact dyadic GEMM", "Prepacked CPU vs. GPU from FP64"],
  ["v3-large-matrices.json", "normal_square_4096", "Gaussian GEMM", "Fastest measured native FP64 CPU"],
];
const byId = id => document.getElementById(id);
const countFormat = new Intl.NumberFormat("en-US");
const nameOrder = new Intl.Collator("en", { numeric: true, sensitivity: "base" });
let records = [];
let filtered = [];
let page = 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// RFC 4180 quoting, including doubled quotes, CRLF and embedded newlines.
function parseCsv(text) {
  const rows = [];
  let row = [], value = "", quoted = false, closed = false;
  const input = text.replace(/^\uFEFF/, "");
  function endField() {
    row.push(value);
    value = "";
    closed = false;
  }
  function endRow() {
    endField();
    if (row.length > 1 || row[0] !== "") rows.push(row);
    row = [];
  }
  for (let i = 0; i < input.length; i++) {
    const char = input[i];
    if (quoted) {
      if (char === '"' && input[i + 1] === '"') {
        value += '"';
        i++;
      } else if (char === '"') {
        quoted = false;
        closed = true;
      } else {
        value += char;
      }
    } else if (char === ",") {
      endField();
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && input[i + 1] === "\n") i++;
      endRow();
    } else if (char === '"' && !value && !closed) {
      quoted = true;
    } else {
      if (closed || char === '"') throw Error("Malformed CSV field");
      value += char;
    }
  }
  if (quoted) throw Error("Unclosed CSV field");
  if (row.length || value || closed) endRow();
  const headers = rows.shift();
  if (!headers || new Set(headers).size !== headers.length) throw Error("Invalid CSV header");
  return rows.map(fields => {
    if (fields.length !== headers.length) throw Error("Inconsistent CSV row width");
    return Object.fromEntries(headers.map((name, i) => [name, fields[i]]));
  });
}

function milliseconds(value) {
  const v = Number(value);
  const digits = v >= 100 ? 2 : v >= 1 ? 3 : 4;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
function ratio(value) { return Number(value).toFixed(2) + "×"; }
function shapeString(shapes) {
  return shapes.map(dims => dims.length ? dims.map(n => countFormat.format(n)).join("×") : "scalar").join(" · ");
}
function searchText(s) {
  return s.toLowerCase().normalize("NFKC").replaceAll(",", "").replaceAll("×", "x").replace(/[-_]/g, " ");
}
function workloadName(r) {
  const id = r.case;
  if (r.source === "public-ordering.json") {
    return ({ sort: "Stable row sort", topk: "Row-wise top-32", argmax: "Global argmax" })[id.split("_")[0]];
  }
  if (id.startsWith("normal_square_")) return "Gaussian square GEMM";
  if (id.startsWith("dyadic_")) {
    const amplitudes = { 4: "15", 12: "2,047", 16: "32,767" };
    return "Dyadic GEMM ±" + amplitudes[id.split("_")[1]];
  }
  if (id.startsWith("rank_")) return "Short-K GEMM · K = " + id.split("_")[1];
  if (id === "narrow_transposed") return "Narrow GEMM · transposed";
  if (id.startsWith("narrow_")) return "Narrow-output GEMM";
  if (id.startsWith("batch_") && r.source.startsWith("v3")) return "Small-matrix batch GEMM";
  if (id === "mv_row" || id === "mv_col") return id === "mv_row" ? "Row-major matrix–vector" : "Column-major matrix–vector";
  const names = {
    sort_rows_1m: "Stable row sort", topk_rows_1m: "Row-wise top-32", argmax_all_8m: "Global argmax",
    argmax_rows_8m: "Row-wise argmax", relu_8m: "ReLU", sum_columns_8m: "Column sum",
    sort_rows: "Row sort", stable_sort: "Stable row sort", max_pool: "Max pooling",
    avg_pool: "Average pooling", add_cpu_roundtrip: "Add · CPU round trip",
    rfft: "Real FFT", mm: "Matrix multiply", bmm: "Batch matrix multiply",
    mlp_forward_backward: "MLP forward + backward", two_layer_forward_backward: "MLP forward + backward",
  };
  if (names[id]) return names[id];
  const normalized = id.replace(/^(?:resident_|host_to_host_)/, "").replace(/_(?:\d.*)$/, "");
  if (names[normalized]) return names[normalized];
  const label = normalized.replaceAll("_", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}
function shortPath(path) {
  const explicit = {
    cpu: "Functional", cpu_out: "Preallocated", cpu_fp64: "Functional", cpu_fp64_out: "Preallocated",
    cpu_add: "Functional", cpu_outer_out: "Outer product · out", cpu_functional: "Functional",
    cpu_prepacked_sgemm: "Prepacked SGEMM", accelerate_dgemm: "Accelerate DGEMM", accelerate_dgemv: "Accelerate DGEMV",
    gpu_from_fp64_out: "Preallocated", mps_030_out: "Preallocated", mps_out: "Preallocated",
    mps_030_roundtrip: "Round trip", new_fp64: "Functional", old_fp64: "Functional",
  };
  if (explicit[path]) return explicit[path];
  const cpu = /^cpu_t(\d+)_(out|functional)$/.exec(path);
  if (cpu) return cpu[1] + " threads · " + (cpu[2] === "out" ? "out" : "functional");
  return "Functional";
}
function scenarioKey(row) {
  if (row.placement === "CPU round trip") return row.timing === "single" ? "transfer-single" : "transfer-batch";
  return row.timing === "single" ? "single" : "batch";
}
function timingLabel(r) {
  return { single: "Single · resident", batch: "Repeated · resident", "transfer-single": "Single · round trip",
    "transfer-batch": "Repeated · round trip" }[r.scenario];
}
function versionLabel(v) { return v === "unversioned archive" ? "Unversioned archive" : "v" + v; }
function enrich(r) {
  if (!/^[\w.-]+\.json$/.test(r.source) || !r.case || !r.version || !["single", "batch-per-call"].includes(r.timing)) {
    throw Error("Missing benchmark metadata");
  }
  r.cpu = Number(r.cpu_median_ms);
  r.gpu = Number(r.gpu_median_ms);
  r.relative = Number(r.cpu_over_gpu);
  if (![r.cpu, r.gpu, r.relative].every(v => Number.isFinite(v) && v > 0)
      || Math.abs(r.cpu / r.gpu - r.relative) > 1e-10 * r.relative
      || r.winner !== (r.cpu > r.gpu ? "GPU" : r.cpu < r.gpu ? "CPU" : "tie")) {
    throw Error("Inconsistent benchmark ratio");
  }
  r.shapes = JSON.parse(r.input_shapes);
  r.shapeLabel = shapeString(r.shapes);
  r.name = workloadName(r);
  r.suite = suites.get(r.source) || r.source;
  r.scenario = scenarioKey(r);
  r.searchKey = searchText([r.case, r.name, r.shapeLabel, r.suite, r.version, r.cpu_baseline, r.cpu_path,
    r.gpu_path, r.input_kind, r.role, r.scenario, r.winner].join(" "));
  return r;
}

function openDetail(r) {
  const dialog = byId("detail");
  byId("detail-suite").textContent = r.suite + " · " + versionLabel(r.version);
  byId("detail-title").textContent = r.name;
  byId("detail-shape").textContent = r.shapeLabel;
  const time = (label, value) => {
    const d = element("div");
    d.append(element("span", "", label), element("strong", "", value));
    return d;
  };
  byId("detail-times").replaceChildren(time("CPU median · ms", milliseconds(r.cpu)),
    time("GPU median · ms", milliseconds(r.gpu)), time("CPU / GPU", ratio(r.relative)));
  const fields = [
    ["Case", r.case],
    ["CPU baseline", r.cpu_baseline + " · " + r.cpu_threads + " threads selected"],
    ["CPU path", r.cpu_path], ["GPU path", r.gpu_path + " · " + r.role],
    ["Timing", timingLabel(r) + " · " + r.calls_per_round + (r.calls_per_round === "1" ? " call" : " calls") + " per timed round"],
    ["Samples", r.cpu_rounds + " CPU rounds · " + r.gpu_rounds + " GPU rounds"],
    ["CPU Q1–Q3", milliseconds(r.cpu_q1_ms) + "–" + milliseconds(r.cpu_q3_ms) + " ms"],
    ["GPU Q1–Q3", milliseconds(r.gpu_q1_ms) + "–" + milliseconds(r.gpu_q3_ms) + " ms"],
  ];
  if (r.gpu_faster_paired_rounds !== "") fields.push(["GPU faster rounds", r.gpu_faster_paired_rounds + " of " + r.gpu_rounds + " paired rounds"]);
  if (r.matched_cpu_path) fields.push(["Matched allocation", ratio(r.matched_cpu_over_gpu) + " · " + r.matched_cpu_path]);
  if (r.reusable_cpu_path) fields.push(["CPU output reuse", ratio(r.reusable_cpu_over_gpu) + " · " + r.reusable_cpu_path]);
  if (r.seed) fields.push(["Seed", r.seed]);
  if (r.input_strides) fields.push(["Input strides", shapeString(JSON.parse(r.input_strides))]);
  if (r.input_kind) fields.push(["Inputs", r.input_kind]);
  if (r.parameters && r.parameters !== "{}") fields.push(["Parameters", r.parameters]);
  if (r.validation) fields.push(["Validation", r.validation]);
  byId("detail-fields").replaceChildren(...fields.map(([label, value]) => {
    const d = element("div");
    d.append(element("dt", "", label), element("dd", "", value));
    return d;
  }));
  byId("raw-link").href = "results/" + encodeURIComponent(r.source);
  byId("pinned-source").href = reportBase + encodeURIComponent(r.source);
  dialog.showModal();
  dialog.scrollTop = 0;
  byId("close").focus();
}

function renderFeatures() {
  const cards = featureSpecs.map(([source, id, title, note]) => {
    const r = records.find(x => x.source === source && x.case === id && x.role === "primary" && x.scenario === "single");
    if (!r) return null;
    const dims = r.shapes[0], size = dims.map(n => countFormat.format(n));
    // The structure-aware CPU study reused the main run's exact seeded input.
    const described = r.parameters ? r : records.find(x => x.case === id && x.seed === r.seed && x.parameters);
    const amplitude = described ? JSON.parse(described.parameters).amplitude : 0;
    const shape = dims.length === 2 && dims[0] === dims[1]
      ? size[0] + " square" + (amplitude ? " · ±" + countFormat.format(amplitude) : "")
      : size.join(" × ");
    const card = element("article", "feature-card");
    card.dataset.winner = r.winner;
    const h = element("h3");
    const button = element("button", "", title);
    button.type = "button";
    button.setAttribute("aria-label", title + ", " + shape + ", " + ratio(r.relative) + ". View run details");
    button.addEventListener("click", () => openDetail(r));
    h.append(button);
    const value = element("p", "feature-ratio", r.relative.toFixed(2));
    value.append(element("small", "", "×"));
    const timings = element("div", "feature-values");
    for (const [name, v] of [["CPU", r.cpu], ["GPU", r.gpu]]) {
      const part = element("div", "feature-timing");
      const words = element("span", "", name + " ");
      words.append(element("em", "", milliseconds(v) + " ms"));
      const track = element("span", "feature-meter");
      track.setAttribute("aria-hidden", "true");
      const bar = element("i");
      bar.style.width = 100 * v / Math.max(r.cpu, r.gpu) + "%";
      track.append(bar);
      part.append(words, track);
      timings.append(part);
    }
    card.append(h, element("p", "feature-shape", shape), value, timings, element("p", "feature-note", note));
    return card;
  }).filter(Boolean);
  byId("featured").replaceChildren(...cards);
}

function makeRow(r) {
  const tr = element("tr");
  tr.setAttribute("role", "row");
  const cell = (label, sub, className, type) => {
    const td = element("td", className);
    td.setAttribute("role", "cell");
    td.append(element(type || "span", className === "num" ? "metric" : "", label));
    if (sub) td.append(element("small", "", sub));
    return td;
  };
  const item = element("td");
  item.setAttribute("role", "cell");
  const open = element("button", "row-button", r.name);
  open.type = "button";
  open.setAttribute("aria-label", r.name + ", " + r.shapeLabel + ". View run details");
  open.addEventListener("click", () => openDetail(r));
  item.append(open, element("small", "shape-cell", r.shapeLabel));
  const experiment = cell(r.suite, versionLabel(r.version) + " · " + r.role);
  const cpu = cell(milliseconds(r.cpu), shortPath(r.cpu_path), "num");
  cpu.prepend(element("span", "mobile-label", "CPU ms"));
  const gpu = cell(milliseconds(r.gpu), shortPath(r.gpu_path), "num");
  gpu.prepend(element("span", "mobile-label", "GPU ms"));
  const relative = element("td", "num");
  relative.setAttribute("role", "cell");
  relative.append(element("span", "mobile-label", "CPU/GPU"), element("span", "score " + r.winner.toLowerCase(), ratio(r.relative)));
  const samples = cell(r.gpu_rounds + " × " + r.calls_per_round, timingLabel(r), "samples", "strong");
  samples.prepend(element("span", "mobile-label", "Rounds × calls"));
  tr.append(item, experiment, cpu, gpu, relative, samples);
  return tr;
}

function outcome() { return byId("filters").elements.outcome.value; }
function sortRows(a, b, sort) {
  let delta = 0;
  if (sort === "ratio-desc") delta = b.relative - a.relative;
  if (sort === "ratio-asc") delta = a.relative - b.relative;
  if (sort === "gpu-asc") delta = a.gpu - b.gpu;
  if (sort === "cpu-asc") delta = a.cpu - b.cpu;
  return delta || nameOrder.compare(a.name, b.name) || nameOrder.compare(a.case, b.case)
    || nameOrder.compare(a.source, b.source) || nameOrder.compare(a.gpu_path, b.gpu_path);
}
function writeUrl() {
  const u = new URL(location.href);
  const current = { outcome: outcome(), paths: byId("paths").value, suite: byId("suite").value,
    scenario: byId("scenario").value, q: byId("search").value, sort: byId("sort").value };
  const defaults = { outcome: "GPU", paths: "primary", suite: "all", scenario: "all", q: "", sort: "ratio-desc" };
  for (const [k, v] of Object.entries(current)) {
    if (v === defaults[k]) u.searchParams.delete(k);
    else u.searchParams.set(k, v);
  }
  history.replaceState(null, "", u);
}
function readUrl() {
  const p = new URLSearchParams(location.search);
  for (const id of ["paths", "suite", "scenario", "sort"]) {
    if ([...byId(id).options].some(opt => opt.value === p.get(id))) byId(id).value = p.get(id);
  }
  const out = p.get("outcome");
  if (["GPU", "all", "CPU"].includes(out)) byId("filters").elements.outcome.value = out;
  byId("search").value = p.get("q") || "";
}
function renderPage() {
  const total = filtered.length, pages = Math.ceil(total / pageSize);
  page = Math.max(0, Math.min(page, pages - 1));
  const start = page * pageSize;
  const stop = Math.min(start + pageSize, total);
  byId("rows").replaceChildren(...filtered.slice(start, stop).map(makeRow));
  byId("table-wrap").hidden = !total;
  byId("empty").hidden = !!total;
  byId("pager").hidden = pages <= 1;
  byId("page-number").textContent = "Page " + (page + 1) + " of " + pages;
  byId("prev").disabled = page === 0;
  byId("next").disabled = page + 1 >= pages;
  byId("status").textContent = total ? "Showing " + (start + 1) + "–" + stop + " of " + total + " comparisons" : "0 comparisons";
}
function applyFilters(updateUrl) {
  const terms = searchText(byId("search").value.trim()).split(/\s+/).filter(Boolean);
  const scope = byId("paths").value, suite = byId("suite").value, scenario = byId("scenario").value;
  const eligible = records.filter(r =>
    (scope !== "primary" || r.role === "primary") && (suite === "all" || r.source === suite)
    && (scenario === "all" || r.scenario === scenario) && terms.every(t => r.searchKey.includes(t)));
  byId("n-gpu").textContent = eligible.filter(r => r.winner === "GPU").length;
  byId("n-cpu").textContent = eligible.filter(r => r.winner === "CPU").length;
  byId("n-all").textContent = eligible.length;
  const order = byId("sort").value;
  filtered = eligible.filter(r => outcome() === "all" || r.winner === outcome()).sort((a, b) => sortRows(a, b, order));
  for (const id of ["h-name", "h-gpu", "h-cpu", "h-ratio"]) byId(id).removeAttribute("aria-sort");
  const [field, direction] = order.split("-");
  byId("h-" + field).setAttribute("aria-sort", direction === "asc" ? "ascending" : "descending");
  page = 0;
  if (updateUrl) writeUrl();
  renderPage();
}
function changePage(step) {
  page += step;
  renderPage();
  byId("results").focus({preventScroll: true});
  byId("results").scrollIntoView({block: "start"});
}
async function load() {
  byId("load-error").hidden = true;
  byId("status").textContent = "Loading results…";
  try {
    const result = await fetch("benchmarks/all-results.csv");
    if (!result.ok) throw Error("Could not load CSV");
    records = parseCsv(await result.text()).map(enrich);
    if (!records.length) throw Error("Empty benchmark archive");
    const sources = new Set(records.map(r => r.source));
    const ordered = [...new Set([...suites.keys(), ...sources])].filter(s => sources.has(s));
    byId("suite").replaceChildren(new Option("All experiments", "all"), ...ordered.map(s => new Option(suites.get(s) || s, s)));
    readUrl();
    renderFeatures();
    applyFilters(false);
  } catch {
    byId("featured").replaceChildren();
    byId("status").textContent = "Measurements unavailable";
    byId("load-error").hidden = false;
    byId("table-wrap").hidden = byId("empty").hidden = byId("pager").hidden = true;
  }
}

byId("filters").addEventListener("submit", e => e.preventDefault());
byId("filters").addEventListener("change", e => { if (records.length && e.target.id !== "search") applyFilters(true); });
byId("search").addEventListener("input", () => { if (records.length) applyFilters(true); });
byId("sort").addEventListener("change", () => { if (records.length) applyFilters(true); });
byId("prev").addEventListener("click", () => changePage(-1));
byId("next").addEventListener("click", () => changePage(1));
byId("reset").addEventListener("click", () => {
  byId("filters").reset();
  byId("sort").value = "ratio-desc";
  applyFilters(true);
  byId("search").focus();
});
byId("close").addEventListener("click", () => byId("detail").close());
byId("detail").addEventListener("click", e => {
  if (e.target !== byId("detail")) return;
  const box = e.target.getBoundingClientRect();
  if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom) e.target.close();
});
byId("retry").addEventListener("click", load);
load();
