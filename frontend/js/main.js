import { api } from "./api.js";
import { ErrorChart } from "./chart.js";
import { css, DIVERGING, ERROR, FORCED, gradientCss, POLICIES, UNCERTAINTY } from "./colors.js";
import { FieldMap } from "./fieldmap.js";

const $ = (s) => document.querySelector(s);
const SPEEDS = [0.5, 1, 2, 4, 8, 16, 32]; // slots per second
const SELECTABLE = ["max_age", "kalman_greedy", "gnn"];
const PRIORITY_MEANING = {
  max_age: "bar = slots since the sensor last reported",
  kalman_greedy: "bar = uncertainty the sink would remove by polling it",
  gnn: "bar = the GNN's probability of picking it first",
  greedy_diverse: "bar = age, discounted by correlation with earlier picks",
  max_variance: "bar = sink's current uncertainty",
  random: "random choice",
};

const state = {
  datasets: [], ds: null, meta: null, id: null,
  policies: ["max_age", "kalman_greedy", "gnn"], mode: "value", speedIdx: 2,
  playing: false, queue: [], fetching: false, done: false, slots: 0,
  maps: {}, last: null, explain: "gnn", generation: 0,
};

// ------------------------------------------------------------------ tooltip
const tip = {
  node: $("#tooltip"),
  show(html, x, y) {
    this.node.innerHTML = html;
    this.node.classList.add("show");
    const r = this.node.getBoundingClientRect();
    this.node.style.left = `${Math.min(window.innerWidth - r.width - 12, x + 16)}px`;
    this.node.style.top = `${Math.min(window.innerHeight - r.height - 12, y + 14)}px`;
  },
  hide() { this.node.classList.remove("show"); },
};
const chart = new ErrorChart($("#chart"), tip);

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 4500);
}
function setStatus(kind, text) {
  $("#status").className = `status ${kind}`;
  $("#status-text").textContent = text;
}

// ------------------------------------------------------------------ controls
async function init() {
  try {
    state.datasets = await api.datasets();
  } catch (e) {
    setStatus("err", "server not reachable");
    $("#clock-line").textContent = "Start the server with:  python -m gnn_aoi serve";
    return;
  }
  const dsSel = $("#dataset");
  dsSel.innerHTML = state.datasets.map((d) => `<option value="${d.name}">${d.title} · ${d.M} sensors</option>`).join("");
  dsSel.onchange = () => { selectDataset(dsSel.value); restart(); };
  $("#day").onchange = restart;
  $("#hour").onchange = restart;
  $("#capacity").oninput = () => { $("#capacity-val").textContent = $("#capacity").value; };
  $("#capacity").onchange = restart;
  $("#speed").oninput = () => { state.speedIdx = +$("#speed").value; showSpeed(); };
  $("#play").onclick = togglePlay;
  $("#stepbtn").onclick = () => { pause(); advance(); };
  $("#restart").onclick = restart;
  document.querySelectorAll("#mode button").forEach((b) => (b.onclick = () => {
    document.querySelectorAll("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    state.mode = b.dataset.mode;
    renderLegend();
    if (state.last) paintMaps(state.last, performance.now());
  }));
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => showView(t.dataset.view)));
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    if (e.code === "ArrowRight") { pause(); advance(); }
    if (e.key === "r" || e.key === "R") restart();
  });
  renderChips();
  showSpeed();
  selectDataset(state.datasets[0].name);
  await restart();
  requestAnimationFrame(frame);
}

function selectDataset(name) {
  state.ds = state.datasets.find((d) => d.name === name);
  const ds = state.ds;
  const firstFull = Math.max(0, ds.days.findIndex((d) => !d.note && d.weekday < 5)); // a complete weekday
  $("#day").innerHTML = ds.days.map((d, i) => {
    const wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d.weekday];
    return `<option value="${d.date}" ${i === firstFull ? "selected" : ""}>${wd} ${d.date}${d.note ? ` · ${d.note}` : ""}</option>`;
  }).join("");
  $("#hour").innerHTML = Array.from({ length: 24 }, (_, h) =>
    `<option value="${h}" ${h === 5 ? "selected" : ""}>${String(h).padStart(2, "0")}:00</option>`).join("");
  const cap = $("#capacity");
  cap.min = ds.C_range[0]; cap.max = ds.C_range[1]; cap.value = ds.default_C;
  $("#capacity-val").textContent = ds.default_C;
}

function renderChips() {
  $("#sched-chips").innerHTML = SELECTABLE.map((n) =>
    `<button class="chip ${state.policies.includes(n) ? "on" : ""} ${n === "gnn" ? "locked" : ""}" data-p="${n}" style="--c:${POLICIES[n].color}" title="${n === "gnn" ? "Always shown" : "Toggle"}"><i></i>${POLICIES[n].short}</button>`).join("");
  document.querySelectorAll("#sched-chips .chip").forEach((c) => (c.onclick = () => {
    const p = c.dataset.p;
    if (p === "gnn") return;
    state.policies = state.policies.includes(p) ? state.policies.filter((x) => x !== p) : SELECTABLE.filter((x) => x === p || state.policies.includes(x));
    renderChips();
    restart();
  }));
}

function showSpeed() {
  const s = SPEEDS[state.speedIdx];
  const minPerSec = s * (state.ds?.slot_minutes || 5);
  $("#speed-val").textContent = `${s}× · ${minPerSec >= 60 ? (minPerSec / 60).toFixed(minPerSec % 60 ? 1 : 0) + " h" : minPerSec + " min"}/s`;
}

function togglePlay() { state.playing ? pause() : play(); }
function play() {
  if (state.done || state.loading) return;
  state.playing = true;
  $("#play").classList.add("playing");
  $("#play .lbl").textContent = "Pause";
  setStatus("live", "live · scheduling");
  tick();
}
function pause() {
  state.playing = false;
  $("#play").classList.remove("playing");
  $("#play .lbl").textContent = "Play";
  if (state.id) setStatus("", "paused");
}

// ------------------------------------------------------------------ session
async function restart() {
  const gen = ++state.generation;
  const wasPlaying = state.playing;
  pause();
  state.loading = true;
  $("#play").disabled = true;
  setStatus("", "loading…");
  $("#clock-line").textContent = "Fitting the sink's model on the training period…";
  try {
    const res = await api.session({
      dataset: state.ds.name, C: +$("#capacity").value, policies: state.policies,
      date: $("#day").value, hour: +$("#hour").value,
    });
    if (gen !== state.generation) return; // a newer restart superseded this one
    Object.assign(state, { id: res.id, meta: res.meta, queue: [], fetching: false, done: false, doneReason: null, slots: 0, last: null });
    if (!state.policies.includes(state.explain)) state.explain = "gnn";
    buildMaps();
    chart.reset(state.policies, res.meta.units);
    $("#log").replaceChildren();
    renderLegend();
    renderExplainTabs();
    renderKpis(null);
    const m = res.meta;
    $("#clock-line").innerHTML = `<b>${m.M}</b> real ${m.quantity.toLowerCase()} sensors · only <b>${m.C}</b> may transmit per ${m.slot_minutes}-minute slot · every sensor must report at least every <b>${m.freshness}</b> slots (${fmtDur(m.freshness * m.slot_minutes)})`;
    setStatus("", "ready");
    state.loading = false;
    $("#play").disabled = false;
    await advance();
    if (wasPlaying || state.slots <= 1) play();
  } catch (e) {
    if (gen === state.generation) { state.loading = false; $("#play").disabled = false; }
    setStatus("err", "error");
    toast(e.message);
  }
}

async function fetchMore() {
  if (state.fetching || state.done || !state.id) return;
  state.fetching = true;
  const gen = state.generation;
  try {
    const n = Math.max(2, Math.min(24, Math.ceil(SPEEDS[state.speedIdx])));
    const res = await api.step(state.id, n);
    if (gen !== state.generation) return;
    state.queue.push(...res.events);
    if (res.done) { state.done = true; state.doneReason = res.reason; }
  } catch (e) {
    if (gen === state.generation) { toast(e.message); pause(); }
  } finally {
    if (gen === state.generation) state.fetching = false;
  }
}

async function advance() {
  if (!state.queue.length) await fetchMore();
  const ev = state.queue.shift();
  if (!ev) { if (state.done) endOfReplay(); return; }
  apply(ev);
  if (state.queue.length < Math.max(4, SPEEDS[state.speedIdx])) fetchMore();
}

function endOfReplay() {
  pause();
  const gap = state.doneReason === "recording gap";
  setStatus("", gap ? "recording gap" : "end of recording");
  const out = (state.ds.outages || []).find(([a]) => a.startsWith(state.meta.replay_until));
  $("#clock-line").innerHTML = gap
    ? `<b style="color:${FORCED}">The real sensor network went offline here</b>${out ? ` (no data from ${out[0]} to ${out[1]})` : ""}. Pick another day or start time and press Restart.`
    : "End of the recording. Pick another day and press Restart.";
  toast(gap ? "Recording gap: the real sensor network was offline" : "End of the recording");
}

function tick() {
  if (!state.playing) return;
  const started = performance.now();
  const gen = state.generation;
  advance().then(() => {
    if (!state.playing || gen !== state.generation) return;
    const wait = Math.max(0, 1000 / SPEEDS[state.speedIdx] - (performance.now() - started));
    setTimeout(tick, wait);
  });
}

// ------------------------------------------------------------------ maps
function buildMaps() {
  const box = $("#maps");
  box.replaceChildren();
  state.maps = {};
  const panels = [{ key: "truth", label: "Measured (ground truth)", color: null }, ...state.policies.map((p) => ({ key: p, label: `Sink's view · ${POLICIES[p].label}`, color: POLICIES[p].color }))];
  box.style.setProperty("--n", panels.length);
  for (const p of panels) {
    const card = document.createElement("div");
    card.className = `card map-card ${p.key === "truth" ? "truth" : ""}`;
    card.style.setProperty("--c", p.color || "#c3c2b7");
    card.innerHTML = `
      <div class="card-head">
        <div class="map-title"><i></i>${p.label}</div>
        <div class="map-stats" id="stats-${p.key}"></div>
      </div>
      <div class="canvas-wrap"></div>
      <div class="map-badge" id="badge-${p.key}"></div>`;
    box.appendChild(card);
    const map = new FieldMap(card.querySelector(".canvas-wrap"), state.meta, {
      color: p.color,
      onHover: (i, e) => sensorTip(i, e),
    });
    state.maps[p.key] = map;
  }
}

function valueT(v) {
  const [lo, mid, hi] = state.meta.value_range;
  const half = Math.max(hi - mid, mid - lo, 1e-6);
  const t = 0.5 + (v - mid) / (2 * half);
  return state.meta.dataset === "pems_bay" ? 1 - t : t; // traffic: slow = red
}
const errScale = () => (state.meta.value_range[2] - state.meta.value_range[0]) / 8;

function paintMaps(ev, now) {
  const truth = ev.truth;
  state.maps.truth.setValues(truth.map(valueT), DIVERGING, now);
  for (const p of state.policies) {
    const d = ev.policies[p];
    let t, fn;
    if (state.mode === "value") { t = d.estimate.map(valueT); fn = DIVERGING; }
    else if (state.mode === "error") { t = d.estimate.map((v, i) => Math.abs(v - truth[i]) / errScale()); fn = ERROR; }
    else { t = d.uncertainty.map((u) => u / errScale()); fn = UNCERTAINTY; }
    state.maps[p].setValues(t, fn, now);
  }
}

function apply(ev) {
  const now = performance.now();
  state.last = ev;
  state.slots += 1;
  const slotMs = 1000 / SPEEDS[state.speedIdx];
  paintMaps(ev, now);
  for (const p of state.policies) {
    const d = ev.policies[p];
    state.maps[p].transmit(d.polled, d.forced, now, Math.min(1100, Math.max(260, slotMs * 0.8)));
  }
  const [date, hm] = ev.time.split(" ");
  $("#clock-date").textContent = new Date(`${date}T00:00`).toLocaleDateString(undefined, { weekday: "long", year: "numeric", month: "short", day: "numeric" });
  $("#clock-hm").textContent = hm;
  $("#progress-bar").style.width = `${Math.min(100, (state.slots / 288) * 100)}%`;
  renderKpis(ev);
  chart.push(hm, ev);
  renderDecision(ev);
  addLog(ev);
  $("#chart-hint").textContent = `${state.slots} slots · ${fmtDur(state.slots * state.meta.slot_minutes)} of real time`;
}

// ------------------------------------------------------------------ side panels
function renderKpis(ev) {
  const box = $("#kpis");
  if (!ev) { box.replaceChildren(); return; }
  const u = state.meta.units;
  const base = ev.policies.max_age?.running_rmse;
  box.innerHTML = state.policies.map((p) => {
    const d = ev.policies[p];
    let delta = "";
    if (base && p !== "max_age") {
      const pct = ((d.running_rmse - base) / base) * 100;
      delta = `<div class="k-delta ${pct < 0 ? "good" : "bad"}">${pct < 0 ? "▼" : "▲"} ${Math.abs(pct).toFixed(1)}% vs Max-AoI</div>`;
    } else if (p === "max_age") delta = `<div class="k-delta" style="color:var(--muted)">baseline</div>`;
    return `<div class="kpi" style="--c:${POLICIES[p].color}"><div class="k-name">${POLICIES[p].short} · avg error</div><div class="k-val">${d.running_rmse.toFixed(3)}<small>${u}</small></div>${delta}</div>`;
  }).join("");
  for (const p of state.policies) {
    const d = ev.policies[p];
    $(`#stats-${p}`).innerHTML = `<span>now <b>${d.rmse.toFixed(2)}${u}</b></span><span>AoI <b>${d.mean_age.toFixed(1)}</b></span><span><b>${d.decision_ms.toFixed(1)}</b> ms</span>`;
    $(`#badge-${p}`).innerHTML = `polled <b>${d.polled.length}</b>${d.forced.length ? ` · <span style="color:${FORCED}">${d.forced.length} forced</span>` : ""}`;
  }
  $("#badge-truth").innerHTML = `<b>${state.meta.title}</b> · ${state.meta.where}`;
}

function renderExplainTabs() {
  const box = $("#explain-tabs");
  box.innerHTML = state.policies.map((p) => `<button data-p="${p}" class="${p === state.explain ? "on" : ""}">${POLICIES[p].short}</button>`).join("");
  box.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    state.explain = b.dataset.p;
    renderExplainTabs();
    if (state.last) renderDecision(state.last);
  }));
}

function renderDecision(ev) {
  const p = state.explain, d = ev.policies[p], m = state.meta;
  const order = new Map(d.polled.map((s, k) => [s, k + 1]));
  const forced = new Set(d.forced);
  const prio = d.priority || d.age.map(() => 0);
  const ids = m.sensors.map((s) => s.id);
  let rows = prio.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, 10);
  for (const s of d.polled) if (!rows.some((r) => r.i === s)) rows.push({ i: s, v: prio[s] });
  rows = rows.sort((a, b) => b.v - a.v).slice(0, 12);
  const maxv = Math.max(1e-9, ...rows.map((r) => r.v));
  const chosen = d.polled.length - d.forced.length;
  $("#decision-summary").innerHTML =
    `<b style="color:${POLICIES[p].color}">${POLICIES[p].short}</b> polled <b>${d.polled.length}</b> of ${m.M} sensors at ${ev.time.split(" ")[1]}: ` +
    (d.forced.length ? `<b style="color:${FORCED}">${d.forced.length} forced</b> (silent for ${m.freshness} slots), ` : "") +
    `<b>${chosen}</b> chosen. <span class="hint">${PRIORITY_MEANING[p]}</span>`;
  $("#decision").innerHTML = rows.map((r, k) => {
    const picked = order.has(r.i);
    const tag = forced.has(r.i) ? `<span class="tag force">forced</span>` : picked ? `<span class="tag pick">pick #${order.get(r.i)}</span>` : `<span></span>`;
    return `<div class="drow ${picked ? "picked" : ""}" style="--c:${POLICIES[p].color}">
      <span class="rank">${k + 1}</span><span class="sid">#${ids[r.i]}</span>
      <span class="bar"><i style="width:${Math.max(2, (r.v / maxv) * 100)}%"></i></span>${tag}</div>`;
  }).join("");
}

function addLog(ev) {
  const ids = state.meta.sensors.map((s) => s.id);
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<div class="t">${ev.time.split(" ")[1]}</div><div class="lines">${state.policies.map((p) => {
    const d = ev.policies[p], f = new Set(d.forced);
    const pills = d.polled.slice(0, 14).map((s) => `<span class="pill ${f.has(s) ? "forced" : ""}" style="--c:${POLICIES[p].color}">${ids[s]}</span>`).join("");
    const more = d.polled.length > 14 ? `<span class="hint">+${d.polled.length - 14}</span>` : "";
    return `<div class="log-line" style="--c:${POLICIES[p].color}"><span class="who"></span>${pills}${more}</div>`;
  }).join("")}</div>`;
  const log = $("#log");
  log.prepend(entry);
  while (log.children.length > 10) log.lastChild.remove();
}

function sensorTip(i, e) {
  if (i < 0 || !state.last) { tip.hide(); return; }
  const ev = state.last, m = state.meta, u = m.units;
  const s = m.sensors[i];
  const rows = state.policies.map((p) => {
    const d = ev.policies[p];
    const err = d.estimate[i] - ev.truth[i];
    return `<div class="r"><span style="color:${POLICIES[p].color}">${POLICIES[p].short}</span><b>${d.estimate[i].toFixed(2)} ± ${d.uncertainty[i].toFixed(2)}</b></div>
      <div class="r"><span>&nbsp;&nbsp;error · age</span><span>${err >= 0 ? "+" : ""}${err.toFixed(2)}${u} · ${d.age[i]} slots</span></div>`;
  }).join("");
  tip.show(`<h4>Sensor #${s.id}</h4><div class="r"><span>Measured</span><b>${ev.truth[i].toFixed(2)} ${u}</b></div>${rows}`, e.clientX, e.clientY);
}

function renderLegend() {
  const m = state.meta;
  if (!m) return;
  const [lo, mid, hi] = m.value_range;
  const half = Math.max(hi - mid, mid - lo);
  let scale;
  if (state.mode === "value") {
    const left = m.dataset === "pems_bay" ? mid + half : mid - half, right = m.dataset === "pems_bay" ? mid - half : mid + half;
    const fn = m.dataset === "pems_bay" ? (t) => DIVERGING(1 - t) : DIVERGING;
    scale = `<span>${left.toFixed(0)} ${m.units}</span><span class="bar" style="background:${gradientCss(fn)}"></span><span>${right.toFixed(0)} ${m.units}</span><span class="hint">${m.quantity}</span>`;
  } else {
    const fn = state.mode === "error" ? ERROR : UNCERTAINTY;
    scale = `<span>0</span><span class="bar" style="background:${gradientCss(fn)}"></span><span>≥ ${errScale().toFixed(1)} ${m.units}</span><span class="hint">${state.mode === "error" ? "|sink estimate − measured|" : "sink's uncertainty (±1σ)"}</span>`;
  }
  const syms = state.policies.map((p) => `<span class="sym" style="--c:${POLICIES[p].color}"><i></i>${POLICIES[p].short} transmission</span>`).join("");
  $("#legend").innerHTML = `<span class="scale">${scale}</span>${syms}<span class="sym" style="--c:${FORCED}"><i></i>forced (freshness rule)</span><span class="sym" style="--c:#fff"><i style="box-shadow:none;border:2px solid #3987e5;background:#0e0e0e"></i>sink (base station)</span>`;
}

function fmtDur(min) {
  if (min < 60) return `${min} min`;
  const h = min / 60;
  return h < 48 ? `${h % 1 ? h.toFixed(1) : h} h` : `${(h / 24).toFixed(1)} days`;
}

// ------------------------------------------------------------------ animation
function frame(now) {
  for (const m of Object.values(state.maps)) m.draw(now);
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ views
let resultsLoaded = false;
function showView(v) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  document.querySelectorAll(".view").forEach((x) => x.classList.toggle("active", x.id === `view-${v}`));
  if (v !== "live") pause();
  if (v === "results" && !resultsLoaded) { resultsLoaded = true; loadResults(); }
}

async function loadResults() {
  let r;
  try { r = await api.results(); } catch (e) { toast(e.message); return; }
  const rm = r.real_main || {}, sig = r.significance || {};
  const pct = (a, b) => ((a - b) / b) * 100;
  const I = rm.intel_lab?.results, P = rm.pems_bay?.results;
  const kpis = [];
  if (I) kpis.push({ v: `${pct(I.gnn.rmse_units[0], I.max_age.rmse_units[0]).toFixed(0)}%`, l: "temperature error vs classic Max-AoI", s: `Intel Lab · ${rm.intel_lab.scenario.M} real sensors` });
  if (sig.intel_lab) kpis.push({ v: `${(sig.intel_lab.kalman_greedy.gnn_win_rate * 100).toFixed(0)}%`, l: "of test windows better than Kalman-greedy", s: `Intel Lab · p = ${sig.intel_lab.kalman_greedy.wilcoxon_p.toPrecision(1)}` });
  if (P) kpis.push({ v: `${rm.pems_bay.scenario.M}`, l: "real freeway sensors scheduled live", s: `PEMS-BAY · ${pct(P.gnn.rmse_units[0], P.max_age.rmse_units[0]).toFixed(1)}% error vs Max-AoI` });
  if (sig.pems_bay) kpis.push({ v: `${(sig.pems_bay.kalman_greedy.gnn_win_rate * 100).toFixed(0)}%`, l: "of test windows better than Kalman-greedy", s: `PEMS-BAY · p = ${sig.pems_bay.kalman_greedy.wilcoxon_p.toPrecision(1)}` });
  $("#big-kpis").innerHTML = kpis.map((k) => `<div class="big-kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div><div class="s">${k.s}</div></div>`).join("");

  const order = ["max_age", "random", "max_variance", "kalman_greedy", "greedy_diverse", "gnn"];
  const cell = (R, n, u) => {
    if (!R?.[n]) return "<td>—</td>";
    const v = R[n].rmse_units[0], d = pct(v, R.max_age.rmse_units[0]);
    return `<td>${v.toFixed(3)} ${u}${n === "max_age" ? "" : `<span class="d">${d > 0 ? "+" : ""}${d.toFixed(1)}%</span>`}</td>`;
  };
  $("#results-table").innerHTML = `<table><thead><tr><th>Scheduler</th><th>Intel Lab · 45 sensors (°C)</th><th>PEMS-BAY · 325 sensors (mph)</th></tr></thead><tbody>${
    order.map((n) => `<tr class="${n === "gnn" ? "ours" : ""}"><td><span class="sq" style="--c:${POLICIES[n].color}"></span>${POLICIES[n].label}</td>${cell(I, n, "°C")}${cell(P, n, "mph")}</tr>`).join("")
  }</tbody></table><p class="hint">RMSE of the sink's estimate across all sensors, averaged over held-out test windows (Intel: ${I?.gnn.n_windows ?? "?"} windows, PEMS-BAY: ${P?.gnn.n_windows ?? "?"}). Percentages are relative to Max-AoI.</p>`;

  const figs = [
    ["real_benchmark", "Real data: every scheduler on the same held-out test days.", true],
    ["poll_map", "Where each scheduler spends its transmissions on the real lab floor.", true],
    ["real_sweeps", "Real data: gain vs channel capacity and network size.", true],
    ["main_benchmark", "Synthetic benchmark (50 sensors, 30 unseen layouts)."],
    ["aoi_vs_error", "Fresher is not the same as better-informed."],
    ["sweeps", "Synthetic sweeps: capacity, correlation, size, interference.", true],
    ["training", "PPO training and held-out validation."],
    ["decision_time", "Decision time per slot."],
  ];
  $("#gallery").innerHTML = figs.map(([f, c, wide]) =>
    `<figure class="card figure ${wide ? "wide" : ""}"><img src="/results/figures/${f}.png" alt="${c}" loading="lazy"><figcaption>${c}</figcaption></figure>`).join("");
}

init();
