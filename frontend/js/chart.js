// Live line chart of the sink's error: thin = this slot, thick = running average.

import { POLICIES } from "./colors.js";

const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};

export class ErrorChart {
  constructor(container, tooltip) {
    this.container = container;
    this.tooltip = tooltip;
    this.svg = el("svg", { role: "img", "aria-label": "Sink error over time" }, container);
    this.svg.addEventListener("mousemove", (e) => this._hover(e));
    this.svg.addEventListener("mouseleave", () => { this.hoverX = null; this.tooltip.hide(); this.render(); });
    new ResizeObserver(() => this.render()).observe(container);
    this.reset([], "");
  }

  reset(names, units) {
    this.names = names;
    this.units = units;
    this.times = [];
    this.now = Object.fromEntries(names.map((n) => [n, []]));
    this.run = Object.fromEntries(names.map((n) => [n, []]));
    this.hoverX = null;
    this.render();
  }

  push(time, event) {
    this.times.push(time);
    for (const n of this.names) {
      const p = event.policies[n];
      this.now[n].push(p.rmse);
      this.run[n].push(p.running_rmse);
    }
    this.render();
  }

  render() {
    const svg = this.svg;
    svg.replaceChildren();
    const W = this.container.clientWidth || 600, H = 250;
    const m = { l: 44, r: 112, t: 10, b: 26 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const n = this.times.length;
    const span = Math.max(48, n);
    let ymax = 0;
    for (const k of this.names) for (const v of this.now[k]) ymax = Math.max(ymax, v);
    ymax = niceMax(ymax || 1);
    const x = (i) => m.l + (i / (span - 1)) * (W - m.l - m.r);
    const y = (v) => H - m.b - (Math.min(v, ymax) / ymax) * (H - m.t - m.b);
    this._geom = { x, m, W, span };

    const grid = el("g", { class: "grid" }, svg), axis = el("g", { class: "axis" }, svg);
    for (let k = 0; k <= 4; k++) {
      const v = (ymax * k) / 4;
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
      el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, axis).textContent = fmt(v);
    }
    el("text", { x: m.l, y: H - 6 }, axis).textContent = this.times[0] || "";
    if (n > 1) el("text", { x: x(n - 1), y: H - 6, "text-anchor": "end" }, axis).textContent = this.times[n - 1];
    el("text", { x: 4, y: 12 }, axis).textContent = this.units;

    const labels = [];
    for (const k of this.names) {
      const c = POLICIES[k].color;
      if (n > 1) {
        el("path", { d: line(this.now[k], x, y), fill: "none", stroke: c, "stroke-opacity": 0.28, "stroke-width": 1.2 }, svg);
        el("path", { d: line(this.run[k], x, y), fill: "none", stroke: c, "stroke-width": k === "gnn" ? 2.8 : 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      }
      if (n) labels.push({ k, c, yv: y(this.run[k][n - 1]), v: this.run[k][n - 1] });
    }
    // direct labels at the line ends, nudged apart
    labels.sort((a, b) => a.yv - b.yv);
    for (let i = 1; i < labels.length; i++) labels[i].yv = Math.max(labels[i].yv, labels[i - 1].yv + 15);
    for (const L of labels) {
      el("circle", { cx: x(n - 1), cy: y(L.v), r: 3.5, fill: L.c, stroke: "#161616", "stroke-width": 2 }, svg);
      const t = el("text", { x: W - m.r + 10, y: L.yv + 4, class: "lbl", fill: "#fff" }, svg);
      t.textContent = `${POLICIES[L.k].short} ${fmt(L.v)}`;
    }
    if (this.hoverX != null && n) {
      const i = Math.min(n - 1, Math.max(0, this.hoverX));
      el("line", { x1: x(i), x2: x(i), y1: m.t, y2: H - m.b, stroke: "#c3c2b7", "stroke-opacity": 0.5 }, svg);
    }
  }

  _hover(e) {
    if (!this._geom || !this.times.length) return;
    const r = this.svg.getBoundingClientRect();
    const { m, W, span } = this._geom;
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((px - m.l) / (W - m.l - m.r)) * (span - 1));
    if (i < 0 || i >= this.times.length) { this.hoverX = null; this.tooltip.hide(); this.render(); return; }
    this.hoverX = i;
    this.render();
    const rows = this.names.map((k) =>
      `<div class="r"><span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${POLICIES[k].color};margin-right:6px"></i>${POLICIES[k].short}</span><b>${fmt(this.now[k][i])} ${this.units}</b></div>`).join("");
    this.tooltip.show(`<h4>${this.times[i]}</h4>${rows}<div class="r" style="margin-top:4px"><span>(error in this slot)</span></div>`, e.clientX, e.clientY);
  }
}

function line(vals, x, y) {
  return vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
}
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  for (const s of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (s * p >= v) return s * p;
  return 10 * p;
}
const fmt = (v) => (v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3));
