// One live map: an interpolated field, the correlation graph, sensors, the
// sink, and transmission packets flying from polled sensors to the sink.

import { css, FORCED } from "./colors.js";

const GRID_W = 112;

export class FieldMap {
  /**
   * @param {HTMLElement} wrap   element the canvas fills
   * @param {object} meta        session meta from the server
   * @param {object} opts        {color: packet colour or null, onHover(i, ev)}
   */
  constructor(wrap, meta, opts = {}) {
    this.wrap = wrap;
    this.meta = meta;
    this.color = opts.color || null;
    this.onHover = opts.onHover || (() => {});
    this.canvas = document.createElement("canvas");
    wrap.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");

    const xs = meta.sensors.map((s) => s.x), ys = meta.sensors.map((s) => s.y);
    this.pos = meta.sensors.map((s) => [s.x, s.y]);
    this.M = this.pos.length;
    const pad = 0.06;
    this.box = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
    const bw = this.box.x1 - this.box.x0, bh = this.box.y1 - this.box.y0;
    this.box.x0 -= bw * pad; this.box.x1 += bw * pad; this.box.y0 -= bh * pad; this.box.y1 += bh * pad;
    this.aspect = (this.box.x1 - this.box.x0) / (this.box.y1 - this.box.y0);
    wrap.style.setProperty("--aspect", this.aspect.toFixed(3));

    this.sink = [xs.reduce((a, b) => a + b, 0) / this.M, ys.reduce((a, b) => a + b, 0) / this.M];
    this.radius = Math.max(2.6, Math.min(6.5, 9 - this.M / 45));
    this.edges = strongestEdges(meta.edges, this.M, 2);
    this._buildGrid();

    this.colors = this.prevColors = null;
    this.colorT0 = 0;
    this.packets = [];
    this.flashes = [];
    this.sinkPulse = [];
    this.hover = -1;

    this.field = document.createElement("canvas");
    this.field.width = this.gw; this.field.height = this.gh;
    this.fctx = this.field.getContext("2d");
    this.fimg = this.fctx.createImageData(this.gw, this.gh);

    new ResizeObserver(() => this._resize()).observe(wrap);
    this._resize();
    this.canvas.addEventListener("mousemove", (e) => this._mouse(e));
    this.canvas.addEventListener("mouseleave", () => { this.hover = -1; this.onHover(-1, null); });
  }

  // --- geometry -----------------------------------------------------------
  _resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = this.wrap.getBoundingClientRect();
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.fimg) this.draw(performance.now()); // resizing clears the canvas; don't wait for the next frame
  }

  px([x, y]) {
    const b = this.box;
    return [((x - b.x0) / (b.x1 - b.x0)) * this.w, (1 - (y - b.y0) / (b.y1 - b.y0)) * this.h];
  }

  _buildGrid() {
    // Gaussian kernel smoothing weights, precomputed per grid cell (sparse).
    const nn = this.pos.map((p, i) => {
      let best = Infinity;
      this.pos.forEach((q, j) => { if (i !== j) best = Math.min(best, Math.hypot(p[0] - q[0], p[1] - q[1])); });
      return best;
    }).sort((a, b) => a - b);
    const h = Math.max(1e-3, 1.8 * nn[Math.floor(nn.length / 2)]);
    const cut2 = (3 * h) ** 2, inv = 1 / (2 * h * h);
    this.gw = GRID_W;
    this.gh = Math.max(24, Math.round(GRID_W / this.aspect));
    const b = this.box;
    this.cells = [];
    for (let gy = 0; gy < this.gh; gy++) {
      for (let gx = 0; gx < this.gw; gx++) {
        const x = b.x0 + ((gx + 0.5) / this.gw) * (b.x1 - b.x0);
        const y = b.y1 - ((gy + 0.5) / this.gh) * (b.y1 - b.y0);
        const idx = [], w = [];
        this.pos.forEach(([sx, sy], i) => {
          const d2 = (sx - x) ** 2 + (sy - y) ** 2;
          if (d2 < cut2) { idx.push(i); w.push(Math.exp(-d2 * inv)); }
        });
        this.cells.push({ idx, w });
      }
    }
  }

  // --- data ---------------------------------------------------------------
  /** t: per-sensor colour parameter in [0,1]; colorFn maps it to rgb. */
  setValues(t, colorFn, now) {
    this.prevColors = this.colors ? this._currentColors(now) : null;
    this.colors = t.map(colorFn);
    this.colorT0 = now;
    const data = this.fimg.data;
    this.cells.forEach((c, k) => {
      let num = 0, den = 0;
      for (let j = 0; j < c.idx.length; j++) { num += c.w[j] * t[c.idx[j]]; den += c.w[j]; }
      const o = k * 4;
      if (den < 1e-4) { data[o + 3] = 0; return; }
      const [r, g, bb] = colorFn(num / den);
      data[o] = r; data[o + 1] = g; data[o + 2] = bb;
      data[o + 3] = Math.round(255 * (1 - Math.exp(-den * 1.6)) * 0.92);
    });
    this.fctx.putImageData(this.fimg, 0, 0);
  }

  _currentColors(now) {
    if (!this.prevColors) return this.colors;
    const f = Math.min(1, (now - this.colorT0) / 350);
    if (f >= 1) return this.colors;
    return this.colors.map((c, i) => c.map((v, k) => Math.round(this.prevColors[i][k] + (v - this.prevColors[i][k]) * f)));
  }

  /** Launch one packet per polled sensor. */
  transmit(polled, forced, now, duration) {
    const fset = new Set(forced);
    polled.forEach((i, k) => {
      const delay = k * Math.min(70, duration * 0.08);
      this.packets.push({ i, forced: fset.has(i), t0: now + delay, dur: duration, bend: (i % 2 ? 1 : -1) * (0.18 + 0.1 * ((i * 7) % 5) / 5) });
      this.flashes.push({ i, t0: now + delay, forced: fset.has(i) });
    });
  }

  // --- drawing ------------------------------------------------------------
  draw(now) {
    const { ctx, w, h } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#121212";
    ctx.fillRect(0, 0, w, h);

    // field
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.globalAlpha = 0.9;
    ctx.drawImage(this.field, 0, 0, w, h);
    ctx.restore();

    // subtle grid
    ctx.strokeStyle = "rgba(255,255,255,0.025)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < w; gx += 28) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke(); }
    for (let gy = 0; gy < h; gy += 28) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke(); }

    // correlation links
    ctx.lineWidth = 1;
    for (const [i, j, wgt] of this.edges) {
      const [ax, ay] = this.px(this.pos[i]), [bx, by] = this.px(this.pos[j]);
      ctx.strokeStyle = `rgba(255,255,255,${0.05 + 0.13 * wgt})`;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }

    const sink = this.px(this.sink);
    if (this.color) this._drawSink(sink, now);

    // sensors
    const cols = this.colors ? this._currentColors(now) : null;
    const r = this.radius;
    for (let i = 0; i < this.M; i++) {
      const [x, y] = this.px(this.pos[i]);
      ctx.beginPath(); ctx.arc(x, y, r + 1.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(10,10,10,0.85)"; ctx.fill();
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = cols ? css(cols[i]) : "#555"; ctx.fill();
      if (i === this.hover) {
        ctx.beginPath(); ctx.arc(x, y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }

    // emission flashes
    this.flashes = this.flashes.filter((f) => now - f.t0 < 700);
    for (const f of this.flashes) {
      const k = (now - f.t0) / 700;
      if (k < 0) continue;
      const [x, y] = this.px(this.pos[f.i]);
      ctx.beginPath(); ctx.arc(x, y, r + 3 + k * 16, 0, Math.PI * 2);
      ctx.strokeStyle = f.forced ? `rgba(250,178,25,${1 - k})` : hexA(this.color, 1 - k);
      ctx.lineWidth = 2; ctx.stroke();
    }

    // packets
    this.packets = this.packets.filter((p) => {
      const k = (now - p.t0) / p.dur;
      if (k >= 1) { this.sinkPulse.push(now); return false; }
      return true;
    });
    for (const p of this.packets) {
      const k = (now - p.t0) / p.dur;
      if (k < 0) continue;
      const a = this.px(this.pos[p.i]);
      const c = ctrl(a, sink, p.bend);
      const col = p.forced ? FORCED : this.color;
      // faint path
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.quadraticCurveTo(c[0], c[1], sink[0], sink[1]);
      ctx.strokeStyle = hexA(col, 0.12 * (1 - k)); ctx.lineWidth = 1; ctx.stroke();
      // trail + head
      for (let s = 6; s >= 0; s--) {
        const kk = ease(Math.max(0, k - s * 0.025));
        const [x, y] = quad(a, c, sink, kk);
        ctx.beginPath(); ctx.arc(x, y, s === 0 ? 3.4 : 2.6 - s * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = hexA(col, s === 0 ? 1 : 0.5 - s * 0.07);
        ctx.fill();
      }
      const [hx, hy] = quad(a, c, sink, ease(k));
      const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 12);
      g.addColorStop(0, hexA(col, 0.55)); g.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(hx, hy, 12, 0, Math.PI * 2); ctx.fill();
    }
  }

  _drawSink([x, y], now) {
    const ctx = this.ctx;
    this.sinkPulse = this.sinkPulse.filter((t) => now - t < 600);
    for (const t of this.sinkPulse) {
      const k = (now - t) / 600;
      ctx.beginPath(); ctx.arc(x, y, 9 + k * 22, 0, Math.PI * 2);
      ctx.strokeStyle = hexA(this.color, 0.5 * (1 - k)); ctx.lineWidth = 1.5; ctx.stroke();
    }
    const breathe = 0.5 + 0.5 * Math.sin(now / 600);
    const g = ctx.createRadialGradient(x, y, 0, x, y, 26);
    g.addColorStop(0, hexA(this.color, 0.35 + 0.15 * breathe)); g.addColorStop(1, hexA(this.color, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 26, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#0e0e0e"; ctx.fill();
    ctx.strokeStyle = this.color; ctx.lineWidth = 2; ctx.stroke();
    // antenna glyph
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(x, y + 4); ctx.lineTo(x, y - 3); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y - 3, 3.2, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
  }

  _mouse(e) {
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < this.M; i++) {
      const [x, y] = this.px(this.pos[i]);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    this.hover = best;
    this.onHover(best, e);
  }
}

function strongestEdges(edges, M, k) {
  const by = Array.from({ length: M }, () => []);
  for (const e of edges) { by[e[0]].push(e); by[e[1]].push(e); }
  const keep = new Set();
  by.forEach((list) => list.sort((a, b) => b[2] - a[2]).slice(0, k).forEach((e) => keep.add(e)));
  return [...keep];
}

const ease = (t) => 1 - (1 - t) ** 3;

function ctrl([ax, ay], [bx, by], bend) {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  return [mx - (by - ay) * bend, my + (bx - ax) * bend];
}

function quad(a, c, b, t) {
  const u = 1 - t;
  return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]];
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${Math.max(0, a)})`;
}
