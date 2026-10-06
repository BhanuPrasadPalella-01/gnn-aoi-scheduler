// Colour system shared by every view. Policy colours match the report figures.

export const POLICIES = {
  gnn: { label: "GNN-PPO (ours)", short: "GNN", color: "#3987e5" },
  max_age: { label: "Max-AoI (classic)", short: "Max-AoI", color: "#d95926" },
  kalman_greedy: { label: "Kalman-greedy", short: "Kalman", color: "#199e70" },
  greedy_diverse: { label: "Corr-diverse (v4)", short: "v4", color: "#c98500" },
  max_variance: { label: "Max-uncertainty", short: "Max-unc", color: "#008300" },
  random: { label: "Random", short: "Random", color: "#9085e9" },
};
export const FORCED = "#fab219";

const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function ramp(stops) {
  const cs = stops.map(hex);
  return (t) => {
    t = Math.min(1, Math.max(0, t)) * (cs.length - 1);
    const i = Math.min(cs.length - 2, Math.floor(t));
    const f = t - i;
    return [0, 1, 2].map((k) => Math.round(cs[i][k] + (cs[i + 1][k] - cs[i][k]) * f));
  };
}

// diverging: cool blue -> neutral -> warm red (reversed for traffic: slow = red)
export const DIVERGING = ramp(["#1c5cab", "#3987e5", "#86b6ef", "#4a4a47", "#ec835a", "#e34948", "#a8262b"]);
// sequential: how wrong / how unsure the sink is
export const ERROR = ramp(["#232322", "#5a2a14", "#d95926", "#fab219", "#fff2c4"]);
export const UNCERTAINTY = ramp(["#232322", "#104281", "#2a78d6", "#86b6ef", "#e8f1fc"]);

export const css = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;

export function gradientCss(fn, n = 12) {
  const stops = Array.from({ length: n }, (_, i) => `${css(fn(i / (n - 1)))} ${(i / (n - 1)) * 100}%`);
  return `linear-gradient(90deg, ${stops.join(",")})`;
}
