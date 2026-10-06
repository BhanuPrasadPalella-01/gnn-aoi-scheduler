"""Static figures. Each reads results/data/*.json and writes results/figures/*.png."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn_aoi.viz import style
from gnn_aoi.viz.style import INK, INK_2, LABELS, MUTED, POLICY_COLORS as COL, POLICY_ORDER, SURFACE

DATA = Path("results/data")
FIGS = Path("results/figures")


def _load(name):
    p = DATA / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"  wrote results/figures/{name}.png")


def _header(fig, title, sub=None):
    """Title + subtitle above the axes, spaced in inches so it works at any figure height."""
    H = fig.get_figheight()
    if sub:
        fig.text(0.01, 1 + 0.12 / H, sub, va="bottom", ha="left", color=MUTED, fontsize=9.5)
    fig.text(0.01, 1 + (0.42 if sub else 0.12) / H, title, va="bottom", ha="left",
             color=INK, fontsize=15, fontweight="bold")


def _bars(ax, results, title, metric="mse", xlabel="Sink estimation error (MSE, lower is better)", fmt="{:.4f}"):
    names = [n for n in POLICY_ORDER if n in results]
    names.sort(key=lambda n: -results[n][metric][0])
    base = results["max_age"][metric][0]
    y = np.arange(len(names))
    for i, n in enumerate(names):
        m, ci = results[n][metric]
        ax.barh(i, m, height=0.62, color=COL[n], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.errorbar(m, i, xerr=ci, color=INK_2, capsize=3, lw=1.2, zorder=4)
        delta = (m - base) / base * 100
        tag = "" if n == "max_age" else f"  {delta:+.0f}% vs B1"
        ax.text(m + ci + base * 0.02, i, fmt.format(m) + tag, va="center", fontsize=9,
                color=INK if n == "gnn" else INK_2, fontweight="bold" if n == "gnn" else None)
    ax.set_yticks(y, [LABELS[n] for n in names])
    for lbl, n in zip(ax.get_yticklabels(), names):
        lbl.set_color(INK if n == "gnn" else INK_2)
        if n == "gnn":
            lbl.set_fontweight("bold")
    ax.set_xlim(0, max(results[n][metric][0] for n in names) * 1.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def fig_main():
    d = _load("main")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, key in zip(axes, ("uniform", "clustered")):
        sc = d[key]["scenario"]
        _bars(ax, d[key]["results"], f"{key.title()} layouts · M={sc['M']}, C={sc['C']}, AoI ≤ {sc['freshness']}")
    fig.tight_layout()
    sc = d["uniform"]["scenario"]
    _header(fig, "Synthetic benchmark: same freshness guarantee, far better information",
            f"Mean ± 95% CI over 30 held-out sensor layouts × 350 slots. Every policy keeps every sensor's age ≤ {sc['freshness']}.")
    _save(fig, "main_benchmark")


def fig_tradeoff():
    d = _load("main")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, key in zip(axes, ("uniform", "clustered")):
        res = d[key]["results"]
        for n in POLICY_ORDER:
            if n not in res:
                continue
            x, y = res[n]["mean_age"][0], res[n]["mse"][0]
            ax.scatter(x, y, s=150 if n == "gnn" else 80, color=COL[n], edgecolor=SURFACE, lw=2,
                       zorder=4 if n == "gnn" else 3, label=LABELS[n])
            if n in ("gnn", "max_age"):
                ax.annotate(LABELS[n], (x, y), xytext=(-10, -16) if n == "gnn" else (8, 4),
                            textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold" if n == "gnn" else None)
        ax.set_xlabel("Mean Age of Information (slots)")
        ax.set_ylabel("Estimation MSE")
        ax.set_title(f"{key.title()} layouts")
        ax.margins(x=0.25, y=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=9.5)
    _header(fig, "Fresher is not the same as better-informed",
            "Max-AoI has the lowest age but high error: it keeps re-polling sensors whose neighbours already tell the sink what they read.")
    _save(fig, "aoi_vs_error")


def _sweep_ax(ax, d, xlabel, title, logx=False, zero_shot_from=None, metric="mse", ylabel=None):
    rows = d["rows"]
    xs = [r["value"] for r in rows]
    for n in ("greedy_diverse", "kalman_greedy", "oracle", "gnn_no_mp", "gnn"):
        if n not in rows[0]["results"]:
            continue
        rel = []
        for r in rows:
            b = np.array(r["results"]["max_age"][f"{metric}_per_layout"])
            p = np.array(r["results"][n][f"{metric}_per_layout"])
            v = (b - p) / b * 100
            rel.append((v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v))))
        m, ci = np.array(rel).T
        ax.plot(xs, m, color=COL[n], lw=2.6 if n == "gnn" else 1.6, marker="o", ms=5 if n == "gnn" else 3.5,
                ls={"oracle": "--", "gnn_no_mp": ":"}.get(n, "-"), label=LABELS[n], zorder=3 if n == "gnn" else 2)
        ax.fill_between(xs, m - ci, m + ci, color=COL[n], alpha=0.12, lw=0)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    if zero_shot_from is not None:
        ax.axvspan(zero_shot_from, max(xs) * 1.05, color="#ffffff", alpha=0.04, lw=0)
        ax.text(zero_shot_from, ax.get_ylim()[1], "  zero-shot (unseen size)", va="top", fontsize=8.5, color=MUTED)
    if logx:
        ax.set_xscale("log")
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or "MSE reduction vs Max-AoI (%)")
    ax.set_title(title)


def fig_sweeps():
    specs = [("capacity", "Channel capacity C (polls/slot), M=50", "Capacity", False, None),
             ("correlation", "Correlation length scale", "Correlation strength", False, None),
             ("scale", "Network size M (C = M/5)", "Generalisation to larger networks", True, 110),
             ("interference", "Interference radius", "Interference constraints", False, None)]
    avail = [s for s in specs if _load(s[0])]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, xl, title, logx, zs) in zip(axes, avail):
        _sweep_ax(ax, _load(name), xl, title, logx, zs)
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=10)
    _header(fig, "Synthetic sweeps: where correlation-aware scheduling pays off",
            "MSE reduction relative to Max-AoI (B1) on the same layouts; bands = 95% CI over 20 held-out layouts (12 for the size sweep). "
             "Above 0 = better than B1.")
    _save(fig, "sweeps")


def fig_compute():
    d = _load("real_scale") or _load("scale")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = [r["value"] for r in d["rows"]]
    for n in ("max_age", "kalman_greedy", "gnn"):
        ys = [r["results"][n]["ms_per_slot"][0] for r in d["rows"]]
        ax.plot(xs, ys, marker="o", color=COL[n], label=LABELS[n])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs, [str(x) for x in xs]); ax.minorticks_off()
    ax.set_xlabel("Network size M"); ax.set_ylabel("Decision time per slot (ms, CPU)")
    ax.set_title("Scheduling cost per 5-minute slot (PEMS-BAY sub-networks)")
    ax.legend(fontsize=9)
    _save(fig, "decision_time")


def fig_training():
    d = _load("training_log")
    if not d:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ppo = d["ppo"]
    it = [p["iter"] for p in ppo]
    r = np.array([p["mean_reward"] for p in ppo]) / (1 - d["config"]["gamma"])
    ax = axes[0]
    ax.plot(it, r, color=COL["gnn"], alpha=0.25, lw=1)
    k = 15
    if len(r) > k:
        ax.plot(it[k - 1:], np.convolve(r, np.ones(k) / k, "valid"), color=COL["gnn"], lw=2.2)
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("−log posterior variance (higher is better)")
    ax.set_title("Training reward (randomised scenarios)")
    ax = axes[1]
    val = [v for v in d["val"]]
    ax.plot([v["iter"] for v in val], [(1 - v["ratio_vs_teacher"]) * 100 for v in val],
            color=COL["gnn"], marker="o", ms=4)
    ax.axhline(0, color=COL["kalman_greedy"], ls="--", lw=1.4)
    ax.text(ax.get_xlim()[0], 0, " Kalman-greedy (starting point)", va="bottom", fontsize=9, color=INK_2)
    b = d.get("best", {})
    if b:
        ax.scatter([b["iter"]], [(1 - b["score"]) * 100], s=140, facecolor="none", edgecolor=INK, lw=1.5, zorder=4)
        ax.annotate("kept checkpoint", (b["iter"], (1 - b["score"]) * 100), xytext=(8, -14),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Error reduction vs Kalman-greedy (%)")
    ax.set_title("Held-out validation")
    fig.tight_layout()
    _save(fig, "training")


def fig_pilot():
    d = _load("pilot")
    if not d:
        return
    rows = d["rows"]
    xs = [r["value"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xs, [r["corr_mae"] for r in rows], marker="o", color=INK_2)
    axes[0].set_xscale("log"); axes[0].set_xticks(xs, [str(x) for x in xs]); axes[0].minorticks_off()
    axes[0].set_xlabel("Pilot length (slots of full observation)")
    axes[0].set_ylabel("Mean |error| of estimated correlation")
    axes[0].set_title("Learning the correlation graph")
    for n in ("max_age", "kalman_greedy", "gnn", "oracle"):
        m = np.array([r["results"][n]["mse"] for r in rows])
        axes[1].plot(xs, m[:, 0], marker="o", color=COL[n], label=LABELS[n], ls="--" if n == "oracle" else "-")
        axes[1].fill_between(xs, m[:, 0] - m[:, 1], m[:, 0] + m[:, 1], color=COL[n], alpha=0.12, lw=0)
    axes[1].set_xscale("log"); axes[1].set_xticks(xs, [str(x) for x in xs]); axes[1].minorticks_off()
    axes[1].set_xlabel("Pilot length"); axes[1].set_ylabel("Estimation MSE")
    axes[1].set_title("…and what a rough graph costs the scheduler")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "pilot_length")


def fig_poll_map():
    d = _load("poll_frequency")
    if not d or "meta" not in d:
        return
    meta = d["meta"]
    pos = np.array([[x["raw_x"], x["raw_y"]] for x in meta["sensors"]])
    M = len(pos)
    strong = {}
    for i, j, w in meta["edges"]:
        strong.setdefault(i, []).append((w, j)); strong.setdefault(j, []).append((w, i))
    fig, axes = plt.subplots(1, len(d["policies"]), figsize=(5.2 * len(d["policies"]), 4.6))
    for ax, (n, p) in zip(axes, d["policies"].items()):
        for i in range(M):
            for w, j in sorted(strong.get(i, []), reverse=True)[:2]:
                ax.plot(*pos[[i, j]].T, color=INK_2, alpha=0.18, lw=0.8, zorder=1)
        share = np.array(p["share"])
        ax.scatter(*pos.T, s=12 + 1800 * share, color=COL[n], alpha=0.8, edgecolor=SURFACE, lw=1.2, zorder=3)
        ax.set_title(f"{p['label']} · error {p['rmse']:.2f} {meta['units']}")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.margins(0.06)
    fig.tight_layout()
    _header(fig, "Where each scheduler spends the channel on the real lab floor",
            f"Intel Berkeley Lab, {meta['M']} temperature sensors at their real positions, one test day ({meta['start_time'][:10]}), "
            f"{meta['C']} polls per 5-minute slot. Bubble area = share of slots polled; lines = strongest learned correlations.")
    _save(fig, "poll_map")


def fig_real_main():
    d = _load("real_main")
    if not d:
        return
    fig, axes = plt.subplots(1, len(d), figsize=(7 * len(d), 4.6))
    for ax, (name, block) in zip(np.atleast_1d(axes), d.items()):
        sc, info = block["scenario"], block["info"]
        _bars(ax, block["results"], f"{info['title']} · {sc['M']} sensors, {sc['C']} polls/slot, AoI ≤ {sc['freshness']}",
              metric="rmse_units", xlabel=f"{info['quantity']} error at the sink (RMSE, {info['units']}, lower is better)",
              fmt="{:.3f}")
    fig.tight_layout()
    a, b = d["intel_lab"], d["pems_bay"]
    _header(fig, "Real sensor data: large gains on the lab floor, small ones on the freeways",
            f"Sink fitted on the training period only; mean ± 95% CI over held-out test windows "
            f"(Intel {a['test_period'][0]} → {a['test_period'][1]}, PEMS-BAY {b['test_period'][0]} → {b['test_period'][1]}).")
    _save(fig, "real_benchmark")


def fig_real_sweeps():
    specs = [("real_capacity_intel_lab", "Polls per slot C (45 sensors)", "Intel Lab: capacity", False),
             ("real_capacity_pems_bay", "Polls per slot C (325 sensors)", "PEMS-BAY: capacity", False),
             ("real_scale", "Sensors in the network M (C = M/8)", "PEMS-BAY: network size", True)]
    avail = [s for s in specs if _load(s[0])]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5.4 * len(avail), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, xl, title, logx) in zip(axes, avail):
        _sweep_ax(ax, _load(name), xl, title, logx, metric="rmse_units", ylabel="Error reduction vs Max-AoI (%)")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=10)
    _header(fig, "Real data: how the gain changes with capacity and network size",
            "RMSE reduction relative to Max-AoI on the same held-out test windows; bands = 95% CI. Above 0 = better than B1.")
    _save(fig, "real_sweeps")


def all_figures():
    style.apply()
    for f in (fig_real_main, fig_real_sweeps, fig_poll_map, fig_main, fig_tradeoff, fig_sweeps,
              fig_compute, fig_training, fig_pilot):
        f()
