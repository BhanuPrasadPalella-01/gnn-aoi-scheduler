"""Hero image + GIF from results/data/replay.json (real Intel Lab data):
the measured temperature field next to what the sink believes under each
scheduler, slot by slot."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from gnn_aoi.simulator import rbf_covariance
from gnn_aoi.viz import style
from gnn_aoi.viz.style import DIVERGING, INK, INK_2, MUTED, POLICY_COLORS as COL, SURFACE

GRID_N = 90


def _interpolator(pos, ls):
    """Kriging weights mapping sensor values -> a dense grid (GP mean with
    the field's own kernel), so the picture shows the field, not just dots."""
    g = np.linspace(0, 1, GRID_N)
    gx, gy = np.meshgrid(g, g)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    K_ss = rbf_covariance(pos, ls, jitter=1e-3)
    d2 = ((grid[:, None] - pos[None]) ** 2).sum(-1)
    K_gs = np.exp(-d2 / (2 * ls**2))
    W = np.linalg.solve(K_ss, K_gs.T).T
    return lambda v: (W @ np.asarray(v)).reshape(GRID_N, GRID_N)


def _strongest_edges(edges, M, k=3):
    """Each sensor's k strongest correlation links: the structure without the hairball."""
    by_node = {i: [] for i in range(M)}
    for i, j, w in edges:
        by_node[i].append((w, i, j)); by_node[j].append((w, i, j))
    keep = {e for n in by_node.values() for e in sorted(n, reverse=True)[:k]}
    return [(i, j, w) for w, i, j in keep]


def _setup(d, names):
    from gnn_aoi.datasets import INFO
    pos = np.array(d["positions"])
    interp = _interpolator(pos, d.get("length_scale", 0.12))
    real = "dataset" in d
    vk, ek = ("truth_units", "estimate_units") if real else ("truth", "estimate")
    err_key = "rmse_units" if real else "mse"
    allv = np.array([s[vk] for s in d["policies"][names[0]]["slots"]])
    mid = float(np.median(allv))
    half = float(np.percentile(np.abs(allv - mid), 98)) or 1.0
    vmin, vmax = mid - half, mid + half
    norm = lambda v: (np.asarray(v) - vmin) / (vmax - vmin)
    base_interp = interp
    interp = lambda v: base_interp(np.asarray(v) - mid) + mid  # far from sensors -> typical value, not 0
    edges = _strongest_edges(d["corr_edges"], len(pos), k=3)
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.1], hspace=0.28, wspace=0.06)
    maps = [fig.add_subplot(gs[0, i]) for i in range(3)]
    strip = fig.add_subplot(gs[1, :])
    T = len(d["policies"][names[0]]["slots"])
    panels = []
    titles = ["Measured" if real else "Ground truth"] + [f"Sink's view · {d['policies'][n]['label']}" for n in names]
    for k, ax in enumerate(maps):
        im = ax.imshow(np.zeros((GRID_N, GRID_N)), origin="lower", extent=(0, 1, 0, 1),
                       cmap=DIVERGING, vmin=vmin, vmax=vmax, interpolation="bilinear")
        for i, j, w in edges:
            ax.plot(*pos[[i, j]].T, color=INK, alpha=0.06 + 0.12 * w, lw=0.7, zorder=2)
        dots = ax.scatter(*pos.T, s=30, color=INK, edgecolor=SURFACE, lw=1, zorder=3)
        rings = ax.scatter([], [], s=200, facecolor="none",
                           edgecolor=COL[names[k - 1]] if k else INK, lw=2.4, zorder=4)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(COL[names[k - 1]] if k else INK_2); sp.set_linewidth(1.6 if k else 1)
        ax.set_title(titles[k], fontsize=11.5, color=INK)
        txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, color=INK,
                      bbox=dict(facecolor=SURFACE, alpha=0.75, edgecolor="none", pad=3))
        panels.append((im, dots, rings, txt))

    lines = {}
    for n in names:
        err = [s[err_key] for s in d["policies"][n]["slots"]]
        strip.plot(err, color=COL[n], alpha=0.25, lw=1)
        lines[n], = strip.plot([], [], color=COL[n], lw=2.2, label=d["policies"][n]["label"])
    cursor = strip.axvline(0, color=INK_2, lw=1)
    strip.set_xlim(0, T - 1)
    strip.set_ylim(0, max(np.percentile([s[err_key] for s in d["policies"][n]["slots"][5:]], 99) for n in names) * 1.15)
    unit = d.get("units", "")
    strip.set_xlabel("Time slot (5 min each)" if real else "Time slot")
    strip.set_ylabel(f"Sink error ({unit})" if real else "Sink MSE")
    strip.legend(loc="upper right", fontsize=9, ncol=2)
    sc = d["scenario"]
    if real:
        info = INFO[d["dataset"]]
        title = f"{info['title']}: live scheduling on real sensor data"
        sub = (f"{sc['M']} real {info['quantity'].lower()} sensors, {sc['C']} polls per 5-minute slot, every sensor "
               f"refreshed within {sc['freshness']} slots. Rings = polled this slot. Colour = {info['quantity'].lower()} ({unit}).")
    else:
        title = "Correlation-aware scheduling, live"
        sub = (f"{sc['M']} sensors, {sc['C']} polls per slot, every sensor refreshed at least every "
               f"{sc['freshness']} slots. Rings = sensors polled this slot. Colour = field value.")
    fig.suptitle(title, x=0.125, ha="left", fontsize=17, fontweight="bold", color=INK, y=0.985)
    fig.text(0.125, 0.925, sub, color=MUTED, fontsize=10)

    def draw(t):
        truth = d["policies"][names[0]]["slots"][t][vk]
        im, dots, rings, txt = panels[0]
        im.set_data(interp(truth))
        dots.set_facecolor(DIVERGING(norm(truth)))
        txt.set_text(d["times"][t].replace("T", "  ") if real else f"slot {t}")
        for k, n in enumerate(names, start=1):
            s = d["policies"][n]["slots"][t]
            im, dots, rings, txt = panels[k]
            im.set_data(interp(s[ek]))
            dots.set_facecolor(DIVERGING(norm(s[ek])))
            rings.set_offsets(pos[s["picks"]] if s["picks"] else np.empty((0, 2)))
            hist = [x[err_key] for x in d["policies"][n]["slots"][: t + 1]]
            if real:
                txt.set_text(f"error now {s[err_key]:.2f}{unit} · avg {np.sqrt(np.mean(np.square(hist))):.2f}{unit}")
            else:
                txt.set_text(f"MSE now {s[err_key]:.3f} · running {np.mean(hist):.3f}")
            lines[n].set_data(range(t + 1), hist)
        cursor.set_xdata([t, t])
        return []

    return fig, draw, T


def _busiest_slot(d, names, key):
    """The slot where the two schedulers' errors differ most (smoothed):
    where scheduling matters."""
    a = np.array([s[key] for s in d["policies"][names[0]]["slots"]])
    b = np.array([s[key] for s in d["policies"][names[1]]["slots"]])
    return int(np.argmax(np.convolve(a - b, np.ones(5) / 5, "same")))


def render(replay="results/data/replay.json", out_dir="results/figures", names=("max_age", "gnn"),
           hero_slot=None, frames=96, fps=8):
    style.apply()
    d = json.loads(Path(replay).read_text())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    key = "rmse_units" if "dataset" in d else "mse"
    if hero_slot is None:
        hero_slot = _busiest_slot(d, names, key)
    fig, draw, T = _setup(d, names)
    draw(min(hero_slot, T - 1))
    fig.savefig(out / "hero.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out / 'hero.png'}")

    fig, draw, T = _setup(d, names)
    first = int(np.clip(hero_slot - frames // 3, 0, max(0, T - frames)))  # open just before the action
    FuncAnimation(fig, draw, frames=range(first, min(T, first + frames)), blit=False).save(
        out / "scheduler.gif", writer=PillowWriter(fps=fps), dpi=58)
    plt.close(fig)
    print(f"  wrote {out / 'scheduler.gif'}")
