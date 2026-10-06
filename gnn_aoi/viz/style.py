"""One visual system for every figure: dark surface, recessive chrome,
fixed colour per policy (validated for colour-vision deficiency; the first
three slots go to the three policies that appear together most)."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

# entity -> colour, never by rank
POLICY_COLORS = {
    "gnn": "#3987e5",            # blue (ours)
    "max_age": "#d95926",        # orange
    "kalman_greedy": "#199e70",  # aqua
    "greedy_diverse": "#c98500", # yellow
    "oracle": "#d55181",         # magenta
    "max_variance": "#008300",   # green
    "random": "#9085e9",         # violet
    "gnn_no_mp": "#e66767",      # red (ablation)
}
POLICY_ORDER = ["random", "max_variance", "max_age", "greedy_diverse", "kalman_greedy", "gnn_no_mp", "gnn", "oracle"]
LABELS = {
    "random": "Random", "max_variance": "Max-uncertainty", "max_age": "Max-AoI (B1)",
    "greedy_diverse": "Corr-diverse (v4)", "kalman_greedy": "Kalman-greedy",
    "gnn": "GNN-PPO (ours)", "oracle": "Oracle-greedy (true Σ)", "gnn_no_mp": "Ablation: no message passing",
}

# sequential (uncertainty) and diverging (field value) maps from the same ramps
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#1a1a19", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#2a78d6", "#5598e7", "#383835", "#e66767", "#e34948"])


def apply():
    mpl.rcParams.update({
        "figure.facecolor": PAGE, "axes.facecolor": SURFACE, "savefig.facecolor": PAGE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "legend.frameon": False, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "savefig.dpi": 170, "savefig.bbox": "tight",
    })
