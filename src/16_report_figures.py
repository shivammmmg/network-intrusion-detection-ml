"""Writes docs/report-figures/fig_sttl_leak.png.

    PYTHONPATH=src python src/16_report_figures.py

Reads data/raw/, not data/processed/.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from config import RAW_TRAIN_CSV, DOCS_DIR
from preprocess import read_raw

OUT_DIR = DOCS_DIR / "report-figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Accuracy of the rule "sttl >= 32 implies attack". See docs/DATA_CARD.md.
STTL_RULE_ACCURACY = 0.921

SURFACE = "#ffffff"
TEXT_PRIMARY = "#000000"
TEXT_SECONDARY = "#444444"
TEXT_MUTED = "#6e6e6e"
GRID = "#d9d9d9"
NORMAL = "#1f77b4"   # matplotlib tab10 C0
ATTACK = "#ff7f0e"   # matplotlib tab10 C1

ROUND_PX = 4.0

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_PRIMARY,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
})


def px_to_data_x(ax, px):
    """Convert a pixel distance to data units on the x axis."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    x_lo, x_hi = ax.get_xlim()
    return (x_hi - x_lo) * px / bbox.width


def px_to_data_y(ax, px):
    """Convert a pixel distance to data units on the y axis."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    y_lo, y_hi = ax.get_ylim()
    return (y_hi - y_lo) * px / bbox.height


def strip_axes(ax, keep_bottom=False):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if keep_bottom:
        ax.spines["bottom"].set_color(GRID)
    else:
        ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=0)


def rounded_hbar(ax, y, x_start, width, height, color, rx, ry, round_end=True):
    """Horizontal bar, square at the baseline and rounded at the data end."""
    if width <= 0:
        return

    y_bottom = y - height / 2
    y_top = y + height / 2

    if (not round_end) or width < rx * 2.0:
        ax.add_patch(Rectangle(
            (x_start, y_bottom), width, height,
            facecolor=color, edgecolor="none", zorder=3,
        ))
        return

    x_end = x_start + width
    corner_y = min(ry, height / 2)
    vertices = [
        (x_start, y_bottom),
        (x_end - rx, y_bottom),
        (x_end, y_bottom),                 # control point
        (x_end, y_bottom + corner_y),
        (x_end, y_top - corner_y),
        (x_end, y_top),                    # control point
        (x_end - rx, y_top),
        (x_start, y_top),
        (x_start, y_bottom),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(vertices, codes), facecolor=color,
                           edgecolor="none", zorder=3))


def figure_sttl_leak(train_raw):
    values = [0, 31, 62, 254]
    normal_counts = []
    attack_counts = []
    for value in values:
        subset = train_raw[train_raw["sttl"] == value]
        normal_counts.append(int((subset["label"] == 0).sum()))
        attack_counts.append(int((subset["label"] == 1).sum()))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y_base = list(range(len(values)))
    offset = 0.19
    peak = max(attack_counts + normal_counts)
    ax.set_xlim(0, peak * 1.22)
    ax.set_ylim(-0.62, len(values) - 0.30)
    rx = px_to_data_x(ax, ROUND_PX)
    ry = px_to_data_y(ax, ROUND_PX)

    for i, value in enumerate(values):
        rounded_hbar(ax, y_base[i] + offset, 0, normal_counts[i], 0.30,
                     NORMAL, rx, ry)
        rounded_hbar(ax, y_base[i] - offset, 0, attack_counts[i], 0.30,
                     ATTACK, rx, ry)
        pad = peak * 0.012
        ax.text(normal_counts[i] + pad, y_base[i] + offset,
                f"{normal_counts[i]:,}", va="center", ha="left",
                fontsize=8, color=TEXT_SECONDARY)
        ax.text(attack_counts[i] + pad, y_base[i] - offset,
                f"{attack_counts[i]:,}", va="center", ha="left",
                fontsize=8, color=TEXT_SECONDARY)

    ax.set_yticks(y_base)
    ax.set_yticklabels([f"sttl = {v}" for v in values], fontsize=9)
    ax.set_xlabel("Flows in the raw training file", fontsize=8.5)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x):,}")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    strip_axes(ax)

    # The decision boundary of the single-rule classifier.
    boundary = 0.5 * (1 + 2)  # between sttl = 31 (index 1) and sttl = 62 (index 2)
    ax.axhline(boundary, color=TEXT_PRIMARY, linewidth=1.0,
               linestyle=(0, (4, 3)), zorder=5)
    ax.text(peak * 1.21, boundary + 0.07,
            f"rule: sttl ≥ 32 implies attack "
            f"({STTL_RULE_ACCURACY * 100:.1f}% training accuracy)",
            ha="right", va="bottom", fontsize=8.5, color=TEXT_PRIMARY)
    ax.text(peak * 1.21, boundary - 0.07,
            "everything below this line the rule calls normal",
            ha="right", va="top", fontsize=7.5, color=TEXT_MUTED)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=NORMAL, edgecolor="none"),
        Rectangle((0, 0), 1, 1, facecolor=ATTACK, edgecolor="none"),
    ]
    ax.legend(handles, ["Class 0 (normal)", "Class 1 (attack)"],
              loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2,
              frameon=False, fontsize=8.5, handlelength=1.1, handleheight=1.1,
              columnspacing=1.4)

    fig.text(0.155, 0.965,
             "One threshold separates the classes without any learning",
             ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY)

    fig.subplots_adjust(left=0.155, right=0.97, top=0.89, bottom=0.24)
    out = OUT_DIR / "fig_sttl_leak.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[written] {out}")


if __name__ == "__main__":
    figure_sttl_leak(read_raw(RAW_TRAIN_CSV))
