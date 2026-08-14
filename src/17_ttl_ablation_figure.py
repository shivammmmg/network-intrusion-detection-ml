"""Writes docs/report-figures/fig_ttl_ablation.png.

    PYTHONPATH=src python src/17_ttl_ablation_figure.py

Values are quoted from experiments/diagnostics/ttl_ablation/ and
experiments/model_analysis/, not recomputed.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import DOCS_DIR

# arm -> (average precision, F1 at the locked threshold)
ARMS = {
    "TTL excluded\n(primary models)": (0.9859, 0.8887),
    "TTL restored\n(ablation refit)": (0.9870, 0.8910),
}

# 95% bootstrap interval, without-TTL model, XGBoost average precision.
AP_INTERVAL = (0.9854, 0.9864)

PANELS = [
    ("Average precision", 0, AP_INTERVAL, "{:.4f}"),
    ("F1 at the locked threshold", 1, None, "{:.4f}"),
]


def draw():
    figure, axes = plt.subplots(1, 2, figsize=(6.5, 3.1))
    names = list(ARMS)

    for axis, (title, index, interval, fmt) in zip(axes, PANELS):
        values = [ARMS[n][index] for n in names]

        if interval is not None:
            axis.axhspan(interval[0], interval[1], color="0.88", zorder=0)

        bars = axis.bar(names, values, width=0.55, color=["C0", "C1"], zorder=2)
        for bar, value in zip(bars, values):
            axis.annotate(
                fmt.format(value),
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )

        low, high = min(values), max(values)
        if interval is not None:
            low, high = min(low, interval[0]), max(high, interval[1])
        pad = max((high - low) * 0.9, 0.0008)
        axis.set_ylim(low - pad, high + pad * 1.6)

        axis.set_title(title, fontsize=10)
        axis.tick_params(axis="x", labelsize=8.5)
        axis.tick_params(axis="y", labelsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.9", zorder=0)
        axis.set_axisbelow(True)
        if interval is not None:
            axis.text(
                -0.34,
                interval[1] + (interval[1] - interval[0]) * 0.30,
                "95% bootstrap\ninterval,\nTTL excluded",
                fontsize=7.5,
                color="0.35",
                va="bottom",
                ha="left",
            )

    axes[0].set_ylabel("Frozen-test score", fontsize=9)
    figure.suptitle(
        "TTL ablation: XGBoost on the frozen test set", fontsize=11, y=0.99
    )
    figure.tight_layout()

    out = DOCS_DIR / "report-figures" / "fig_ttl_ablation.png"
    figure.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    draw()
