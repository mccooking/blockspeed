"""The headline artifact: quality-vs-passes curves.

Every system (AR baseline, teacher at many steps, students at few) becomes a
series of (forward passes per token, quality score) points on one plot.
"""

from __future__ import annotations

from pathlib import Path


def quality_vs_passes(
    series: list[dict],
    ylabel: str = "quality",
    title: str = "Quality vs forward passes per token",
    higher_is_better: bool = True,
    out_png: str | Path | None = None,
):
    """series: [{"label": str, "passes_per_token": [x...], "quality": [y...]}, ...]

    x-axis is log-scale passes per token: AR = 1.0, a 32-token block at 8
    steps = 0.25. Left is faster; the game is staying high while moving left.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in series:
        ax.plot(
            s["passes_per_token"],
            s["quality"],
            marker=s.get("marker", "o"),
            ls=s.get("ls", "-"),
            label=s["label"],
        )
    ax.set_xscale("log")
    ax.set_xlabel("forward passes per generated token (lower = faster)")
    ax.set_ylabel(ylabel + ("" if higher_is_better else " (lower = better)"))
    ax.set_title(title)
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.annotate("AR baseline (1 pass/token)", xy=(1.0, 0.02), xycoords=("data", "axes fraction"),
                rotation=90, va="bottom", ha="right", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return fig
