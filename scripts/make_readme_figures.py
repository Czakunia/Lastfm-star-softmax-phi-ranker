#!/usr/bin/env python3
"""README landing figures: prediction goal, method, sealed ranking."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"

ROSE = "#D96B76"
ROSE_LIGHT = "#F3C4C8"
ROSE_DARK = "#9B1C2C"
NAVY = "#D6E4F0"
NAVY_LINE = "#7A9BB5"
INK = "#2C1819"
MUTED = "#6B4A4E"
WHITE = "#FFFFFF"
GRID = "#E6E8EB"

HGT_HARD_SEEDS = [0.202174, 0.202932, 0.197613, 0.201272, 0.203313]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": WHITE,
            "savefig.pad_inches": 0.12,
            "pdf.fonttype": 42,
        }
    )


def _round(ax, xy, w, h, fc, ec, lw=1.1, r=0.08, z=2):
    p = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, p0, p1, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.4,
            color=color,
            zorder=4,
        )
    )


def fig_goal() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.15))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.15)
    ax.axis("off")

    ax.text(0.15, 2.88, "Prediction goal", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        0.15,
        2.58,
        "For each Last-FM* listener, rank unseen artists so the next listens land in the top 20.",
        fontsize=9.5,
        color=MUTED,
    )

    _round(ax, (0.15, 0.28), 3.15, 2.10, ROSE_LIGHT, ROSE, r=0.12)
    ax.text(0.35, 2.08, "Observed", fontsize=8.5, fontweight="bold", color=ROSE_DARK)
    ax.text(0.35, 1.78, "user  u", fontsize=10, fontweight="bold", color=INK)
    for i, name in enumerate(("Radiohead", "Bjork", "Miles Davis", "...")):
        y = 1.42 - 0.28 * i
        ax.add_patch(Circle((0.55, y + 0.08), 0.07, facecolor=ROSE, edgecolor="none", zorder=3))
        ax.text(0.72, y, name, fontsize=9, color=INK, va="center")
    ax.text(0.35, 0.40, r"$H_u$  (train history)", fontsize=8, color=MUTED)

    _arrow(ax, (3.42, 1.33), (4.05, 1.33), ROSE_DARK)

    _round(ax, (4.15, 0.28), 2.70, 2.10, NAVY, NAVY_LINE, r=0.12)
    ax.text(4.35, 2.08, "Ask", fontsize=8.5, fontweight="bold", color="#3D5A73")
    ax.text(4.35, 1.62, "Which unseen\nartists next?", fontsize=11, fontweight="bold", color=INK)
    ax.text(4.35, 0.88, r"$X \notin H_u$", fontsize=10, color=INK)
    ax.text(4.35, 0.48, "full catalogue,\ncandidates $C_{CF}$", fontsize=8, color=MUTED)

    _arrow(ax, (6.97, 1.33), (7.60, 1.33), ROSE_DARK)

    _round(ax, (7.70, 0.28), 3.30, 2.10, WHITE, ROSE, r=0.12, lw=1.4)
    ax.text(7.90, 2.08, "Output", fontsize=8.5, fontweight="bold", color=ROSE_DARK)
    ax.text(7.90, 1.72, "ranked list  Top-20", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(7.90, 1.28, "1.  $X_{(1)}$", fontsize=9, color=INK)
    ax.text(7.90, 1.00, "2.  $X_{(2)}$", fontsize=9, color=INK)
    ax.text(7.90, 0.72, "3.  $X_{(3)}$  ...", fontsize=9, color=INK)
    ax.text(7.90, 0.40, "metric: NDCG@20", fontsize=8, color=MUTED)

    fig.savefig(OUT / "goal.png")
    fig.savefig(OUT / "goal.svg")
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.55))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.55)
    ax.axis("off")

    ax.text(0.15, 4.28, "How the ranker works", fontsize=13, fontweight="bold", color=INK)
    ax.text(
        0.15,
        3.98,
        "No learned weights.  UserKNN retrieves; Pearson φ scores; softmax aggregates.",
        fontsize=9.5,
        color=MUTED,
    )

    panels = [
        (0.15, "1", "Retrieve", NAVY, NAVY_LINE),
        (3.85, "2", "Score", ROSE_LIGHT, ROSE),
        (7.55, "3", "Rank", WHITE, ROSE_DARK),
    ]
    for x0, num, title, fc, ec in panels:
        _round(ax, (x0, 0.22), 3.50, 3.55, fc, ec, r=0.12, lw=1.2)
        ax.add_patch(Circle((x0 + 0.32, 3.42), 0.16, facecolor=ROSE_DARK, edgecolor="none", zorder=5))
        ax.text(x0 + 0.32, 3.42, num, ha="center", va="center", color=WHITE, fontsize=9, fontweight="bold", zorder=6)
        ax.text(x0 + 0.58, 3.34, title, fontsize=12, fontweight="bold", color=INK, va="center")

    ax.text(
        0.35,
        2.85,
        "200 nearest users\nby cosine of binary\nhistories (overlap ≥ 1)",
        fontsize=9,
        color=INK,
        va="top",
    )
    ax.text(
        0.35,
        1.70,
        "C$_{CF}$(u) = union of the\nneighbours' histories,\nminus H$_u$."
        "\n\nItems never retrieved\nstay at −∞.",
        fontsize=8.5,
        color=MUTED,
        va="top",
    )

    ax.text(
        4.05,
        2.85,
        "For each candidate X and\neach history item h:",
        fontsize=9,
        color=INK,
        va="top",
    )
    ax.text(
        4.05,
        1.85,
        r"$\phi(h,X)$  Pearson 2×2"
        "\n"
        r"$\alpha=\mathrm{softmax}(\phi/\tau)$"
        "\n"
        r"$s(u,X)=\sum_h \alpha_h\,\phi(h,X)$",
        fontsize=9.5,
        color=INK,
        va="top",
    )
    ax.text(
        4.05,
        0.48,
        "default τ = 1    optional τ = 0.5",
        fontsize=8,
        color=ROSE_DARK,
    )

    ax.text(
        7.75,
        2.85,
        "Sort $C_{CF}(u)$ by $s(u,X)$\nand take the top 20.",
        fontsize=9,
        color=INK,
        va="top",
    )
    ax.text(
        7.75,
        1.85,
        "Sealed extra, τ = 1\nNDCG@20  =  0.2151\n\nτ = 0.5 stays 2nd vs RP3β\nNDCG@20  =  0.2185",
        fontsize=9,
        color=INK,
        va="top",
    )

    _arrow(ax, (3.68, 2.00), (3.82, 2.00), ROSE_DARK)
    _arrow(ax, (7.38, 2.00), (7.52, 2.00), ROSE_DARK)

    fig.savefig(OUT / "pipeline.png")
    fig.savefig(OUT / "pipeline.svg")
    plt.close(fig)


def fig_ranking() -> None:
    rows = [
        ("TopPop", 0.0091, "repro"),
        ("KGIN", 0.092, "lit"),
        ("UserKNN", 0.1622, "repro"),
        ("HGT + stats (rand. neg.)", 0.1718, "ours"),
        ("EASE", 0.191, "lit"),
        ("HGT + stats (hard neg.)", 0.2015, "ours"),
        ("P3α", 0.2035, "repro"),
        ("ItemKNN", 0.2051, "repro"),
        ("softmax-φ  τ=1", 0.2151, "phi"),
        ("RP3β", 0.2318, "repro"),
    ]
    names = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows], dtype=float)
    kinds = [r[2] for r in rows]
    colors = []
    for k in kinds:
        if k == "phi":
            colors.append(ROSE_DARK)
        elif k == "ours":
            colors.append(ROSE)
        else:
            colors.append(ROSE_LIGHT)

    fig, ax = plt.subplots(figsize=(11.2, 4.15))
    x = np.arange(len(rows))
    bars = ax.bar(x, vals, width=0.72, color=colors, edgecolor=WHITE, linewidth=0.5, zorder=3)
    for bar, kind in zip(bars, kinds):
        if kind == "lit":
            bar.set_hatch("///")
            bar.set_edgecolor(ROSE)
        if kind == "phi":
            bar.set_edgecolor(ROSE_DARK)
            bar.set_linewidth(0.6)

    ax.scatter(
        np.full(len(HGT_HARD_SEEDS), 5) + np.linspace(-0.14, 0.14, len(HGT_HARD_SEEDS)),
        HGT_HARD_SEEDS,
        s=16,
        zorder=5,
        color=WHITE,
        edgecolors=ROSE_DARK,
        linewidths=0.7,
    )

    for i, (v, kind) in enumerate(zip(vals, kinds)):
        lab = f"{v:.3f}" if kind != "phi" else f"{v:.4f}"
        ax.text(
            i,
            v + 0.006,
            lab,
            ha="center",
            va="bottom",
            fontsize=8,
            color=ROSE_DARK if kind in {"phi", "ours"} else MUTED,
            fontweight="bold" if kind == "phi" else "normal",
        )

    ax.axhline(0.2, color=INK, ls="--", lw=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylim(0.0, 0.275)
    ax.set_ylabel("Sealed extra  NDCG@20")
    ax.yaxis.grid(True, lw=0.4, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Last-FM* sealed extra  ·  this package in rose (τ = 1)", loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)

    handles = [
        Patch(facecolor=ROSE_DARK, edgecolor=WHITE, label="This package (softmax-φ, τ=1)"),
        Patch(facecolor=ROSE, edgecolor=WHITE, label="HGT + stats (other package)"),
        Patch(facecolor=ROSE_LIGHT, edgecolor=WHITE, label="Reproduced classical"),
        Patch(facecolor=ROSE_LIGHT, edgecolor=ROSE, hatch="///", label="Literature (not re-run)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=7.8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "ranking.png")
    fig.savefig(OUT / "ranking.svg")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_goal()
    fig_pipeline()
    fig_ranking()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
