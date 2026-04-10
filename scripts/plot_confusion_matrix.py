#!/usr/bin/env python3
"""
从 SafeEar 导出的混淆矩阵 CSV 绘制热力图（论文 / 答辩用）。

用法:
  python scripts/plot_confusion_matrix.py \
    --csv Exps/ASVspoof19_len_e30/test_fixed_confusion_matrix.csv \
    --out figures/cm_len_e30_test_fixed.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_cm_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    # 2x2: rows true_bonafide, true_spoof; cols pred bonafide, pred spoof
    mat = df.iloc[:2, :2].values.astype(np.int64)
    if mat.shape != (2, 2):
        raise ValueError(f"expected 2x2, got {mat.shape}")
    return mat


def _far_frr_from_cm(cm: np.ndarray) -> tuple[float, float]:
    """
    与本 repo 指标口径一致：label==0 bonafide, label==1 spoof

    cm layout:
      [[TN, FP],
       [FN, TP]]

    - FAR（攻击假接受）= FN/(FN+TP)：spoof 被误判为 bonafide
    - FRR（真人假拒绝）= FP/(TN+FP)：bonafide 被误判为 spoof
    """
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return far, frr


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    title: str = "Confusion matrix",
    out_path: str | None = None,
    dpi: int = 220,
    figsize: tuple[float, float] = (5.8, 4.9),
    cmap: str = "Blues",
    show: bool = False,
    show_metrics: bool = True,
) -> None:
    total = int(cm.sum())

    mpl.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(cm, cmap=cmap, interpolation="nearest", vmin=0)

    # Gridlines
    ax.set_xticks(np.arange(cm.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    row_labels = ["True: Bonafide", "True: Spoof"]
    col_labels = ["Pred: Bonafide", "Pred: Spoof"]
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)

    # Cell annotations: count + percent of total
    vmax = int(cm.max()) if cm.size else 0
    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            pct = 100.0 * v / total if total else 0.0
            color = "white" if (vmax > 0 and v > 0.55 * vmax) else "black"
            ax.text(
                j,
                i,
                f"{v}\n({pct:.2f}%)",
                ha="center",
                va="center",
                fontsize=12,
                color=color,
                fontweight="semibold",
            )

    ax.set_title(title, fontsize=13, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=90, labelpad=10)

    if show_metrics:
        far, frr = _far_frr_from_cm(cm)
        # Put metrics close to the matrix (no big blank gap).
        ax.text(
            0.5,
            -0.10,
            f"FAR (spoof→bonafide) = {far:.4f}    |    FRR (bonafide→spoof) = {frr:.4f}",
            ha="center",
            va="top",
            fontsize=10.5,
            color="#37474f",
            transform=ax.transAxes,
        )
        fig.subplots_adjust(bottom=0.18)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot confusion matrix from SafeEar CSV")
    p.add_argument("--csv", required=True, help="Path to *_confusion_matrix.csv")
    p.add_argument("--out", default="", help="Output PNG path (default: same dir as CSV, .png)")
    p.add_argument("--title", default="", help="Figure title")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--show", action="store_true", help="Show interactive window")
    p.add_argument(
        "--no-metrics",
        action="store_true",
        help="Do not print FAR/FRR below the figure",
    )
    # Backward compat with the checkpoint script CLI
    p.add_argument(
        "--no-tpr-tnr",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        print(f"File not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    cm = load_cm_csv(args.csv)
    title = args.title or os.path.basename(args.csv).replace(".csv", "").replace("_", " ")
    out = args.out or os.path.splitext(args.csv)[0] + ".png"

    plot_confusion_matrix(
        cm,
        title=title,
        out_path=out,
        dpi=args.dpi,
        show=args.show,
        show_metrics=not (args.no_metrics or args.no_tpr_tnr),
    )


if __name__ == "__main__":
    main()

