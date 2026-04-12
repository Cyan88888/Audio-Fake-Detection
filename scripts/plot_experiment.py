#!/usr/bin/env python3
"""
从 Lightning 实验目录读取 metrics.csv（及可选的 roc/pr npy、测试集标签分数），生成论文用图。

默认读取 csv_logs 下版本号最大的 version_*/metrics.csv。
混淆矩阵依赖测试阶段导出的 ``test_labels.npy``、``test_prob_bonafide.npy``（由 ``TransformerSpoofTrainer``
在 ``save_score_path`` 非空时写入）；若缺失，请先重新运行 ``trainer.test()``。

仅依赖标准库 csv + json + numpy + matplotlib，无需 pandas。

用法:
  python scripts/plot_experiment.py --exp Exps/TransformerSpoof19_wavlm_e30
  python scripts/plot_experiment.py --exp Exps/TransformerSpoof19_wavlm_e30 --out-dir figures/wavlm_e30
  python scripts/plot_experiment.py --exp Exps/... --cm-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_exp_dir(exp: str) -> Path:
    p = Path(exp)
    if not p.is_absolute():
        p = _repo_root() / p
    return p.resolve()


def _find_latest_metrics_csv(exp_dir: Path) -> Path | None:
    base = exp_dir / "csv_logs"
    if not base.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for child in base.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"version_(\d+)$", child.name)
        if not m:
            continue
        ver = int(m.group(1))
        csv_path = child / "metrics.csv"
        if not csv_path.is_file():
            continue
        cand = (ver, csv_path)
        if best is None or ver > best[0]:
            best = cand
    return best[1] if best else None


def _f(row: dict[str, str], key: str) -> float | None:
    v = row.get(key, "").strip()
    if v == "" or v.lower() == "nan":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(row: dict[str, str], key: str) -> int | None:
    v = row.get(key, "").strip()
    if v == "":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _load_metrics_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [dict(r) for r in reader]
    return list(fieldnames), rows


def _plot_metrics_csv(csv_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, rows = _load_metrics_rows(csv_path)
    saved: list[Path] = []

    steps: list[int] = []
    train_loss_step: list[float] = []
    for r in rows:
        s = _i(r, "step")
        tl = _f(r, "train_loss_step")
        if s is not None and tl is not None:
            steps.append(s)
            train_loss_step.append(tl)

    if steps:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(steps, train_loss_step, lw=0.8, alpha=0.85, color="C0")
        ax.set_xlabel("step")
        ax.set_ylabel("train_loss")
        ax.set_title("Training loss (per step)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "train_loss_step.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    epochs: list[int] = []
    val_eer: list[float] = []
    val_mindcf: list[float] = []
    val_roc: list[float] = []
    val_acc: list[float] = []
    val_lr: list[float] = []
    for r in rows:
        eer = _f(r, "val_eer")
        ep = _i(r, "epoch")
        if eer is None or ep is None:
            continue
        epochs.append(ep)
        val_eer.append(eer)
        val_mindcf.append(_f(r, "val_minDCF") or float("nan"))
        val_roc.append(_f(r, "val_roc_auc") or float("nan"))
        val_acc.append(_f(r, "val_acc") or float("nan"))
        lr = _f(r, "lr")
        val_lr.append(lr if lr is not None else float("nan"))

    if epochs:
        series = [
            (val_eer, "val_eer", "Validation EER"),
            (val_mindcf, "val_minDCF", "Validation minDCF"),
            (val_roc, "val_roc_auc", "Validation ROC-AUC"),
            (val_acc, "val_acc", "Validation accuracy"),
        ]
        series = [(y, ylab, title) for y, ylab, title in series if not all(np.isnan(y))]
        if series:
            n = len(series)
            fig, axes = plt.subplots(n, 1, figsize=(8, 2.2 * n), squeeze=False)
            axes = axes.ravel()
            for ax, (y, ylab, title) in zip(axes, series):
                yy = np.asarray(y, dtype=np.float64)
                ax.plot(epochs, yy, marker="o", ms=3, lw=1)
                ax.set_xlabel("epoch")
                ax.set_ylabel(ylab)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = out_dir / "validation_metrics.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

    lr_epochs = [e for e, lr in zip(epochs, val_lr) if not np.isnan(lr)]
    lr_vals = [lr for lr in val_lr if not np.isnan(lr)]
    if lr_epochs:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(lr_epochs, lr_vals, marker="o", ms=3, color="C3")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning rate (logged at validation)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "lr.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    ep_te: list[int] = []
    train_loss_epoch: list[float] = []
    for r in rows:
        tle = _f(r, "train_loss_epoch")
        ep = _i(r, "epoch")
        if tle is not None and ep is not None:
            ep_te.append(ep)
            train_loss_epoch.append(tle)
    if ep_te:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(ep_te, train_loss_epoch, marker="o", ms=3, color="C2")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train_loss_epoch")
        ax.set_title("Training loss (per epoch)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / "train_loss_epoch.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


def _plot_roc_pr(exp_dir: Path, out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    roc_path = exp_dir / "roc_curve.npy"
    if roc_path.is_file():
        roc = np.load(roc_path)
        if roc.ndim == 2 and roc.shape[0] >= 2:
            fpr, tpr = roc[0], roc[1]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(fpr, tpr, lw=2, label="ROC")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.35)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.set_title("ROC curve (test set)")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = out_dir / "roc.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

    pr_path = exp_dir / "pr_curve.npy"
    if pr_path.is_file():
        pr = np.load(pr_path)
        if pr.ndim == 2 and pr.shape[0] >= 2:
            recall, precision = pr[0], pr[1]
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(recall, precision, lw=2, color="C1")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("PR curve (test set)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p = out_dir / "pr.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(p)

    return saved


def _default_cm_threshold(exp_dir: Path) -> float:
    meta = exp_dir / "test_cm_meta.json"
    if meta.is_file():
        try:
            with meta.open(encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("threshold_bonafide", 0.5))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return 0.5


def _binary_confusion_matrix(
    y_true: np.ndarray, y_score_bonafide: np.ndarray, threshold_bonafide: float
) -> np.ndarray:
    """与 ``binary_metrics.compute_binary_classification_metrics`` 相同判决规则。"""
    y_true = y_true.astype(np.int64).ravel()
    y_score_bonafide = y_score_bonafide.astype(np.float64).ravel()
    if y_true.shape != y_score_bonafide.shape:
        raise ValueError(
            f"labels/scores length mismatch: {y_true.shape} vs {y_score_bonafide.shape}"
        )
    y_pred = (y_score_bonafide < threshold_bonafide).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _far_frr_from_cm(cm: np.ndarray) -> tuple[float, float]:
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    far = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    frr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    return far, frr


def _plot_confusion_matrix(
    exp_dir: Path, out_dir: Path, threshold_bonafide: float
) -> list[Path]:
    labels_path = exp_dir / "test_labels.npy"
    scores_path = exp_dir / "test_prob_bonafide.npy"
    if not labels_path.is_file() or not scores_path.is_file():
        print(
            "提示: 跳过混淆矩阵（未找到 test_labels.npy / test_prob_bonafide.npy）。"
            "请使用含 save_score_path 的配置重新运行测试以导出。",
            file=sys.stderr,
        )
        return []

    y_true = np.load(labels_path)
    y_score = np.load(scores_path)
    cm = _binary_confusion_matrix(y_true, y_score, threshold_bonafide)
    total = int(cm.sum())
    if total == 0:
        print("警告: 混淆矩阵样本数为 0，跳过", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=150)
    vmax = max(int(cm.max()), 1)
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest", vmin=0, vmax=vmax)

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

    for i in range(2):
        for j in range(2):
            v = int(cm[i, j])
            pct = 100.0 * v / total
            color = "white" if v > 0.55 * vmax else "black"
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

    ax.set_title(
        f"Confusion matrix (test, P(bonafide) ≥ {threshold_bonafide:.3f} → bonafide)",
        fontsize=12,
        pad=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.set_ylabel("Count", rotation=90, labelpad=10)

    far, frr = _far_frr_from_cm(cm)
    ax.text(
        0.5,
        -0.12,
        f"FAR (spoof→bonafide) = {far:.4f}   |   FRR (bonafide→spoof) = {frr:.4f}",
        ha="center",
        va="top",
        fontsize=10,
        color="#37474f",
        transform=ax.transAxes,
    )
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    p = out_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [p]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Lightning experiment metrics and ROC/PR.")
    parser.add_argument(
        "--exp",
        type=str,
        default="Exps/TransformerSpoof19_wavlm_e30",
        help="实验目录（相对仓库根或绝对路径）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="图片输出目录，默认 <exp>/plots",
    )
    parser.add_argument(
        "--cm-threshold",
        type=float,
        default=None,
        help="混淆矩阵判决阈值 P(bonafide)；默认读取 test_cm_meta.json，否则 0.5",
    )
    args = parser.parse_args()

    exp_dir = _resolve_exp_dir(args.exp)
    if not exp_dir.is_dir():
        print(f"错误: 实验目录不存在: {exp_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "plots")
    csv_path = _find_latest_metrics_csv(exp_dir)
    if csv_path is None:
        print(f"错误: 未找到 {exp_dir}/csv_logs/version_*/metrics.csv", file=sys.stderr)
        return 1

    print(f"metrics: {csv_path}")
    print(f"output:  {out_dir}")

    all_saved = _plot_metrics_csv(csv_path, out_dir)
    all_saved += _plot_roc_pr(exp_dir, out_dir)

    th = args.cm_threshold
    if th is None:
        th = _default_cm_threshold(exp_dir)
    all_saved += _plot_confusion_matrix(exp_dir, out_dir, th)

    for p in all_saved:
        print(f"  saved {p}")

    if not all_saved:
        print("警告: 未生成任何图片（检查 metrics.csv 是否为空）", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
