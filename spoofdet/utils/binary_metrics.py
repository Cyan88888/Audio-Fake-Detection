"""Binary classification metrics for spoof detection (bonafide vs spoof)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_min_dcf(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
):
    """
    minDCF = min_t (c_miss * P_miss(t) * p_target + c_fa * P_fa(t) * (1 - p_target))

    Here:
    - target_scores: scores for bonafide trials (label==0)
    - nontarget_scores: scores for spoof trials (label==1)
    - decision rule: accept target if score >= threshold
    """
    target_scores = _to_numpy(target_scores).astype(np.float64)
    nontarget_scores = _to_numpy(nontarget_scores).astype(np.float64)

    all_scores = np.concatenate([target_scores, nontarget_scores])
    labels = np.concatenate(
        [np.ones(target_scores.size, dtype=np.int32), np.zeros(nontarget_scores.size, dtype=np.int32)]
    )

    order = np.argsort(all_scores, kind="mergesort")
    scores_sorted = all_scores[order]
    labels_sorted = labels[order]

    tar_total = float(target_scores.size)
    non_total = float(nontarget_scores.size)

    tar_below = np.cumsum(labels_sorted)
    non_below = np.cumsum(1 - labels_sorted)

    p_miss = np.concatenate(([0.0], tar_below / tar_total))
    p_fa = np.concatenate(([1.0], (non_total - non_below) / non_total))

    thresholds = np.concatenate(([scores_sorted[0] - 1e-6], scores_sorted))

    dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
    min_idx = int(np.argmin(dcf))
    return float(dcf[min_idx]), float(thresholds[min_idx])


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_score_bonafide: np.ndarray,
    threshold_bonafide: float = 0.5,
    pos_label: int = 1,
):
    """
    Detector outputs bonafide probability (class 0) as score.

    Args:
        y_true: 0=bonafide, 1=spoof
        y_score_bonafide: P(bonafide) in [0,1]
        threshold_bonafide: predict bonafide if score >= threshold, else spoof
        pos_label: positive label for Precision/Recall/F1/ROC/PR (default: spoof=1)
    """
    y_true = _to_numpy(y_true).astype(np.int64)
    y_score_bonafide = _to_numpy(y_score_bonafide).astype(np.float64)

    y_score_spoof = 1.0 - y_score_bonafide

    y_pred = (y_score_bonafide < threshold_bonafide).astype(np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=pos_label, average="binary", zero_division=0
    )

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score_spoof, pos_label=pos_label)
    roc_auc = float(auc(fpr, tpr))

    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_true, y_score_spoof, pos_label=pos_label
    )
    ap = float(average_precision_score(y_true, y_score_spoof, pos_label=pos_label))

    # PAR (Presentation Attack Pass Rate): spoof accepted as bonafide.
    # PRR (Presentation Real Pass Rate): bonafide accepted as bonafide.
    # FAR/FRR (biometric naming):
    # - FAR: attack/impostor accepted as genuine == PAR
    # - FRR: genuine rejected as attack == 1 - PRR
    spoof_mask = y_true == 1
    bona_mask = y_true == 0
    spoof_total = int(np.sum(spoof_mask))
    bona_total = int(np.sum(bona_mask))
    spoof_pass = int(np.sum((y_pred == 0) & spoof_mask))
    bona_pass = int(np.sum((y_pred == 0) & bona_mask))
    par = float(spoof_pass / spoof_total) if spoof_total > 0 else 0.0
    prr = float(bona_pass / bona_total) if bona_total > 0 else 0.0
    far = par
    frr = float(1.0 - prr)

    return {
        "acc": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "par": par,
        "prr": prr,
        "far": far,
        "frr": frr,
        "roc_auc": roc_auc,
        "pr_ap": ap,
        "roc_curve": (fpr.astype(np.float64), tpr.astype(np.float64), roc_thresholds.astype(np.float64)),
        "pr_curve": (pr_recall.astype(np.float64), pr_precision.astype(np.float64), pr_thresholds.astype(np.float64)),
    }
