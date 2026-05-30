"""t-DCF utilities for ASVspoof2019 LA (CM + ASV tandem evaluation)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _asv_error_rates(target_scores: np.ndarray, nontarget_scores: np.ndarray, spoof_scores: np.ndarray) -> Tuple[float, float, float]:
    """Compute ASV error rates at ASV EER threshold."""
    tar = np.asarray(target_scores, dtype=np.float64)
    non = np.asarray(nontarget_scores, dtype=np.float64)
    spoof = np.asarray(spoof_scores, dtype=np.float64)
    all_scores = np.concatenate([tar, non])
    labels = np.concatenate([np.ones(tar.size, dtype=np.int32), np.zeros(non.size, dtype=np.int32)])

    order = np.argsort(all_scores, kind="mergesort")
    s = all_scores[order]
    y = labels[order]
    tar_total = float(tar.size)
    non_total = float(non.size)
    tar_below = np.cumsum(y)
    non_below = np.cumsum(1 - y)
    p_miss = np.concatenate(([0.0], tar_below / tar_total))
    p_fa = np.concatenate(([1.0], (non_total - non_below) / non_total))
    idx = int(np.argmin(np.abs(p_miss - p_fa)))
    thr = np.concatenate(([s[0] - 1e-6], s))[idx]

    p_miss_asv = float(np.mean(tar < thr))
    p_fa_asv = float(np.mean(non >= thr))
    p_miss_spoof_asv = float(np.mean(spoof < thr))
    return p_miss_asv, p_fa_asv, p_miss_spoof_asv


def compute_min_tdcf_from_cm_bonafide_score(
    cm_bonafide_scores: np.ndarray,
    cm_labels: np.ndarray,
    asv_target_scores: np.ndarray,
    asv_nontarget_scores: np.ndarray,
    asv_spoof_scores: np.ndarray,
    p_tar: float = 0.9405,
    p_non: float = 0.0095,
    p_spoof: float = 0.05,
    c_miss_asv: float = 1.0,
    c_fa_asv: float = 10.0,
    c_miss_cm: float = 1.0,
    c_fa_cm: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute normalized min t-DCF for CM scores.
    `cm_bonafide_scores`: higher means more bonafide; labels: 0 bonafide, 1 spoof.
    """
    cm_scores = np.asarray(cm_bonafide_scores, dtype=np.float64)
    cm_labels = np.asarray(cm_labels, dtype=np.int64)
    bona = cm_scores[cm_labels == 0]
    spoof = cm_scores[cm_labels == 1]
    if bona.size == 0 or spoof.size == 0:
        return float("nan"), float("nan")

    p_miss_asv, p_fa_asv, p_miss_spoof_asv = _asv_error_rates(
        np.asarray(asv_target_scores, dtype=np.float64),
        np.asarray(asv_nontarget_scores, dtype=np.float64),
        np.asarray(asv_spoof_scores, dtype=np.float64),
    )

    c1 = p_tar * (c_miss_cm - c_miss_asv * p_miss_asv) - p_non * c_fa_asv * p_fa_asv
    c2 = c_fa_cm * p_spoof * (1.0 - p_miss_spoof_asv)
    if c1 <= 0 or c2 <= 0:
        return float("nan"), float("nan")

    all_scores = np.concatenate([bona, spoof])
    labels = np.concatenate(
        [np.ones(bona.size, dtype=np.int32), np.zeros(spoof.size, dtype=np.int32)]
    )  # 1 bonafide, 0 spoof
    order = np.argsort(all_scores, kind="mergesort")
    s = all_scores[order]
    y = labels[order]
    bona_total = float(bona.size)
    spoof_total = float(spoof.size)
    bona_below = np.cumsum(y)
    spoof_below = np.cumsum(1 - y)
    p_miss_cm = np.concatenate(([0.0], bona_below / bona_total))
    p_fa_cm = np.concatenate(([1.0], (spoof_total - spoof_below) / spoof_total))
    thresholds = np.concatenate(([s[0] - 1e-6], s))

    tdcf = c1 * p_miss_cm + c2 * p_fa_cm
    tdcf_norm = tdcf / min(c1, c2)
    idx = int(np.argmin(tdcf_norm))
    return float(tdcf_norm[idx]), float(thresholds[idx])


def load_asvspoof2019_la_asv_scores(la_root: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load ASV target/nontarget/spoof scores from official ASVspoof2019 LA files.
    Returns {"dev": {...}, "eval": {...}}.
    """
    root = Path(la_root)
    prot_dir = root / "ASVspoof2019_LA_asv_protocols"
    score_dir = root / "ASVspoof2019_LA_asv_scores"

    def _load_one(split: str) -> Dict[str, np.ndarray]:
        prot = prot_dir / f"ASVspoof2019.LA.asv.{split}.gi.trl.txt"
        scr = score_dir / f"ASVspoof2019.LA.asv.{split}.gi.trl.scores.txt"
        with prot.open("r", encoding="utf-8") as f:
            prot_lines = [ln.strip().split() for ln in f if ln.strip()]
        with scr.open("r", encoding="utf-8") as f:
            scr_lines = [ln.strip().split() for ln in f if ln.strip()]
        if len(prot_lines) != len(scr_lines):
            raise RuntimeError(f"ASV protocol/score length mismatch for {split}: {len(prot_lines)} vs {len(scr_lines)}")

        tar, non, spoof = [], [], []
        for p, s in zip(prot_lines, scr_lines):
            key = p[-1].lower()
            score = float(s[-1])
            if key == "target":
                tar.append(score)
            elif key == "nontarget":
                non.append(score)
            elif key == "spoof":
                spoof.append(score)
        return {
            "target": np.asarray(tar, dtype=np.float64),
            "nontarget": np.asarray(non, dtype=np.float64),
            "spoof": np.asarray(spoof, dtype=np.float64),
        }

    return {"dev": _load_one("dev"), "eval": _load_one("eval")}
