#!/usr/bin/env python3
"""Calibrate test metrics using validation threshold and write back to test_results.json.

This script reads:
- Exps/.../test_results.json
- Exps/.../test_labels.npy
- Exps/.../test_prob_bonafide.npy

Then computes threshold-dependent metrics on test set using threshold_bonafide
selected with priority:
1) val_min_tDCF_th
2) val_minDCF_th

- FAR / FRR / ACC / F1

And writes non-destructive fields into test_results.json with `test_calib_` prefix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _collect_result_files(root: Path) -> List[Path]:
    return sorted(p for p in root.glob("**/test_results.json") if p.is_file())


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Labels:
      0 = bonafide
      1 = spoof
    Prediction:
      y_pred=1 means spoof, y_pred=0 means bonafide
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    bona = y_true == 0
    spoof = y_true == 1

    tp = int(np.sum((y_pred == 1) & spoof))  # spoof correctly rejected
    fn = int(np.sum((y_pred == 0) & spoof))  # spoof accepted as bona
    tn = int(np.sum((y_pred == 0) & bona))   # bona correctly accepted
    fp = int(np.sum((y_pred == 1) & bona))   # bona rejected as spoof

    acc = _safe_div(tp + tn, y_true.size)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    far = _safe_div(fn, int(np.sum(spoof)))  # spoof pass rate
    frr = _safe_div(fp, int(np.sum(bona)))   # bona reject rate

    return {
        "acc": acc,
        "f1": f1,
        "far": far,
        "frr": frr,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_threshold(val0: Dict[str, object]) -> Tuple[float, str]:
    """Return (threshold_bonafide, source_name) by priority."""
    for key in ("val_min_tDCF_th", "val_minDCF_th"):
        raw = val0.get(key, None)
        if raw is None:
            continue
        try:
            th = float(raw)
        except Exception:
            continue
        if np.isfinite(th):
            return th, key
    raise ValueError("No finite threshold found in val_min_tDCF_th / val_minDCF_th")


def _process_one(result_json: Path, dry_run: bool = False) -> Tuple[str, str]:
    exp_dir = result_json.parent
    labels_path = exp_dir / "test_labels.npy"
    probs_path = exp_dir / "test_prob_bonafide.npy"

    if not labels_path.is_file() or not probs_path.is_file():
        return ("skip", f"{exp_dir}: missing test_labels.npy or test_prob_bonafide.npy")

    data = json.loads(result_json.read_text(encoding="utf-8"))
    val_list = data.get("validate") or []
    test_list = data.get("test") or []
    if not val_list or not isinstance(val_list[0], dict) or not test_list or not isinstance(test_list[0], dict):
        return ("skip", f"{exp_dir}: invalid validate/test structure")

    val0 = val_list[0]
    test0 = test_list[0]

    try:
        th_bona, th_source = _select_threshold(val0)
    except ValueError:
        return ("skip", f"{exp_dir}: no finite val_min_tDCF_th / val_minDCF_th")

    y_true = np.load(labels_path)
    y_score_bona = np.load(probs_path)
    if y_true.shape[0] != y_score_bona.shape[0]:
        return ("skip", f"{exp_dir}: labels/prob size mismatch")

    # rule: bonafide if score >= threshold_bonafide else spoof
    y_pred = (y_score_bona < th_bona).astype(np.int64)
    m = _compute_binary_metrics(y_true, y_pred)

    # Non-destructive write-back fields
    test0["test_calib_source"] = th_source
    test0["test_calib_threshold_bonafide"] = float(th_bona)
    test0["test_calib_threshold_spoof"] = float(1.0 - th_bona)
    test0["test_calib_acc"] = m["acc"]
    test0["test_calib_f1"] = m["f1"]
    test0["test_calib_FAR"] = m["far"]
    test0["test_calib_FRR"] = m["frr"]
    test0["test_calib_tp_spoof"] = int(m["tp"])
    test0["test_calib_tn_bonafide"] = int(m["tn"])
    test0["test_calib_fp_bonafide_to_spoof"] = int(m["fp"])
    test0["test_calib_fn_spoof_to_bonafide"] = int(m["fn"])

    if not dry_run:
        result_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return ("ok", f"{exp_dir}: updated with calibrated metrics")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute test FAR/FRR/ACC/F1 using validation threshold and append to test_results.json "
            "(priority: val_min_tDCF_th > val_minDCF_th)"
        ),
    )
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Single experiment directory, e.g. Exps/Ablation_Pool_max",
    )
    parser.add_argument(
        "--exps-root",
        default="Exps",
        help="Root directory to scan when --exp-dir is not provided",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print actions, do not write file")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    targets: List[Path]
    if args.exp_dir:
        exp_dir = (repo_root / args.exp_dir).resolve()
        targets = [exp_dir / "test_results.json"]
    else:
        exps_root = (repo_root / args.exps_root).resolve()
        targets = _collect_result_files(exps_root)

    if not targets:
        print("[WARN] no test_results.json found")
        return 1

    ok = 0
    skip = 0
    for rp in targets:
        if not rp.is_file():
            skip += 1
            print(f"[SKIP] {rp}: file not found")
            continue
        status, msg = _process_one(rp, dry_run=args.dry_run)
        if status == "ok":
            ok += 1
            print(f"[OK]   {msg}")
        else:
            skip += 1
            print(f"[SKIP] {msg}")

    print(f"[DONE] ok={ok}, skip={skip}, total={len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
