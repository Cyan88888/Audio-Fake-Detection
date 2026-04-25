#!/usr/bin/env python3
"""Aggregate experiment test results into CSV/Markdown tables.

Usage example:
  python scripts/aggregate_test_results.py --exps-dir Exps --out-dir Exps/_summary
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def collect_result_files(exps_dir: Path) -> List[Path]:
    return sorted(p for p in exps_dir.glob("**/test_results.json") if p.is_file())


def _extract_key_in_block(text: str, block: str, key: str) -> str:
    """
    Extract `key` under a top-level YAML `block` by simple indentation scan.
    Falls back to empty string when missing.
    """
    m = re.search(rf"(?m)^{re.escape(block)}:\s*$", text)
    if not m:
        return ""
    tail = text[m.end() :]
    for line in tail.splitlines():
        if line and not line.startswith(" ") and not line.startswith("\t"):
            break
        s = line.strip()
        if s.startswith(f"{key}:"):
            return s.split(":", 1)[1].strip().strip("'\"")
    return ""


def parse_experiment_config(exp_dir: Path) -> Dict[str, object]:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.is_file():
        return {"config_file": ""}
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    row: Dict[str, object] = {"config_file": str(cfg_path)}

    # experiment id
    row["cfg_exp_name"] = _extract_key_in_block(text, "exp", "name")
    row["cfg_exp_dir"] = _extract_key_in_block(text, "exp", "dir")

    # model structure
    row["cfg_pooling"] = _extract_key_in_block(text, "detect_model", "pooling")
    row["cfg_positional_embedding"] = _extract_key_in_block(text, "detect_model", "positional_embedding")
    row["cfg_num_layers"] = _extract_key_in_block(text, "detect_model", "num_layers")
    row["cfg_num_heads"] = _extract_key_in_block(text, "detect_model", "num_heads")
    row["cfg_dropout_rate"] = _extract_key_in_block(text, "detect_model", "dropout_rate")
    row["cfg_attention_dropout"] = _extract_key_in_block(text, "detect_model", "attention_dropout")
    row["cfg_stochastic_depth_rate"] = _extract_key_in_block(text, "detect_model", "stochastic_depth_rate")
    row["cfg_sequence_length"] = _extract_key_in_block(text, "detect_model", "sequence_length")
    row["cfg_mlp_ratio"] = _extract_key_in_block(text, "detect_model", "mlp_ratio")

    # training / regularization
    row["cfg_lr"] = _extract_key_in_block(text, "system", "lr")
    row["cfg_weight_decay"] = _extract_key_in_block(text, "system", "weight_decay")
    row["cfg_warmup_epochs_ratio"] = _extract_key_in_block(text, "system", "warmup_epochs_ratio")
    row["cfg_feat_norm_mode"] = _extract_key_in_block(text, "system", "feat_norm_mode")
    row["cfg_label_smoothing"] = _extract_key_in_block(text, "system", "label_smoothing")
    row["cfg_aug_time_mask_prob"] = _extract_key_in_block(text, "system", "aug_time_mask_prob")
    row["cfg_aug_time_mask_max_frames"] = _extract_key_in_block(text, "system", "aug_time_mask_max_frames")
    row["cfg_aug_chunk_shuffle_prob"] = _extract_key_in_block(text, "system", "aug_chunk_shuffle_prob")
    row["cfg_aug_chunk_size"] = _extract_key_in_block(text, "system", "aug_chunk_size")
    row["cfg_aug_feat_dropout_prob"] = _extract_key_in_block(text, "system", "aug_feat_dropout_prob")
    row["cfg_threshold_mode"] = _extract_key_in_block(text, "system", "threshold_mode")
    row["cfg_threshold_fixed_bonafide"] = _extract_key_in_block(text, "system", "threshold_fixed_bonafide")

    # runtime
    row["cfg_batch_size"] = _extract_key_in_block(text, "datamodule", "batch_size")
    row["cfg_max_epochs"] = _extract_key_in_block(text, "trainer", "max_epochs")
    row["cfg_gradient_clip_val"] = _extract_key_in_block(text, "trainer", "gradient_clip_val")
    row["cfg_eval_crop_mode"] = _extract_key_in_block(text, "DataClass_dict", "eval_crop_mode")
    row["cfg_max_len"] = _extract_key_in_block(text, "DataClass_dict", "max_len")
    return row


def flatten_result(fp: Path) -> Dict[str, object]:
    data = json.loads(fp.read_text(encoding="utf-8"))
    val = (data.get("validate") or [{}])[0]
    test = (data.get("test") or [{}])[0]
    exp_dir = fp.parent
    row: Dict[str, object] = {
        "exp_name": exp_dir.name,
        "exp_path": str(exp_dir),
        "result_file": str(fp),
        "ckpt_path": data.get("ckpt_path", ""),
    }
    row.update(parse_experiment_config(exp_dir))
    for k, v in val.items():
        row[k] = v
    for k, v in test.items():
        row[k] = v
    return row


def ordered_columns(rows: List[Dict[str, object]]) -> List[str]:
    base = [
        "exp_name",
        "cfg_pooling",
        "cfg_positional_embedding",
        "cfg_num_layers",
        "cfg_num_heads",
        "cfg_lr",
        "cfg_weight_decay",
        "cfg_label_smoothing",
        "cfg_feat_norm_mode",
        "cfg_aug_time_mask_prob",
        "cfg_aug_time_mask_max_frames",
        "cfg_aug_chunk_shuffle_prob",
        "cfg_aug_feat_dropout_prob",
        "ckpt_path",
        "test_eer",
        "test_minDCF",
        "test_min_tDCF",
        "test_FAR",
        "test_FRR",
        "test_PAR",
        "test_PRR",
        "test_acc",
        "test_f1",
    ]
    keys = set()
    for r in rows:
        keys.update(r.keys())
    ordered = [k for k in base if k in keys]
    rest = sorted(k for k in keys if k not in ordered)
    return ordered + rest


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_markdown(path: Path, rows: List[Dict[str, object]]) -> None:
    # Compact thesis-facing table with key test metrics.
    show_cols = [
        "exp_name",
        "cfg_pooling",
        "cfg_label_smoothing",
        "cfg_aug_time_mask_prob",
        "cfg_feat_norm_mode",
        "test_eer",
        "test_minDCF",
        "test_min_tDCF",
        "test_FAR",
        "test_FRR",
    ]
    present_cols = [c for c in show_cols if any(c in r for r in rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Experiment Summary ({datetime.now().isoformat(timespec='seconds')})")
    lines.append("")
    lines.append("| " + " | ".join(present_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(present_cols)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in present_cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_rows(rows: List[Dict[str, object]]) -> Tuple[Dict[str, object], Dict[str, object]]:
    rows_eer = [r for r in rows if isinstance(r.get("test_eer"), (float, int))]
    rows_dcf = [r for r in rows if isinstance(r.get("test_minDCF"), (float, int))]
    best_eer = min(rows_eer, key=lambda r: float(r["test_eer"])) if rows_eer else {}
    best_dcf = min(rows_dcf, key=lambda r: float(r["test_minDCF"])) if rows_dcf else {}
    return best_eer, best_dcf


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate all Exps/*/test_results.json into summary tables.")
    parser.add_argument("--exps-dir", default="Exps", help="Experiment root directory")
    parser.add_argument("--out-dir", default="Exps/_summary", help="Output directory for summary files")
    parser.add_argument("--sort-by", default="test_eer", help="Column used to sort rows ascending")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    exps_dir = (repo_root / args.exps_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    files = collect_result_files(exps_dir)
    if not files:
        print(f"[WARN] no test_results.json found under: {exps_dir}")
        return 1

    rows = [flatten_result(p) for p in files]
    if rows and args.sort_by in rows[0]:
        rows.sort(key=lambda r: float(r.get(args.sort_by, 1e9)) if isinstance(r.get(args.sort_by), (int, float)) else 1e9)

    cols = ordered_columns(rows)
    csv_path = out_dir / "all_test_results.csv"
    md_path = out_dir / "all_test_results.md"
    json_path = out_dir / "all_test_results.json"

    write_csv(csv_path, rows, cols)
    write_markdown(md_path, rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    best_eer, best_dcf = best_rows(rows)
    print(f"[DONE] aggregated {len(rows)} experiment(s)")
    print(f"[OUT]  csv:  {csv_path}")
    print(f"[OUT]  md:   {md_path}")
    print(f"[OUT]  json: {json_path}")
    if best_eer:
        print(f"[BEST] by test_eer: {best_eer.get('exp_name')} ({best_eer.get('test_eer')})")
    if best_dcf:
        print(f"[BEST] by test_minDCF: {best_dcf.get('exp_name')} ({best_dcf.get('test_minDCF')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
