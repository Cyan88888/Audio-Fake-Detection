#!/usr/bin/env python3
"""Batch-run `test.py` for multiple configs and checkpoints.

Usage example:
  python scripts/run_all_tests.py --config-dir config --python python
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

def list_config_files(config_dir: Path, pattern: str) -> List[Path]:
    return sorted(p for p in config_dir.glob(pattern) if p.is_file())


def parse_exp_dir_from_config(cfg_path: Path) -> Optional[Path]:
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    m_block = re.search(r"(?m)^exp:\s*$", text)
    if not m_block:
        return None
    tail = text[m_block.end() :]
    exp_dir = None
    exp_name = None
    for line in tail.splitlines():
        # Stop when leaving `exp` block.
        if line and not line.startswith(" ") and not line.startswith("\t"):
            break
        s = line.strip()
        if s.startswith("dir:"):
            exp_dir = s.split(":", 1)[1].strip().strip("'\"")
        elif s.startswith("name:"):
            exp_name = s.split(":", 1)[1].strip().strip("'\"")
    if not exp_dir or not exp_name:
        return None
    return (cfg_path.parent.parent / str(exp_dir) / str(exp_name)).resolve()


def score_ckpt_name(ckpt: Path) -> float:
    """Lower score is better; try val_eer from filename first."""
    name = ckpt.name
    if "val_eer=" in name:
        try:
            tail = name.split("val_eer=", 1)[1]
            num = tail.split(".ckpt", 1)[0]
            return float(num)
        except Exception:
            pass
    if name.startswith("last"):
        return 1e9
    return 1e8


def choose_checkpoint(exp_dir: Path, strategy: str) -> Optional[Path]:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    cands = sorted(ckpt_dir.glob("*.ckpt"))
    if not cands:
        return None

    if strategy == "last":
        for c in cands:
            if c.name.startswith("last"):
                return c
        return cands[-1]

    if strategy == "best":
        return sorted(cands, key=score_ckpt_name)[0]

    # auto
    best = sorted(cands, key=score_ckpt_name)[0]
    return best


def progress_line(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[------------------------] 0/0"
    done = max(0, min(done, total))
    fill = int(width * done / total)
    bar = "#" * fill + "-" * (width - fill)
    return f"[{bar}] {done}/{total}"


def run_one(
    python_bin: str,
    repo_root: Path,
    cfg_path: Path,
    ckpt_path: Path,
    dry_run: bool,
    stream_log: bool,
) -> Dict[str, object]:
    cmd = [
        python_bin,
        "test.py",
        "--conf_dir",
        str(cfg_path.relative_to(repo_root)),
        "--ckpt_path",
        str(ckpt_path.relative_to(repo_root)),
    ]
    started_at = datetime.now().isoformat(timespec="seconds")
    if dry_run:
        return {
            "config": str(cfg_path),
            "checkpoint": str(ckpt_path),
            "status": "dry_run",
            "returncode": 0,
            "command": cmd,
            "started_at": started_at,
            "finished_at": started_at,
        }

    if stream_log:
        merged_tail: deque[str] = deque(maxlen=40)
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"      {line}")
            merged_tail.append(line)
        proc.wait()
        stdout_tail = "\n".join(merged_tail)
        stderr_tail = ""
    else:
        proc2 = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
        proc = proc2
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])

    finished_at = datetime.now().isoformat(timespec="seconds")
    return {
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "command": cmd,
        "started_at": started_at,
        "finished_at": finished_at,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto run test.py for all experiment configs.")
    parser.add_argument("--config-dir", default="config", help="Config directory, default: config")
    parser.add_argument("--pattern", default="*.yaml", help="Glob pattern under config-dir")
    parser.add_argument(
        "--ckpt-strategy",
        default="auto",
        choices=["auto", "best", "last"],
        help="Checkpoint selection strategy",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--only", nargs="*", default=None, help="Only run configs whose filename contains any keyword")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip configs whose filename contains any keyword")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument(
        "--no-stream-log",
        action="store_true",
        help="Disable real-time test.py logs (fallback to tail-only capture).",
    )
    parser.add_argument(
        "--report",
        default="Exps/_batch_test_report.json",
        help="Where to write run report JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_dir = (repo_root / args.config_dir).resolve()
    if not config_dir.is_dir():
        print(f"[ERROR] config dir not found: {config_dir}")
        return 2

    cfg_files = list_config_files(config_dir, args.pattern)
    if args.only:
        keys = [k.lower() for k in args.only]
        cfg_files = [p for p in cfg_files if any(k in p.name.lower() for k in keys)]
    if args.skip:
        keys = [k.lower() for k in args.skip]
        cfg_files = [p for p in cfg_files if not any(k in p.name.lower() for k in keys)]

    print(f"[INFO] found {len(cfg_files)} config(s)")
    stream_log = not args.no_stream_log

    results: List[Dict[str, object]] = []
    for idx, cfg_path in enumerate(cfg_files, start=1):
        print(f"[PROGRESS] {progress_line(idx - 1, len(cfg_files))}")
        exp_dir = parse_exp_dir_from_config(cfg_path)
        if exp_dir is None:
            print(f"[{idx}/{len(cfg_files)}] SKIP {cfg_path.name}: cannot parse exp.dir/exp.name")
            results.append({"config": str(cfg_path), "status": "skip_no_exp"})
            continue
        ckpt = choose_checkpoint(exp_dir, args.ckpt_strategy)
        if ckpt is None:
            print(f"[{idx}/{len(cfg_files)}] SKIP {cfg_path.name}: no checkpoint under {exp_dir / 'checkpoints'}")
            results.append({"config": str(cfg_path), "status": "skip_no_ckpt", "exp_dir": str(exp_dir)})
            continue

        print(f"[{idx}/{len(cfg_files)}] RUN  {cfg_path.name}  ckpt={ckpt.name}")
        item = run_one(args.python, repo_root, cfg_path, ckpt, args.dry_run, stream_log=stream_log)
        results.append(item)
        if item.get("status") == "failed":
            print(f"    -> FAILED (returncode={item.get('returncode')})")
        else:
            print("    -> OK")
    print(f"[PROGRESS] {progress_line(len(cfg_files), len(cfg_files))}")

    report_path = (repo_root / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "config_dir": str(config_dir),
        "count_total": len(results),
        "count_ok": sum(1 for x in results if x.get("status") == "ok"),
        "count_failed": sum(1 for x in results if x.get("status") == "failed"),
        "count_skipped": sum(1 for x in results if str(x.get("status", "")).startswith("skip")),
        "items": results,
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
