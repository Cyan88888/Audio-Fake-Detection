from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .. import config
from ..db.models import HistoryRecord
from ..db.repository import repository


def save_prediction_items(job_id: str, items: List[Dict]) -> None:
    rows = []
    for item in items:
        rows.append(
            HistoryRecord(
                id=None,
                job_id=job_id,
                filename=item["filename"],
                pred_label=item["pred_label"],
                prob_spoof=float(item["prob_spoof"]),
                prob_bonafide=float(item["prob_bonafide"]),
                threshold=float(item["threshold"]),
                decision_by_threshold=item["decision_by_threshold"],
                model_version=item["model_version"],
                inference_time_ms=float(item["inference_time_ms"]),
                created_at=item["created_at"],
                payload_json=json.dumps(item, ensure_ascii=False),
            )
        )
    repository.insert_history(rows)


def list_history(limit: int = 100) -> List[Dict]:
    rows = repository.list_history(limit=limit)
    out = []
    for r in rows:
        payload = json.loads(r["payload_json"])
        payload["history_id"] = r["id"]
        payload["job_id"] = r["job_id"]
        out.append(payload)
    return out


def export_history_csv(limit: int = 100) -> Path:
    items = list_history(limit=limit)
    output = config.STORAGE_DIR / "history_export.csv"
    fieldnames = [
        "history_id",
        "job_id",
        "filename",
        "pred_label",
        "prob_bonafide",
        "prob_spoof",
        "threshold",
        "decision_by_threshold",
        "model_version",
        "inference_time_ms",
        "created_at",
    ]
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in items:
            writer.writerow({k: i.get(k, "") for k in fieldnames})
    return output


def export_history_json(limit: int = 100) -> Path:
    items = list_history(limit=limit)
    output = config.STORAGE_DIR / "history_export.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return output
