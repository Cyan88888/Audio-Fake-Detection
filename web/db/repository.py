from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List

from .. import config
from .models import AuditRecord, HistoryRecord


class SqliteRepository:
    def __init__(self) -> None:
        self.db_path = config.STORAGE_DIR / "web_app.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    pred_label TEXT NOT NULL,
                    prob_spoof REAL NOT NULL,
                    prob_bonafide REAL NOT NULL,
                    threshold REAL NOT NULL,
                    decision_by_threshold TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def insert_history(self, records: Iterable[HistoryRecord]) -> None:
        rows = [
            (
                r.job_id,
                r.filename,
                r.pred_label,
                r.prob_spoof,
                r.prob_bonafide,
                r.threshold,
                r.decision_by_threshold,
                r.model_version,
                r.inference_time_ms,
                r.created_at,
                r.payload_json,
            )
            for r in records
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO prediction_history (
                    job_id, filename, pred_label, prob_spoof, prob_bonafide, threshold,
                    decision_by_threshold, model_version, inference_time_ms, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def list_history(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM prediction_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cur.fetchall()

    def insert_audit(self, row: AuditRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (action, actor, detail, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (row.action, row.actor, row.detail, row.created_at),
            )
            conn.commit()

    def list_audit(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT * FROM audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            return cur.fetchall()


repository = SqliteRepository()
