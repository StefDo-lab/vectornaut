# -*- coding: utf-8 -*-
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


def get_data_dir() -> str:
    return os.path.abspath(os.environ.get("VECTORNAUT_DATA_DIR", "."))


def data_path(*parts: str) -> str:
    return os.path.join(get_data_dir(), *parts)


def history_dir() -> str:
    return data_path("history")


def reports_dir() -> str:
    return data_path("reports")


def eval_runs_dir() -> str:
    return data_path("eval_runs")


def sqlite_path() -> str:
    return data_path("vectornaut.sqlite3")


def _connect() -> sqlite3.Connection:
    os.makedirs(get_data_dir(), exist_ok=True)
    conn = sqlite3.connect(sqlite_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history_runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            design_name TEXT NOT NULL,
            query TEXT NOT NULL,
            domain TEXT NOT NULL,
            solver_method TEXT NOT NULL,
            performance_gain_pct REAL,
            relative_error REAL,
            history_path TEXT NOT NULL,
            report_path TEXT
        )
        """
    )
    conn.commit()
    return conn


def index_history_run(run_id: str, history_path_value: str, report_path_value: str, run_data: Dict[str, Any]) -> None:
    miner = run_data.get("miner", {}) if isinstance(run_data.get("miner"), dict) else {}
    auditor = run_data.get("auditor", {}) if isinstance(run_data.get("auditor"), dict) else {}
    simulator = run_data.get("simulator", {}) if isinstance(run_data.get("simulator"), dict) else {}

    created_at = datetime.fromtimestamp(os.path.getmtime(history_path_value)).isoformat(timespec="seconds")
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO history_runs (
                id, created_at, design_name, query, domain, solver_method,
                performance_gain_pct, relative_error, history_path, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                miner.get("design_name") or "Unknown Design",
                run_data.get("query") or "",
                miner.get("domain") or "",
                simulator.get("solver_method") or auditor.get("solver_method") or "",
                simulator.get("performance_gain_pct"),
                simulator.get("relative_error"),
                os.path.abspath(history_path_value),
                os.path.abspath(report_path_value) if report_path_value else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_indexed_history(limit: int = 30) -> List[Dict[str, Any]]:
    if not os.path.exists(sqlite_path()):
        return []

    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, created_at, design_name, query, domain, solver_method,
                   performance_gain_pct, relative_error, history_path, report_path
            FROM history_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
