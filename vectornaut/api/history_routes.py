# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from vectornaut.storage import history_dir, list_indexed_history, load_json_file


router = APIRouter(prefix="/api/history", tags=["history"])


def _safe_history_id(run_id: str) -> str:
    return os.path.basename(run_id).replace(".json", "")


def _history_file_for_id(run_id: str) -> str:
    safe_id = _safe_history_id(run_id)
    return os.path.join(history_dir(), f"{safe_id}.json")


def _summarize_history_run(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    miner = data.get("miner") or data.get("miner_stage") or {}
    auditor = data.get("auditor") or data.get("auditor_stage") or {}
    simulator = data.get("simulator") or data.get("simulator_stage") or {}
    stat = os.stat(path)

    return {
        "id": os.path.basename(path).replace(".json", ""),
        "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "design_name": miner.get("design_name") or "Unknown Design",
        "query": data.get("query") or "",
        "domain": miner.get("domain") or "",
        "solver_method": simulator.get("solver_method") or auditor.get("solver_method") or "",
        "performance_gain_pct": simulator.get("performance_gain_pct"),
        "relative_error": simulator.get("relative_error"),
    }


@router.get("")
async def list_history(limit: int = 30):
    try:
        indexed_runs = list_indexed_history(limit=limit)
        if indexed_runs:
            return JSONResponse(content={"runs": indexed_runs}, media_type="application/json; charset=utf-8")

        if not os.path.exists(history_dir()):
            return JSONResponse(content={"runs": []}, media_type="application/json; charset=utf-8")

        files = [
            os.path.join(history_dir(), name)
            for name in os.listdir(history_dir())
            if name.endswith(".json")
        ]
        files.sort(key=lambda path: os.path.getmtime(path), reverse=True)

        runs = []
        for path in files[: max(1, min(limit, 200))]:
            try:
                runs.append(_summarize_history_run(path))
            except Exception as run_err:
                runs.append({
                    "id": os.path.basename(path).replace(".json", ""),
                    "created_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
                    "design_name": "Unreadable run",
                    "error": str(run_err),
                })

        return JSONResponse(content={"runs": runs}, media_type="application/json; charset=utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{run_id}")
async def get_history_run(run_id: str):
    try:
        path = _history_file_for_id(run_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"History run not found: {run_id}")
        data = load_json_file(path)
        return JSONResponse(content=data, media_type="application/json; charset=utf-8")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
