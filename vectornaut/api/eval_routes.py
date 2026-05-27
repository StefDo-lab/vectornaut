# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vectornaut.config import MinerOutput
from vectornaut.evaluation import (
    build_eval_trace,
    evaluate_run_output,
    list_eval_records,
    load_eval_record,
    load_history_record,
    make_eval_run_id,
    persist_eval_record,
    utcish_now,
)
from vectornaut.pipeline import PipelineRunRequest, run_pipeline
from vectornaut.reporting import archive_run_data


router = APIRouter(prefix="/api", tags=["eval"])


class EvalRunRequest(BaseModel):
    name: Optional[str] = None
    query: str
    epochs: int = 80
    is_mock: bool = True
    override_parameters: Optional[Dict[str, float]] = None
    previous_miner_output: Optional[MinerOutput] = None
    max_optimization_rounds: int = 2
    criteria: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class EvalBatchRequest(BaseModel):
    cases: List[EvalRunRequest]
    stop_on_failure: bool = False


class EvalJudgeRequest(BaseModel):
    run: Dict[str, Any]
    criteria: Dict[str, Any] = Field(default_factory=dict)


class DebugReplayRequest(BaseModel):
    history_file: Optional[str] = None
    eval_run_id: Optional[str] = None
    run: Optional[Dict[str, Any]] = None
    rerun: bool = False
    criteria: Dict[str, Any] = Field(default_factory=dict)
    epochs: Optional[int] = None
    is_mock: Optional[bool] = None


async def _execute_eval_run(req: EvalRunRequest) -> Dict[str, Any]:
    run_id = make_eval_run_id()
    started_at = utcish_now()
    start_perf = time.perf_counter()
    output_data: Optional[Dict[str, Any]] = None
    status = "ok"
    error_message: Optional[str] = None

    run_request = PipelineRunRequest(
        query=req.query,
        epochs=req.epochs,
        is_mock=req.is_mock,
        override_parameters=req.override_parameters,
        previous_miner_output=req.previous_miner_output,
        max_optimization_rounds=req.max_optimization_rounds,
    )

    try:
        output_data = run_pipeline(run_request)
        archive_run_data(output_data)
    except Exception as exc:
        status = "failed"
        error_message = str(exc)

    duration_ms = int((time.perf_counter() - start_perf) * 1000)
    finished_at = utcish_now()
    request_data = req.model_dump()
    trace = build_eval_trace(
        run_id=run_id,
        request_data=request_data,
        run_data=output_data,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        status=status,
        error=error_message,
    )
    evaluation = evaluate_run_output(output_data or {}, req.criteria)
    if status == "failed":
        evaluation["passed"] = False

    record = {
        "run_id": run_id,
        "name": req.name,
        "tags": req.tags,
        "status": status,
        "trace": trace,
        "evaluation": evaluation,
        "output": output_data,
        "error": error_message,
    }
    record["storage_path"] = persist_eval_record(record)
    return record


@router.post("/eval/run")
async def eval_run(req: EvalRunRequest):
    record = await _execute_eval_run(req)
    return JSONResponse(content=record, media_type="application/json; charset=utf-8")


@router.post("/eval/batch")
async def eval_batch(req: EvalBatchRequest):
    records = []
    for case in req.cases:
        record = await _execute_eval_run(case)
        records.append(record)
        if req.stop_on_failure and not record.get("evaluation", {}).get("passed", False):
            break

    passed_count = sum(1 for record in records if record.get("evaluation", {}).get("passed", False))
    response = {
        "success": passed_count == len(records) and len(records) == len(req.cases),
        "requested": len(req.cases),
        "executed": len(records),
        "passed": passed_count,
        "failed": len(records) - passed_count,
        "records": records,
    }
    return JSONResponse(content=response, media_type="application/json; charset=utf-8")


@router.post("/eval/judge")
async def eval_judge(req: EvalJudgeRequest):
    evaluation = evaluate_run_output(req.run, req.criteria)
    return JSONResponse(content=evaluation, media_type="application/json; charset=utf-8")


@router.get("/eval/runs")
async def list_eval_runs(limit: int = 50):
    records = list_eval_records(limit=limit)
    return JSONResponse(content={"runs": records}, media_type="application/json; charset=utf-8")


@router.get("/eval/runs/{run_id}")
async def get_eval_run(run_id: str):
    try:
        record = load_eval_record(run_id)
        return JSONResponse(content=record, media_type="application/json; charset=utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/debug/replay")
async def debug_replay(req: DebugReplayRequest):
    try:
        source_data: Optional[Dict[str, Any]] = None
        source_label = "inline"

        if req.run is not None:
            source_data = req.run
        elif req.eval_run_id:
            record = load_eval_record(req.eval_run_id)
            source_label = req.eval_run_id
            source_data = record.get("output")
            if source_data is None:
                source_data = record
        elif req.history_file:
            source_label = req.history_file
            source_data = load_history_record(req.history_file)
        else:
            raise HTTPException(status_code=400, detail="Provide one of: run, eval_run_id, or history_file.")

        if not req.rerun:
            evaluation = evaluate_run_output(source_data or {}, req.criteria)
            replay_run_id = make_eval_run_id()
            replay_record = {
                "run_id": replay_run_id,
                "name": f"replay:{source_label}",
                "status": "replayed",
                "trace": build_eval_trace(
                    run_id=replay_run_id,
                    request_data={"source": source_label, "rerun": False},
                    run_data=source_data,
                    started_at=utcish_now(),
                    finished_at=utcish_now(),
                    duration_ms=0,
                    status="replayed",
                ),
                "evaluation": evaluation,
                "output": source_data,
                "error": None,
            }
            replay_record["storage_path"] = persist_eval_record(replay_record)
            return JSONResponse(content=replay_record, media_type="application/json; charset=utf-8")

        query = None
        if source_data:
            query = source_data.get("query")
        if not query and req.eval_run_id:
            record = load_eval_record(req.eval_run_id)
            query = get_nested_request_query(record)
        if not query:
            raise HTTPException(status_code=400, detail="Cannot rerun this record because it does not contain the original query.")

        eval_req = EvalRunRequest(
            name=f"rerun:{source_label}",
            query=query,
            epochs=req.epochs or int(source_data.get("epochs", 80) if source_data else 80),
            is_mock=req.is_mock if req.is_mock is not None else bool(source_data.get("is_mock", True) if source_data else True),
            criteria=req.criteria,
            tags=["debug-replay"],
        )
        record = await _execute_eval_run(eval_req)
        return JSONResponse(content=record, media_type="application/json; charset=utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


def get_nested_request_query(record: Dict[str, Any]) -> Optional[str]:
    trace = record.get("trace", {})
    request = trace.get("request", {}) if isinstance(trace, dict) else {}
    query = request.get("query") if isinstance(request, dict) else None
    return query if isinstance(query, str) and query.strip() else None
