# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vectornaut.config import MinerOutput
from vectornaut.pipeline import ConceptsExhaustedError, PipelineRunRequest, compare_solvers, run_pipeline
from vectornaut.reporting import archive_run_data


router = APIRouter(prefix="/api", tags=["run"])


class RunRequest(PipelineRunRequest):
    previous_miner_output: Optional[MinerOutput] = None


class SolverCompareRequest(BaseModel):
    current_run: Dict[str, Any]
    methods: Optional[List[str]] = None
    epochs: int = 200
    is_mock: bool = False


@router.post("/run")
async def run_discovery_loop(req: RunRequest):
    try:
        # A request rejected as physically infeasible is a regular result
        # (success=false, status="rejected") and is archived like any other run.
        response_data = run_pipeline(req)
        archive_run_data(response_data)
        return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")
    except ConceptsExhaustedError as e:
        # Every concept failed: the request was processed but produced no usable result.
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "status": "failed",
                "detail": str(e),
                "error": {
                    "type": "concepts_exhausted",
                    "message": str(e),
                    "attempts": e.attempts,
                    "failed_concepts": e.failed_concepts,
                },
                "query": req.query,
            },
            media_type="application/json; charset=utf-8",
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/solver-compare")
async def compare_solver_methods(req: SolverCompareRequest):
    try:
        response_data = compare_solvers(
            current_run=req.current_run,
            methods=req.methods,
            epochs=req.epochs,
            is_mock=req.is_mock,
        )
        return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
