# -*- coding: utf-8 -*-
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from vectornaut.config import MinerOutput
from vectornaut.pipeline import PipelineRunRequest, run_pipeline
from vectornaut.reporting import archive_run_data


router = APIRouter(prefix="/api", tags=["run"])


class RunRequest(PipelineRunRequest):
    previous_miner_output: Optional[MinerOutput] = None


@router.post("/run")
async def run_discovery_loop(req: RunRequest):
    try:
        response_data = run_pipeline(req)
        archive_run_data(response_data)
        return JSONResponse(content=response_data, media_type="application/json; charset=utf-8")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
