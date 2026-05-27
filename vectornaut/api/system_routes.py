# -*- coding: utf-8 -*-
import os
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from vectornaut.storage import get_data_dir, sqlite_path


router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health() -> JSONResponse:
    payload: Dict[str, Any] = {
        "status": "ok",
        "service": "vectornaut",
        "data_dir": get_data_dir(),
        "sqlite_index": sqlite_path(),
        "sqlite_index_exists": os.path.exists(sqlite_path()),
    }
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8")
