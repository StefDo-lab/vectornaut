# -*- coding: utf-8 -*-
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Ensure correct pathing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectornaut.api.chat_routes import router as chat_router
from vectornaut.api.eval_routes import router as eval_router
from vectornaut.api.history_routes import router as history_router
from vectornaut.api.run_routes import router as run_router

# Load environment vars
if load_dotenv:
    load_dotenv()

app = FastAPI(
    title="Vectornaut Omni API",
    description="Backend API services orchestrating Miner, Auditor, and Dynamic Solver stages."
)
app.include_router(run_router)
app.include_router(chat_router)
app.include_router(eval_router)
app.include_router(history_router)

# Mount static folder for frontend UI (will be served at root "/")
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    print("[*] Launching Vectornaut 2.0 Web Server on http://0.0.0.0:8080 ...")
    uvicorn.run("web_server:app", host="0.0.0.0", port=8080, reload=False)
