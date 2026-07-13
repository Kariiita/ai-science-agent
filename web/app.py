"""FastAPI HTTP service wrapping the AutoResearcher agent.

Usage:
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api import AutoResearcher

app = FastAPI(title="V-SciAgent AutoResearcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_API_KEY = os.environ.get("API_KEY", "")
_PROJECTS_BASE = Path(_PROJECT_ROOT)
_active_researchers: dict = {}
_cycle_results: dict = {}


def _check_api_key(x_api_key: Optional[str]):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _get_researcher(project_id: str) -> AutoResearcher:
    project_dir = _PROJECTS_BASE / project_id
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found: " + project_id)
    if project_id not in _active_researchers:
        _active_researchers[project_id] = AutoResearcher(str(project_dir))
    return _active_researchers[project_id]


class ProjectCreate(BaseModel):
    name: str
    brief: str
    target_metrics: dict = {}
    dataset_path: str = "data"


class CycleRequest(BaseModel):
    project_id: str
    n_cycles: int = 1


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/project")
async def create_project(req: ProjectCreate, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    project_dir = _PROJECTS_BASE / req.name
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "PROJECT_BRIEF.md").write_text(
        "# " + req.name + "\n\n" + req.brief + "\n", encoding="utf-8")
    metrics_yaml = ""
    for key, target in req.target_metrics.items():
        direction = "higher" if "delta" in key else "lower"
        metrics_yaml += '    - key: "%s"\n      target: %s\n      direction: "%s"\n' % (key, target, direction)
    config = (
        'project:\n  name: "%s"\n  brief: "PROJECT_BRIEF.md"\n  workspace: "."\n\n'
        'goals:\n  metrics:\n%s  stop_on_achieved: true\n\n'
        'agent:\n  provider: "dashscope"\n  model: "auto"\n  max_cycles: 10\n'
    ) % (req.name, metrics_yaml)
    (project_dir / "config.yaml").write_text(config, encoding="utf-8")
    return {"project_id": req.name, "path": str(project_dir)}


@app.post("/api/cycle")
async def run_cycle(req: CycleRequest, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    researcher = _get_researcher(req.project_id)

    def _run():
        try:
            if req.n_cycles == 1:
                _cycle_results[req.project_id] = researcher.run_one_cycle()
            else:
                results = researcher.run_n_cycles(req.n_cycles)
                _cycle_results[req.project_id] = results[-1] if results else {}
        except Exception as e:
            _cycle_results[req.project_id] = {"error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started", "project_id": req.project_id, "n_cycles": req.n_cycles}


@app.get("/api/status/{project_id}")
async def get_status(project_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    researcher = _get_researcher(project_id)
    status = researcher.get_status()
    if project_id in _cycle_results:
        status["last_cycle_result"] = _cycle_results[project_id]
    return status


@app.get("/api/history/{project_id}")
async def get_history(project_id: str, limit: int = 20, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    researcher = _get_researcher(project_id)
    return {"project_id": project_id, "history": researcher.get_experiment_history(limit=limit)}


@app.get("/api/results/{project_id}")
async def get_results(project_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    researcher = _get_researcher(project_id)
    history = researcher.get_experiment_history(limit=1)
    report_path = _PROJECTS_BASE / project_id / "RESEARCH_REPORT.md"
    report_text = ""
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
    return {"project_id": project_id, "latest_metrics": history[0] if history else {}, "report": report_text}


@app.get("/api/agents/trace/{project_id}")
async def get_agent_trace(project_id: str, limit: int = 50, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    trace_path = _PROJECTS_BASE / project_id / "logs" / "agent_trace.jsonl"
    traces = []
    if trace_path.exists():
        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines[-limit:]:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return {"project_id": project_id, "traces": traces}


@app.get("/api/verify/{project_id}")
async def get_verify(project_id: str, x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    verify_path = _PROJECTS_BASE / project_id / "logs" / "verify_result.json"
    if verify_path.exists():
        return json.loads(verify_path.read_text(encoding="utf-8"))
    return {"project_id": project_id, "message": "No verification results yet"}


@app.websocket("/ws/status/{project_id}")
async def ws_status(websocket: WebSocket, project_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                researcher = _get_researcher(project_id)
                await websocket.send_json(researcher.get_status())
            except HTTPException:
                await websocket.send_json({"error": "Project not found"})
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
