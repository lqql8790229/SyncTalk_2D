"""FastAPI application for SyncTalk service."""

import uuid
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .schemas import (
    InferenceRequest, InferenceResponse,
    TrainRequest, TrainResponse,
    TaskStatusResponse, TaskStatus, HealthResponse,
)
from ..configs.base import SyncTalkConfig
from ..inference.engine import InferenceEngine
from .. import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SyncTalk API",
    description="Commercial-grade 2D lip-sync video generation API",
    version=__version__,
)

tasks: Dict[str, dict] = {}
executor = ThreadPoolExecutor(max_workers=2)


def _run_inference(task_id: str, request: InferenceRequest):
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        config = SyncTalkConfig.from_resolution(
            request.resolution, asr_mode=request.asr_mode
        )
        engine = InferenceEngine(config)
        result_path = engine.generate(
            name=request.name,
            audio_path=request.audio_path,
            checkpoint_path=request.checkpoint_path,
        )
        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["result_path"] = result_path
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        tasks[task_id]["status"] = TaskStatus.FAILED
        tasks[task_id]["error"] = str(e)


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu = torch.cuda.is_available()
    return HealthResponse(
        status="healthy",
        version=__version__,
        gpu_available=gpu,
        device="cuda" if gpu else "cpu",
    )


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def create_inference(request: InferenceRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "result_path": None,
        "error": None,
    }
    executor.submit(_run_inference, task_id, request)
    return InferenceResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Inference task submitted",
    )


@app.get("/api/v1/inference/{task_id}", response_model=TaskStatusResponse)
async def get_inference_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result_path=task.get("result_path"),
        error=task.get("error"),
    )


@app.get("/api/v1/inference/{task_id}/video")
async def download_video(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is {task['status']}")
    path = task["result_path"]
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4")
