"""FastAPI application for SyncTalk service."""

import os
import uuid
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

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
from ..training.trainer import Trainer
from ..models.export import get_model_info
from ..models.unet import UNet
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


def _run_training(task_id: str, request: TrainRequest):
    try:
        tasks[task_id]["status"] = TaskStatus.PROCESSING
        config = SyncTalkConfig.from_resolution(
            request.resolution, asr_mode=request.asr_mode
        )
        config.train.epochs = request.epochs
        config.train.batch_size = request.batch_size
        config.train.use_syncnet = request.use_syncnet

        dataset_dir = f"./dataset/{request.name}"
        save_dir = f"./checkpoint/{request.name}"
        syncnet_dir = f"./syncnet_ckpt/{request.name}"

        trainer = Trainer(config)
        trainer.train_full(dataset_dir, save_dir, syncnet_dir)

        tasks[task_id]["status"] = TaskStatus.COMPLETED
        tasks[task_id]["result_path"] = save_dir
    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}", exc_info=True)
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
        "type": "inference",
        "result_path": None,
        "error": None,
    }
    executor.submit(_run_inference, task_id, request)
    return InferenceResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="Inference task submitted",
    )


@app.post("/api/v1/train", response_model=TrainResponse)
async def create_training(request: TrainRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "type": "training",
        "result_path": None,
        "error": None,
    }
    executor.submit(_run_training, task_id, request)
    return TrainResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"Training task submitted for '{request.name}'",
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        result_path=task.get("result_path"),
        error=task.get("error"),
    )


@app.get("/api/v1/inference/{task_id}", response_model=TaskStatusResponse)
async def get_inference_status(task_id: str):
    return await get_task_status(task_id)


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


@app.get("/api/v1/projects")
async def list_projects():
    """List available projects/datasets."""
    dataset_dir = Path("./dataset")
    projects = []
    if dataset_dir.exists():
        for d in sorted(dataset_dir.iterdir()):
            if d.is_dir():
                has_video = any(d.glob("*.mp4"))
                has_checkpoint = Path(f"./checkpoint/{d.name}").exists()
                projects.append({
                    "name": d.name,
                    "has_video": has_video,
                    "has_checkpoint": has_checkpoint,
                })
    return {"projects": projects}


@app.get("/api/v1/models/info")
async def model_info():
    """Get model architecture information."""
    configs = {}
    for res in [160, 328]:
        from ..configs.base import ModelConfig
        cfg = ModelConfig(resolution=res)
        model = UNet.from_config(cfg)
        info = get_model_info(model)
        info["resolution"] = res
        info["n_down_layers"] = cfg.n_down_layers
        configs[f"{res}px"] = info
    return configs
