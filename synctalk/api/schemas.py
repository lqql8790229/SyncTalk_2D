"""API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceRequest(BaseModel):
    name: str = Field(..., description="Dataset/project name")
    audio_path: str = Field(..., description="Path to audio WAV file")
    checkpoint_path: Optional[str] = Field(None, description="Model checkpoint path")
    resolution: int = Field(328, description="Resolution (160 or 328)")
    asr_mode: str = Field("ave", description="ASR mode (ave, hubert, wenet)")


class InferenceResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result_path: Optional[str] = None
    error: Optional[str] = None
    progress: float = 0.0


class TrainRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    resolution: int = Field(328, description="Resolution")
    epochs: int = Field(100, description="Training epochs")
    batch_size: int = Field(8, description="Batch size")
    asr_mode: str = Field("ave", description="ASR mode")
    use_syncnet: bool = Field(True, description="Use SyncNet loss")


class TrainResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    device: str
