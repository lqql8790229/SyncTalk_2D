"""Cloud API server for SyncTalk.

Handles: user auth, character metadata, usage tracking.
Does NOT handle: GPU inference (runs on user's local machine).
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from .database import get_db, init_db, User, CharacterModel, UsageLog
from .auth import (
    hash_password, verify_password, create_access_token, get_current_user_id,
)
from .. import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SyncTalk Cloud API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    logger.info("Database initialized")


# ── Schemas ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    plan: str

class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    plan: str
    created_at: datetime

class CharacterCreate(BaseModel):
    name: str
    resolution: int = 328
    asr_mode: str = "ave"

class CharacterResponse(BaseModel):
    id: str
    name: str
    resolution: int
    asr_mode: str
    status: str
    frame_count: int
    file_size_mb: float
    created_at: datetime

class HeartbeatRequest(BaseModel):
    character_id: str = ""
    mode: str = ""
    fps_avg: float = 0
    gpu_model: str = ""


# ── Auth Routes ──────────────────────────────────────────

@app.post("/api/v1/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(400, "Email already registered")

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        display_name=req.display_name or req.email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token, user_id=user.id,
        email=user.email, plan=user.plan,
    )


@app.post("/api/v1/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token(user.id, user.email)
    return TokenResponse(
        access_token=token, user_id=user.id,
        email=user.email, plan=user.plan,
    )


@app.get("/api/v1/auth/me", response_model=UserResponse)
def get_me(user_id: str = Depends(get_current_user_id),
           db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    return UserResponse(
        id=user.id, email=user.email,
        display_name=user.display_name,
        plan=user.plan, created_at=user.created_at,
    )


# ── Character Routes ─────────────────────────────────────

PLAN_LIMITS = {"free": 1, "pro": 5, "business": 999}

@app.get("/api/v1/characters", response_model=list[CharacterResponse])
def list_characters(user_id: str = Depends(get_current_user_id),
                    db: Session = Depends(get_db)):
    chars = db.query(CharacterModel).filter(
        CharacterModel.user_id == user_id
    ).order_by(CharacterModel.created_at.desc()).all()
    return [CharacterResponse(
        id=c.id, name=c.name, resolution=c.resolution,
        asr_mode=c.asr_mode, status=c.status,
        frame_count=c.frame_count, file_size_mb=c.file_size_mb,
        created_at=c.created_at,
    ) for c in chars]


@app.post("/api/v1/characters", response_model=CharacterResponse)
def create_character(req: CharacterCreate,
                     user_id: str = Depends(get_current_user_id),
                     db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    limit = PLAN_LIMITS.get(user.plan, 1)
    count = db.query(CharacterModel).filter(
        CharacterModel.user_id == user_id
    ).count()
    if count >= limit:
        raise HTTPException(
            403, f"Character limit reached ({limit} for {user.plan} plan)"
        )

    char = CharacterModel(
        user_id=user_id, name=req.name,
        resolution=req.resolution, asr_mode=req.asr_mode,
    )
    db.add(char)
    db.commit()
    db.refresh(char)

    return CharacterResponse(
        id=char.id, name=char.name, resolution=char.resolution,
        asr_mode=char.asr_mode, status=char.status,
        frame_count=char.frame_count, file_size_mb=char.file_size_mb,
        created_at=char.created_at,
    )


@app.delete("/api/v1/characters/{character_id}")
def delete_character(character_id: str,
                     user_id: str = Depends(get_current_user_id),
                     db: Session = Depends(get_db)):
    char = db.query(CharacterModel).filter(
        CharacterModel.id == character_id,
        CharacterModel.user_id == user_id,
    ).first()
    if not char:
        raise HTTPException(404, "Character not found")
    db.delete(char)
    db.commit()
    return {"status": "deleted"}


# ── Usage Routes ─────────────────────────────────────────

@app.post("/api/v1/usage/heartbeat")
def heartbeat(req: HeartbeatRequest,
              user_id: str = Depends(get_current_user_id),
              db: Session = Depends(get_db)):
    log = UsageLog(
        user_id=user_id, character_id=req.character_id,
        mode=req.mode, fps_avg=req.fps_avg, gpu_model=req.gpu_model,
    )
    db.add(log)
    db.commit()
    return {"status": "ok"}


@app.get("/api/v1/health")
def health():
    return {"status": "healthy", "version": __version__, "service": "cloud"}
