"""Database connection and ORM models using SQLite (dev) / PostgreSQL (prod).

Uses SQLite for development and local testing. Switch DATABASE_URL
environment variable to PostgreSQL for production.
"""

import os
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime,
    ForeignKey, Text, Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///synctalk.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), default="")
    plan = Column(String(20), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    characters = relationship("CharacterModel", back_populates="user", cascade="all,delete")


class CharacterModel(Base):
    __tablename__ = "characters"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    resolution = Column(Integer, default=328)
    asr_mode = Column(String(20), default="ave")
    status = Column(String(20), default="created")
    frame_count = Column(Integer, default=0)
    checkpoint_hash = Column(String(64), default="")
    model_url = Column(String(500), default="")
    thumbnail_url = Column(String(500), default="")
    file_size_mb = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="characters")


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(String(36), primary_key=True, default=gen_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    character_id = Column(String(36), default="")
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime)
    duration_min = Column(Float, default=0)
    mode = Column(String(20), default="")
    fps_avg = Column(Float, default=0)
    gpu_model = Column(String(100), default="")


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
