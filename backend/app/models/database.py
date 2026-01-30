"""SQLAlchemy database models."""

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import AsyncGenerator

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Boolean,
    create_engine,
    TypeDecorator,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.config import get_settings

settings = get_settings()


class GUID(TypeDecorator):
    """Platform-independent GUID type using String for SQLite compatibility."""

    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid.UUID(value)
        return value


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class VideoStatus(str, PyEnum):
    """Video processing status."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"


class AnalysisStatus(str, PyEnum):
    """Analysis job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    DETECTING_SCENES = "detecting_scenes"
    ANALYZING_VISUALS = "analyzing_visuals"
    DETECTING_HOOKS = "detecting_hooks"
    SCORING_VIRALITY = "scoring_virality"
    SUGGESTING_CLIPS = "suggesting_clips"
    COMPLETED = "completed"
    FAILED = "failed"


class Video(Base):
    """Video file metadata."""

    __tablename__ = "videos"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Source info
    source_url = Column(String(2048), nullable=True)
    source_platform = Column(String(50), nullable=True)  # youtube, vimeo, etc.
    original_filename = Column(String(500), nullable=True)

    # File info
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)

    # Metadata
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    thumbnail_path = Column(String(1000), nullable=True)

    # Status
    status = Column(Enum(VideoStatus), default=VideoStatus.PENDING)
    error_message = Column(Text, nullable=True)

    # Relationships
    analysis_jobs = relationship("AnalysisJob", back_populates="video", cascade="all, delete-orphan")


class AnalysisJob(Base):
    """Video analysis job tracking."""

    __tablename__ = "analysis_jobs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    video_id = Column(GUID(), ForeignKey("videos.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Status
    status = Column(Enum(AnalysisStatus), default=AnalysisStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0-100
    current_stage = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)

    # Configuration
    config = Column(JSON, default=dict)  # Analysis options

    # Results summary
    total_duration = Column(Float, nullable=True)
    word_count = Column(Integer, nullable=True)
    speaker_count = Column(Integer, nullable=True)
    scene_count = Column(Integer, nullable=True)
    clip_count = Column(Integer, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="analysis_jobs")
    transcription = relationship(
        "Transcription", back_populates="analysis_job", uselist=False, cascade="all, delete-orphan"
    )
    diarization = relationship(
        "Diarization", back_populates="analysis_job", uselist=False, cascade="all, delete-orphan"
    )
    scenes = relationship("Scene", back_populates="analysis_job", cascade="all, delete-orphan")
    clips = relationship(
        "ClipSuggestion", back_populates="analysis_job", cascade="all, delete-orphan"
    )
    hooks = relationship("Hook", back_populates="analysis_job", cascade="all, delete-orphan")


class Transcription(Base):
    """Full transcript with word-level timestamps."""

    __tablename__ = "transcriptions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Full text
    full_text = Column(Text, nullable=False)
    language = Column(String(10), nullable=True)
    language_probability = Column(Float, nullable=True)

    # Stats
    word_count = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="transcription")
    segments = relationship(
        "TranscriptionSegment", back_populates="transcription", cascade="all, delete-orphan"
    )


class TranscriptionSegment(Base):
    """Individual transcription segment with timestamps."""

    __tablename__ = "transcription_segments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    transcription_id = Column(GUID(), ForeignKey("transcriptions.id"), nullable=False)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    # Content
    text = Column(Text, nullable=False)
    words = Column(JSON, nullable=True)  # [{word, start, end, probability}]

    # Confidence
    avg_log_prob = Column(Float, nullable=True)
    no_speech_prob = Column(Float, nullable=True)

    # Speaker (filled after diarization merge)
    speaker_id = Column(String(50), nullable=True)

    # Relationships
    transcription = relationship("Transcription", back_populates="segments")


class Diarization(Base):
    """Speaker diarization results."""

    __tablename__ = "diarizations"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Stats
    speaker_count = Column(Integer, default=0)
    speaker_labels = Column(JSON, default=list)  # ["SPEAKER_00", "SPEAKER_01"]

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="diarization")
    segments = relationship(
        "DiarizationSegment", back_populates="diarization", cascade="all, delete-orphan"
    )


class DiarizationSegment(Base):
    """Individual speaker segment."""

    __tablename__ = "diarization_segments"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    diarization_id = Column(GUID(), ForeignKey("diarizations.id"), nullable=False)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    # Speaker
    speaker_id = Column(String(50), nullable=False)

    # Relationships
    diarization = relationship("Diarization", back_populates="segments")


class Scene(Base):
    """Detected scene/shot boundary."""

    __tablename__ = "scenes"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    # Scene info
    scene_index = Column(Integer, nullable=False)
    scene_type = Column(String(50), nullable=True)  # hard_cut, fade, dissolve

    # Visual analysis
    keyframe_path = Column(String(1000), nullable=True)
    dominant_colors = Column(JSON, nullable=True)
    detected_objects = Column(JSON, nullable=True)  # YOLO results
    motion_intensity = Column(Float, nullable=True)  # 0-1

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="scenes")


class ClipSuggestion(Base):
    """AI-suggested clip with virality score."""

    __tablename__ = "clip_suggestions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)

    # Content
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    transcript_excerpt = Column(Text, nullable=True)

    # Scores (0-100)
    virality_score = Column(Float, nullable=False)
    emotional_resonance = Column(Float, default=0.0)
    shareability = Column(Float, default=0.0)
    uniqueness = Column(Float, default=0.0)
    hook_strength = Column(Float, default=0.0)
    production_quality = Column(Float, default=0.0)

    # Export
    exported = Column(Boolean, default=False)
    export_path = Column(String(1000), nullable=True)

    # Ranking
    rank = Column(Integer, nullable=True)

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="clips")


class Hook(Base):
    """Potential hook/opener detected in video."""

    __tablename__ = "hooks"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)

    # Timing
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    # Content
    text = Column(Text, nullable=True)
    hook_type = Column(String(50), nullable=True)  # question, statement, action, statistic

    # Scores (0-100)
    hook_score = Column(Float, nullable=False)
    curiosity_gap = Column(Float, default=0.0)
    emotional_trigger = Column(Float, default=0.0)
    clarity = Column(Float, default=0.0)
    visual_interest = Column(Float, default=0.0)

    # Position
    is_video_start = Column(Boolean, default=False)
    rank = Column(Integer, nullable=True)

    # Relationships
    analysis_job = relationship("AnalysisJob", back_populates="hooks")


# Database engines and sessions
# Handle SQLite vs PostgreSQL connection args
is_sqlite = "sqlite" in settings.database_url

if is_sqlite:
    async_engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args={"check_same_thread": False},
    )
    sync_engine = create_engine(
        settings.database_sync_url,
        echo=settings.debug,
        connect_args={"check_same_thread": False},
    )
else:
    async_engine = create_async_engine(settings.database_url, echo=settings.debug)
    sync_engine = create_engine(settings.database_sync_url, echo=settings.debug)

AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with AsyncSessionLocal() as session:
        yield session


def get_sync_session():
    """Get sync database session (for Celery workers)."""
    session = SyncSessionLocal()
    try:
        yield session
    finally:
        session.close()


async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
