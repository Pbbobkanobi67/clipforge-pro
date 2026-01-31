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


# ============== OpusClip Pro Features ==============


class CaptionStyleType(str, PyEnum):
    """Caption animation style types."""

    KARAOKE = "karaoke"  # Word-by-word highlight
    BOUNCE = "bounce"  # Words bounce in
    TYPEWRITER = "typewriter"  # Letters appear one by one
    FADE = "fade"  # Words fade in
    GLOW = "glow"  # Glowing highlight effect
    WAVE = "wave"  # Wave animation
    STATIC = "static"  # No animation


class CaptionPosition(str, PyEnum):
    """Caption vertical position."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


class CaptionStyle(Base):
    """Caption style configuration for animated captions."""

    __tablename__ = "caption_styles"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Basic info
    name = Column(String(100), nullable=False)
    is_default = Column(Boolean, default=False)

    # Animation
    style_type = Column(Enum(CaptionStyleType), default=CaptionStyleType.KARAOKE)
    animation_duration = Column(Float, default=0.1)  # Seconds per word/character

    # Font settings
    font_family = Column(String(100), default="Arial")
    font_size = Column(Integer, default=48)
    font_weight = Column(String(20), default="bold")  # normal, bold, etc.

    # Colors (hex format)
    text_color = Column(String(20), default="#FFFFFF")
    highlight_color = Column(String(20), default="#FFFF00")  # For karaoke effect
    background_color = Column(String(20), nullable=True)  # Optional background box
    stroke_color = Column(String(20), default="#000000")
    stroke_width = Column(Integer, default=2)

    # Position
    position = Column(Enum(CaptionPosition), default=CaptionPosition.BOTTOM)
    margin_bottom = Column(Integer, default=50)  # Pixels from edge
    margin_horizontal = Column(Integer, default=40)

    # Display settings
    words_per_line = Column(Integer, default=6)
    max_lines = Column(Integer, default=2)

    # Relationships
    brand_templates = relationship("BrandTemplate", back_populates="caption_style")


class LogoPosition(str, PyEnum):
    """Logo position on video."""

    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"


class BrandTemplate(Base):
    """Brand template for consistent video styling."""

    __tablename__ = "brand_templates"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Basic info
    name = Column(String(100), nullable=False)
    is_default = Column(Boolean, default=False)

    # Logo settings
    logo_path = Column(String(1000), nullable=True)
    logo_position = Column(Enum(LogoPosition), default=LogoPosition.TOP_RIGHT)
    logo_size = Column(Integer, default=100)  # Width in pixels
    logo_opacity = Column(Float, default=1.0)  # 0.0 to 1.0
    logo_margin = Column(Integer, default=20)

    # Brand colors (hex format)
    primary_color = Column(String(20), default="#FFFFFF")
    secondary_color = Column(String(20), default="#000000")
    accent_color = Column(String(20), default="#FFFF00")

    # Caption style reference
    caption_style_id = Column(GUID(), ForeignKey("caption_styles.id"), nullable=True)

    # Outro settings
    outro_enabled = Column(Boolean, default=False)
    outro_duration = Column(Float, default=3.0)  # Seconds
    outro_text = Column(String(500), nullable=True)
    outro_cta_text = Column(String(200), nullable=True)  # Call to action
    outro_background_color = Column(String(20), default="#000000")

    # Relationships
    caption_style = relationship("CaptionStyle", back_populates="brand_templates")
    editor_projects = relationship("EditorProject", back_populates="brand_template")


class AspectRatio(str, PyEnum):
    """Target aspect ratios for reframing."""

    PORTRAIT_9_16 = "9:16"  # TikTok, Reels, Shorts
    SQUARE_1_1 = "1:1"  # Instagram feed
    LANDSCAPE_16_9 = "16:9"  # YouTube
    LANDSCAPE_4_3 = "4:3"  # Standard
    PORTRAIT_4_5 = "4:5"  # Instagram portrait


class TrackingMode(str, PyEnum):
    """Subject tracking mode for reframing."""

    SPEAKER = "speaker"  # Follow speaker/face
    ACTION = "action"  # Follow main action
    CENTER = "center"  # Center crop
    MANUAL = "manual"  # User-defined keyframes


class ReframeConfig(Base):
    """AI reframe configuration for smart cropping."""

    __tablename__ = "reframe_configs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    clip_id = Column(GUID(), ForeignKey("clip_suggestions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Target dimensions
    aspect_ratio = Column(Enum(AspectRatio), default=AspectRatio.PORTRAIT_9_16)
    target_width = Column(Integer, default=1080)
    target_height = Column(Integer, default=1920)

    # Tracking settings
    tracking_mode = Column(Enum(TrackingMode), default=TrackingMode.SPEAKER)
    smooth_factor = Column(Float, default=0.3)  # Smoothing for camera movement

    # Keyframes for crop positions
    # Format: [{"time": 0.0, "x": 100, "y": 0, "width": 608, "height": 1080}, ...]
    keyframes = Column(JSON, default=list)

    # Generated crop data from analysis
    # Format: {"frames": [{"time": 0.0, "x": 100, "y": 0, "confidence": 0.95}, ...]}
    crop_data = Column(JSON, default=dict)

    # Processing status
    processed = Column(Boolean, default=False)
    export_path = Column(String(1000), nullable=True)

    # Relationships
    clip = relationship("ClipSuggestion", backref="reframe_configs")


class Thumbnail(Base):
    """Generated thumbnail for a clip."""

    __tablename__ = "thumbnails"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    clip_id = Column(GUID(), ForeignKey("clip_suggestions.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Source frame
    source_timestamp = Column(Float, nullable=False)  # Timestamp in video
    source_frame_path = Column(String(1000), nullable=True)  # Raw frame image

    # Output
    output_path = Column(String(1000), nullable=True)  # Final thumbnail with overlays

    # Text overlay settings
    text_overlay = Column(String(500), nullable=True)
    text_position = Column(String(50), default="center")  # top, center, bottom
    text_style = Column(JSON, default=dict)  # {"font_size": 48, "color": "#FFFFFF", ...}

    # Analysis scores
    has_face = Column(Boolean, default=False)
    face_emotion = Column(String(50), nullable=True)  # happy, surprised, neutral, etc.
    engagement_score = Column(Float, default=0.0)  # 0-100 based on visual appeal
    composition_score = Column(Float, default=0.0)  # Rule of thirds, etc.

    # Selection
    is_selected = Column(Boolean, default=False)  # User's chosen thumbnail
    rank = Column(Integer, nullable=True)  # Auto-ranked position

    # Relationships
    clip = relationship("ClipSuggestion", backref="thumbnails")


class EditorProject(Base):
    """Timeline editor project for multi-clip editing."""

    __tablename__ = "editor_projects"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    analysis_job_id = Column(GUID(), ForeignKey("analysis_jobs.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Project info
    name = Column(String(200), nullable=False)

    # Timeline data
    # Format: {"tracks": [{"type": "video", "clips": [{"clip_id": "...", "start": 0, "duration": 30}]}]}
    timeline_data = Column(JSON, default=dict)

    # Export settings
    export_resolution = Column(String(20), default="1080p")
    export_format = Column(String(10), default="mp4")
    export_fps = Column(Float, default=30.0)
    brand_template_id = Column(GUID(), ForeignKey("brand_templates.id"), nullable=True)

    # Status
    rendered = Column(Boolean, default=False)
    render_path = Column(String(1000), nullable=True)
    render_progress = Column(Float, default=0.0)

    # Relationships
    analysis_job = relationship("AnalysisJob", backref="editor_projects")
    brand_template = relationship("BrandTemplate", back_populates="editor_projects")


class SocialPlatform(str, PyEnum):
    """Supported social media platforms."""

    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"


class PostStatus(str, PyEnum):
    """Status of a scheduled post."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    POSTING = "posting"
    POSTED = "posted"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SocialConnection(Base):
    """OAuth connection to a social media platform."""

    __tablename__ = "social_connections"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Platform info
    platform = Column(Enum(SocialPlatform), nullable=False)
    account_name = Column(String(200), nullable=True)
    account_id = Column(String(200), nullable=True)

    # OAuth tokens (should be encrypted in production)
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    scheduled_posts = relationship("ScheduledPost", back_populates="connection")


class ScheduledPost(Base):
    """Scheduled social media post."""

    __tablename__ = "scheduled_posts"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    clip_id = Column(GUID(), ForeignKey("clip_suggestions.id"), nullable=False)
    connection_id = Column(GUID(), ForeignKey("social_connections.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Schedule
    scheduled_time = Column(DateTime, nullable=False)
    timezone = Column(String(50), default="UTC")

    # Content
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    hashtags = Column(JSON, default=list)  # ["#viral", "#fyp", ...]

    # Status
    status = Column(Enum(PostStatus), default=PostStatus.PENDING)
    posted_at = Column(DateTime, nullable=True)
    platform_post_id = Column(String(200), nullable=True)  # ID from the platform
    error_message = Column(Text, nullable=True)

    # Relationships
    clip = relationship("ClipSuggestion", backref="scheduled_posts")
    connection = relationship("SocialConnection", back_populates="scheduled_posts")


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
