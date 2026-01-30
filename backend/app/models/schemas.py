"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict


# ============== Video Schemas ==============


class VideoCreate(BaseModel):
    """Schema for video upload."""

    title: Optional[str] = None
    description: Optional[str] = None


class VideoDownloadRequest(BaseModel):
    """Schema for video download from URL."""

    url: str = Field(..., description="URL to download video from")
    title: Optional[str] = None
    output_dir: Optional[str] = Field(None, description="Custom output directory for downloaded video")


class VideoResponse(BaseModel):
    """Schema for video response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    source_url: Optional[str] = None
    source_platform: Optional[str] = None
    original_filename: Optional[str] = None
    file_path: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    title: Optional[str] = None
    description: Optional[str] = None
    thumbnail_path: Optional[str] = None
    status: str
    error_message: Optional[str] = None


# ============== Analysis Schemas ==============


class AnalysisStartRequest(BaseModel):
    """Schema for starting analysis."""

    transcribe: bool = Field(default=True, description="Enable transcription")
    diarize: bool = Field(default=True, description="Enable speaker diarization")
    detect_scenes: bool = Field(default=True, description="Enable scene detection")
    analyze_visuals: bool = Field(default=True, description="Enable YOLO visual analysis")
    detect_hooks: bool = Field(default=True, description="Enable hook detection")
    suggest_clips: bool = Field(default=True, description="Enable clip suggestions")
    llm_analysis: bool = Field(default=True, description="Enable LLM-powered analysis")
    whisper_model: str = Field(default="small", description="Whisper model to use")
    min_clip_duration: float = Field(default=15.0, description="Minimum clip duration in seconds")
    max_clip_duration: float = Field(default=90.0, description="Maximum clip duration in seconds")
    target_clips: int = Field(default=5, description="Target number of clips to suggest")


class AnalysisJobResponse(BaseModel):
    """Schema for analysis job response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    progress: float
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    config: dict
    total_duration: Optional[float] = None
    word_count: Optional[int] = None
    speaker_count: Optional[int] = None
    scene_count: Optional[int] = None
    clip_count: Optional[int] = None


class AnalysisStatusResponse(BaseModel):
    """Schema for analysis status response with progress details."""

    job_id: UUID
    status: str
    progress: float
    current_stage: Optional[str] = None
    stages_completed: list[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    eta_seconds: Optional[float] = None


# ============== Transcription Schemas ==============


class WordTimestamp(BaseModel):
    """Schema for word-level timestamp."""

    word: str
    start: float
    end: float
    probability: Optional[float] = None


class TranscriptionSegmentResponse(BaseModel):
    """Schema for transcription segment."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    start_time: float
    end_time: float
    text: str
    words: Optional[list[WordTimestamp]] = None
    speaker_id: Optional[str] = None
    avg_log_prob: Optional[float] = None


class TranscriptionResponse(BaseModel):
    """Schema for full transcription."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_text: str
    language: Optional[str] = None
    language_probability: Optional[float] = None
    word_count: int
    duration_seconds: Optional[float] = None
    segments: list[TranscriptionSegmentResponse] = Field(default_factory=list)


# ============== Diarization Schemas ==============


class DiarizationSegmentResponse(BaseModel):
    """Schema for diarization segment."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    start_time: float
    end_time: float
    speaker_id: str


class DiarizationResponse(BaseModel):
    """Schema for diarization results."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    speaker_count: int
    speaker_labels: list[str]
    segments: list[DiarizationSegmentResponse] = Field(default_factory=list)


# ============== Scene Schemas ==============


class SceneResponse(BaseModel):
    """Schema for scene detection result."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    start_time: float
    end_time: float
    duration: float
    scene_index: int
    scene_type: Optional[str] = None
    keyframe_path: Optional[str] = None
    dominant_colors: Optional[list[str]] = None
    detected_objects: Optional[list[dict]] = None
    motion_intensity: Optional[float] = None


# ============== Hook Schemas ==============


class HookResponse(BaseModel):
    """Schema for hook detection result."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    start_time: float
    end_time: float
    text: Optional[str] = None
    hook_type: Optional[str] = None
    hook_score: float
    curiosity_gap: float
    emotional_trigger: float
    clarity: float
    visual_interest: float
    is_video_start: bool
    rank: Optional[int] = None


# ============== Clip Schemas ==============


class ViralityScore(BaseModel):
    """Schema for virality score breakdown."""

    total: float = Field(..., ge=0, le=100, description="Total virality score (0-100)")
    emotional_resonance: float = Field(default=0.0, ge=0, le=20)
    shareability: float = Field(default=0.0, ge=0, le=20)
    uniqueness: float = Field(default=0.0, ge=0, le=20)
    hook_strength: float = Field(default=0.0, ge=0, le=20)
    production_quality: float = Field(default=0.0, ge=0, le=20)


class ClipSuggestionResponse(BaseModel):
    """Schema for clip suggestion."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    start_time: float
    end_time: float
    duration: float
    title: Optional[str] = None
    description: Optional[str] = None
    transcript_excerpt: Optional[str] = None
    virality_score: float
    emotional_resonance: float
    shareability: float
    uniqueness: float
    hook_strength: float
    production_quality: float
    exported: bool
    export_path: Optional[str] = None
    rank: Optional[int] = None


class ClipExportRequest(BaseModel):
    """Schema for clip export request."""

    clip_id: UUID
    format: str = Field(default="mp4", description="Output format")
    resolution: Optional[str] = Field(default=None, description="Output resolution (e.g., '1080p')")
    include_captions: bool = Field(default=False, description="Burn in captions")


# ============== Full Analysis Response ==============


class AnalysisFullResponse(BaseModel):
    """Schema for complete analysis results."""

    job: AnalysisJobResponse
    video: VideoResponse
    transcription: Optional[TranscriptionResponse] = None
    diarization: Optional[DiarizationResponse] = None
    scenes: list[SceneResponse] = Field(default_factory=list)
    hooks: list[HookResponse] = Field(default_factory=list)
    clips: list[ClipSuggestionResponse] = Field(default_factory=list)
