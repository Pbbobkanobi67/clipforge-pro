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


# ============== Caption Style Schemas ==============


class CaptionStyleCreate(BaseModel):
    """Schema for creating a caption style."""

    name: str = Field(..., min_length=1, max_length=100)
    style_type: str = Field(default="karaoke", description="karaoke, bounce, typewriter, fade, glow, wave, static")
    animation_duration: float = Field(default=0.1, ge=0.01, le=1.0)
    font_family: str = Field(default="Arial", max_length=100)
    font_size: int = Field(default=48, ge=12, le=200)
    font_weight: str = Field(default="bold", max_length=20)
    text_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: str = Field(default="#FFFF00", pattern=r"^#[0-9A-Fa-f]{6}$")
    background_color: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_width: int = Field(default=2, ge=0, le=10)
    position: str = Field(default="bottom", description="top, center, bottom")
    margin_bottom: int = Field(default=50, ge=0, le=500)
    margin_horizontal: int = Field(default=40, ge=0, le=500)
    words_per_line: int = Field(default=6, ge=1, le=20)
    max_lines: int = Field(default=2, ge=1, le=5)
    is_default: bool = Field(default=False)


class CaptionStyleUpdate(BaseModel):
    """Schema for updating a caption style."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    style_type: Optional[str] = None
    animation_duration: Optional[float] = Field(None, ge=0.01, le=1.0)
    font_family: Optional[str] = Field(None, max_length=100)
    font_size: Optional[int] = Field(None, ge=12, le=200)
    font_weight: Optional[str] = Field(None, max_length=20)
    text_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    background_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_width: Optional[int] = Field(None, ge=0, le=10)
    position: Optional[str] = None
    margin_bottom: Optional[int] = Field(None, ge=0, le=500)
    margin_horizontal: Optional[int] = Field(None, ge=0, le=500)
    words_per_line: Optional[int] = Field(None, ge=1, le=20)
    max_lines: Optional[int] = Field(None, ge=1, le=5)
    is_default: Optional[bool] = None


class CaptionStyleResponse(BaseModel):
    """Schema for caption style response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    style_type: str
    animation_duration: float
    font_family: str
    font_size: int
    font_weight: str
    text_color: str
    highlight_color: str
    background_color: Optional[str] = None
    stroke_color: str
    stroke_width: int
    position: str
    margin_bottom: int
    margin_horizontal: int
    words_per_line: int
    max_lines: int
    is_default: bool


class CaptionPreviewRequest(BaseModel):
    """Schema for caption preview request."""

    clip_id: UUID
    style_id: Optional[UUID] = None  # Use default if not provided
    preview_duration: float = Field(default=5.0, ge=1.0, le=30.0)


# ============== Brand Template Schemas ==============


class BrandTemplateCreate(BaseModel):
    """Schema for creating a brand template."""

    name: str = Field(..., min_length=1, max_length=100)
    is_default: bool = Field(default=False)

    # Logo settings
    logo_position: str = Field(default="top_right", description="top_left, top_right, bottom_left, bottom_right, center")
    logo_size: int = Field(default=100, ge=20, le=500)
    logo_opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    logo_margin: int = Field(default=20, ge=0, le=200)

    # Brand colors
    primary_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    accent_color: str = Field(default="#FFFF00", pattern=r"^#[0-9A-Fa-f]{6}$")

    # Caption style
    caption_style_id: Optional[UUID] = None

    # Outro settings
    outro_enabled: bool = Field(default=False)
    outro_duration: float = Field(default=3.0, ge=1.0, le=10.0)
    outro_text: Optional[str] = Field(None, max_length=500)
    outro_cta_text: Optional[str] = Field(None, max_length=200)
    outro_background_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")


class BrandTemplateUpdate(BaseModel):
    """Schema for updating a brand template."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_default: Optional[bool] = None
    logo_position: Optional[str] = None
    logo_size: Optional[int] = Field(None, ge=20, le=500)
    logo_opacity: Optional[float] = Field(None, ge=0.0, le=1.0)
    logo_margin: Optional[int] = Field(None, ge=0, le=200)
    primary_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    accent_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")
    caption_style_id: Optional[UUID] = None
    outro_enabled: Optional[bool] = None
    outro_duration: Optional[float] = Field(None, ge=1.0, le=10.0)
    outro_text: Optional[str] = Field(None, max_length=500)
    outro_cta_text: Optional[str] = Field(None, max_length=200)
    outro_background_color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")


class BrandTemplateResponse(BaseModel):
    """Schema for brand template response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    is_default: bool
    logo_path: Optional[str] = None
    logo_position: str
    logo_size: int
    logo_opacity: float
    logo_margin: int
    primary_color: str
    secondary_color: str
    accent_color: str
    caption_style_id: Optional[UUID] = None
    outro_enabled: bool
    outro_duration: float
    outro_text: Optional[str] = None
    outro_cta_text: Optional[str] = None
    outro_background_color: str


# ============== Reframe Schemas ==============


class ReframeKeyframe(BaseModel):
    """Schema for a reframe keyframe."""

    time: float = Field(..., ge=0)
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class ReframeRequest(BaseModel):
    """Schema for generating reframe config."""

    aspect_ratio: str = Field(default="9:16", description="9:16, 1:1, 16:9, 4:3, 4:5")
    tracking_mode: str = Field(default="speaker", description="speaker, action, center, manual")
    smooth_factor: float = Field(default=0.3, ge=0.0, le=1.0)


class ReframeUpdate(BaseModel):
    """Schema for updating reframe config."""

    aspect_ratio: Optional[str] = None
    tracking_mode: Optional[str] = None
    smooth_factor: Optional[float] = Field(None, ge=0.0, le=1.0)
    keyframes: Optional[list[ReframeKeyframe]] = None


class ReframeResponse(BaseModel):
    """Schema for reframe config response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    clip_id: UUID
    created_at: datetime
    updated_at: datetime
    aspect_ratio: str
    target_width: int
    target_height: int
    tracking_mode: str
    smooth_factor: float
    keyframes: list[dict] = Field(default_factory=list)
    crop_data: dict = Field(default_factory=dict)
    processed: bool
    export_path: Optional[str] = None


# ============== Thumbnail Schemas ==============


class ThumbnailGenerateRequest(BaseModel):
    """Schema for generating thumbnails."""

    count: int = Field(default=5, ge=1, le=20, description="Number of thumbnails to generate")
    prefer_faces: bool = Field(default=True, description="Prioritize frames with faces")
    timestamps: Optional[list[float]] = Field(default=None, description="Specific timestamps to extract")


class ThumbnailTextOverlay(BaseModel):
    """Schema for thumbnail text overlay."""

    text: str = Field(..., min_length=1, max_length=500)
    position: str = Field(default="center", description="top, center, bottom")
    font_size: int = Field(default=48, ge=12, le=200)
    color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_color: str = Field(default="#000000", pattern=r"^#[0-9A-Fa-f]{6}$")
    stroke_width: int = Field(default=2, ge=0, le=10)


class ThumbnailUpdate(BaseModel):
    """Schema for updating a thumbnail."""

    text_overlay: Optional[str] = Field(None, max_length=500)
    text_position: Optional[str] = None
    text_style: Optional[dict] = None
    is_selected: Optional[bool] = None


class ThumbnailResponse(BaseModel):
    """Schema for thumbnail response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    clip_id: UUID
    created_at: datetime
    source_timestamp: float
    source_frame_path: Optional[str] = None
    output_path: Optional[str] = None
    text_overlay: Optional[str] = None
    text_position: str
    text_style: dict = Field(default_factory=dict)
    has_face: bool
    face_emotion: Optional[str] = None
    engagement_score: float
    composition_score: float
    is_selected: bool
    rank: Optional[int] = None


# ============== Editor Project Schemas ==============


class TimelineClip(BaseModel):
    """Schema for a clip in the timeline."""

    clip_id: UUID
    start: float = Field(..., ge=0, description="Start position in timeline (seconds)")
    duration: float = Field(..., gt=0, description="Duration in timeline (seconds)")
    trim_start: float = Field(default=0, ge=0, description="Trim from clip start")
    trim_end: float = Field(default=0, ge=0, description="Trim from clip end")


class TimelineTrack(BaseModel):
    """Schema for a timeline track."""

    type: str = Field(default="video", description="video, audio, caption")
    clips: list[TimelineClip] = Field(default_factory=list)


class TimelineData(BaseModel):
    """Schema for timeline data."""

    tracks: list[TimelineTrack] = Field(default_factory=list)
    duration: float = Field(default=0, ge=0)


class EditorProjectCreate(BaseModel):
    """Schema for creating an editor project."""

    analysis_job_id: UUID
    name: str = Field(..., min_length=1, max_length=200)
    export_resolution: str = Field(default="1080p")
    export_format: str = Field(default="mp4")
    export_fps: float = Field(default=30.0, ge=1.0, le=120.0)
    brand_template_id: Optional[UUID] = None


class EditorProjectUpdate(BaseModel):
    """Schema for updating an editor project."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    timeline_data: Optional[TimelineData] = None
    export_resolution: Optional[str] = None
    export_format: Optional[str] = None
    export_fps: Optional[float] = Field(None, ge=1.0, le=120.0)
    brand_template_id: Optional[UUID] = None


class EditorProjectResponse(BaseModel):
    """Schema for editor project response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    analysis_job_id: UUID
    created_at: datetime
    updated_at: datetime
    name: str
    timeline_data: dict = Field(default_factory=dict)
    export_resolution: str
    export_format: str
    export_fps: float
    brand_template_id: Optional[UUID] = None
    rendered: bool
    render_path: Optional[str] = None
    render_progress: float


# ============== Social/Scheduler Schemas ==============


class SocialConnectionCreate(BaseModel):
    """Schema for creating a social connection."""

    platform: str = Field(..., description="youtube, tiktok, instagram, twitter, facebook, linkedin")


class SocialConnectionResponse(BaseModel):
    """Schema for social connection response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    platform: str
    account_name: Optional[str] = None
    account_id: Optional[str] = None
    is_active: bool
    last_used_at: Optional[datetime] = None


class ScheduledPostCreate(BaseModel):
    """Schema for creating a scheduled post."""

    clip_id: UUID
    connection_id: UUID
    scheduled_time: datetime
    timezone: str = Field(default="UTC")
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    hashtags: list[str] = Field(default_factory=list)


class ScheduledPostUpdate(BaseModel):
    """Schema for updating a scheduled post."""

    scheduled_time: Optional[datetime] = None
    timezone: Optional[str] = None
    title: Optional[str] = Field(None, max_length=500)
    description: Optional[str] = None
    hashtags: Optional[list[str]] = None
    status: Optional[str] = None


class ScheduledPostResponse(BaseModel):
    """Schema for scheduled post response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    clip_id: UUID
    connection_id: UUID
    created_at: datetime
    scheduled_time: datetime
    timezone: str
    title: Optional[str] = None
    description: Optional[str] = None
    hashtags: list[str] = Field(default_factory=list)
    status: str
    posted_at: Optional[datetime] = None
    platform_post_id: Optional[str] = None
    error_message: Optional[str] = None


# ============== Enhanced Export Schemas ==============


class EnhancedClipExportRequest(BaseModel):
    """Schema for enhanced clip export with all options."""

    format: str = Field(default="mp4")
    resolution: Optional[str] = Field(default=None, description="720p, 1080p, 4k")

    # Caption options
    include_captions: bool = Field(default=False)
    caption_style_id: Optional[UUID] = None

    # Reframe options
    reframe: bool = Field(default=False)
    aspect_ratio: Optional[str] = Field(default=None, description="9:16, 1:1, 16:9, 4:3, 4:5")

    # Branding options
    brand_template_id: Optional[UUID] = None
    include_logo: bool = Field(default=False)
    include_outro: bool = Field(default=False)


class XMLExportRequest(BaseModel):
    """Schema for XML export request."""

    format: str = Field(default="fcpxml", description="fcpxml or premiere")
    include_markers: bool = Field(default=True)
    include_captions: bool = Field(default=True)
