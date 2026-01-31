"""Caption style CRUD and preview endpoints."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    CaptionStyle,
    CaptionStyleType,
    CaptionPosition,
    ClipSuggestion,
    AnalysisJob,
    Video,
    Transcription,
    TranscriptionSegment,
    get_async_session,
)
from app.models.schemas import (
    CaptionStyleCreate,
    CaptionStyleUpdate,
    CaptionStyleResponse,
    CaptionPreviewRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/styles", response_model=CaptionStyleResponse)
async def create_caption_style(
    data: CaptionStyleCreate,
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new caption style."""
    # Validate enums
    try:
        style_type = CaptionStyleType(data.style_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style_type. Must be one of: {[e.value for e in CaptionStyleType]}"
        )

    try:
        position = CaptionPosition(data.position)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid position. Must be one of: {[e.value for e in CaptionPosition]}"
        )

    # If this is set as default, unset other defaults
    if data.is_default:
        result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.is_default == True)
        )
        existing_defaults = result.scalars().all()
        for style in existing_defaults:
            style.is_default = False

    # Create new style
    style = CaptionStyle(
        name=data.name,
        style_type=style_type,
        animation_duration=data.animation_duration,
        font_family=data.font_family,
        font_size=data.font_size,
        font_weight=data.font_weight,
        text_color=data.text_color,
        highlight_color=data.highlight_color,
        background_color=data.background_color,
        stroke_color=data.stroke_color,
        stroke_width=data.stroke_width,
        position=position,
        margin_bottom=data.margin_bottom,
        margin_horizontal=data.margin_horizontal,
        words_per_line=data.words_per_line,
        max_lines=data.max_lines,
        is_default=data.is_default,
    )

    session.add(style)
    await session.commit()
    await session.refresh(style)

    logger.info(f"Created caption style: {style.id} ({style.name})")
    return CaptionStyleResponse.model_validate(style)


@router.get("/styles", response_model=list[CaptionStyleResponse])
async def list_caption_styles(
    session: AsyncSession = Depends(get_async_session),
):
    """List all caption styles."""
    result = await session.execute(
        select(CaptionStyle).order_by(CaptionStyle.created_at.desc())
    )
    styles = result.scalars().all()
    return [CaptionStyleResponse.model_validate(s) for s in styles]


@router.get("/styles/{style_id}", response_model=CaptionStyleResponse)
async def get_caption_style(
    style_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific caption style."""
    result = await session.execute(
        select(CaptionStyle).filter(CaptionStyle.id == style_id)
    )
    style = result.scalar_one_or_none()

    if not style:
        raise HTTPException(status_code=404, detail="Caption style not found")

    return CaptionStyleResponse.model_validate(style)


@router.put("/styles/{style_id}", response_model=CaptionStyleResponse)
async def update_caption_style(
    style_id: uuid.UUID,
    data: CaptionStyleUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    """Update a caption style."""
    result = await session.execute(
        select(CaptionStyle).filter(CaptionStyle.id == style_id)
    )
    style = result.scalar_one_or_none()

    if not style:
        raise HTTPException(status_code=404, detail="Caption style not found")

    # Update fields if provided
    update_data = data.model_dump(exclude_unset=True)

    # Validate enums if present
    if "style_type" in update_data:
        try:
            update_data["style_type"] = CaptionStyleType(update_data["style_type"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style_type. Must be one of: {[e.value for e in CaptionStyleType]}"
            )

    if "position" in update_data:
        try:
            update_data["position"] = CaptionPosition(update_data["position"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid position. Must be one of: {[e.value for e in CaptionPosition]}"
            )

    # Handle default flag
    if update_data.get("is_default"):
        other_defaults = await session.execute(
            select(CaptionStyle).filter(
                CaptionStyle.is_default == True,
                CaptionStyle.id != style_id
            )
        )
        for other in other_defaults.scalars().all():
            other.is_default = False

    for key, value in update_data.items():
        setattr(style, key, value)

    await session.commit()
    await session.refresh(style)

    logger.info(f"Updated caption style: {style_id}")
    return CaptionStyleResponse.model_validate(style)


@router.delete("/styles/{style_id}")
async def delete_caption_style(
    style_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a caption style."""
    result = await session.execute(
        select(CaptionStyle).filter(CaptionStyle.id == style_id)
    )
    style = result.scalar_one_or_none()

    if not style:
        raise HTTPException(status_code=404, detail="Caption style not found")

    await session.delete(style)
    await session.commit()

    logger.info(f"Deleted caption style: {style_id}")
    return {"status": "deleted", "id": str(style_id)}


@router.post("/preview")
async def generate_caption_preview(
    request: CaptionPreviewRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate a preview video with animated captions.

    Returns the path to the preview video once ready.
    """
    # Get clip
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == request.clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get caption style (use default if not specified)
    style_config = {}
    if request.style_id:
        style_result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.id == request.style_id)
        )
        style = style_result.scalar_one_or_none()
        if style:
            style_config = {
                "style_type": style.style_type.value,
                "animation_duration": style.animation_duration,
                "font_family": style.font_family,
                "font_size": style.font_size,
                "font_weight": style.font_weight,
                "text_color": style.text_color,
                "highlight_color": style.highlight_color,
                "background_color": style.background_color,
                "stroke_color": style.stroke_color,
                "stroke_width": style.stroke_width,
                "position": style.position.value,
                "margin_bottom": style.margin_bottom,
                "margin_horizontal": style.margin_horizontal,
                "words_per_line": style.words_per_line,
                "max_lines": style.max_lines,
            }
    else:
        # Try to get default style
        default_result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.is_default == True)
        )
        default_style = default_result.scalar_one_or_none()
        if default_style:
            style_config = {
                "style_type": default_style.style_type.value,
                "animation_duration": default_style.animation_duration,
                "font_family": default_style.font_family,
                "font_size": default_style.font_size,
                "font_weight": default_style.font_weight,
                "text_color": default_style.text_color,
                "highlight_color": default_style.highlight_color,
                "background_color": default_style.background_color,
                "stroke_color": default_style.stroke_color,
                "stroke_width": default_style.stroke_width,
                "position": default_style.position.value,
                "margin_bottom": default_style.margin_bottom,
                "margin_horizontal": default_style.margin_horizontal,
                "words_per_line": default_style.words_per_line,
                "max_lines": default_style.max_lines,
            }

    # Get video and transcription
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == clip.analysis_job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    video_result = await session.execute(
        select(Video).filter(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get transcription segments for the clip
    trans_result = await session.execute(
        select(Transcription).filter(Transcription.analysis_job_id == job.id)
    )
    transcription = trans_result.scalar_one_or_none()

    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")

    segments_result = await session.execute(
        select(TranscriptionSegment)
        .filter(TranscriptionSegment.transcription_id == transcription.id)
        .filter(TranscriptionSegment.start_time >= clip.start_time)
        .filter(TranscriptionSegment.end_time <= clip.end_time)
        .order_by(TranscriptionSegment.start_time)
    )
    segments = segments_result.scalars().all()

    # Convert segments to dict format
    segment_dicts = [
        {
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "text": seg.text,
            "words": seg.words or [],
        }
        for seg in segments
    ]

    # Generate preview
    from app.services.caption_service import get_caption_service

    caption_service = get_caption_service()
    preview_dir = settings.cache_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_path = str(preview_dir / f"caption_preview_{clip.id}_{uuid.uuid4().hex[:8]}.mp4")

    preview_duration = min(request.preview_duration, clip.duration)

    await caption_service.generate_preview(
        segments=segment_dicts,
        video_path=video.file_path,
        output_path=preview_path,
        style_config=style_config,
        duration=preview_duration,
        clip_start_offset=clip.start_time,
    )

    logger.info(f"Generated caption preview: {preview_path}")

    return {
        "status": "completed",
        "preview_path": preview_path,
        "duration": preview_duration,
        "style_config": style_config,
    }


@router.get("/preview/{filename}")
async def download_caption_preview(filename: str):
    """Download a generated caption preview."""
    preview_dir = settings.cache_path / "previews"
    preview_path = preview_dir / filename

    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")

    return FileResponse(
        path=preview_path,
        filename=filename,
        media_type="video/mp4",
    )
