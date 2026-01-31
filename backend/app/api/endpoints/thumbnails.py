"""Thumbnail generation and management endpoints."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    Thumbnail,
    ClipSuggestion,
    AnalysisJob,
    Video,
    get_async_session,
)
from app.models.schemas import (
    ThumbnailGenerateRequest,
    ThumbnailUpdate,
    ThumbnailResponse,
    ThumbnailTextOverlay,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# Create thumbnails directory
THUMBNAILS_DIR = settings.storage_path / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{clip_id}/generate", response_model=list[ThumbnailResponse])
async def generate_thumbnails(
    clip_id: uuid.UUID,
    request: ThumbnailGenerateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate thumbnail candidates for a clip.

    Extracts frames at strategic points and scores them for engagement potential.
    """
    # Get clip
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get video
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

    # Delete existing thumbnails for this clip
    existing_result = await session.execute(
        select(Thumbnail).filter(Thumbnail.clip_id == clip_id)
    )
    for thumb in existing_result.scalars().all():
        # Delete files
        if thumb.source_frame_path:
            Path(thumb.source_frame_path).unlink(missing_ok=True)
        if thumb.output_path:
            Path(thumb.output_path).unlink(missing_ok=True)
        await session.delete(thumb)

    # Generate thumbnails
    from app.services.thumbnail_service import get_thumbnail_service

    thumbnail_service = get_thumbnail_service()

    output_dir = THUMBNAILS_DIR / str(clip_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    thumb_data = await thumbnail_service.generate_thumbnails(
        video_path=video.file_path,
        clip_start=clip.start_time,
        clip_end=clip.end_time,
        output_dir=str(output_dir),
        count=request.count,
        prefer_faces=request.prefer_faces,
        specific_timestamps=request.timestamps,
    )

    # Save to database
    thumbnails = []
    for data in thumb_data:
        thumb = Thumbnail(
            clip_id=clip_id,
            source_timestamp=data["timestamp"],
            source_frame_path=data["path"],
            output_path=data["path"],  # Initially same as source
            has_face=data["has_face"],
            face_emotion=data.get("face_emotion"),
            engagement_score=data["engagement_score"],
            composition_score=data["composition_score"],
            rank=data["rank"],
            is_selected=data["rank"] == 1,  # Select best by default
        )
        session.add(thumb)
        thumbnails.append(thumb)

    await session.commit()

    # Refresh all thumbnails
    for thumb in thumbnails:
        await session.refresh(thumb)

    logger.info(f"Generated {len(thumbnails)} thumbnails for clip: {clip_id}")
    return [ThumbnailResponse.model_validate(t) for t in thumbnails]


@router.get("/{clip_id}", response_model=list[ThumbnailResponse])
async def get_clip_thumbnails(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get all thumbnails for a clip."""
    result = await session.execute(
        select(Thumbnail)
        .filter(Thumbnail.clip_id == clip_id)
        .order_by(Thumbnail.rank)
    )
    thumbnails = result.scalars().all()

    if not thumbnails:
        raise HTTPException(status_code=404, detail="No thumbnails found for clip")

    return [ThumbnailResponse.model_validate(t) for t in thumbnails]


@router.get("/detail/{thumbnail_id}", response_model=ThumbnailResponse)
async def get_thumbnail(
    thumbnail_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific thumbnail."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return ThumbnailResponse.model_validate(thumb)


@router.put("/{thumbnail_id}", response_model=ThumbnailResponse)
async def update_thumbnail(
    thumbnail_id: uuid.UUID,
    data: ThumbnailUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    """Update a thumbnail (e.g., add text overlay)."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    update_data = data.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        setattr(thumb, key, value)

    await session.commit()
    await session.refresh(thumb)

    logger.info(f"Updated thumbnail: {thumbnail_id}")
    return ThumbnailResponse.model_validate(thumb)


@router.post("/{thumbnail_id}/text-overlay", response_model=ThumbnailResponse)
async def add_text_overlay(
    thumbnail_id: uuid.UUID,
    overlay: ThumbnailTextOverlay,
    session: AsyncSession = Depends(get_async_session),
):
    """Add text overlay to a thumbnail."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    if not thumb.source_frame_path or not Path(thumb.source_frame_path).exists():
        raise HTTPException(status_code=400, detail="Source frame not found")

    # Generate output path
    output_dir = Path(thumb.source_frame_path).parent
    output_filename = f"overlay_{thumbnail_id}.jpg"
    output_path = str(output_dir / output_filename)

    # Add text overlay
    from app.services.thumbnail_service import get_thumbnail_service

    thumbnail_service = get_thumbnail_service()

    await thumbnail_service.add_text_overlay(
        input_path=thumb.source_frame_path,
        output_path=output_path,
        text=overlay.text,
        position=overlay.position,
        font_size=overlay.font_size,
        color=overlay.color,
        stroke_color=overlay.stroke_color,
        stroke_width=overlay.stroke_width,
    )

    # Update thumbnail
    thumb.output_path = output_path
    thumb.text_overlay = overlay.text
    thumb.text_position = overlay.position
    thumb.text_style = {
        "font_size": overlay.font_size,
        "color": overlay.color,
        "stroke_color": overlay.stroke_color,
        "stroke_width": overlay.stroke_width,
    }

    await session.commit()
    await session.refresh(thumb)

    logger.info(f"Added text overlay to thumbnail: {thumbnail_id}")
    return ThumbnailResponse.model_validate(thumb)


@router.post("/{thumbnail_id}/select", response_model=ThumbnailResponse)
async def select_thumbnail(
    thumbnail_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Select a thumbnail as the primary choice for its clip."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Deselect other thumbnails for this clip
    other_result = await session.execute(
        select(Thumbnail).filter(
            Thumbnail.clip_id == thumb.clip_id,
            Thumbnail.id != thumbnail_id
        )
    )
    for other in other_result.scalars().all():
        other.is_selected = False

    # Select this thumbnail
    thumb.is_selected = True

    await session.commit()
    await session.refresh(thumb)

    logger.info(f"Selected thumbnail: {thumbnail_id} for clip: {thumb.clip_id}")
    return ThumbnailResponse.model_validate(thumb)


@router.get("/{thumbnail_id}/download")
async def download_thumbnail(
    thumbnail_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Download a thumbnail image."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Use output_path if available (has overlay), otherwise source
    file_path = thumb.output_path or thumb.source_frame_path

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found")

    return FileResponse(
        path=file_path,
        filename=f"thumbnail_{thumbnail_id}.jpg",
        media_type="image/jpeg",
    )


@router.delete("/{thumbnail_id}")
async def delete_thumbnail(
    thumbnail_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a thumbnail."""
    result = await session.execute(
        select(Thumbnail).filter(Thumbnail.id == thumbnail_id)
    )
    thumb = result.scalar_one_or_none()

    if not thumb:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    # Delete files
    if thumb.source_frame_path:
        Path(thumb.source_frame_path).unlink(missing_ok=True)
    if thumb.output_path and thumb.output_path != thumb.source_frame_path:
        Path(thumb.output_path).unlink(missing_ok=True)

    await session.delete(thumb)
    await session.commit()

    logger.info(f"Deleted thumbnail: {thumbnail_id}")
    return {"status": "deleted", "id": str(thumbnail_id)}
