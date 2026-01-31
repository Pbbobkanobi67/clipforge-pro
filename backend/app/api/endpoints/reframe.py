"""AI Reframe endpoints for smart video cropping."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    ReframeConfig,
    AspectRatio,
    TrackingMode,
    ClipSuggestion,
    AnalysisJob,
    Video,
    get_async_session,
)
from app.models.schemas import (
    ReframeRequest,
    ReframeUpdate,
    ReframeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# Aspect ratio to dimensions mapping
ASPECT_DIMENSIONS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
    "4:3": (1440, 1080),
    "4:5": (1080, 1350),
}


@router.post("/{clip_id}", response_model=ReframeResponse)
async def generate_reframe(
    clip_id: uuid.UUID,
    request: ReframeRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate AI reframe configuration for a clip.

    Analyzes the video to generate smart crop keyframes for the target aspect ratio.
    """
    # Get clip
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Validate aspect ratio
    try:
        aspect_ratio = AspectRatio(request.aspect_ratio)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect_ratio. Must be one of: {[e.value for e in AspectRatio]}"
        )

    # Validate tracking mode
    try:
        tracking_mode = TrackingMode(request.tracking_mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tracking_mode. Must be one of: {[e.value for e in TrackingMode]}"
        )

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

    # Check if reframe config already exists for this clip
    existing_result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    existing = existing_result.scalar_one_or_none()

    # Get target dimensions
    target_width, target_height = ASPECT_DIMENSIONS.get(request.aspect_ratio, (1080, 1920))

    # Analyze video for reframe
    from app.services.reframe_service import get_reframe_service

    reframe_service = get_reframe_service()

    # Analyze the clip portion of the video
    crop_data = await reframe_service.analyze_video_for_reframe(
        video_path=video.file_path,
        aspect_ratio=request.aspect_ratio,
        tracking_mode=request.tracking_mode,
        sample_interval=0.5,
    )

    # Filter keyframes to clip timerange
    keyframes = [
        kf for kf in crop_data.get("keyframes", [])
        if clip.start_time <= kf["time"] <= clip.end_time
    ]

    if existing:
        # Update existing config
        existing.aspect_ratio = aspect_ratio
        existing.target_width = target_width
        existing.target_height = target_height
        existing.tracking_mode = tracking_mode
        existing.smooth_factor = request.smooth_factor
        existing.keyframes = keyframes
        existing.crop_data = crop_data
        existing.processed = False
        existing.export_path = None

        await session.commit()
        await session.refresh(existing)

        logger.info(f"Updated reframe config: {existing.id}")
        return ReframeResponse.model_validate(existing)

    else:
        # Create new config
        config = ReframeConfig(
            clip_id=clip_id,
            aspect_ratio=aspect_ratio,
            target_width=target_width,
            target_height=target_height,
            tracking_mode=tracking_mode,
            smooth_factor=request.smooth_factor,
            keyframes=keyframes,
            crop_data=crop_data,
            processed=False,
        )

        session.add(config)
        await session.commit()
        await session.refresh(config)

        logger.info(f"Created reframe config: {config.id}")
        return ReframeResponse.model_validate(config)


@router.get("/{clip_id}", response_model=ReframeResponse)
async def get_reframe_config(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get the reframe configuration for a clip."""
    result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Reframe config not found")

    return ReframeResponse.model_validate(config)


@router.put("/{clip_id}", response_model=ReframeResponse)
async def update_reframe_config(
    clip_id: uuid.UUID,
    data: ReframeUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    """Update reframe configuration (e.g., manual keyframe adjustments)."""
    result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    config = result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Reframe config not found")

    update_data = data.model_dump(exclude_unset=True)

    # Validate aspect ratio if present
    if "aspect_ratio" in update_data:
        try:
            update_data["aspect_ratio"] = AspectRatio(update_data["aspect_ratio"])
            target_width, target_height = ASPECT_DIMENSIONS.get(
                update_data["aspect_ratio"].value, (1080, 1920)
            )
            update_data["target_width"] = target_width
            update_data["target_height"] = target_height
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aspect_ratio. Must be one of: {[e.value for e in AspectRatio]}"
            )

    # Validate tracking mode if present
    if "tracking_mode" in update_data:
        try:
            update_data["tracking_mode"] = TrackingMode(update_data["tracking_mode"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tracking_mode. Must be one of: {[e.value for e in TrackingMode]}"
            )

    # Convert keyframes from Pydantic to dict if present
    if "keyframes" in update_data and update_data["keyframes"]:
        update_data["keyframes"] = [
            kf.model_dump() if hasattr(kf, "model_dump") else kf
            for kf in update_data["keyframes"]
        ]

    # Reset processed status when config changes
    config.processed = False
    config.export_path = None

    for key, value in update_data.items():
        setattr(config, key, value)

    await session.commit()
    await session.refresh(config)

    logger.info(f"Updated reframe config for clip: {clip_id}")
    return ReframeResponse.model_validate(config)


@router.post("/{clip_id}/preview")
async def generate_reframe_preview(
    clip_id: uuid.UUID,
    duration: float = 5.0,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate a short preview of the reframed clip.
    """
    # Get reframe config
    config_result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    config = config_result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Reframe config not found. Generate one first.")

    # Get clip and video
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

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

    # Generate preview
    from app.services.reframe_service import get_reframe_service

    reframe_service = get_reframe_service()

    preview_dir = settings.cache_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_path = str(preview_dir / f"reframe_preview_{clip_id}_{uuid.uuid4().hex[:8]}.mp4")
    preview_duration = min(duration, clip.duration)

    await reframe_service.generate_reframed_video(
        video_path=video.file_path,
        output_path=preview_path,
        keyframes=config.keyframes or [],
        target_width=config.target_width,
        target_height=config.target_height,
        smooth_factor=config.smooth_factor,
        start_time=clip.start_time,
        end_time=clip.start_time + preview_duration,
    )

    logger.info(f"Generated reframe preview: {preview_path}")

    return {
        "status": "completed",
        "preview_path": preview_path,
        "duration": preview_duration,
        "aspect_ratio": config.aspect_ratio.value,
        "dimensions": f"{config.target_width}x{config.target_height}",
    }


@router.post("/{clip_id}/export")
async def export_reframed_clip(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export the full reframed clip.
    """
    # Get reframe config
    config_result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    config = config_result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Reframe config not found. Generate one first.")

    # Check if already exported
    if config.processed and config.export_path and Path(config.export_path).exists():
        return {
            "status": "completed",
            "export_path": config.export_path,
        }

    # Get clip and video
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

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

    # Export reframed clip
    from app.services.reframe_service import get_reframe_service

    reframe_service = get_reframe_service()

    export_path = str(settings.clip_storage_path / f"reframed_{clip_id}.mp4")

    await reframe_service.generate_reframed_video(
        video_path=video.file_path,
        output_path=export_path,
        keyframes=config.keyframes or [],
        target_width=config.target_width,
        target_height=config.target_height,
        smooth_factor=config.smooth_factor,
        start_time=clip.start_time,
        end_time=clip.end_time,
    )

    # Update config
    config.processed = True
    config.export_path = export_path
    await session.commit()

    logger.info(f"Exported reframed clip: {export_path}")

    return {
        "status": "completed",
        "export_path": export_path,
        "aspect_ratio": config.aspect_ratio.value,
        "dimensions": f"{config.target_width}x{config.target_height}",
    }


@router.get("/{clip_id}/download")
async def download_reframed_clip(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Download the exported reframed clip."""
    config_result = await session.execute(
        select(ReframeConfig).filter(ReframeConfig.clip_id == clip_id)
    )
    config = config_result.scalar_one_or_none()

    if not config:
        raise HTTPException(status_code=404, detail="Reframe config not found")

    if not config.processed or not config.export_path:
        raise HTTPException(status_code=400, detail="Clip not yet exported")

    export_path = Path(config.export_path)
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=export_path,
        filename=f"reframed_{clip_id}.mp4",
        media_type="video/mp4",
    )
