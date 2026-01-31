"""Clip suggestion and export endpoints."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import AnalysisJob, ClipSuggestion, Video, CaptionStyle, BrandTemplate, get_async_session
from app.models.schemas import ClipExportRequest, ClipSuggestionResponse, EnhancedClipExportRequest

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/{job_id}", response_model=list[ClipSuggestionResponse])
async def get_clips(
    job_id: uuid.UUID,
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum virality score"),
    limit: int = Query(10, ge=1, le=50, description="Maximum clips to return"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get suggested clips for an analysis job.

    Returns clips sorted by virality score (descending).
    """
    # Verify job exists
    job_result = await session.execute(select(AnalysisJob).filter(AnalysisJob.id == job_id))
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Query clips
    query = select(ClipSuggestion).filter(ClipSuggestion.analysis_job_id == job_id)

    if min_score is not None:
        query = query.filter(ClipSuggestion.virality_score >= min_score)

    query = query.order_by(ClipSuggestion.virality_score.desc()).limit(limit)

    result = await session.execute(query)
    clips = result.scalars().all()

    return [ClipSuggestionResponse.model_validate(c) for c in clips]


@router.get("/detail/{clip_id}", response_model=ClipSuggestionResponse)
async def get_clip_detail(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get detailed information about a specific clip."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    clip = result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    return ClipSuggestionResponse.model_validate(clip)


@router.post("/{clip_id}/export", response_model=ClipSuggestionResponse)
async def export_clip(
    clip_id: uuid.UUID,
    request: ClipExportRequest = ClipExportRequest(clip_id=uuid.uuid4()),
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export a clip as a video file.

    Creates a new video file with the specified clip boundaries.
    Optionally burns in captions from the transcription.
    """
    # Get clip with job and video
    result = await session.execute(
        select(ClipSuggestion)
        .filter(ClipSuggestion.id == clip_id)
        .options()
    )
    clip = result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get video through job
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == clip.analysis_job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    video_result = await session.execute(select(Video).filter(Video.id == job.video_id))
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Source video not found")

    # Generate output path
    output_filename = f"clip_{clip_id}.{request.format}"
    output_path = settings.clip_storage_path / output_filename

    # Check if already exported
    if clip.exported and clip.export_path and Path(clip.export_path).exists():
        return ClipSuggestionResponse.model_validate(clip)

    # Export in background
    if background_tasks:
        background_tasks.add_task(
            _export_clip_task,
            clip_id=clip_id,
            video_path=video.file_path,
            output_path=str(output_path),
            start_time=clip.start_time,
            end_time=clip.end_time,
            format=request.format,
            resolution=request.resolution,
            include_captions=request.include_captions,
            job_id=job.id,
        )
    else:
        # Synchronous export for immediate response
        await _export_clip_task(
            clip_id=clip_id,
            video_path=video.file_path,
            output_path=str(output_path),
            start_time=clip.start_time,
            end_time=clip.end_time,
            format=request.format,
            resolution=request.resolution,
            include_captions=request.include_captions,
            job_id=job.id,
        )

    # Update clip record
    clip.export_path = str(output_path)
    clip.exported = True
    await session.commit()
    await session.refresh(clip)

    return ClipSuggestionResponse.model_validate(clip)


async def _export_clip_task(
    clip_id: uuid.UUID,
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    format: str,
    resolution: Optional[str],
    include_captions: bool,
    job_id: uuid.UUID,
):
    """Background task to export clip."""
    from app.services.video_service import VideoService
    from app.models.database import SyncSessionLocal

    video_service = VideoService()
    session = SyncSessionLocal()

    try:
        # Get captions if needed
        captions = None
        if include_captions:
            from app.models.database import Transcription, TranscriptionSegment

            transcription = (
                session.query(Transcription)
                .filter(Transcription.analysis_job_id == job_id)
                .first()
            )
            if transcription:
                segments = (
                    session.query(TranscriptionSegment)
                    .filter(TranscriptionSegment.transcription_id == transcription.id)
                    .filter(TranscriptionSegment.start_time >= start_time)
                    .filter(TranscriptionSegment.end_time <= end_time)
                    .all()
                )
                captions = [
                    {
                        "start": seg.start_time - start_time,
                        "end": seg.end_time - start_time,
                        "text": seg.text,
                    }
                    for seg in segments
                ]

        # Export clip
        await video_service.export_clip(
            input_path=video_path,
            output_path=output_path,
            start_time=start_time,
            end_time=end_time,
            resolution=resolution,
            captions=captions,
        )

        # Update database
        clip = session.query(ClipSuggestion).filter(ClipSuggestion.id == clip_id).first()
        if clip:
            clip.export_path = output_path
            clip.exported = True
            session.commit()

        logger.info(f"Clip exported: {clip_id} -> {output_path}")

    except Exception as e:
        logger.error(f"Failed to export clip {clip_id}: {e}")
        raise
    finally:
        session.close()


@router.post("/{clip_id}/export/enhanced", response_model=ClipSuggestionResponse)
async def export_clip_enhanced(
    clip_id: uuid.UUID,
    request: EnhancedClipExportRequest,
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export a clip with enhanced options including captions, branding, and reframe.

    Supports:
    - Animated caption styles (karaoke, bounce, etc.)
    - Brand templates with logo overlay and outro
    - Aspect ratio reframing (9:16, 1:1, 16:9)
    """
    # Get clip
    result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get job and video
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == clip.analysis_job_id)
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    video_result = await session.execute(select(Video).filter(Video.id == job.video_id))
    video = video_result.scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=404, detail="Source video not found")

    # Get caption style if specified
    caption_style = None
    if request.include_captions and request.caption_style_id:
        style_result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.id == request.caption_style_id)
        )
        caption_style = style_result.scalar_one_or_none()

    # Get brand template if specified
    brand_template = None
    if request.brand_template_id:
        brand_result = await session.execute(
            select(BrandTemplate).filter(BrandTemplate.id == request.brand_template_id)
        )
        brand_template = brand_result.scalar_one_or_none()

    # Build output filename with options
    suffix_parts = []
    if request.aspect_ratio:
        suffix_parts.append(request.aspect_ratio.replace(":", "x"))
    if caption_style:
        suffix_parts.append("captions")
    if brand_template:
        suffix_parts.append("branded")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_filename = f"clip_{clip_id}{suffix}.{request.format}"
    output_path = settings.clip_storage_path / output_filename

    # Export with enhanced options
    if background_tasks:
        background_tasks.add_task(
            _export_clip_enhanced_task,
            clip_id=clip_id,
            video_path=video.file_path,
            output_path=str(output_path),
            start_time=clip.start_time,
            end_time=clip.end_time,
            format=request.format,
            resolution=request.resolution,
            include_captions=request.include_captions,
            caption_style_id=str(request.caption_style_id) if request.caption_style_id else None,
            brand_template_id=str(request.brand_template_id) if request.brand_template_id else None,
            aspect_ratio=request.aspect_ratio,
            job_id=job.id,
        )
    else:
        await _export_clip_enhanced_task(
            clip_id=clip_id,
            video_path=video.file_path,
            output_path=str(output_path),
            start_time=clip.start_time,
            end_time=clip.end_time,
            format=request.format,
            resolution=request.resolution,
            include_captions=request.include_captions,
            caption_style_id=str(request.caption_style_id) if request.caption_style_id else None,
            brand_template_id=str(request.brand_template_id) if request.brand_template_id else None,
            aspect_ratio=request.aspect_ratio,
            job_id=job.id,
        )

    # Update clip record
    clip.export_path = str(output_path)
    clip.exported = True
    await session.commit()
    await session.refresh(clip)

    return ClipSuggestionResponse.model_validate(clip)


async def _export_clip_enhanced_task(
    clip_id: uuid.UUID,
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    format: str,
    resolution: Optional[str],
    include_captions: bool,
    caption_style_id: Optional[str],
    brand_template_id: Optional[str],
    aspect_ratio: Optional[str],
    job_id: uuid.UUID,
):
    """Background task to export clip with enhanced options."""
    from app.services.video_service import VideoService
    from app.services.caption_service import CaptionService
    from app.models.database import SyncSessionLocal, CaptionStyle, BrandTemplate, Transcription, TranscriptionSegment

    video_service = VideoService()
    session = SyncSessionLocal()

    try:
        # Get caption style
        caption_style = None
        if caption_style_id:
            caption_style = session.query(CaptionStyle).filter(CaptionStyle.id == caption_style_id).first()

        # Get brand template
        brand_template = None
        if brand_template_id:
            brand_template = session.query(BrandTemplate).filter(BrandTemplate.id == brand_template_id).first()

        # Get captions if needed
        captions = None
        ass_path = None
        if include_captions:
            transcription = (
                session.query(Transcription)
                .filter(Transcription.analysis_job_id == job_id)
                .first()
            )
            if transcription:
                segments = (
                    session.query(TranscriptionSegment)
                    .filter(TranscriptionSegment.transcription_id == transcription.id)
                    .filter(TranscriptionSegment.start_time >= start_time)
                    .filter(TranscriptionSegment.end_time <= end_time)
                    .order_by(TranscriptionSegment.start_time)
                    .all()
                )

                if caption_style and segments:
                    # Generate ASS subtitles with style
                    caption_service = CaptionService()
                    ass_content = caption_service.generate_ass_subtitles(
                        segments=[{
                            "start": seg.start_time - start_time,
                            "end": seg.end_time - start_time,
                            "text": seg.text,
                            "words": seg.words,
                        } for seg in segments],
                        style=caption_style,
                    )
                    ass_path = output_path.replace(f".{format}", ".ass")
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
                else:
                    captions = [
                        {
                            "start": seg.start_time - start_time,
                            "end": seg.end_time - start_time,
                            "text": seg.text,
                        }
                        for seg in segments
                    ]

        # Build export options
        export_kwargs = {
            "input_path": video_path,
            "output_path": output_path,
            "start_time": start_time,
            "end_time": end_time,
            "resolution": resolution,
        }

        if ass_path:
            export_kwargs["ass_path"] = ass_path
        elif captions:
            export_kwargs["captions"] = captions

        if aspect_ratio:
            export_kwargs["aspect_ratio"] = aspect_ratio

        if brand_template:
            if brand_template.logo_path:
                export_kwargs["logo_path"] = brand_template.logo_path
                export_kwargs["logo_position"] = brand_template.logo_position
                export_kwargs["logo_size"] = brand_template.logo_size
                export_kwargs["logo_opacity"] = brand_template.logo_opacity

        # Use enhanced export if available, otherwise fall back to basic
        if hasattr(video_service, 'export_clip_enhanced'):
            await video_service.export_clip_enhanced(**export_kwargs)
        else:
            await video_service.export_clip(**export_kwargs)

        # Update database
        clip = session.query(ClipSuggestion).filter(ClipSuggestion.id == clip_id).first()
        if clip:
            clip.export_path = output_path
            clip.exported = True
            session.commit()

        logger.info(f"Enhanced clip exported: {clip_id} -> {output_path}")

    except Exception as e:
        logger.error(f"Failed to export enhanced clip {clip_id}: {e}")
        raise
    finally:
        session.close()


@router.get("/{clip_id}/download")
async def download_clip(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Download an exported clip file."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    clip = result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if not clip.exported or not clip.export_path:
        raise HTTPException(status_code=400, detail="Clip not yet exported")

    file_path = Path(clip.export_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=file_path,
        filename=f"clip_{clip_id}{file_path.suffix}",
        media_type="video/mp4",
    )
