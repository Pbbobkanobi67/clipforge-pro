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
from app.models.database import AnalysisJob, ClipSuggestion, ClipExport, Video, CaptionStyle, BrandTemplate, get_async_session
from app.models.schemas import (
    ClipExportRequest, ClipSuggestionResponse, EnhancedClipExportRequest,
    BatchExportRequest, BatchExportResponse,
    ClipUpdateRequest, ClipDuplicateRequest, ClipCreateRequest, ClipExportHistoryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/batch/export", response_model=BatchExportResponse)
async def batch_export_clips(
    request: BatchExportRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export multiple clips at once with the same settings.
    Processes clips sequentially and returns a summary.
    """
    results = []
    succeeded = 0
    failed = 0

    for clip_id in request.clip_ids:
        try:
            enhanced_req = EnhancedClipExportRequest(
                format=request.format,
                include_captions=request.include_captions,
                caption_style_id=request.caption_style_id,
                remove_fillers=request.remove_fillers,
                add_emojis=request.add_emojis,
                keyword_highlight=request.keyword_highlight,
                brand_template_id=request.brand_template_id,
                aspect_ratio=request.aspect_ratio,
                auto_zoom=request.auto_zoom,
                remove_silence=request.remove_silence,
                video_fade=request.video_fade,
                split_layout=request.split_layout,
                split_ratio=request.split_ratio,
                separator_color=request.separator_color,
            )

            result = await export_clip_enhanced(
                clip_id=clip_id,
                request=enhanced_req,
                session=session,
            )
            results.append({
                "clip_id": str(clip_id),
                "status": "success",
                "export_path": result.export_path,
            })
            succeeded += 1
        except Exception as e:
            logger.error(f"Batch export failed for clip {clip_id}: {e}")
            results.append({
                "clip_id": str(clip_id),
                "status": "failed",
                "error": str(e),
            })
            failed += 1

    return BatchExportResponse(
        total=len(request.clip_ids),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@router.post("/create", response_model=ClipSuggestionResponse)
async def create_clip(
    request: ClipCreateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Manually create a clip from a video with specified time boundaries."""
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")

    # Find latest analysis job for this video
    job_result = await session.execute(
        select(AnalysisJob)
        .filter(AnalysisJob.video_id == request.video_id)
        .order_by(AnalysisJob.created_at.desc())
        .limit(1)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="No analysis job found for this video. Analyze the video first.")

    clip = ClipSuggestion(
        id=uuid.uuid4(),
        analysis_job_id=job.id,
        start_time=request.start_time,
        end_time=request.end_time,
        duration=request.end_time - request.start_time,
        title=request.title or f"Manual clip {request.start_time:.1f}s-{request.end_time:.1f}s",
        description=request.description,
        virality_score=0,
        emotional_resonance=0,
        shareability=0,
        uniqueness=0,
        hook_strength=0,
        production_quality=0,
        is_manual=True,
        version_label=request.version_label,
    )
    session.add(clip)
    await session.commit()
    await session.refresh(clip)
    return ClipSuggestionResponse.model_validate(clip)


@router.put("/manage/{clip_id}", response_model=ClipSuggestionResponse)
async def update_clip(
    clip_id: uuid.UUID,
    request: ClipUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """Edit a clip's boundaries, title, description, or version label."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    boundaries_changed = False
    if request.start_time is not None:
        clip.start_time = request.start_time
        boundaries_changed = True
    if request.end_time is not None:
        clip.end_time = request.end_time
        boundaries_changed = True
    if request.title is not None:
        clip.title = request.title
    if request.description is not None:
        clip.description = request.description
    if request.version_label is not None:
        clip.version_label = request.version_label

    if boundaries_changed:
        clip.duration = clip.end_time - clip.start_time
        if clip.duration <= 0:
            raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
        clip.exported = False
        clip.export_path = None

    await session.commit()
    await session.refresh(clip)
    return ClipSuggestionResponse.model_validate(clip)


@router.post("/manage/{clip_id}/duplicate", response_model=ClipSuggestionResponse)
async def duplicate_clip(
    clip_id: uuid.UUID,
    request: ClipDuplicateRequest = ClipDuplicateRequest(),
    session: AsyncSession = Depends(get_async_session),
):
    """Duplicate a clip for comparison with different settings."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    original = result.scalar_one_or_none()
    if not original:
        raise HTTPException(status_code=404, detail="Clip not found")

    copy = ClipSuggestion(
        id=uuid.uuid4(),
        analysis_job_id=original.analysis_job_id,
        start_time=original.start_time,
        end_time=original.end_time,
        duration=original.duration,
        title=f"{original.title or 'Clip'} (copy)",
        description=original.description,
        transcript_excerpt=original.transcript_excerpt,
        virality_score=original.virality_score,
        emotional_resonance=original.emotional_resonance,
        shareability=original.shareability,
        uniqueness=original.uniqueness,
        hook_strength=original.hook_strength,
        production_quality=original.production_quality,
        rank=None,
        exported=False,
        export_path=None,
        parent_clip_id=clip_id,
        version_label=request.version_label,
        is_manual=original.is_manual,
    )
    session.add(copy)
    await session.commit()
    await session.refresh(copy)
    return ClipSuggestionResponse.model_validate(copy)


@router.delete("/manage/{clip_id}")
async def delete_clip(
    clip_id: uuid.UUID,
    delete_files: bool = Query(False, description="Also delete export files from disk"),
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a clip and its export history."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    clip = result.scalar_one_or_none()
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if delete_files:
        # Delete export files from disk
        exports_result = await session.execute(
            select(ClipExport).filter(ClipExport.clip_id == clip_id)
        )
        for export in exports_result.scalars().all():
            try:
                Path(export.export_path).unlink(missing_ok=True)
            except Exception:
                pass
        if clip.export_path:
            try:
                Path(clip.export_path).unlink(missing_ok=True)
            except Exception:
                pass

    await session.delete(clip)
    await session.commit()
    return {"status": "deleted", "clip_id": str(clip_id)}


@router.get("/manage/{clip_id}/exports", response_model=list[ClipExportHistoryResponse])
async def get_clip_exports(
    clip_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get export history for a clip."""
    # Verify clip exists
    clip_result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    if not clip_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Clip not found")

    result = await session.execute(
        select(ClipExport)
        .filter(ClipExport.clip_id == clip_id)
        .order_by(ClipExport.created_at.desc())
    )
    exports = result.scalars().all()
    return [ClipExportHistoryResponse.model_validate(e) for e in exports]


@router.get("/manage/{clip_id}/exports/{export_id}/download")
async def download_clip_export(
    clip_id: uuid.UUID,
    export_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Download a specific historical export version."""
    result = await session.execute(
        select(ClipExport)
        .filter(ClipExport.id == export_id, ClipExport.clip_id == clip_id)
    )
    export = result.scalar_one_or_none()
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")

    file_path = Path(export.export_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / export.export_path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=f"clip_{clip_id}_{export_id}{file_path.suffix}",
        media_type="video/mp4",
    )


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
    request: ClipExportRequest = ClipExportRequest(),
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
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Ensure clips directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already exported
    if clip.exported and clip.export_path:
        existing = Path(clip.export_path)
        if not existing.is_absolute():
            existing = Path.cwd() / clip.export_path
        if existing.exists():
            return ClipSuggestionResponse.model_validate(clip)

    # Compute timestamp offset for trimmed analysis
    config = job.config or {}
    time_offset = config.get("clip_start_time", 0) or 0

    # Resolve video path
    video_path = Path(video.file_path)
    if not video_path.is_absolute():
        video_path = Path.cwd() / video.file_path

    # Export args (with offset-adjusted times for ffmpeg, original times for DB queries)
    export_kwargs = dict(
        clip_id=clip_id,
        video_path=str(video_path),
        output_path=str(output_path),
        start_time=clip.start_time + time_offset,
        end_time=clip.end_time + time_offset,
        format=request.format,
        resolution=request.resolution,
        include_captions=request.include_captions,
        job_id=job.id,
        db_start_time=clip.start_time if time_offset else None,
        db_end_time=clip.end_time if time_offset else None,
    )

    # Export in background
    if background_tasks:
        background_tasks.add_task(_export_clip_task, **export_kwargs)
        # Set export_path but NOT exported=True -- task will set it when done
        clip.export_path = str(output_path)
        await session.commit()
        await session.refresh(clip)
    else:
        # Synchronous export for immediate response
        await _export_clip_task(**export_kwargs)
        # File exists now, safe to mark exported
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
    db_start_time: Optional[float] = None,
    db_end_time: Optional[float] = None,
):
    """Background task to export clip.

    start_time/end_time: times relative to the source video file (offset-adjusted).
    db_start_time/db_end_time: original times in the DB (for caption segment queries).
    If db times not provided, assumes start_time/end_time match DB values.
    """
    from app.services.video_service import VideoService
    from app.models.database import SyncSessionLocal

    video_service = VideoService()
    session = SyncSessionLocal()

    # For caption queries, use DB times (no offset) if provided
    caption_start = db_start_time if db_start_time is not None else start_time
    caption_end = db_end_time if db_end_time is not None else end_time

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
                    .filter(TranscriptionSegment.start_time >= caption_start)
                    .filter(TranscriptionSegment.end_time <= caption_end)
                    .all()
                )
                captions = [
                    {
                        "start": seg.start_time - caption_start,
                        "end": seg.end_time - caption_start,
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

        # Update database + record export history
        clip = session.query(ClipSuggestion).filter(ClipSuggestion.id == clip_id).first()
        if clip:
            clip.export_path = output_path
            clip.exported = True

            # Record export in history
            file_size = None
            try:
                file_size = Path(output_path).stat().st_size
            except Exception:
                pass
            export_record = ClipExport(
                id=uuid.uuid4(),
                clip_id=clip_id,
                export_path=output_path,
                format=format,
                file_size_bytes=file_size,
                settings_json="{}",
            )
            session.add(export_record)
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
    if request.include_broll:
        suffix_parts.append("broll")
    if request.split_layout:
        suffix_parts.append("split")

    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    output_filename = f"clip_{clip_id}{suffix}.{request.format}"
    output_path = settings.clip_storage_path / output_filename
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute timestamp offset for trimmed analysis
    config = job.config or {}
    time_offset = config.get("clip_start_time", 0) or 0

    # Resolve video path
    video_path = Path(video.file_path)
    if not video_path.is_absolute():
        video_path = Path.cwd() / video.file_path

    # Build enhanced export args (with offset-adjusted times for ffmpeg, original for DB queries)
    enhanced_kwargs = dict(
        clip_id=clip_id,
        video_path=str(video_path),
        output_path=str(output_path),
        start_time=clip.start_time + time_offset,
        end_time=clip.end_time + time_offset,
        format=request.format,
        resolution=request.resolution,
        include_captions=request.include_captions,
        caption_style_id=str(request.caption_style_id) if request.caption_style_id else None,
        brand_template_id=str(request.brand_template_id) if request.brand_template_id else None,
        aspect_ratio=request.aspect_ratio,
        job_id=job.id,
        include_broll=request.include_broll,
        broll_transition_duration=request.broll_transition_duration,
        db_start_time=clip.start_time if time_offset else None,
        db_end_time=clip.end_time if time_offset else None,
        remove_fillers=request.remove_fillers,
        add_emojis=request.add_emojis,
        keyword_highlight=request.keyword_highlight,
        auto_zoom=request.auto_zoom,
        remove_silence=request.remove_silence,
        video_fade=request.video_fade,
        split_layout=request.split_layout,
        split_ratio=request.split_ratio,
        separator_color=request.separator_color,
    )

    # Export synchronously so file exists before response
    await _export_clip_enhanced_task(**enhanced_kwargs)

    # File exists now, safe to mark exported
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
    include_broll: bool = False,
    broll_transition_duration: float = 0.5,
    db_start_time: Optional[float] = None,
    db_end_time: Optional[float] = None,
    remove_fillers: bool = True,
    add_emojis: bool = False,
    keyword_highlight: bool = True,
    auto_zoom: bool = False,
    remove_silence: bool = False,
    video_fade: bool = False,
    split_layout: bool = False,
    split_ratio: float = 0.65,
    separator_color: str = "#333333",
):
    """Background task to export clip with enhanced options.

    start_time/end_time: times relative to the source video (offset-adjusted).
    db_start_time/db_end_time: original DB times for caption segment queries.
    """
    from app.services.video_service import VideoService
    from app.services.caption_service import CaptionService
    from app.models.database import SyncSessionLocal, CaptionStyle, BrandTemplate, Transcription, TranscriptionSegment, BRollSuggestion, BRollAsset

    video_service = VideoService()
    session = SyncSessionLocal()

    # For caption DB queries, use un-offset times if provided
    caption_start = db_start_time if db_start_time is not None else start_time
    caption_end = db_end_time if db_end_time is not None else end_time

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
                    .filter(TranscriptionSegment.start_time >= caption_start)
                    .filter(TranscriptionSegment.end_time <= caption_end)
                    .order_by(TranscriptionSegment.start_time)
                    .all()
                )

                if caption_style and segments:
                    # Generate ASS subtitles with style
                    caption_service = CaptionService()
                    ass_path = output_path.replace(f".{format}", ".ass")

                    # Convert caption style to dict
                    style_config = {
                        "style_type": caption_style.style_type,
                        "font_family": caption_style.font_family,
                        "font_size": caption_style.font_size,
                        "font_weight": caption_style.font_weight,
                        "text_color": caption_style.text_color,
                        "highlight_color": caption_style.highlight_color,
                        "background_color": caption_style.background_color,
                        "stroke_color": caption_style.stroke_color,
                        "stroke_width": caption_style.stroke_width,
                        "position": caption_style.position,
                        "margin_bottom": caption_style.margin_bottom,
                        "animation_duration": caption_style.animation_duration,
                        "words_per_line": caption_style.words_per_line,
                        "keyword_highlight": keyword_highlight,
                    }

                    await caption_service.generate_ass_subtitles(
                        segments=[{
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "text": seg.text,
                            "words": seg.words,
                        } for seg in segments],
                        output_path=ass_path,
                        style_config=style_config,
                        clip_start_offset=caption_start,
                        remove_fillers=remove_fillers,
                        add_emojis=add_emojis,
                    )
                else:
                    captions = [
                        {
                            "start": seg.start_time - caption_start,
                            "end": seg.end_time - caption_start,
                            "text": seg.text,
                        }
                        for seg in segments
                    ]

        # Split layout: generate split video first, then skip standard crop/reframe
        if split_layout:
            from app.services.split_layout_service import get_split_layout_service

            split_service = get_split_layout_service()
            layout_analysis = await split_service.analyze_layout(
                video_path=video_path,
                start_time=start_time,
                end_time=end_time,
            )

            await split_service.generate_split_video(
                video_path=video_path,
                output_path=output_path,
                layout_analysis=layout_analysis,
                split_ratio=split_ratio,
                separator_color=separator_color,
                start_time=start_time,
                end_time=end_time,
            )

            # Update database + record export history
            import json as json_mod
            clip = session.query(ClipSuggestion).filter(ClipSuggestion.id == clip_id).first()
            if clip:
                clip.export_path = output_path
                clip.exported = True
                file_size = None
                try:
                    file_size = Path(output_path).stat().st_size
                except Exception:
                    pass
                settings_snapshot = json_mod.dumps({
                    "format": format,
                    "split_layout": True,
                    "split_ratio": split_ratio,
                    "separator_color": separator_color,
                    "layout_type": layout_analysis.get("layout_type"),
                })
                export_record = ClipExport(
                    id=uuid.uuid4(),
                    clip_id=clip_id,
                    export_path=output_path,
                    format=format,
                    file_size_bytes=file_size,
                    settings_json=settings_snapshot,
                )
                session.add(export_record)
                session.commit()

            logger.info(f"Split layout clip exported: {clip_id} -> {output_path}")
            return  # Skip standard export pipeline

        # Build export options
        export_kwargs = {
            "input_path": video_path,
            "output_path": output_path,
            "start_time": start_time,
            "end_time": end_time,
            "resolution": resolution,
        }

        if ass_path:
            export_kwargs["ass_subtitle_path"] = ass_path
        elif captions:
            export_kwargs["captions"] = captions

        if aspect_ratio:
            # Use ReframeService to calculate crop dimensions
            from app.services.reframe_service import ReframeService
            reframe_service = ReframeService()
            crop_data = await reframe_service.analyze_video_for_reframe(
                video_path=video_path,
                aspect_ratio=aspect_ratio,
                tracking_mode="speaker",  # Default to speaker tracking
            )
            export_kwargs["crop_x"] = crop_data["keyframes"][0]["x"] if crop_data["keyframes"] else (crop_data["source_width"] - crop_data["crop_width"]) // 2
            export_kwargs["crop_y"] = crop_data["keyframes"][0]["y"] if crop_data["keyframes"] else (crop_data["source_height"] - crop_data["crop_height"]) // 2
            export_kwargs["crop_width"] = crop_data["crop_width"]
            export_kwargs["crop_height"] = crop_data["crop_height"]
            export_kwargs["scale_width"] = crop_data["target_width"]
            export_kwargs["scale_height"] = crop_data["target_height"]

        if brand_template:
            if brand_template.logo_path:
                export_kwargs["logo_path"] = brand_template.logo_path
                export_kwargs["logo_position"] = brand_template.logo_position
                export_kwargs["logo_size"] = brand_template.logo_size
                export_kwargs["logo_opacity"] = brand_template.logo_opacity

        # Auto-zoom: gentle zoom effect for visual interest
        if auto_zoom:
            export_kwargs["auto_zoom"] = True

        # Silence removal: remove dead air pauses
        if remove_silence:
            export_kwargs["remove_silence"] = True

        # Video fade transitions
        if video_fade:
            export_kwargs["video_fade"] = True

        # Use enhanced export if available, otherwise fall back to basic
        if hasattr(video_service, 'export_clip_enhanced'):
            await video_service.export_clip_enhanced(**export_kwargs)
        else:
            await video_service.export_clip(**export_kwargs)

        # Apply B-roll if requested
        if include_broll:
            from app.services.broll_integration_service import get_broll_integration_service

            # Query approved B-roll suggestions for this clip
            approved_suggestions = (
                session.query(BRollSuggestion)
                .filter(BRollSuggestion.clip_id == clip_id)
                .filter(BRollSuggestion.is_approved == True)
                .order_by(BRollSuggestion.start_time)
                .all()
            )

            if approved_suggestions:
                insertions = []
                for s in approved_suggestions:
                    if s.asset_id:
                        asset = session.query(BRollAsset).filter(BRollAsset.id == s.asset_id).first()
                        if asset and asset.local_path:
                            insertions.append({
                                "start_time": s.start_time,
                                "duration": s.duration,
                                "asset_path": asset.local_path,
                                "mode": s.insert_mode.value if s.insert_mode else "full_replace",
                                "transition": s.transition_type or "crossfade",
                            })

                if insertions:
                    # Apply B-roll to the exported clip
                    broll_service = get_broll_integration_service()
                    temp_output = output_path.replace(f".{format}", f"_temp.{format}")

                    # Rename current output to temp
                    import shutil
                    shutil.move(output_path, temp_output)

                    # Apply B-roll
                    await broll_service.apply_broll(
                        input_path=temp_output,
                        output_path=output_path,
                        broll_insertions=insertions,
                        transition_duration=broll_transition_duration,
                    )

                    # Clean up temp file
                    Path(temp_output).unlink(missing_ok=True)
                    logger.info(f"Applied {len(insertions)} B-roll insertions to clip {clip_id}")

        # Update database + record export history
        clip = session.query(ClipSuggestion).filter(ClipSuggestion.id == clip_id).first()
        if clip:
            clip.export_path = output_path
            clip.exported = True

            # Record export in history with settings snapshot
            import json
            file_size = None
            try:
                file_size = Path(output_path).stat().st_size
            except Exception:
                pass
            settings_snapshot = json.dumps({
                "format": format,
                "resolution": resolution,
                "include_captions": include_captions,
                "caption_style_id": caption_style_id,
                "brand_template_id": brand_template_id,
                "aspect_ratio": aspect_ratio,
                "include_broll": include_broll,
                "remove_fillers": remove_fillers,
                "add_emojis": add_emojis,
                "keyword_highlight": keyword_highlight,
                "auto_zoom": auto_zoom,
                "remove_silence": remove_silence,
                "video_fade": video_fade,
            })
            export_record = ClipExport(
                id=uuid.uuid4(),
                clip_id=clip_id,
                export_path=output_path,
                format=format,
                file_size_bytes=file_size,
                settings_json=settings_snapshot,
            )
            session.add(export_record)
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
    """Download an exported clip file. Exports on-demand if not yet exported."""
    result = await session.execute(select(ClipSuggestion).filter(ClipSuggestion.id == clip_id))
    clip = result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Resolve relative export path
    file_path = None
    if clip.export_path:
        file_path = Path(clip.export_path)
        if not file_path.is_absolute():
            file_path = Path.cwd() / clip.export_path

    # If file doesn't exist, do on-demand export
    if not file_path or not file_path.exists():
        # Get job and video for export
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

        # Compute timestamp offset for trimmed analysis
        config = job.config or {}
        time_offset = config.get("clip_start_time", 0) or 0

        output_filename = f"clip_{clip_id}.mp4"
        output_path = settings.clip_storage_path / output_filename
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        # Ensure clips directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve video path
        video_path = Path(video.file_path)
        if not video_path.is_absolute():
            video_path = Path.cwd() / video.file_path

        # Export synchronously with offset-adjusted timestamps
        try:
            await _export_clip_task(
                clip_id=clip_id,
                video_path=str(video_path),
                output_path=str(output_path),
                start_time=clip.start_time + time_offset,
                end_time=clip.end_time + time_offset,
                format="mp4",
                resolution=None,
                include_captions=False,
                job_id=job.id,
            )
        except Exception as e:
            logger.error(f"On-demand export failed for clip {clip_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Export failed: {e}")

        # Update clip record
        clip.export_path = str(output_path)
        clip.exported = True
        await session.commit()

        file_path = output_path

    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Export completed but file not found")

    return FileResponse(
        path=str(file_path),
        filename=f"clip_{clip_id}{file_path.suffix}",
        media_type="video/mp4",
    )


