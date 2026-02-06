"""B-Roll suggestion and integration endpoints."""

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.database import (
    AnalysisJob,
    BRollAsset,
    BRollDetectionReason,
    BRollInsertMode,
    BRollSuggestion,
    ClipSuggestion,
    Transcription,
    TranscriptionSegment,
    Video,
    get_async_session,
)
from app.models.schemas import (
    BRollApproveRequest,
    BRollAssetResponse,
    BRollExportRequest,
    BRollGenerateRequest,
    BRollSearchRequest,
    BRollSearchResult,
    BRollSelectAssetRequest,
    BRollSuggestionResponse,
)
from app.services.broll_detection_service import get_broll_detection_service
from app.services.broll_integration_service import get_broll_integration_service
from app.services.broll_search_service import get_broll_search_service
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/{clip_id}", response_model=list[BRollSuggestionResponse])
async def get_broll_suggestions(
    clip_id: uuid.UUID,
    approved_only: bool = Query(False, description="Only return approved suggestions"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get B-roll suggestions for a clip.

    Returns all detected B-roll insertion points with their metadata.
    """
    # Verify clip exists
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Query suggestions
    query = select(BRollSuggestion).filter(BRollSuggestion.clip_id == clip_id)

    if approved_only:
        query = query.filter(BRollSuggestion.is_approved == True)

    query = query.options(selectinload(BRollSuggestion.asset))
    query = query.order_by(BRollSuggestion.start_time)

    result = await session.execute(query)
    suggestions = result.scalars().all()

    return [BRollSuggestionResponse.model_validate(s) for s in suggestions]


@router.post("/{clip_id}/generate", response_model=list[BRollSuggestionResponse])
async def generate_broll_suggestions(
    clip_id: uuid.UUID,
    request: BRollGenerateRequest = BRollGenerateRequest(),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Generate AI-powered B-roll suggestions for a clip.

    Analyzes the transcript and visual data to identify optimal
    insertion points for stock footage.
    """
    # Get clip with job
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get job for transcription access
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == clip.analysis_job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Get transcription segments for this clip
    transcription_result = await session.execute(
        select(Transcription).filter(Transcription.analysis_job_id == job.id)
    )
    transcription = transcription_result.scalar_one_or_none()

    if not transcription:
        raise HTTPException(status_code=400, detail="No transcription available for this clip")

    # Get segments within clip bounds
    segments_result = await session.execute(
        select(TranscriptionSegment)
        .filter(TranscriptionSegment.transcription_id == transcription.id)
        .filter(TranscriptionSegment.start_time >= clip.start_time)
        .filter(TranscriptionSegment.end_time <= clip.end_time)
        .order_by(TranscriptionSegment.start_time)
    )
    segments = segments_result.scalars().all()

    # Convert to dicts with times relative to clip start
    segment_dicts = [
        {
            "start_time": seg.start_time - clip.start_time,
            "end_time": seg.end_time - clip.start_time,
            "text": seg.text,
            "words": seg.words,
        }
        for seg in segments
    ]

    # Get visual analysis if available (from scenes)
    visual_analysis = None  # TODO: Get from Scene analysis if available

    # Get LLM service if requested
    llm_service = None
    if request.use_llm:
        try:
            llm_service = get_llm_service()
        except Exception as e:
            logger.warning(f"Could not initialize LLM service: {e}")

    # Detect B-roll opportunities
    detection_service = get_broll_detection_service()
    opportunities = await detection_service.detect_broll_opportunities(
        clip_id=clip_id,
        transcription_segments=segment_dicts,
        visual_analysis=visual_analysis,
        llm_service=llm_service,
        max_suggestions=request.max_suggestions,
        min_duration=request.min_duration,
        max_duration=request.max_duration,
    )

    # Clear existing suggestions for this clip
    await session.execute(
        select(BRollSuggestion).filter(BRollSuggestion.clip_id == clip_id)
    )
    existing = await session.execute(
        select(BRollSuggestion).filter(BRollSuggestion.clip_id == clip_id)
    )
    for s in existing.scalars().all():
        await session.delete(s)

    # Create new suggestions
    created_suggestions = []
    for opp in opportunities:
        # Map detection reason to enum
        reason_map = {
            "abstract_concept": BRollDetectionReason.ABSTRACT_CONCEPT,
            "topic_change": BRollDetectionReason.TOPIC_CHANGE,
            "visual_gap": BRollDetectionReason.VISUAL_GAP,
            "transition": BRollDetectionReason.TRANSITION,
        }

        suggestion = BRollSuggestion(
            clip_id=clip_id,
            start_time=opp["start_time"],
            end_time=opp["end_time"],
            duration=opp["duration"],
            detection_reason=reason_map.get(opp["detection_reason"], BRollDetectionReason.ABSTRACT_CONCEPT),
            transcript_context=opp.get("transcript_context"),
            keywords=opp.get("keywords", []),
            search_queries=opp.get("search_queries", []),
            relevance_score=opp.get("relevance_score", 0.5),
            confidence=opp.get("confidence", 0.5),
            rank=opp.get("rank"),
        )

        session.add(suggestion)
        created_suggestions.append(suggestion)

    await session.commit()

    # Refresh to get IDs
    for s in created_suggestions:
        await session.refresh(s)

    return [BRollSuggestionResponse.model_validate(s) for s in created_suggestions]


@router.get("/search", response_model=list[BRollSearchResult])
async def search_stock_footage(
    q: str = Query(..., min_length=1, description="Search query"),
    providers: list[str] = Query(default=["pexels", "pixabay"], description="Providers to search"),
    min_duration: float = Query(default=3.0, ge=0, description="Minimum duration"),
    max_duration: float = Query(default=30.0, ge=1, description="Maximum duration"),
    orientation: str = Query(default="landscape", description="Video orientation"),
    limit: int = Query(default=10, ge=1, le=50, description="Results per provider"),
):
    """
    Search for stock footage across providers.

    Returns video metadata and preview URLs.
    """
    search_service = get_broll_search_service()

    results = await search_service.search(
        queries=[q],
        providers=providers,
        min_duration=min_duration,
        max_duration=max_duration,
        orientation=orientation,
        limit=limit,
    )

    return [BRollSearchResult.model_validate(r) for r in results]


@router.post("/suggestion/{suggestion_id}/select-asset", response_model=BRollSuggestionResponse)
async def select_asset_for_suggestion(
    suggestion_id: uuid.UUID,
    request: BRollSelectAssetRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Select a stock footage asset for a B-roll suggestion.

    Downloads the asset if not already cached and associates it with the suggestion.
    """
    # Get suggestion
    result = await session.execute(
        select(BRollSuggestion)
        .filter(BRollSuggestion.id == suggestion_id)
        .options(selectinload(BRollSuggestion.asset))
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="B-roll suggestion not found")

    # Check if asset already exists
    asset_result = await session.execute(
        select(BRollAsset).filter(
            BRollAsset.provider == request.provider,
            BRollAsset.provider_id == request.provider_id,
        )
    )
    asset = asset_result.scalar_one_or_none()

    search_service = get_broll_search_service()

    if not asset:
        # Download and create asset
        try:
            local_path = await search_service.download_asset(
                provider=request.provider,
                provider_id=request.provider_id,
                download_url=request.download_url,
                title=request.title,
            )

            asset = BRollAsset(
                provider=request.provider,
                provider_id=request.provider_id,
                provider_url=request.download_url,
                title=request.title,
                tags=request.tags,
                duration=request.duration,
                width=request.width,
                height=request.height,
                local_path=local_path,
            )
            session.add(asset)
            await session.flush()

        except Exception as e:
            logger.error(f"Failed to download asset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download asset: {str(e)}")

    # Update suggestion
    suggestion.asset_id = asset.id
    suggestion.insert_mode = BRollInsertMode(request.insert_mode)
    suggestion.transition_type = request.transition_type

    await session.commit()
    await session.refresh(suggestion)

    # Reload with asset
    result = await session.execute(
        select(BRollSuggestion)
        .filter(BRollSuggestion.id == suggestion_id)
        .options(selectinload(BRollSuggestion.asset))
    )
    suggestion = result.scalar_one()

    return BRollSuggestionResponse.model_validate(suggestion)


@router.post("/suggestion/{suggestion_id}/approve", response_model=BRollSuggestionResponse)
async def approve_suggestion(
    suggestion_id: uuid.UUID,
    request: BRollApproveRequest = BRollApproveRequest(),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Approve or unapprove a B-roll suggestion for export.

    Only approved suggestions will be included when exporting with B-roll.
    """
    result = await session.execute(
        select(BRollSuggestion)
        .filter(BRollSuggestion.id == suggestion_id)
        .options(selectinload(BRollSuggestion.asset))
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="B-roll suggestion not found")

    if request.is_approved and not suggestion.asset_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot approve suggestion without selected asset"
        )

    suggestion.is_approved = request.is_approved
    await session.commit()
    await session.refresh(suggestion)

    return BRollSuggestionResponse.model_validate(suggestion)


@router.post("/{clip_id}/export")
async def export_with_broll(
    clip_id: uuid.UUID,
    request: BRollExportRequest = BRollExportRequest(),
    background_tasks: BackgroundTasks = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export a clip with approved B-roll insertions.

    Applies all approved B-roll suggestions with specified transitions.
    """
    # Get clip with job
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
    )
    clip = clip_result.scalar_one_or_none()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # Get job and video
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
        raise HTTPException(status_code=404, detail="Source video not found")

    # Get approved B-roll suggestions
    suggestions_result = await session.execute(
        select(BRollSuggestion)
        .filter(BRollSuggestion.clip_id == clip_id)
        .filter(BRollSuggestion.is_approved == True)
        .options(selectinload(BRollSuggestion.asset))
        .order_by(BRollSuggestion.start_time)
    )
    suggestions = suggestions_result.scalars().all()

    if not suggestions:
        raise HTTPException(status_code=400, detail="No approved B-roll suggestions")

    # Build insertion list
    insertions = []
    for s in suggestions:
        if s.asset and s.asset.local_path:
            insertions.append({
                "start_time": s.start_time,
                "duration": s.duration,
                "asset_path": s.asset.local_path,
                "mode": s.insert_mode.value if s.insert_mode else "full_replace",
                "transition": s.transition_type or "crossfade",
            })

    if not insertions:
        raise HTTPException(status_code=400, detail="No valid B-roll assets found")

    # Generate output path
    output_filename = f"clip_{clip_id}_broll.{request.format}"
    output_path = settings.clip_storage_path / output_filename

    # First extract the clip from the source video
    from app.services.video_service import VideoService
    video_service = VideoService()

    temp_clip_path = settings.clip_storage_path / f"temp_clip_{clip_id}.mp4"

    await video_service.export_clip(
        input_path=video.file_path,
        output_path=str(temp_clip_path),
        start_time=clip.start_time,
        end_time=clip.end_time,
        resolution=request.resolution,
    )

    # Apply B-roll
    integration_service = get_broll_integration_service()
    await integration_service.apply_broll(
        input_path=str(temp_clip_path),
        output_path=str(output_path),
        broll_insertions=insertions,
        transition_duration=request.transition_duration,
    )

    # Clean up temp file
    temp_clip_path.unlink(missing_ok=True)

    # Update clip record
    clip.export_path = str(output_path)
    clip.exported = True
    await session.commit()

    return {
        "status": "success",
        "clip_id": str(clip_id),
        "output_path": str(output_path),
        "broll_count": len(insertions),
    }


@router.get("/suggestion/{suggestion_id}/preview")
async def get_broll_preview(
    suggestion_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get a preview frame showing B-roll insertion.

    Returns a PNG image showing how the B-roll will appear.
    """
    # Get suggestion with asset
    result = await session.execute(
        select(BRollSuggestion)
        .filter(BRollSuggestion.id == suggestion_id)
        .options(selectinload(BRollSuggestion.asset))
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="B-roll suggestion not found")

    if not suggestion.asset or not suggestion.asset.local_path:
        raise HTTPException(status_code=400, detail="No asset selected for this suggestion")

    # Get clip and video for main video path
    clip_result = await session.execute(
        select(ClipSuggestion).filter(ClipSuggestion.id == suggestion.clip_id)
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
        raise HTTPException(status_code=404, detail="Source video not found")

    # Generate preview frame
    integration_service = get_broll_integration_service()
    preview_bytes = await integration_service.get_broll_preview_frame(
        main_video_path=video.file_path,
        broll_path=suggestion.asset.local_path,
        timestamp=clip.start_time + suggestion.start_time,
        mode=suggestion.insert_mode.value if suggestion.insert_mode else "full_replace",
    )

    return Response(content=preview_bytes, media_type="image/png")


@router.delete("/suggestion/{suggestion_id}")
async def delete_suggestion(
    suggestion_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a B-roll suggestion."""
    result = await session.execute(
        select(BRollSuggestion).filter(BRollSuggestion.id == suggestion_id)
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="B-roll suggestion not found")

    await session.delete(suggestion)
    await session.commit()

    return {"status": "deleted", "id": str(suggestion_id)}
