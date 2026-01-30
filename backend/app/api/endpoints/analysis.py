"""Analysis control endpoints."""

import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.models.database import (
    AnalysisJob,
    AnalysisStatus,
    Video,
    VideoStatus,
    get_async_session,
)
from app.models.schemas import (
    AnalysisFullResponse,
    AnalysisJobResponse,
    AnalysisStartRequest,
    AnalysisStatusResponse,
    TranscriptionResponse,
    TranscriptionSegmentResponse,
    DiarizationResponse,
    DiarizationSegmentResponse,
    SceneResponse,
    HookResponse,
    ClipSuggestionResponse,
    VideoResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/{video_id}/start", response_model=AnalysisJobResponse)
async def start_analysis(
    video_id: uuid.UUID,
    request: AnalysisStartRequest = AnalysisStartRequest(),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Start AI analysis on a video.

    Triggers the analysis pipeline:
    1. Transcription (faster-whisper)
    2. Speaker diarization (pyannote)
    3. Scene detection (OpenCV)
    4. Visual analysis (YOLO)
    5. Hook detection
    6. Virality scoring
    7. Clip suggestions
    """
    # Check video exists and is ready
    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if video.status != VideoStatus.READY:
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready for analysis. Status: {video.status}",
        )

    # Check for existing running analysis
    existing_result = await session.execute(
        select(AnalysisJob).filter(
            AnalysisJob.video_id == video_id,
            AnalysisJob.status.in_([AnalysisStatus.PENDING, AnalysisStatus.PROCESSING]),
        )
    )
    existing = existing_result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Analysis already in progress for this video. Job ID: {existing.id}",
        )

    # Create analysis job
    job = AnalysisJob(
        video_id=video_id,
        status=AnalysisStatus.PENDING,
        progress=0.0,
        config=request.model_dump(),
    )

    session.add(job)
    await session.commit()
    await session.refresh(job)

    # Try Celery first, fall back to direct async execution
    celery_available = False
    try:
        # Quick Redis check before attempting Celery
        import redis
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1)
        r.ping()

        from app.workers.tasks import run_analysis_pipeline
        run_analysis_pipeline.delay(str(job.id))
        celery_available = True
        logger.info(f"Analysis job queued via Celery: {job.id}")
    except (ImportError, redis.ConnectionError, redis.TimeoutError):
        logger.info("Redis/Celery not available, using direct execution")
    except Exception as e:
        logger.warning(f"Celery failed ({e}), using direct execution")

    # Fallback: Run pipeline directly in background
    if not celery_available:
        asyncio.create_task(
            _run_analysis_directly(str(job.id), video.file_path, request.model_dump())
        )
        logger.info(f"Analysis job started directly: {job.id}")

    return job


async def _run_analysis_directly(job_id: str, video_path: str, config: dict):
    """Run the analysis pipeline directly without Celery."""
    from app.models.database import SyncSessionLocal
    from app.services.analysis_pipeline import run_pipeline_async, update_job_status

    # Resolve relative path to absolute
    video_path_resolved = Path(video_path)
    if not video_path_resolved.is_absolute():
        video_path_resolved = Path.cwd() / video_path
    video_path = str(video_path_resolved)

    session = SyncSessionLocal()
    try:
        result = await run_pipeline_async(session, job_id, video_path, config)

        if result.get("success"):
            update_job_status(session, job_id, AnalysisStatus.COMPLETED, 100, "Completed")
            logger.info(f"Analysis pipeline completed for job {job_id}")
        else:
            update_job_status(
                session, job_id, AnalysisStatus.FAILED, result.get("progress", 0),
                error=result.get("error", "Unknown error")
            )
    except Exception as e:
        logger.exception(f"Direct pipeline failed for job {job_id}: {e}")
        update_job_status(session, job_id, AnalysisStatus.FAILED, 0, error=str(e))
    finally:
        session.close()


@router.get("/{job_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get current status and progress of an analysis job."""
    result = await session.execute(select(AnalysisJob).filter(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Determine completed stages based on status
    status_order = [
        AnalysisStatus.PENDING,
        AnalysisStatus.PROCESSING,
        AnalysisStatus.TRANSCRIBING,
        AnalysisStatus.DIARIZING,
        AnalysisStatus.DETECTING_SCENES,
        AnalysisStatus.ANALYZING_VISUALS,
        AnalysisStatus.DETECTING_HOOKS,
        AnalysisStatus.SCORING_VIRALITY,
        AnalysisStatus.SUGGESTING_CLIPS,
        AnalysisStatus.COMPLETED,
    ]

    current_index = (
        status_order.index(job.status) if job.status in status_order else 0
    )
    stages_completed = [s.value for s in status_order[1:current_index]]

    return AnalysisStatusResponse(
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        current_stage=job.current_stage,
        stages_completed=stages_completed,
        error_message=job.error_message,
    )


@router.get("/{job_id}/stream")
async def stream_analysis_status(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Stream analysis progress via Server-Sent Events (SSE).

    Returns real-time updates as the analysis progresses.
    """
    # Verify job exists
    result = await session.execute(select(AnalysisJob).filter(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for analysis progress."""
        from app.models.database import AsyncSessionLocal

        last_progress = -1
        last_status = None

        while True:
            async with AsyncSessionLocal() as new_session:
                result = await new_session.execute(
                    select(AnalysisJob).filter(AnalysisJob.id == job_id)
                )
                job = result.scalar_one_or_none()

                if not job:
                    yield f"data: {{'error': 'Job not found'}}\n\n"
                    break

                # Send update if progress changed
                if job.progress != last_progress or job.status != last_status:
                    last_progress = job.progress
                    last_status = job.status

                    import json

                    data = {
                        "job_id": str(job.id),
                        "status": job.status.value,
                        "progress": job.progress,
                        "current_stage": job.current_stage,
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                # Stop if completed or failed
                if job.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                    if job.status == AnalysisStatus.FAILED:
                        data = {
                            "job_id": str(job.id),
                            "status": job.status.value,
                            "error": job.error_message,
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    break

            await asyncio.sleep(1)  # Poll every second

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/full", response_model=AnalysisFullResponse)
async def get_full_analysis(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get complete analysis results including all data."""
    # Load job with all relationships
    result = await session.execute(
        select(AnalysisJob)
        .filter(AnalysisJob.id == job_id)
        .options(
            selectinload(AnalysisJob.video),
            selectinload(AnalysisJob.transcription),
            selectinload(AnalysisJob.diarization),
            selectinload(AnalysisJob.scenes),
            selectinload(AnalysisJob.hooks),
            selectinload(AnalysisJob.clips),
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Load nested relationships
    transcription_response = None
    if job.transcription:
        # Load segments
        from app.models.database import TranscriptionSegment

        seg_result = await session.execute(
            select(TranscriptionSegment).filter(
                TranscriptionSegment.transcription_id == job.transcription.id
            )
        )
        segments = seg_result.scalars().all()

        transcription_response = TranscriptionResponse(
            id=job.transcription.id,
            full_text=job.transcription.full_text,
            language=job.transcription.language,
            language_probability=job.transcription.language_probability,
            word_count=job.transcription.word_count,
            duration_seconds=job.transcription.duration_seconds,
            segments=[TranscriptionSegmentResponse.model_validate(s) for s in segments],
        )

    diarization_response = None
    if job.diarization:
        from app.models.database import DiarizationSegment

        seg_result = await session.execute(
            select(DiarizationSegment).filter(
                DiarizationSegment.diarization_id == job.diarization.id
            )
        )
        segments = seg_result.scalars().all()

        diarization_response = DiarizationResponse(
            id=job.diarization.id,
            speaker_count=job.diarization.speaker_count,
            speaker_labels=job.diarization.speaker_labels,
            segments=[DiarizationSegmentResponse.model_validate(s) for s in segments],
        )

    return AnalysisFullResponse(
        job=AnalysisJobResponse.model_validate(job),
        video=VideoResponse.model_validate(job.video),
        transcription=transcription_response,
        diarization=diarization_response,
        scenes=[SceneResponse.model_validate(s) for s in job.scenes],
        hooks=[HookResponse.model_validate(h) for h in job.hooks],
        clips=[ClipSuggestionResponse.model_validate(c) for c in job.clips],
    )


@router.delete("/{job_id}")
async def cancel_analysis(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Cancel a running analysis job."""
    result = await session.execute(select(AnalysisJob).filter(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    if job.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )

    # Try to revoke Celery task
    try:
        from app.workers.celery_app import celery_app

        celery_app.control.revoke(str(job_id), terminate=True)
    except Exception as e:
        logger.warning(f"Failed to revoke Celery task: {e}")

    job.status = AnalysisStatus.FAILED
    job.error_message = "Cancelled by user"
    await session.commit()

    return {"message": "Analysis cancelled", "job_id": str(job_id)}
