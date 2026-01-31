"""Export endpoints for XML/EDL formats."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    AnalysisJob,
    Video,
    ClipSuggestion,
    Transcription,
    TranscriptionSegment,
    get_async_session,
)
from app.models.schemas import XMLExportRequest

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# Export directory
EXPORT_DIR = settings.storage_path / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/{job_id}/xml/fcpxml")
async def export_fcpxml(
    job_id: uuid.UUID,
    request: XMLExportRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export clips to Final Cut Pro X XML format (FCPXML).

    Creates an FCPXML file that can be imported into Final Cut Pro X.
    """
    # Get analysis job
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Get video
    video_result = await session.execute(
        select(Video).filter(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get clips
    clips_result = await session.execute(
        select(ClipSuggestion)
        .filter(ClipSuggestion.analysis_job_id == job_id)
        .order_by(ClipSuggestion.rank)
    )
    clips = clips_result.scalars().all()

    if not clips:
        raise HTTPException(status_code=404, detail="No clips found for this job")

    # Convert clips to dict format
    clip_dicts = [
        {
            "start_time": c.start_time,
            "end_time": c.end_time,
            "title": c.title or f"Clip {i+1}",
            "transcript_excerpt": c.transcript_excerpt,
        }
        for i, c in enumerate(clips)
    ]

    # Get captions if requested
    captions = None
    if request.include_captions:
        trans_result = await session.execute(
            select(Transcription).filter(Transcription.analysis_job_id == job_id)
        )
        transcription = trans_result.scalar_one_or_none()

        if transcription:
            segments_result = await session.execute(
                select(TranscriptionSegment)
                .filter(TranscriptionSegment.transcription_id == transcription.id)
                .order_by(TranscriptionSegment.start_time)
            )
            segments = segments_result.scalars().all()
            captions = [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "text": s.text,
                }
                for s in segments
            ]

    # Export
    from app.services.xml_export_service import get_xml_export_service
    xml_service = get_xml_export_service()

    output_filename = f"export_{job_id}_fcpxml.fcpxml"
    output_path = str(EXPORT_DIR / output_filename)

    await xml_service.export_fcpxml(
        clips=clip_dicts,
        video_path=video.file_path,
        output_path=output_path,
        project_name=video.title or "Video Extract Pro Export",
        fps=video.fps or 30.0,
        width=video.width or 1920,
        height=video.height or 1080,
        include_markers=request.include_markers,
        include_captions=request.include_captions,
        captions=captions,
    )

    logger.info(f"Exported FCPXML: {output_path}")

    return {
        "status": "completed",
        "format": "fcpxml",
        "export_path": output_path,
        "filename": output_filename,
        "clip_count": len(clips),
    }


@router.post("/{job_id}/xml/premiere")
async def export_premiere_xml(
    job_id: uuid.UUID,
    request: XMLExportRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export clips to Adobe Premiere Pro XML format.

    Creates an XML file that can be imported into Adobe Premiere Pro.
    """
    # Get analysis job
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Get video
    video_result = await session.execute(
        select(Video).filter(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get clips
    clips_result = await session.execute(
        select(ClipSuggestion)
        .filter(ClipSuggestion.analysis_job_id == job_id)
        .order_by(ClipSuggestion.rank)
    )
    clips = clips_result.scalars().all()

    if not clips:
        raise HTTPException(status_code=404, detail="No clips found for this job")

    # Convert clips to dict format
    clip_dicts = [
        {
            "start_time": c.start_time,
            "end_time": c.end_time,
            "title": c.title or f"Clip {i+1}",
            "transcript_excerpt": c.transcript_excerpt,
        }
        for i, c in enumerate(clips)
    ]

    # Get captions if requested
    captions = None
    if request.include_captions:
        trans_result = await session.execute(
            select(Transcription).filter(Transcription.analysis_job_id == job_id)
        )
        transcription = trans_result.scalar_one_or_none()

        if transcription:
            segments_result = await session.execute(
                select(TranscriptionSegment)
                .filter(TranscriptionSegment.transcription_id == transcription.id)
                .order_by(TranscriptionSegment.start_time)
            )
            segments = segments_result.scalars().all()
            captions = [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "text": s.text,
                }
                for s in segments
            ]

    # Export
    from app.services.xml_export_service import get_xml_export_service
    xml_service = get_xml_export_service()

    output_filename = f"export_{job_id}_premiere.xml"
    output_path = str(EXPORT_DIR / output_filename)

    await xml_service.export_premiere_xml(
        clips=clip_dicts,
        video_path=video.file_path,
        output_path=output_path,
        project_name=video.title or "Video Extract Pro Export",
        fps=video.fps or 30.0,
        width=video.width or 1920,
        height=video.height or 1080,
        include_markers=request.include_markers,
        include_captions=request.include_captions,
        captions=captions,
    )

    logger.info(f"Exported Premiere XML: {output_path}")

    return {
        "status": "completed",
        "format": "premiere",
        "export_path": output_path,
        "filename": output_filename,
        "clip_count": len(clips),
    }


@router.post("/{job_id}/edl")
async def export_edl(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Export clips to EDL (Edit Decision List) format.

    Creates a simple text EDL file compatible with most NLEs.
    """
    # Get analysis job
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Get video
    video_result = await session.execute(
        select(Video).filter(Video.id == job.video_id)
    )
    video = video_result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Get clips
    clips_result = await session.execute(
        select(ClipSuggestion)
        .filter(ClipSuggestion.analysis_job_id == job_id)
        .order_by(ClipSuggestion.rank)
    )
    clips = clips_result.scalars().all()

    if not clips:
        raise HTTPException(status_code=404, detail="No clips found for this job")

    # Convert clips to dict format
    clip_dicts = [
        {
            "start_time": c.start_time,
            "end_time": c.end_time,
            "title": c.title or f"Clip {i+1}",
            "transcript_excerpt": c.transcript_excerpt,
        }
        for i, c in enumerate(clips)
    ]

    # Export
    from app.services.xml_export_service import get_xml_export_service
    xml_service = get_xml_export_service()

    output_filename = f"export_{job_id}.edl"
    output_path = str(EXPORT_DIR / output_filename)

    await xml_service.export_edl(
        clips=clip_dicts,
        output_path=output_path,
        project_name=video.title or "Video Extract Pro Export",
        fps=video.fps or 30.0,
    )

    logger.info(f"Exported EDL: {output_path}")

    return {
        "status": "completed",
        "format": "edl",
        "export_path": output_path,
        "filename": output_filename,
        "clip_count": len(clips),
    }


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download an exported file."""
    # Security: only allow files in export directory
    file_path = EXPORT_DIR / filename

    # Validate path doesn't escape export directory
    try:
        file_path = file_path.resolve()
        EXPORT_DIR.resolve()
        if not str(file_path).startswith(str(EXPORT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    # Determine media type
    media_type = "application/xml"
    if filename.endswith(".edl"):
        media_type = "text/plain"
    elif filename.endswith(".fcpxml"):
        media_type = "application/xml"

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )
