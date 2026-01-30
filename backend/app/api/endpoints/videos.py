"""Video upload and download endpoints."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Video, VideoStatus, get_async_session
from app.models.schemas import VideoCreate, VideoDownloadRequest, VideoResponse
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=VideoResponse)
async def upload_video(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Upload a video file for analysis.

    Accepts common video formats (mp4, webm, mkv, mov, avi).
    """
    # Validate file type
    allowed_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".flv"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Generate unique file path
    video_id = uuid.uuid4()
    file_name = f"{video_id}{file_ext}"
    file_path = settings.video_storage_path / file_name

    # Save file
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # Get file size
    file_size = file_path.stat().st_size

    # Check size limit
    if file_size > settings.max_video_size_mb * 1024 * 1024:
        file_path.unlink()  # Delete file
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_video_size_mb}MB",
        )

    # Extract video metadata
    video_service = VideoService()
    try:
        metadata = await video_service.get_video_metadata(str(file_path))
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
        metadata = {}

    # Check duration limit
    duration = metadata.get("duration", 0)
    if duration > settings.max_video_duration_minutes * 60:
        file_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Video too long. Maximum duration: {settings.max_video_duration_minutes} minutes",
        )

    # Create database record
    video = Video(
        id=video_id,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size_bytes=file_size,
        duration_seconds=metadata.get("duration"),
        width=metadata.get("width"),
        height=metadata.get("height"),
        fps=metadata.get("fps"),
        title=title or metadata.get("title") or file.filename,
        description=description,
        status=VideoStatus.READY,
    )

    session.add(video)
    await session.commit()
    await session.refresh(video)

    logger.info(f"Video uploaded: {video.id}")
    return video


@router.post("/download", response_model=VideoResponse)
async def download_video(
    request: VideoDownloadRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Download a video from URL (YouTube, Vimeo, etc.) for analysis.

    Uses yt-dlp to download the video.
    """
    video_service = VideoService()

    # Detect platform
    platform = video_service.detect_platform(request.url)

    # Create video record (pending)
    video_id = uuid.uuid4()
    video = Video(
        id=video_id,
        source_url=request.url,
        source_platform=platform,
        title=request.title or "Downloading...",
        status=VideoStatus.DOWNLOADING,
        file_path="",  # Will be set after download
    )

    session.add(video)
    await session.commit()
    await session.refresh(video)

    # Determine output directory
    output_dir = request.output_dir if request.output_dir else str(settings.video_storage_path)

    # Start download in background
    background_tasks.add_task(
        _download_video_task,
        video_id=video_id,
        url=request.url,
        output_dir=output_dir,
    )

    return video


def _download_video_task(video_id: uuid.UUID, url: str, output_dir: str):
    """Background task to download video (runs in thread pool)."""
    import asyncio
    from app.models.database import SyncSessionLocal

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_service = VideoService()
    session = SyncSessionLocal()

    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            return

        # Run async download in new event loop (since we're in a thread)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                video_service.download_video(
                    url=url,
                    output_dir=output_dir,
                    video_id=str(video_id),
                )
            )
        finally:
            loop.close()

        # Update video record
        video.file_path = result["file_path"]
        video.file_size_bytes = result.get("file_size")
        video.duration_seconds = result.get("duration")
        video.width = result.get("width")
        video.height = result.get("height")
        video.fps = result.get("fps")
        video.title = result.get("title", video.title)
        video.description = result.get("description")
        video.thumbnail_path = result.get("thumbnail_path")
        video.status = VideoStatus.READY

        session.commit()
        logger.info(f"Video downloaded: {video_id}")

    except Exception as e:
        error_msg = str(e) if str(e) else repr(e)
        logger.error(f"Failed to download video {video_id}: {error_msg}")
        video = session.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = VideoStatus.ERROR
            video.error_message = error_msg or "Unknown download error"
            session.commit()
    finally:
        session.close()


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get video details by ID."""
    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return video


@router.delete("/{video_id}")
async def delete_video(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a video and its associated files."""
    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete files
    if video.file_path:
        file_path = Path(video.file_path)
        if file_path.exists():
            file_path.unlink()

    if video.thumbnail_path:
        thumb_path = Path(video.thumbnail_path)
        if thumb_path.exists():
            thumb_path.unlink()

    # Delete database record
    await session.delete(video)
    await session.commit()

    return {"message": "Video deleted", "video_id": str(video_id)}
