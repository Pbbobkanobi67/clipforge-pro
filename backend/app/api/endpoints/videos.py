"""Video upload and download endpoints."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import asyncio
import base64
import mimetypes
import subprocess

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import Video, VideoStatus, get_async_session
from app.models.schemas import VideoCreate, VideoDownloadRequest, VideoResponse
from app.services.video_service import VideoService

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.get("/", response_model=list[VideoResponse])
async def list_videos(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_async_session),
):
    """List all uploaded videos, newest first, with analysis status."""
    result = await session.execute(
        select(Video).order_by(Video.created_at.desc()).offset(offset).limit(limit)
    )
    videos = result.scalars().all()
    return videos


@router.get("/library")
async def get_video_library(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get video library with analysis status and clip counts.

    Returns enriched video data for the library view.
    """
    from app.models.database import AnalysisJob, AnalysisStatus, ClipSuggestion

    result = await session.execute(
        select(Video).order_by(Video.created_at.desc()).offset(offset).limit(limit)
    )
    videos = result.scalars().all()

    library = []
    for video in videos:
        # Get latest analysis job
        job_result = await session.execute(
            select(AnalysisJob)
            .filter(AnalysisJob.video_id == video.id)
            .order_by(AnalysisJob.created_at.desc())
            .limit(1)
        )
        job = job_result.scalar_one_or_none()

        # Get clip count if analysis is complete
        clip_count = 0
        if job:
            clip_result = await session.execute(
                select(ClipSuggestion)
                .filter(ClipSuggestion.analysis_job_id == job.id)
            )
            clip_count = len(clip_result.scalars().all())

        library.append({
            "id": str(video.id),
            "title": video.title or video.original_filename or "Untitled",
            "original_filename": video.original_filename,
            "source_url": video.source_url,
            "created_at": video.created_at.isoformat() if video.created_at else None,
            "duration_seconds": video.duration_seconds,
            "width": video.width,
            "height": video.height,
            "status": video.status.value if video.status else "unknown",
            "analysis_status": job.status.value if job else None,
            "analysis_progress": job.progress if job else None,
            "clip_count": clip_count,
            "thumbnail_path": video.thumbnail_path,
        })

    return {"videos": library, "total": len(library), "offset": offset}


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
    import subprocess
    from app.models.database import SyncSessionLocal
    from app.services.video_service import YTDLP_CMD

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    video_service = VideoService()
    session = SyncSessionLocal()

    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            return

        logger.info(f"Starting download for video {video_id}: {url}")

        # Step 1: Quick metadata fetch (no download) to update title immediately
        try:
            meta_cmd = [
                YTDLP_CMD,
                "--no-download",
                "--print-json",
                "--no-warnings",
                "--no-playlist",
                url,
            ]
            meta_result = subprocess.run(meta_cmd, capture_output=True, timeout=30)
            if meta_result.returncode == 0:
                import json
                meta_info = json.loads(meta_result.stdout.decode())
                video.title = meta_info.get("title", video.title)
                video.duration_seconds = meta_info.get("duration")
                video.width = meta_info.get("width")
                video.height = meta_info.get("height")
                video.fps = meta_info.get("fps")
                video.description = meta_info.get("description")
                session.commit()
                logger.info(f"Metadata fetched for {video_id}: {video.title}")
        except Exception as e:
            logger.warning(f"Metadata fetch failed for {video_id}, continuing with download: {e}")

        # Step 2: Download the actual file
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

        # Update video record with final info
        video.file_path = result["file_path"]
        video.file_size_bytes = result.get("file_size")
        video.duration_seconds = result.get("duration") or video.duration_seconds
        video.width = result.get("width") or video.width
        video.height = result.get("height") or video.height
        video.fps = result.get("fps") or video.fps
        video.title = result.get("title") or video.title
        video.description = result.get("description") or video.description
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


@router.get("/{video_id}/stream")
async def stream_video(
    video_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
):
    """Stream video file with Range request support for seeking."""
    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    file_path = Path(video.file_path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / video.file_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = file_path.stat().st_size
    content_type = mimetypes.guess_type(str(file_path))[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        # Parse range: "bytes=START-END"
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        async def ranged_file():
            async with aiofiles.open(file_path, "rb") as f:
                await f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(1024 * 1024, remaining)
                    data = await f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            ranged_file(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )
    else:
        async def full_file():
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(1024 * 1024):
                    yield chunk

        return StreamingResponse(
            full_file(),
            media_type=content_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
            },
        )


@router.get("/{video_id}/thumbnails")
async def get_timeline_thumbnails(
    video_id: uuid.UUID,
    count: int = Query(default=20, ge=1, le=60),
    session: AsyncSession = Depends(get_async_session),
):
    """Generate evenly-spaced frame thumbnails for the timeline strip."""
    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if not video.duration_seconds:
        raise HTTPException(status_code=400, detail="Video duration unknown")

    video_path = Path(video.file_path)
    if not video_path.is_absolute():
        video_path = Path.cwd() / video.file_path

    # Cache directory for this video's thumbnails
    cache_dir = Path(settings.cache_path) / "timeline_thumbs" / str(video_id)
    if not cache_dir.is_absolute():
        cache_dir = Path.cwd() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    duration = video.duration_seconds
    thumbnails = []

    for i in range(count):
        timestamp = duration * i / count
        thumb_path = cache_dir / f"thumb_{i:03d}_{count}.jpg"

        # Generate if not cached
        if not thumb_path.exists():
            try:
                cmd = [
                    "ffmpeg",
                    "-ss", str(timestamp),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", "scale=160:-1",
                    "-q:v", "5",
                    "-update", "1",
                    "-y",
                    str(thumb_path),
                ]
                result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
                if result.returncode != 0:
                    logger.warning(f"ffmpeg failed for thumb at {timestamp}s: {result.stderr.decode()[:200]}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to extract thumbnail at {timestamp}s: {e}")
                continue

        if thumb_path.exists():
            data = thumb_path.read_bytes()
            thumbnails.append({
                "timestamp": round(timestamp, 2),
                "data": base64.b64encode(data).decode(),
            })

    return {"thumbnails": thumbnails, "duration": duration, "count": len(thumbnails)}


@router.get("/{video_id}/thumbnail")
async def get_video_thumbnail(
    video_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Serve video thumbnail image."""
    from fastapi.responses import FileResponse

    result = await session.execute(select(Video).filter(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video or not video.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    thumb_path = Path(video.thumbnail_path)
    if not thumb_path.is_absolute():
        thumb_path = Path.cwd() / video.thumbnail_path
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file not found on disk")

    content_type = mimetypes.guess_type(str(thumb_path))[0] or "image/webp"
    return FileResponse(str(thumb_path), media_type=content_type)


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
