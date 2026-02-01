"""Timeline editor project endpoints."""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    EditorProject,
    AnalysisJob,
    ClipSuggestion,
    Video,
    BrandTemplate,
    get_async_session,
)
from app.models.schemas import (
    EditorProjectCreate,
    EditorProjectUpdate,
    EditorProjectResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


@router.post("/projects", response_model=EditorProjectResponse)
async def create_editor_project(
    data: EditorProjectCreate,
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new editor project."""
    # Verify analysis job exists
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == data.analysis_job_id)
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")

    # Verify brand template if provided
    if data.brand_template_id:
        brand_result = await session.execute(
            select(BrandTemplate).filter(BrandTemplate.id == data.brand_template_id)
        )
        if not brand_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Brand template not found")

    # Create project with empty timeline
    project = EditorProject(
        analysis_job_id=data.analysis_job_id,
        name=data.name,
        timeline_data={"tracks": [], "duration": 0},
        export_resolution=data.export_resolution,
        export_format=data.export_format,
        export_fps=data.export_fps,
        brand_template_id=data.brand_template_id,
    )

    session.add(project)
    await session.commit()
    await session.refresh(project)

    logger.info(f"Created editor project: {project.id} ({project.name})")
    return EditorProjectResponse.model_validate(project)


@router.get("/projects", response_model=list[EditorProjectResponse])
async def list_editor_projects(
    analysis_job_id: uuid.UUID = None,
    session: AsyncSession = Depends(get_async_session),
):
    """List all editor projects, optionally filtered by analysis job."""
    query = select(EditorProject).order_by(EditorProject.updated_at.desc())

    if analysis_job_id:
        query = query.filter(EditorProject.analysis_job_id == analysis_job_id)

    result = await session.execute(query)
    projects = result.scalars().all()
    return [EditorProjectResponse.model_validate(p) for p in projects]


@router.get("/projects/{project_id}", response_model=EditorProjectResponse)
async def get_editor_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific editor project."""
    result = await session.execute(
        select(EditorProject).filter(EditorProject.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Editor project not found")

    return EditorProjectResponse.model_validate(project)


@router.put("/projects/{project_id}", response_model=EditorProjectResponse)
async def update_editor_project(
    project_id: uuid.UUID,
    data: EditorProjectUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    """Update an editor project (including timeline data)."""
    result = await session.execute(
        select(EditorProject).filter(EditorProject.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Editor project not found")

    update_data = data.model_dump(exclude_unset=True)

    # Helper to convert UUIDs to strings recursively for JSON storage
    def convert_uuids(obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_uuids(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_uuids(item) for item in obj]
        return obj

    # Convert timeline_data Pydantic model to dict if present
    if "timeline_data" in update_data and update_data["timeline_data"]:
        if hasattr(update_data["timeline_data"], "model_dump"):
            update_data["timeline_data"] = update_data["timeline_data"].model_dump()
        # Convert UUIDs to strings for JSON serialization
        update_data["timeline_data"] = convert_uuids(update_data["timeline_data"])

    # Validate brand template if being updated
    if update_data.get("brand_template_id"):
        brand_result = await session.execute(
            select(BrandTemplate).filter(BrandTemplate.id == update_data["brand_template_id"])
        )
        if not brand_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Brand template not found")

    # Reset rendered status when timeline changes
    if "timeline_data" in update_data:
        project.rendered = False
        project.render_path = None
        project.render_progress = 0

    for key, value in update_data.items():
        setattr(project, key, value)

    await session.commit()
    await session.refresh(project)

    logger.info(f"Updated editor project: {project_id}")
    return EditorProjectResponse.model_validate(project)


@router.delete("/projects/{project_id}")
async def delete_editor_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete an editor project."""
    result = await session.execute(
        select(EditorProject).filter(EditorProject.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Editor project not found")

    # Delete rendered file if exists
    if project.render_path:
        Path(project.render_path).unlink(missing_ok=True)

    await session.delete(project)
    await session.commit()

    logger.info(f"Deleted editor project: {project_id}")
    return {"status": "deleted", "id": str(project_id)}


@router.post("/projects/{project_id}/render")
async def render_editor_project(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Render the editor project timeline to a video file.
    """
    # Get project
    result = await session.execute(
        select(EditorProject).filter(EditorProject.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Editor project not found")

    # Check if already rendered
    if project.rendered and project.render_path and Path(project.render_path).exists():
        return {
            "status": "completed",
            "render_path": project.render_path,
        }

    # Validate timeline
    from app.services.editor_service import get_editor_service
    editor_service = get_editor_service()

    errors = editor_service.validate_timeline(project.timeline_data)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Get source video
    job_result = await session.execute(
        select(AnalysisJob).filter(AnalysisJob.id == project.analysis_job_id)
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

    # Get clip paths
    clip_ids = []
    for track in project.timeline_data.get("tracks", []):
        for clip in track.get("clips", []):
            clip_ids.append(clip.get("clip_id"))

    # For now, all clips use the same source video
    # In a real implementation, clips might have been exported separately
    clip_paths = {}
    for clip_id in clip_ids:
        # Check if clip has been exported
        clip_result = await session.execute(
            select(ClipSuggestion).filter(ClipSuggestion.id == clip_id)
        )
        clip = clip_result.scalar_one_or_none()
        if clip and clip.export_path and Path(clip.export_path).exists():
            clip_paths[str(clip_id)] = clip.export_path
        else:
            # Use source video with timing info
            clip_paths[str(clip_id)] = video.file_path

    # Get branding if configured
    logo_path = None
    logo_position = "top_right"
    logo_size = 100
    logo_margin = 20

    if project.brand_template_id:
        brand_result = await session.execute(
            select(BrandTemplate).filter(BrandTemplate.id == project.brand_template_id)
        )
        brand = brand_result.scalar_one_or_none()
        if brand and brand.logo_path:
            logo_path = brand.logo_path
            logo_position = brand.logo_position.value if brand.logo_position else "top_right"
            logo_size = brand.logo_size
            logo_margin = brand.logo_margin

    # Prepare output path
    render_path = str(settings.storage_path / "renders" / f"project_{project_id}.{project.export_format}")
    Path(render_path).parent.mkdir(parents=True, exist_ok=True)

    # Render
    try:
        project.render_progress = 10
        await session.commit()

        await editor_service.render_timeline(
            timeline_data=project.timeline_data,
            clip_paths=clip_paths,
            output_path=render_path,
            resolution=project.export_resolution,
            fps=project.export_fps,
            format=project.export_format,
            logo_path=logo_path,
            logo_position=logo_position,
            logo_size=logo_size,
            logo_margin=logo_margin,
        )

        project.rendered = True
        project.render_path = render_path
        project.render_progress = 100
        await session.commit()

        logger.info(f"Rendered editor project: {project_id}")
        return {
            "status": "completed",
            "render_path": render_path,
        }

    except Exception as e:
        logger.error(f"Render failed for project {project_id}: {e}")
        project.render_progress = 0
        await session.commit()
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")


@router.get("/projects/{project_id}/download")
async def download_rendered_project(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Download the rendered project video."""
    result = await session.execute(
        select(EditorProject).filter(EditorProject.id == project_id)
    )
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Editor project not found")

    if not project.rendered or not project.render_path:
        raise HTTPException(status_code=400, detail="Project not yet rendered")

    render_path = Path(project.render_path)
    if not render_path.exists():
        raise HTTPException(status_code=404, detail="Render file not found")

    return FileResponse(
        path=render_path,
        filename=f"{project.name}.{project.export_format}",
        media_type=f"video/{project.export_format}",
    )
