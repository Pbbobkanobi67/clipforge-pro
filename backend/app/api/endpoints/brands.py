"""Brand template CRUD and logo upload endpoints."""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.database import (
    BrandTemplate,
    CaptionStyle,
    LogoPosition,
    get_async_session,
)
from app.models.schemas import (
    BrandTemplateCreate,
    BrandTemplateUpdate,
    BrandTemplateResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# Create brand assets directory
BRAND_ASSETS_DIR = settings.storage_path / "brand_assets"
BRAND_ASSETS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", response_model=BrandTemplateResponse)
async def create_brand_template(
    data: BrandTemplateCreate,
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new brand template."""
    # Validate logo position
    try:
        logo_position = LogoPosition(data.logo_position)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid logo_position. Must be one of: {[e.value for e in LogoPosition]}"
        )

    # Validate caption style if provided
    if data.caption_style_id:
        style_result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.id == data.caption_style_id)
        )
        if not style_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Caption style not found")

    # If this is set as default, unset other defaults
    if data.is_default:
        result = await session.execute(
            select(BrandTemplate).filter(BrandTemplate.is_default == True)
        )
        for brand in result.scalars().all():
            brand.is_default = False

    # Create template
    template = BrandTemplate(
        name=data.name,
        is_default=data.is_default,
        logo_position=logo_position,
        logo_size=data.logo_size,
        logo_opacity=data.logo_opacity,
        logo_margin=data.logo_margin,
        primary_color=data.primary_color,
        secondary_color=data.secondary_color,
        accent_color=data.accent_color,
        caption_style_id=data.caption_style_id,
        outro_enabled=data.outro_enabled,
        outro_duration=data.outro_duration,
        outro_text=data.outro_text,
        outro_cta_text=data.outro_cta_text,
        outro_background_color=data.outro_background_color,
    )

    session.add(template)
    await session.commit()
    await session.refresh(template)

    logger.info(f"Created brand template: {template.id} ({template.name})")
    return BrandTemplateResponse.model_validate(template)


@router.get("", response_model=list[BrandTemplateResponse])
async def list_brand_templates(
    session: AsyncSession = Depends(get_async_session),
):
    """List all brand templates."""
    result = await session.execute(
        select(BrandTemplate).order_by(BrandTemplate.created_at.desc())
    )
    templates = result.scalars().all()
    return [BrandTemplateResponse.model_validate(t) for t in templates]


@router.get("/{template_id}", response_model=BrandTemplateResponse)
async def get_brand_template(
    template_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Get a specific brand template."""
    result = await session.execute(
        select(BrandTemplate).filter(BrandTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Brand template not found")

    return BrandTemplateResponse.model_validate(template)


@router.put("/{template_id}", response_model=BrandTemplateResponse)
async def update_brand_template(
    template_id: uuid.UUID,
    data: BrandTemplateUpdate,
    session: AsyncSession = Depends(get_async_session),
):
    """Update a brand template."""
    result = await session.execute(
        select(BrandTemplate).filter(BrandTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Brand template not found")

    update_data = data.model_dump(exclude_unset=True)

    # Validate logo position if present
    if "logo_position" in update_data:
        try:
            update_data["logo_position"] = LogoPosition(update_data["logo_position"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid logo_position. Must be one of: {[e.value for e in LogoPosition]}"
            )

    # Validate caption style if provided
    if update_data.get("caption_style_id"):
        style_result = await session.execute(
            select(CaptionStyle).filter(CaptionStyle.id == update_data["caption_style_id"])
        )
        if not style_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Caption style not found")

    # Handle default flag
    if update_data.get("is_default"):
        other_defaults = await session.execute(
            select(BrandTemplate).filter(
                BrandTemplate.is_default == True,
                BrandTemplate.id != template_id
            )
        )
        for other in other_defaults.scalars().all():
            other.is_default = False

    for key, value in update_data.items():
        setattr(template, key, value)

    await session.commit()
    await session.refresh(template)

    logger.info(f"Updated brand template: {template_id}")
    return BrandTemplateResponse.model_validate(template)


@router.delete("/{template_id}")
async def delete_brand_template(
    template_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete a brand template."""
    result = await session.execute(
        select(BrandTemplate).filter(BrandTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Brand template not found")

    # Delete associated logo file if exists
    if template.logo_path:
        logo_path = Path(template.logo_path)
        if logo_path.exists():
            logo_path.unlink()

    await session.delete(template)
    await session.commit()

    logger.info(f"Deleted brand template: {template_id}")
    return {"status": "deleted", "id": str(template_id)}


@router.post("/{template_id}/logo", response_model=BrandTemplateResponse)
async def upload_logo(
    template_id: uuid.UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
):
    """Upload a logo image for a brand template."""
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/webp", "image/svg+xml"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )

    # Get template
    result = await session.execute(
        select(BrandTemplate).filter(BrandTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Brand template not found")

    # Delete old logo if exists
    if template.logo_path:
        old_path = Path(template.logo_path)
        if old_path.exists():
            old_path.unlink()

    # Save new logo
    file_ext = Path(file.filename).suffix if file.filename else ".png"
    logo_filename = f"logo_{template_id}{file_ext}"
    logo_path = BRAND_ASSETS_DIR / logo_filename

    with open(logo_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update template
    template.logo_path = str(logo_path)
    await session.commit()
    await session.refresh(template)

    logger.info(f"Uploaded logo for brand template: {template_id}")
    return BrandTemplateResponse.model_validate(template)


@router.delete("/{template_id}/logo", response_model=BrandTemplateResponse)
async def delete_logo(
    template_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
):
    """Delete the logo from a brand template."""
    result = await session.execute(
        select(BrandTemplate).filter(BrandTemplate.id == template_id)
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(status_code=404, detail="Brand template not found")

    # Delete logo file
    if template.logo_path:
        logo_path = Path(template.logo_path)
        if logo_path.exists():
            logo_path.unlink()

    template.logo_path = None
    await session.commit()
    await session.refresh(template)

    logger.info(f"Deleted logo from brand template: {template_id}")
    return BrandTemplateResponse.model_validate(template)
