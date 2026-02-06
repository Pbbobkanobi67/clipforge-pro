"""Main API router combining all endpoint routers."""

from fastapi import APIRouter

from app.api.endpoints import (
    videos,
    analysis,
    clips,
    captions,
    brands,
    reframe,
    thumbnails,
    editor,
    export,
    broll,
)

api_router = APIRouter()

# Core endpoints
api_router.include_router(videos.router, prefix="/videos", tags=["Videos"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(clips.router, prefix="/clips", tags=["Clips"])

# OpusClip Pro features
api_router.include_router(captions.router, prefix="/captions", tags=["Captions"])
api_router.include_router(brands.router, prefix="/brands", tags=["Brands"])
api_router.include_router(reframe.router, prefix="/reframe", tags=["AI Reframe"])
api_router.include_router(thumbnails.router, prefix="/thumbnails", tags=["Thumbnails"])
api_router.include_router(editor.router, prefix="/editor", tags=["Timeline Editor"])
api_router.include_router(export.router, prefix="/export", tags=["Export"])
api_router.include_router(broll.router, prefix="/broll", tags=["AI B-Roll"])
