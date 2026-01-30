"""Main API router combining all endpoint routers."""

from fastapi import APIRouter

from app.api.endpoints import videos, analysis, clips

api_router = APIRouter()

api_router.include_router(videos.router, prefix="/videos", tags=["Videos"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(clips.router, prefix="/clips", tags=["Clips"])
