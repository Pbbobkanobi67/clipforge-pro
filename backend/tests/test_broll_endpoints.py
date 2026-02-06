"""Tests for B-Roll API endpoints."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    AnalysisJob,
    AnalysisStatus,
    BRollAsset,
    BRollDetectionReason,
    BRollInsertMode,
    BRollSuggestion,
    ClipSuggestion,
    Transcription,
    TranscriptionSegment,
    Video,
    VideoStatus,
)


class TestBRollEndpoints:
    """Test cases for B-Roll API endpoints."""

    @pytest.fixture
    async def sample_video(self, async_session: AsyncSession) -> Video:
        """Create a sample video in the database."""
        video = Video(
            id=uuid.uuid4(),
            file_path="/test/video.mp4",
            status=VideoStatus.READY,
            duration_seconds=60.0,
            width=1920,
            height=1080,
        )
        async_session.add(video)
        await async_session.commit()
        await async_session.refresh(video)
        return video

    @pytest.fixture
    async def sample_job(
        self, async_session: AsyncSession, sample_video: Video
    ) -> AnalysisJob:
        """Create a sample analysis job."""
        job = AnalysisJob(
            id=uuid.uuid4(),
            video_id=sample_video.id,
            status=AnalysisStatus.COMPLETED,
        )
        async_session.add(job)
        await async_session.commit()
        await async_session.refresh(job)
        return job

    @pytest.fixture
    async def sample_clip(
        self, async_session: AsyncSession, sample_job: AnalysisJob
    ) -> ClipSuggestion:
        """Create a sample clip suggestion."""
        clip = ClipSuggestion(
            id=uuid.uuid4(),
            analysis_job_id=sample_job.id,
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            virality_score=85.0,
        )
        async_session.add(clip)
        await async_session.commit()
        await async_session.refresh(clip)
        return clip

    @pytest.fixture
    async def sample_transcription(
        self, async_session: AsyncSession, sample_job: AnalysisJob
    ) -> Transcription:
        """Create a sample transcription."""
        transcription = Transcription(
            id=uuid.uuid4(),
            analysis_job_id=sample_job.id,
            full_text="This is about data analytics and technology trends.",
            language="en",
            word_count=8,
        )
        async_session.add(transcription)
        await async_session.commit()

        # Add segments
        segment = TranscriptionSegment(
            id=uuid.uuid4(),
            transcription_id=transcription.id,
            start_time=0.0,
            end_time=5.0,
            text="This is about data analytics and technology trends.",
        )
        async_session.add(segment)
        await async_session.commit()

        await async_session.refresh(transcription)
        return transcription

    @pytest.fixture
    async def sample_broll_suggestion(
        self, async_session: AsyncSession, sample_clip: ClipSuggestion
    ) -> BRollSuggestion:
        """Create a sample B-roll suggestion."""
        suggestion = BRollSuggestion(
            id=uuid.uuid4(),
            clip_id=sample_clip.id,
            start_time=5.0,
            end_time=10.0,
            duration=5.0,
            detection_reason=BRollDetectionReason.ABSTRACT_CONCEPT,
            transcript_context="data analytics and technology",
            keywords=["data", "analytics", "technology"],
            search_queries=["technology background", "data visualization"],
            relevance_score=0.8,
            confidence=0.75,
            is_approved=False,
        )
        async_session.add(suggestion)
        await async_session.commit()
        await async_session.refresh(suggestion)
        return suggestion

    @pytest.fixture
    async def sample_broll_asset(
        self, async_session: AsyncSession
    ) -> BRollAsset:
        """Create a sample B-roll asset."""
        asset = BRollAsset(
            id=uuid.uuid4(),
            provider="pexels",
            provider_id="123456",
            provider_url="https://www.pexels.com/video/123456/",
            title="Technology Background",
            tags=["technology", "abstract"],
            duration=15.0,
            width=1920,
            height=1080,
            local_path="/storage/broll/test.mp4",
        )
        async_session.add(asset)
        await async_session.commit()
        await async_session.refresh(asset)
        return asset

    # ============== GET /broll/{clip_id} ==============

    @pytest.mark.asyncio
    async def test_get_broll_suggestions_success(
        self,
        client: AsyncClient,
        sample_clip: ClipSuggestion,
        sample_broll_suggestion: BRollSuggestion,
    ):
        """Test getting B-roll suggestions for a clip."""
        response = await client.get(f"/api/v1/broll/{sample_clip.id}")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == str(sample_broll_suggestion.id)
        assert data[0]["detection_reason"] == "abstract_concept"

    @pytest.mark.asyncio
    async def test_get_broll_suggestions_clip_not_found(self, client: AsyncClient):
        """Test getting B-roll for non-existent clip."""
        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/broll/{fake_id}")

        assert response.status_code == 404
        assert "Clip not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_broll_suggestions_approved_only(
        self,
        client: AsyncClient,
        async_session: AsyncSession,
        sample_clip: ClipSuggestion,
        sample_broll_suggestion: BRollSuggestion,
    ):
        """Test filtering for approved suggestions only."""
        # Create an approved suggestion
        approved = BRollSuggestion(
            id=uuid.uuid4(),
            clip_id=sample_clip.id,
            start_time=15.0,
            end_time=20.0,
            duration=5.0,
            detection_reason=BRollDetectionReason.TOPIC_CHANGE,
            relevance_score=0.9,
            confidence=0.8,
            is_approved=True,
        )
        async_session.add(approved)
        await async_session.commit()

        response = await client.get(
            f"/api/v1/broll/{sample_clip.id}",
            params={"approved_only": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["is_approved"] is True

    # ============== POST /broll/{clip_id}/generate ==============

    @pytest.mark.asyncio
    async def test_generate_broll_suggestions(
        self,
        client: AsyncClient,
        sample_clip: ClipSuggestion,
        sample_transcription: Transcription,
    ):
        """Test generating B-roll suggestions."""
        with patch(
            "app.api.endpoints.broll.get_broll_detection_service"
        ) as mock_get_service:
            mock_service = MagicMock()
            mock_service.detect_broll_opportunities = AsyncMock(
                return_value=[
                    {
                        "start_time": 5.0,
                        "end_time": 10.0,
                        "duration": 5.0,
                        "detection_reason": "abstract_concept",
                        "transcript_context": "data analytics",
                        "keywords": ["data"],
                        "search_queries": ["data visualization"],
                        "relevance_score": 0.8,
                        "confidence": 0.7,
                        "rank": 1,
                    }
                ]
            )
            mock_get_service.return_value = mock_service

            response = await client.post(
                f"/api/v1/broll/{sample_clip.id}/generate",
                json={"max_suggestions": 5, "use_llm": False},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["detection_reason"] == "abstract_concept"

    @pytest.mark.asyncio
    async def test_generate_broll_no_transcription(
        self, client: AsyncClient, sample_clip: ClipSuggestion
    ):
        """Test generating B-roll without transcription fails."""
        response = await client.post(
            f"/api/v1/broll/{sample_clip.id}/generate",
            json={"max_suggestions": 5},
        )

        assert response.status_code == 400
        assert "transcription" in response.json()["detail"].lower()

    # ============== GET /broll/search ==============

    @pytest.mark.asyncio
    async def test_search_stock_footage(self, client: AsyncClient):
        """Test searching for stock footage."""
        with patch(
            "app.api.endpoints.broll.get_broll_search_service"
        ) as mock_get_service:
            mock_service = MagicMock()
            mock_service.search = AsyncMock(
                return_value=[
                    {
                        "provider": "pexels",
                        "provider_id": "123",
                        "provider_url": "https://pexels.com/video/123",
                        "download_url": "https://example.com/video.mp4",
                        "thumbnail_url": "https://example.com/thumb.jpg",
                        "title": "Tech Video",
                        "tags": ["technology"],
                        "duration": 15.0,
                        "width": 1920,
                        "height": 1080,
                    }
                ]
            )
            mock_get_service.return_value = mock_service

            response = await client.get(
                "/api/v1/broll/search",
                params={"q": "technology"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["provider"] == "pexels"

    @pytest.mark.asyncio
    async def test_search_stock_footage_empty_query(self, client: AsyncClient):
        """Test search with empty query fails."""
        response = await client.get(
            "/api/v1/broll/search",
            params={"q": ""},
        )

        assert response.status_code == 422  # Validation error

    # ============== POST /broll/suggestion/{id}/select-asset ==============

    @pytest.mark.asyncio
    async def test_select_asset_for_suggestion(
        self,
        client: AsyncClient,
        sample_broll_suggestion: BRollSuggestion,
    ):
        """Test selecting an asset for a suggestion."""
        with patch(
            "app.api.endpoints.broll.get_broll_search_service"
        ) as mock_get_service:
            mock_service = MagicMock()
            mock_service.download_asset = AsyncMock(
                return_value="/storage/broll/downloaded.mp4"
            )
            mock_get_service.return_value = mock_service

            response = await client.post(
                f"/api/v1/broll/suggestion/{sample_broll_suggestion.id}/select-asset",
                json={
                    "provider": "pexels",
                    "provider_id": "123456",
                    "download_url": "https://example.com/video.mp4",
                    "title": "Tech Video",
                    "duration": 15.0,
                    "width": 1920,
                    "height": 1080,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["asset_id"] is not None

    @pytest.mark.asyncio
    async def test_select_asset_suggestion_not_found(self, client: AsyncClient):
        """Test selecting asset for non-existent suggestion."""
        fake_id = uuid.uuid4()
        response = await client.post(
            f"/api/v1/broll/suggestion/{fake_id}/select-asset",
            json={
                "provider": "pexels",
                "provider_id": "123",
                "download_url": "https://example.com/video.mp4",
            },
        )

        assert response.status_code == 404

    # ============== POST /broll/suggestion/{id}/approve ==============

    @pytest.mark.asyncio
    async def test_approve_suggestion_without_asset(
        self,
        client: AsyncClient,
        sample_broll_suggestion: BRollSuggestion,
    ):
        """Test approving suggestion without asset fails."""
        response = await client.post(
            f"/api/v1/broll/suggestion/{sample_broll_suggestion.id}/approve",
            json={"is_approved": True},
        )

        assert response.status_code == 400
        assert "asset" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_approve_suggestion_with_asset(
        self,
        client: AsyncClient,
        async_session: AsyncSession,
        sample_broll_suggestion: BRollSuggestion,
        sample_broll_asset: BRollAsset,
    ):
        """Test approving suggestion with asset."""
        # Associate asset with suggestion
        sample_broll_suggestion.asset_id = sample_broll_asset.id
        await async_session.commit()

        response = await client.post(
            f"/api/v1/broll/suggestion/{sample_broll_suggestion.id}/approve",
            json={"is_approved": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_approved"] is True

    @pytest.mark.asyncio
    async def test_unapprove_suggestion(
        self,
        client: AsyncClient,
        async_session: AsyncSession,
        sample_broll_suggestion: BRollSuggestion,
        sample_broll_asset: BRollAsset,
    ):
        """Test unapproving a suggestion."""
        sample_broll_suggestion.asset_id = sample_broll_asset.id
        sample_broll_suggestion.is_approved = True
        await async_session.commit()

        response = await client.post(
            f"/api/v1/broll/suggestion/{sample_broll_suggestion.id}/approve",
            json={"is_approved": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_approved"] is False

    # ============== DELETE /broll/suggestion/{id} ==============

    @pytest.mark.asyncio
    async def test_delete_suggestion(
        self,
        client: AsyncClient,
        sample_broll_suggestion: BRollSuggestion,
    ):
        """Test deleting a suggestion."""
        response = await client.delete(
            f"/api/v1/broll/suggestion/{sample_broll_suggestion.id}"
        )

        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_suggestion_not_found(self, client: AsyncClient):
        """Test deleting non-existent suggestion."""
        fake_id = uuid.uuid4()
        response = await client.delete(f"/api/v1/broll/suggestion/{fake_id}")

        assert response.status_code == 404


class TestBRollExport:
    """Test cases for B-Roll export endpoint."""

    @pytest.fixture
    async def approved_setup(
        self,
        async_session: AsyncSession,
    ) -> dict:
        """Set up approved B-roll for export testing."""
        # Create video
        video = Video(
            id=uuid.uuid4(),
            file_path="/test/video.mp4",
            status=VideoStatus.READY,
            duration_seconds=60.0,
        )
        async_session.add(video)

        # Create job
        job = AnalysisJob(
            id=uuid.uuid4(),
            video_id=video.id,
            status=AnalysisStatus.COMPLETED,
        )
        async_session.add(job)

        # Create clip
        clip = ClipSuggestion(
            id=uuid.uuid4(),
            analysis_job_id=job.id,
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            virality_score=85.0,
        )
        async_session.add(clip)

        # Create asset
        asset = BRollAsset(
            id=uuid.uuid4(),
            provider="pexels",
            provider_id="123",
            local_path="/storage/broll/test.mp4",
            duration=10.0,
        )
        async_session.add(asset)

        # Create approved suggestion
        suggestion = BRollSuggestion(
            id=uuid.uuid4(),
            clip_id=clip.id,
            start_time=5.0,
            end_time=10.0,
            duration=5.0,
            detection_reason=BRollDetectionReason.ABSTRACT_CONCEPT,
            asset_id=asset.id,
            is_approved=True,
        )
        async_session.add(suggestion)

        await async_session.commit()

        return {
            "video": video,
            "job": job,
            "clip": clip,
            "asset": asset,
            "suggestion": suggestion,
        }

    @pytest.mark.asyncio
    async def test_export_with_broll_no_approved(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        """Test export fails when no approved suggestions."""
        # Create clip without approved suggestions
        video = Video(
            id=uuid.uuid4(),
            file_path="/test/video.mp4",
            status=VideoStatus.READY,
        )
        async_session.add(video)

        job = AnalysisJob(
            id=uuid.uuid4(),
            video_id=video.id,
            status=AnalysisStatus.COMPLETED,
        )
        async_session.add(job)

        clip = ClipSuggestion(
            id=uuid.uuid4(),
            analysis_job_id=job.id,
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            virality_score=85.0,
        )
        async_session.add(clip)
        await async_session.commit()

        response = await client.post(
            f"/api/v1/broll/{clip.id}/export",
            json={},
        )

        assert response.status_code == 400
        assert "approved" in response.json()["detail"].lower()
