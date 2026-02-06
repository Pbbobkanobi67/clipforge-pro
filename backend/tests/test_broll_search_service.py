"""Tests for BRollSearchService."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.broll_search_service import (
    BRollSearchService,
    get_broll_search_service,
)


class TestBRollSearchService:
    """Test cases for BRollSearchService."""

    @pytest.fixture
    def service(self) -> BRollSearchService:
        """Create a BRollSearchService instance."""
        return BRollSearchService()

    @pytest.fixture
    def mock_pexels_response(self) -> dict:
        """Mock Pexels API response."""
        return {
            "videos": [
                {
                    "id": 123456,
                    "duration": 15,
                    "url": "https://www.pexels.com/video/123456/",
                    "image": "https://images.pexels.com/videos/123456/thumbnail.jpg",
                    "video_files": [
                        {"link": "https://player.vimeo.com/123456_720p.mp4", "width": 1280, "height": 720},
                        {"link": "https://player.vimeo.com/123456_1080p.mp4", "width": 1920, "height": 1080},
                    ],
                    "video_pictures": [
                        {"picture": "https://images.pexels.com/videos/123456/thumb.jpg"}
                    ],
                },
                {
                    "id": 789012,
                    "duration": 8,
                    "url": "https://www.pexels.com/video/789012/",
                    "image": "https://images.pexels.com/videos/789012/thumbnail.jpg",
                    "video_files": [
                        {"link": "https://player.vimeo.com/789012_1080p.mp4", "width": 1920, "height": 1080},
                    ],
                    "video_pictures": [],
                },
            ]
        }

    @pytest.fixture
    def mock_pixabay_response(self) -> dict:
        """Mock Pixabay API response."""
        return {
            "hits": [
                {
                    "id": 111222,
                    "duration": 12,
                    "pageURL": "https://pixabay.com/videos/111222/",
                    "tags": "technology, digital, abstract",
                    "picture_id": "abc123",
                    "videos": {
                        "large": {"url": "https://cdn.pixabay.com/111222_large.mp4", "width": 1920, "height": 1080},
                        "medium": {"url": "https://cdn.pixabay.com/111222_medium.mp4", "width": 1280, "height": 720},
                    },
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_search_pexels(self, service: BRollSearchService, mock_pexels_response: dict):
        """Test Pexels search functionality."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_pexels_response)

        with patch.object(service, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_get_session.return_value = mock_session

            results = await service._search_pexels(
                query="technology",
                orientation="landscape",
                limit=10,
                min_duration=3.0,
                max_duration=30.0,
            )

        assert len(results) == 2
        assert results[0]["provider"] == "pexels"
        assert results[0]["provider_id"] == "123456"
        assert results[0]["duration"] == 15

    @pytest.mark.asyncio
    async def test_search_pixabay(self, service: BRollSearchService, mock_pixabay_response: dict):
        """Test Pixabay search functionality."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_pixabay_response)

        with patch.object(service, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_get_session.return_value = mock_session

            results = await service._search_pixabay(
                query="technology",
                orientation="landscape",
                limit=10,
                min_duration=3.0,
                max_duration=30.0,
            )

        assert len(results) == 1
        assert results[0]["provider"] == "pixabay"
        assert results[0]["provider_id"] == "111222"
        assert "technology" in results[0]["tags"]

    @pytest.mark.asyncio
    async def test_search_filters_by_duration(self, service: BRollSearchService, mock_pexels_response: dict):
        """Test that search filters by duration."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_pexels_response)

        with patch.object(service, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_get_session.return_value = mock_session

            # Set min duration to filter out shorter videos
            results = await service._search_pexels(
                query="technology",
                orientation="landscape",
                limit=10,
                min_duration=10.0,  # Filter out 8s video
                max_duration=30.0,
            )

        # Only the 15s video should remain
        assert len(results) == 1
        assert results[0]["duration"] == 15

    @pytest.mark.asyncio
    async def test_search_handles_api_error(self, service: BRollSearchService):
        """Test handling of API errors."""
        mock_response = AsyncMock()
        mock_response.status = 500

        with patch.object(service, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_get_session.return_value = mock_session

            results = await service._search_pexels(
                query="technology",
                orientation="landscape",
                limit=10,
                min_duration=3.0,
                max_duration=30.0,
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_search_multiple_queries(
        self,
        service: BRollSearchService,
        mock_pexels_response: dict,
        mock_pixabay_response: dict,
    ):
        """Test searching with multiple queries."""
        # Mock both providers
        with patch.object(service, "_search_pexels", new_callable=AsyncMock) as mock_pexels, \
             patch.object(service, "_search_pixabay", new_callable=AsyncMock) as mock_pixabay:

            mock_pexels.return_value = [
                {"provider": "pexels", "provider_id": "123", "duration": 10}
            ]
            mock_pixabay.return_value = [
                {"provider": "pixabay", "provider_id": "456", "duration": 12}
            ]

            results = await service.search(
                queries=["technology", "data"],
                providers=["pexels", "pixabay"],
            )

        # Should have results from both providers for both queries
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_deduplicates_results(self, service: BRollSearchService):
        """Test that duplicate results are removed."""
        with patch.object(service, "_search_pexels", new_callable=AsyncMock) as mock_pexels:
            # Return duplicate results from different queries
            mock_pexels.return_value = [
                {"provider": "pexels", "provider_id": "123", "duration": 10},
                {"provider": "pexels", "provider_id": "123", "duration": 10},  # Duplicate
            ]

            results = await service.search(
                queries=["technology"],
                providers=["pexels"],
            )

        # Duplicates should be removed
        unique_ids = set(r["provider_id"] for r in results)
        assert len(results) == len(unique_ids)

    def test_get_best_video_file_prefers_1080p(self, service: BRollSearchService):
        """Test that 1080p video is preferred."""
        video_files = [
            {"link": "https://example.com/480p.mp4", "width": 854, "height": 480},
            {"link": "https://example.com/720p.mp4", "width": 1280, "height": 720},
            {"link": "https://example.com/1080p.mp4", "width": 1920, "height": 1080},
        ]

        best = service._get_best_video_file(video_files)

        assert best["height"] == 1080

    def test_get_best_video_file_falls_back_to_720p(self, service: BRollSearchService):
        """Test fallback to 720p when 1080p not available."""
        video_files = [
            {"link": "https://example.com/480p.mp4", "width": 854, "height": 480},
            {"link": "https://example.com/720p.mp4", "width": 1280, "height": 720},
        ]

        best = service._get_best_video_file(video_files)

        assert best["height"] == 720

    def test_get_best_video_file_empty_list(self, service: BRollSearchService):
        """Test handling of empty video files list."""
        best = service._get_best_video_file([])

        assert best is None

    def test_get_best_pixabay_file(self, service: BRollSearchService):
        """Test Pixabay video file selection."""
        videos = {
            "large": {"url": "https://example.com/large.mp4", "width": 1920, "height": 1080},
            "medium": {"url": "https://example.com/medium.mp4", "width": 1280, "height": 720},
            "small": {"url": "https://example.com/small.mp4", "width": 640, "height": 360},
        }

        best = service._get_best_pixabay_file(videos)

        assert "large" in best["url"]

    @pytest.mark.asyncio
    async def test_download_asset(self, service: BRollSearchService, tmp_path: Path):
        """Test downloading an asset."""
        mock_content = b"fake video content for testing"

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=mock_content)

        with patch.object(service, "_get_session") as mock_get_session, \
             patch("app.services.broll_search_service.settings") as mock_settings:

            mock_settings.broll_storage_path = tmp_path

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            ))
            mock_get_session.return_value = mock_session

            local_path = await service.download_asset(
                provider="pexels",
                provider_id="123456",
                download_url="https://example.com/video.mp4",
                title="Test Video",
            )

        assert Path(local_path).exists()
        assert Path(local_path).read_bytes() == mock_content

    @pytest.mark.asyncio
    async def test_download_asset_caches(self, service: BRollSearchService, tmp_path: Path):
        """Test that downloaded assets are cached."""
        import hashlib

        mock_content = b"cached video content"
        download_url = "https://example.com/video.mp4"

        with patch("app.services.broll_search_service.settings") as mock_settings:
            mock_settings.broll_storage_path = tmp_path

            # Calculate the same hash the service uses
            url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]

            # Create file manually to simulate cache with correct filename
            cached_file = tmp_path / f"pexels_123456_{url_hash}_Test_Video.mp4"
            cached_file.write_bytes(mock_content)

            # Mock session that should NOT be called if cache works
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=b"new downloaded content")

            with patch.object(service, "_get_session") as mock_get_session:
                mock_session = MagicMock()
                mock_session.get = MagicMock(return_value=AsyncMock(
                    __aenter__=AsyncMock(return_value=mock_response),
                    __aexit__=AsyncMock(return_value=None),
                ))
                mock_get_session.return_value = mock_session

                local_path = await service.download_asset(
                    provider="pexels",
                    provider_id="123456",
                    download_url=download_url,
                    title="Test Video",
                )

        # Should return cached file, not download new one
        assert Path(local_path).read_bytes() == mock_content
        # Session.get should not have been called since file was cached
        mock_session.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_session(self, service: BRollSearchService):
        """Test closing the aiohttp session."""
        # Create a session
        service._session = MagicMock()
        service._session.closed = False
        service._session.close = AsyncMock()

        await service.close()

        service._session.close.assert_called_once()


class TestServiceSingleton:
    """Test service singleton pattern."""

    def test_get_broll_search_service(self):
        """Test singleton service getter."""
        service1 = get_broll_search_service()
        service2 = get_broll_search_service()

        assert service1 is service2
        assert isinstance(service1, BRollSearchService)
