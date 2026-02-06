"""Pytest fixtures for ClipForge Pro backend tests."""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

# Set test environment before importing app modules
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["DATABASE_SYNC_URL"] = "sqlite:///:memory:"
os.environ["PEXELS_API_KEY"] = "test_pexels_key"
os.environ["PIXABAY_API_KEY"] = "test_pixabay_key"

from app.main import app
from app.models.database import Base, get_async_session


# Create test database engines
test_async_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
)

TestAsyncSessionLocal = async_sessionmaker(
    test_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestAsyncSessionLocal() as session:
        yield session

    async with test_async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(async_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database session override."""

    async def override_get_session():
        yield async_session

    app.dependency_overrides[get_async_session] = override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sync_client() -> Generator[TestClient, None, None]:
    """Create a synchronous test client."""
    with TestClient(app) as c:
        yield c


# ============== Sample Data Fixtures ==============


@pytest.fixture
def sample_transcript_segments() -> list[dict]:
    """Sample transcription segments for testing."""
    return [
        {
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Today we're going to talk about data analytics and technology trends.",
            "words": [
                {"word": "Today", "start": 0.0, "end": 0.3},
                {"word": "we're", "start": 0.3, "end": 0.5},
                {"word": "going", "start": 0.5, "end": 0.7},
                {"word": "to", "start": 0.7, "end": 0.8},
                {"word": "talk", "start": 0.8, "end": 1.0},
                {"word": "about", "start": 1.0, "end": 1.2},
                {"word": "data", "start": 1.2, "end": 1.5},
                {"word": "analytics", "start": 1.5, "end": 2.0},
                {"word": "and", "start": 2.0, "end": 2.2},
                {"word": "technology", "start": 2.2, "end": 2.8},
                {"word": "trends", "start": 2.8, "end": 3.2},
            ],
        },
        {
            "start_time": 5.5,
            "end_time": 10.0,
            "text": "The future of artificial intelligence is really exciting.",
            "words": [],
        },
        {
            "start_time": 10.5,
            "end_time": 15.0,
            "text": "Let me show you this chart with the growth statistics.",
            "words": [],
        },
        {
            "start_time": 15.5,
            "end_time": 20.0,
            "text": "Now let's switch to a completely different topic about cooking.",
            "words": [],
        },
        {
            "start_time": 20.5,
            "end_time": 25.0,
            "text": "Making pasta requires the right ingredients and technique.",
            "words": [],
        },
    ]


@pytest.fixture
def sample_visual_analysis() -> dict:
    """Sample visual analysis data for testing."""
    return {
        "timeline": [
            {"time": 0.0, "objects": [{"label": "person", "confidence": 0.95}]},
            {"time": 2.0, "objects": [{"label": "person", "confidence": 0.93}]},
            {"time": 4.0, "objects": []},  # Visual gap
            {"time": 6.0, "objects": []},  # Visual gap
            {"time": 8.0, "objects": [{"label": "person", "confidence": 0.91}]},
            {"time": 10.0, "objects": [{"label": "laptop", "confidence": 0.88}]},
        ]
    }


@pytest.fixture
def sample_broll_search_results() -> list[dict]:
    """Sample B-roll search results for testing."""
    return [
        {
            "provider": "pexels",
            "provider_id": "123456",
            "provider_url": "https://www.pexels.com/video/123456/",
            "download_url": "https://player.vimeo.com/external/123456.hd.mp4",
            "thumbnail_url": "https://images.pexels.com/videos/123456/thumbnail.jpg",
            "title": "Technology Background",
            "tags": ["technology", "abstract", "digital"],
            "duration": 15.0,
            "width": 1920,
            "height": 1080,
        },
        {
            "provider": "pixabay",
            "provider_id": "789012",
            "provider_url": "https://pixabay.com/videos/789012/",
            "download_url": "https://cdn.pixabay.com/video/789012.mp4",
            "thumbnail_url": "https://cdn.pixabay.com/video/789012_thumb.jpg",
            "title": "Data Visualization",
            "tags": ["data", "charts", "business"],
            "duration": 10.0,
            "width": 1920,
            "height": 1080,
        },
    ]


# ============== Mock Fixtures ==============


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Create a mock LLM service."""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value="technology background\ndata visualization\nabstract patterns")
    return mock


@pytest.fixture
def mock_aiohttp_session() -> MagicMock:
    """Create a mock aiohttp session for API testing."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "videos": [
            {
                "id": 123456,
                "duration": 15,
                "url": "https://www.pexels.com/video/123456/",
                "image": "https://images.pexels.com/videos/123456/thumbnail.jpg",
                "video_files": [
                    {"link": "https://player.vimeo.com/123456.hd.mp4", "width": 1920, "height": 1080}
                ],
                "video_pictures": [{"picture": "https://images.pexels.com/videos/123456/thumb.jpg"}],
            }
        ]
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock(return_value=mock_response)
    return mock_session


@pytest.fixture
def temp_video_path(tmp_path: Path) -> Path:
    """Create a temporary video file path."""
    video_path = tmp_path / "test_video.mp4"
    # Create a minimal valid file (in real tests, you'd use a real video)
    video_path.write_bytes(b"fake video content")
    return video_path


@pytest.fixture
def temp_broll_path(tmp_path: Path) -> Path:
    """Create a temporary B-roll video file path."""
    broll_path = tmp_path / "test_broll.mp4"
    broll_path.write_bytes(b"fake broll content")
    return broll_path
