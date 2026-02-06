"""B-Roll search service for finding stock footage from various providers."""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import aiohttp

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BRollSearchService:
    """Service for searching and downloading stock footage from various providers."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def search(
        self,
        queries: list[str],
        providers: list[str] = None,
        min_duration: float = 3.0,
        max_duration: float = 30.0,
        orientation: str = "landscape",
        limit: int = 10,
    ) -> list[dict]:
        """
        Search for stock footage across providers.

        Args:
            queries: List of search queries
            providers: List of providers to search (default: pexels, pixabay)
            min_duration: Minimum video duration in seconds
            max_duration: Maximum video duration in seconds
            orientation: Video orientation (landscape, portrait, square)
            limit: Maximum results per query per provider

        Returns:
            List of search results with metadata
        """
        if providers is None:
            providers = ["pexels", "pixabay"]

        all_results = []
        search_tasks = []

        for query in queries:
            for provider in providers:
                if provider == "pexels" and settings.pexels_api_key:
                    search_tasks.append(
                        self._search_pexels(query, orientation, limit, min_duration, max_duration)
                    )
                elif provider == "pixabay" and settings.pixabay_api_key:
                    search_tasks.append(
                        self._search_pixabay(query, orientation, limit, min_duration, max_duration)
                    )

        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Search error: {result}")
                elif result:
                    all_results.extend(result)

        # Deduplicate by provider_id
        seen = set()
        unique_results = []
        for r in all_results:
            key = f"{r['provider']}:{r['provider_id']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return unique_results

    async def _search_pexels(
        self,
        query: str,
        orientation: str,
        limit: int,
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Search Pexels for stock videos."""
        session = await self._get_session()

        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": settings.pexels_api_key}
        params = {
            "query": query,
            "orientation": orientation,
            "per_page": min(limit, 80),  # Pexels max is 80
        }

        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Pexels API error: {response.status}")
                    return []

                data = await response.json()
                results = []

                for video in data.get("videos", []):
                    duration = video.get("duration", 0)

                    # Filter by duration
                    if duration < min_duration or duration > max_duration:
                        continue

                    # Get the best quality video file
                    video_files = video.get("video_files", [])
                    best_file = self._get_best_video_file(video_files)

                    if not best_file:
                        continue

                    # Get thumbnail
                    thumbnail = video.get("image", "")
                    video_pictures = video.get("video_pictures", [])
                    if video_pictures:
                        thumbnail = video_pictures[0].get("picture", thumbnail)

                    results.append({
                        "provider": "pexels",
                        "provider_id": str(video.get("id")),
                        "provider_url": video.get("url", ""),
                        "download_url": best_file.get("link", ""),
                        "thumbnail_url": thumbnail,
                        "title": f"Pexels Video {video.get('id')}",
                        "tags": [],  # Pexels doesn't provide tags in search
                        "duration": duration,
                        "width": best_file.get("width"),
                        "height": best_file.get("height"),
                    })

                return results

        except Exception as e:
            logger.error(f"Pexels search error: {e}")
            return []

    async def _search_pixabay(
        self,
        query: str,
        orientation: str,
        limit: int,
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Search Pixabay for stock videos."""
        session = await self._get_session()

        # Map orientation
        orientation_map = {
            "landscape": "horizontal",
            "portrait": "vertical",
            "square": "all",
        }

        url = "https://pixabay.com/api/videos/"
        params = {
            "key": settings.pixabay_api_key,
            "q": quote_plus(query),
            "per_page": min(limit, 200),  # Pixabay max is 200
            "safesearch": "true",
        }

        # Only add orientation if not 'all'
        if orientation in orientation_map and orientation_map[orientation] != "all":
            params["orientation"] = orientation_map[orientation]

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Pixabay API error: {response.status}")
                    return []

                data = await response.json()
                results = []

                for video in data.get("hits", []):
                    duration = video.get("duration", 0)

                    # Filter by duration
                    if duration < min_duration or duration > max_duration:
                        continue

                    # Get the best quality video
                    videos = video.get("videos", {})
                    best_file = self._get_best_pixabay_file(videos)

                    if not best_file:
                        continue

                    # Parse tags
                    tags = [t.strip() for t in video.get("tags", "").split(",") if t.strip()]

                    results.append({
                        "provider": "pixabay",
                        "provider_id": str(video.get("id")),
                        "provider_url": video.get("pageURL", ""),
                        "download_url": best_file.get("url", ""),
                        "thumbnail_url": f"https://i.vimeocdn.com/video/{video.get('picture_id')}_640x360.jpg",
                        "title": video.get("tags", "").split(",")[0] if video.get("tags") else f"Pixabay Video {video.get('id')}",
                        "tags": tags,
                        "duration": duration,
                        "width": best_file.get("width"),
                        "height": best_file.get("height"),
                    })

                return results

        except Exception as e:
            logger.error(f"Pixabay search error: {e}")
            return []

    def _get_best_video_file(self, video_files: list[dict]) -> Optional[dict]:
        """Get the best quality video file from Pexels response."""
        if not video_files:
            return None

        # Prefer 1080p or 720p
        preferred_heights = [1080, 720, 480]

        for height in preferred_heights:
            for vf in video_files:
                if vf.get("height") == height:
                    return vf

        # Fall back to largest file
        return max(video_files, key=lambda x: x.get("height", 0))

    def _get_best_pixabay_file(self, videos: dict) -> Optional[dict]:
        """Get the best quality video file from Pixabay response."""
        if not videos:
            return None

        # Prefer large, then medium, then small
        for quality in ["large", "medium", "small", "tiny"]:
            if quality in videos and videos[quality].get("url"):
                return videos[quality]

        return None

    async def download_asset(
        self,
        provider: str,
        provider_id: str,
        download_url: str,
        title: Optional[str] = None,
    ) -> str:
        """
        Download a stock footage asset to local storage.

        Args:
            provider: Provider name (pexels, pixabay)
            provider_id: Provider's video ID
            download_url: URL to download the video
            title: Optional title for filename

        Returns:
            Local file path
        """
        session = await self._get_session()

        # Create storage directory
        storage_path = settings.broll_storage_path
        storage_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        url_hash = hashlib.md5(download_url.encode()).hexdigest()[:8]
        safe_title = "".join(c for c in (title or "video")[:30] if c.isalnum() or c in " -_").strip()
        safe_title = safe_title.replace(" ", "_")
        filename = f"{provider}_{provider_id}_{url_hash}_{safe_title}.mp4"
        local_path = storage_path / filename

        # Check if already downloaded
        if local_path.exists():
            logger.info(f"Asset already cached: {local_path}")
            return str(local_path)

        try:
            logger.info(f"Downloading B-roll asset: {download_url}")

            async with session.get(download_url) as response:
                if response.status != 200:
                    raise ValueError(f"Download failed with status {response.status}")

                content = await response.read()

                with open(local_path, "wb") as f:
                    f.write(content)

            logger.info(f"Downloaded B-roll asset: {local_path}")
            return str(local_path)

        except Exception as e:
            logger.error(f"Failed to download asset: {e}")
            raise

    async def download_thumbnail(
        self,
        thumbnail_url: str,
        provider: str,
        provider_id: str,
    ) -> Optional[str]:
        """Download thumbnail for an asset."""
        if not thumbnail_url:
            return None

        session = await self._get_session()

        # Create thumbnails directory
        thumbnail_dir = settings.broll_storage_path / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        ext = Path(thumbnail_url).suffix or ".jpg"
        if "?" in ext:
            ext = ext.split("?")[0]
        filename = f"{provider}_{provider_id}_thumb{ext}"
        local_path = thumbnail_dir / filename

        # Check if already downloaded
        if local_path.exists():
            return str(local_path)

        try:
            async with session.get(thumbnail_url) as response:
                if response.status != 200:
                    return None

                content = await response.read()

                with open(local_path, "wb") as f:
                    f.write(content)

            return str(local_path)

        except Exception as e:
            logger.warning(f"Failed to download thumbnail: {e}")
            return None

    async def get_cache_size(self) -> int:
        """Get current cache size in bytes."""
        storage_path = settings.broll_storage_path
        if not storage_path.exists():
            return 0

        total_size = 0
        for f in storage_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

        return total_size

    async def cleanup_cache(self, max_size_bytes: Optional[int] = None):
        """Remove oldest cached files to stay under size limit."""
        if max_size_bytes is None:
            max_size_bytes = int(settings.broll_max_cache_size_gb * 1024 * 1024 * 1024)

        storage_path = settings.broll_storage_path
        if not storage_path.exists():
            return

        # Get all files with their stats
        files = []
        for f in storage_path.rglob("*.mp4"):
            if f.is_file():
                stat = f.stat()
                files.append((f, stat.st_mtime, stat.st_size))

        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])

        # Calculate current size
        current_size = sum(f[2] for f in files)

        # Remove oldest files until under limit
        while current_size > max_size_bytes and files:
            oldest = files.pop(0)
            try:
                oldest[0].unlink()
                current_size -= oldest[2]
                logger.info(f"Removed cached B-roll: {oldest[0]}")
            except Exception as e:
                logger.warning(f"Failed to remove cached file: {e}")


# Singleton instance
_search_service: Optional[BRollSearchService] = None


def get_broll_search_service() -> BRollSearchService:
    """Get B-roll search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = BRollSearchService()
    return _search_service
