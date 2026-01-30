"""Video processing service using yt-dlp and ffmpeg."""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Use venv's yt-dlp if available, otherwise fall back to system
_VENV_YTDLP = Path(sys.executable).parent / "yt-dlp.exe"
YTDLP_CMD = str(_VENV_YTDLP) if _VENV_YTDLP.exists() else "yt-dlp"


def _run_subprocess(cmd: list[str]) -> tuple[int, bytes, bytes]:
    """Run subprocess synchronously and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stdout, result.stderr


class VideoService:
    """Service for video file operations."""

    def detect_platform(self, url: str) -> Optional[str]:
        """Detect video platform from URL."""
        url_lower = url.lower()
        platforms = {
            "youtube": ["youtube.com", "youtu.be"],
            "vimeo": ["vimeo.com"],
            "twitter": ["twitter.com", "x.com"],
            "tiktok": ["tiktok.com"],
            "instagram": ["instagram.com"],
            "twitch": ["twitch.tv"],
            "facebook": ["facebook.com", "fb.watch"],
            "dailymotion": ["dailymotion.com"],
        }
        for platform, domains in platforms.items():
            if any(domain in url_lower for domain in domains):
                return platform
        return "other"

    async def get_video_metadata(self, file_path: str) -> dict[str, Any]:
        """Extract video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]

        try:
            # Use thread pool for subprocess (works on Windows)
            returncode, stdout, stderr = await asyncio.to_thread(_run_subprocess, cmd)

            if returncode != 0:
                logger.warning(f"ffprobe failed: {stderr.decode()}")
                return {}

            data = json.loads(stdout.decode())
            video_stream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {},
            )
            format_info = data.get("format", {})

            return {
                "duration": float(format_info.get("duration", 0)),
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "0") else float(video_stream.get("r_frame_rate", 0)),
                "codec": video_stream.get("codec_name"),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "title": format_info.get("tags", {}).get("title"),
            }
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return {}

    async def download_video(
        self,
        url: str,
        output_dir: str,
        video_id: str,
    ) -> dict[str, Any]:
        """Download video using yt-dlp."""
        output_template = str(Path(output_dir) / f"{video_id}.%(ext)s")

        cmd = [
            YTDLP_CMD,
            "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "-o", output_template,
            "--write-thumbnail",
            "--no-playlist",
            "--print-json",
            "--no-warnings",
            "--extractor-args", "youtube:player_client=android",
            url,
        ]

        try:
            # Use thread pool for subprocess (works on Windows)
            returncode, stdout, stderr = await asyncio.to_thread(_run_subprocess, cmd)

            if returncode != 0:
                error = stderr.decode()
                logger.error(f"yt-dlp failed: {error}")
                raise Exception(f"Download failed: {error}")

            # Parse JSON output
            info = json.loads(stdout.decode())

            # Find downloaded file - always use output_dir since _filename may be relative
            ext = info.get("ext", "mp4")
            file_path = str(Path(output_dir) / f"{video_id}.{ext}")

            # Find thumbnail
            thumbnail_path = None
            for ext in ["jpg", "png", "webp"]:
                thumb = Path(output_dir) / f"{video_id}.{ext}"
                if thumb.exists():
                    thumbnail_path = str(thumb)
                    break

            return {
                "file_path": file_path,
                "file_size": info.get("filesize") or info.get("filesize_approx"),
                "duration": info.get("duration"),
                "width": info.get("width"),
                "height": info.get("height"),
                "fps": info.get("fps"),
                "title": info.get("title"),
                "description": info.get("description"),
                "thumbnail_path": thumbnail_path,
            }

        except json.JSONDecodeError:
            # Sometimes yt-dlp doesn't output JSON, find file manually
            for ext in ["mp4", "webm", "mkv"]:
                file_path = Path(output_dir) / f"{video_id}.{ext}"
                if file_path.exists():
                    metadata = await self.get_video_metadata(str(file_path))
                    return {
                        "file_path": str(file_path),
                        "file_size": file_path.stat().st_size,
                        **metadata,
                    }
            raise Exception("Downloaded file not found")

    async def extract_audio(
        self,
        video_path: str,
        output_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """Extract audio from video file for transcription."""
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            output_path,
        ]

        # Use thread pool for subprocess (works on Windows)
        returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)

        if returncode != 0:
            raise Exception(f"Audio extraction failed: {stderr.decode()}")

        return output_path

    async def export_clip(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        resolution: Optional[str] = None,
        captions: Optional[list[dict]] = None,
    ) -> str:
        """Export a clip from video with optional captions."""
        duration = end_time - start_time

        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
        ]

        # Add resolution scaling if specified
        if resolution:
            height = int(resolution.replace("p", ""))
            cmd.extend(["-vf", f"scale=-2:{height}"])

        # Add captions as subtitles (burn-in)
        if captions:
            # Create temporary SRT file
            srt_path = output_path + ".srt"
            self._create_srt(captions, srt_path)
            cmd.extend(["-vf", f"subtitles={srt_path}"])

        cmd.append(output_path)

        # Use thread pool for subprocess (works on Windows)
        returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)

        if returncode != 0:
            raise Exception(f"Clip export failed: {stderr.decode()}")

        # Cleanup temp SRT
        if captions:
            Path(output_path + ".srt").unlink(missing_ok=True)

        return output_path

    def _create_srt(self, captions: list[dict], output_path: str) -> None:
        """Create SRT subtitle file from captions."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, cap in enumerate(captions, 1):
                start = self._format_srt_time(cap["start"])
                end = self._format_srt_time(cap["end"])
                f.write(f"{i}\n{start} --> {end}\n{cap['text']}\n\n")

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    async def extract_frame(
        self,
        video_path: str,
        output_path: str,
        timestamp: float,
    ) -> str:
        """Extract a single frame at specified timestamp."""
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-y",
            output_path,
        ]

        # Use thread pool for subprocess (works on Windows)
        returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)

        if returncode != 0:
            raise Exception(f"Frame extraction failed: {stderr.decode()}")

        return output_path
