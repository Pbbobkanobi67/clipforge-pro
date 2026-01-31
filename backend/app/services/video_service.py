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

    async def export_clip_enhanced(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        resolution: Optional[str] = None,
        # Animated captions
        ass_subtitle_path: Optional[str] = None,
        # Reframe / crop
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
        scale_width: Optional[int] = None,
        scale_height: Optional[int] = None,
        # Logo overlay
        logo_path: Optional[str] = None,
        logo_position: str = "top_right",
        logo_size: int = 100,
        logo_opacity: float = 1.0,
        logo_margin: int = 20,
        # Outro
        outro_enabled: bool = False,
        outro_duration: float = 3.0,
        outro_text: Optional[str] = None,
        outro_cta: Optional[str] = None,
        outro_bg_color: str = "#000000",
    ) -> str:
        """
        Enhanced clip export with animated captions, reframe, logo, and outro.

        Args:
            input_path: Source video path
            output_path: Output video path
            start_time: Clip start time
            end_time: Clip end time
            resolution: Target resolution (e.g., "1080p")
            ass_subtitle_path: Path to ASS subtitle file for animated captions
            crop_x, crop_y, crop_width, crop_height: Crop parameters for reframe
            scale_width, scale_height: Final output dimensions
            logo_path: Path to logo image
            logo_position: Logo position (top_left, top_right, bottom_left, bottom_right)
            logo_size: Logo width in pixels
            logo_opacity: Logo opacity (0.0-1.0)
            logo_margin: Logo margin from edges
            outro_enabled: Whether to add outro
            outro_duration: Outro duration in seconds
            outro_text: Main outro text
            outro_cta: Call-to-action text
            outro_bg_color: Outro background color

        Returns:
            Path to exported clip
        """
        duration = end_time - start_time
        filter_parts = []
        inputs = ["-i", input_path]
        input_count = 1

        # Build video filter chain
        video_filters = []

        # 1. Crop for reframe (if specified)
        if crop_width and crop_height:
            crop_x = crop_x or 0
            crop_y = crop_y or 0
            video_filters.append(f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}")

        # 2. Scale (resolution or reframe target)
        if scale_width and scale_height:
            video_filters.append(f"scale={scale_width}:{scale_height}")
        elif resolution:
            height = int(resolution.replace("p", ""))
            video_filters.append(f"scale=-2:{height}")

        # 3. ASS subtitles (animated captions)
        if ass_subtitle_path and Path(ass_subtitle_path).exists():
            # Escape path for ffmpeg filter
            escaped_path = ass_subtitle_path.replace("\\", "/").replace(":", "\\:")
            video_filters.append(f"ass='{escaped_path}'")

        # Combine main video filters
        main_video_filter = ",".join(video_filters) if video_filters else None

        # 4. Logo overlay (requires complex filter)
        if logo_path and Path(logo_path).exists():
            inputs.extend(["-i", logo_path])
            logo_input = input_count
            input_count += 1

            # Calculate logo position
            position_map = {
                "top_left": f"{logo_margin}:{logo_margin}",
                "top_right": f"W-w-{logo_margin}:{logo_margin}",
                "bottom_left": f"{logo_margin}:H-h-{logo_margin}",
                "bottom_right": f"W-w-{logo_margin}:H-h-{logo_margin}",
                "center": "(W-w)/2:(H-h)/2",
            }
            overlay_pos = position_map.get(logo_position, position_map["top_right"])

            # Build complex filter for logo
            if main_video_filter:
                # Apply video filters first, then overlay logo
                filter_complex = (
                    f"[0:v]{main_video_filter}[main];"
                    f"[{logo_input}:v]scale={logo_size}:-1,format=rgba,"
                    f"colorchannelmixer=aa={logo_opacity}[logo];"
                    f"[main][logo]overlay={overlay_pos}[outv]"
                )
            else:
                filter_complex = (
                    f"[{logo_input}:v]scale={logo_size}:-1,format=rgba,"
                    f"colorchannelmixer=aa={logo_opacity}[logo];"
                    f"[0:v][logo]overlay={overlay_pos}[outv]"
                )

            use_filter_complex = True
        else:
            use_filter_complex = False
            filter_complex = None

        # Build command
        cmd = ["ffmpeg", "-ss", str(start_time)]
        cmd.extend(inputs)
        cmd.extend(["-t", str(duration)])

        if use_filter_complex:
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", "[outv]", "-map", "0:a?"])
        elif main_video_filter:
            cmd.extend(["-vf", main_video_filter])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-y",
        ])

        # If outro is enabled, we need a two-pass approach
        if outro_enabled and outro_text:
            # First export main clip to temp file
            temp_clip = output_path + ".temp.mp4"
            cmd.append(temp_clip)

            returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)
            if returncode != 0:
                raise Exception(f"Clip export failed: {stderr.decode()}")

            # Then create outro and concatenate
            await self._add_outro(
                clip_path=temp_clip,
                output_path=output_path,
                outro_duration=outro_duration,
                outro_text=outro_text,
                outro_cta=outro_cta,
                outro_bg_color=outro_bg_color,
            )

            # Cleanup temp file
            Path(temp_clip).unlink(missing_ok=True)
        else:
            cmd.append(output_path)
            returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)
            if returncode != 0:
                raise Exception(f"Clip export failed: {stderr.decode()}")

        logger.info(f"Enhanced clip exported: {output_path}")
        return output_path

    async def _add_outro(
        self,
        clip_path: str,
        output_path: str,
        outro_duration: float,
        outro_text: str,
        outro_cta: Optional[str],
        outro_bg_color: str,
    ) -> str:
        """Add outro screen to clip using concat demuxer."""
        import tempfile

        # Get clip dimensions
        metadata = await self.get_video_metadata(clip_path)
        width = metadata.get("width", 1920)
        height = metadata.get("height", 1080)
        fps = metadata.get("fps", 30)

        # Create outro video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            outro_path = tmp.name

        # Convert hex color to ffmpeg format
        bg_color = outro_bg_color.lstrip("#")

        # Build drawtext filter for outro
        text_parts = []

        # Main text
        if outro_text:
            text_parts.append(
                f"drawtext=text='{outro_text}':"
                f"fontsize={height//15}:fontcolor=white:"
                f"x=(w-tw)/2:y=(h-th)/2-{height//10}:"
                f"borderw=2:bordercolor=black"
            )

        # CTA text
        if outro_cta:
            text_parts.append(
                f"drawtext=text='{outro_cta}':"
                f"fontsize={height//20}:fontcolor=yellow:"
                f"x=(w-tw)/2:y=(h-th)/2+{height//10}:"
                f"borderw=2:bordercolor=black"
            )

        filter_str = ",".join(text_parts) if text_parts else "null"

        # Generate outro video
        cmd = [
            "ffmpeg",
            "-f", "lavfi",
            "-i", f"color=c=0x{bg_color}:s={width}x{height}:d={outro_duration}:r={fps}",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo:d={outro_duration}",
            "-vf", filter_str,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            outro_path,
        ]

        returncode, _, stderr = await asyncio.to_thread(_run_subprocess, cmd)
        if returncode != 0:
            logger.warning(f"Outro generation failed: {stderr.decode()}")
            # Fall back to just copying the clip
            import shutil
            shutil.copy(clip_path, output_path)
            return output_path

        # Concatenate clip and outro
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_list = f.name
            f.write(f"file '{clip_path}'\n")
            f.write(f"file '{outro_path}'\n")

        concat_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-y",
            output_path,
        ]

        returncode, _, stderr = await asyncio.to_thread(_run_subprocess, concat_cmd)

        # Cleanup
        Path(outro_path).unlink(missing_ok=True)
        Path(concat_list).unlink(missing_ok=True)

        if returncode != 0:
            raise Exception(f"Concat failed: {stderr.decode()}")

        return output_path
