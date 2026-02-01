"""Timeline editor service for multi-clip rendering."""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EditorService:
    """Service for timeline-based video editing and rendering."""

    def __init__(self):
        self._cache_dir = settings.cache_path / "editor"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def render_timeline(
        self,
        timeline_data: dict,
        clip_paths: dict[str, str],  # clip_id -> file_path mapping
        output_path: str,
        resolution: str = "1080p",
        fps: float = 30.0,
        format: str = "mp4",
        # Branding options
        logo_path: Optional[str] = None,
        logo_position: str = "top_right",
        logo_size: int = 100,
        logo_margin: int = 20,
        # Caption options
        ass_subtitle_paths: Optional[dict[str, str]] = None,  # clip_id -> ass_path mapping
    ) -> str:
        """
        Render a timeline project to a single video file.

        Args:
            timeline_data: Timeline structure with tracks and clips
            clip_paths: Mapping of clip IDs to source video paths
            output_path: Output video path
            resolution: Target resolution
            fps: Frame rate
            format: Output format
            logo_path: Optional logo to overlay
            logo_position: Logo position
            logo_size: Logo size
            logo_margin: Logo margin
            ass_subtitle_paths: Optional ASS subtitle paths per clip

        Returns:
            Path to rendered video
        """
        from app.services.video_service import VideoService, _run_subprocess

        video_service = VideoService()

        # Parse resolution
        height = int(resolution.replace("p", ""))
        width = int(height * 16 / 9)  # Assume 16:9

        # Get tracks from timeline
        tracks = timeline_data.get("tracks", [])
        video_track = next((t for t in tracks if t.get("type") == "video"), None)

        if not video_track or not video_track.get("clips"):
            raise ValueError("No video clips in timeline")

        clips = video_track["clips"]

        # Process each clip
        temp_files = []
        concat_entries = []

        try:
            for i, clip_data in enumerate(clips):
                clip_id = str(clip_data.get("clip_id"))
                source_path = clip_paths.get(clip_id)

                if not source_path or not Path(source_path).exists():
                    logger.warning(f"Clip not found: {clip_id}")
                    continue

                # Get clip trim settings
                trim_start = clip_data.get("trim_start", 0)
                trim_end = clip_data.get("trim_end", 0)
                duration = clip_data.get("duration", 0)

                # Calculate actual timestamps
                # Get source video duration
                metadata = await video_service.get_video_metadata(source_path)
                source_duration = metadata.get("duration", 0)

                if source_duration > 0:
                    actual_start = trim_start
                    actual_end = source_duration - trim_end
                    actual_duration = actual_end - actual_start
                else:
                    actual_start = 0
                    actual_end = duration
                    actual_duration = duration

                # Create temp file for processed clip
                temp_path = str(self._cache_dir / f"temp_clip_{i}_{clip_id[:8]}.mp4")
                temp_files.append(temp_path)

                # Build filter
                filters = [f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                           f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
                           f"fps={fps}"]

                # Add subtitles if available
                if ass_subtitle_paths and clip_id in ass_subtitle_paths:
                    ass_path = ass_subtitle_paths[clip_id]
                    if Path(ass_path).exists():
                        escaped_path = ass_path.replace("\\", "/").replace(":", "\\:")
                        filters.append(f"ass='{escaped_path}'")

                filter_str = ",".join(filters)

                # Process clip
                cmd = [
                    "ffmpeg",
                    "-ss", str(actual_start),
                    "-i", source_path,
                    "-t", str(actual_duration),
                    "-vf", filter_str,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-ar", "44100",
                    "-y",
                    temp_path,
                ]

                returncode, _, stderr = await asyncio.to_thread(
                    lambda: _run_subprocess(cmd)
                )

                if returncode != 0:
                    logger.error(f"Failed to process clip {clip_id}: {stderr.decode()}")
                    continue

                # Use just the filename since concat file is in the same directory
                temp_filename = Path(temp_path).name
                concat_entries.append(f"file '{temp_filename}'")

            if not concat_entries:
                raise ValueError("No clips could be processed")

            # Create concat list file
            concat_list_path = str(self._cache_dir / "concat_list.txt")
            Path(concat_list_path).write_text("\n".join(concat_entries))
            temp_files.append(concat_list_path)

            # Concatenate all clips
            if logo_path and Path(logo_path).exists():
                # With logo overlay
                concat_output = str(self._cache_dir / "concat_temp.mp4")
                temp_files.append(concat_output)

                concat_cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    "-y",
                    concat_output,
                ]

                returncode, _, stderr = await asyncio.to_thread(
                    lambda: _run_subprocess(concat_cmd)
                )

                if returncode != 0:
                    raise Exception(f"Concat failed: {stderr.decode()}")

                # Add logo overlay
                position_map = {
                    "top_left": f"{logo_margin}:{logo_margin}",
                    "top_right": f"W-w-{logo_margin}:{logo_margin}",
                    "bottom_left": f"{logo_margin}:H-h-{logo_margin}",
                    "bottom_right": f"W-w-{logo_margin}:H-h-{logo_margin}",
                }
                overlay_pos = position_map.get(logo_position, position_map["top_right"])

                logo_cmd = [
                    "ffmpeg",
                    "-i", concat_output,
                    "-i", logo_path,
                    "-filter_complex",
                    f"[1:v]scale={logo_size}:-1[logo];[0:v][logo]overlay={overlay_pos}",
                    "-c:v", "libx264",
                    "-c:a", "copy",
                    "-y",
                    output_path,
                ]

                returncode, _, stderr = await asyncio.to_thread(
                    lambda: _run_subprocess(logo_cmd)
                )

                if returncode != 0:
                    raise Exception(f"Logo overlay failed: {stderr.decode()}")
            else:
                # Simple concat
                concat_cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c", "copy",
                    "-y",
                    output_path,
                ]

                returncode, _, stderr = await asyncio.to_thread(
                    lambda: _run_subprocess(concat_cmd)
                )

                if returncode != 0:
                    raise Exception(f"Concat failed: {stderr.decode()}")

            logger.info(f"Timeline rendered: {output_path}")
            return output_path

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                Path(temp_file).unlink(missing_ok=True)

    async def get_timeline_duration(self, timeline_data: dict) -> float:
        """Calculate total duration of a timeline."""
        tracks = timeline_data.get("tracks", [])
        max_duration = 0.0

        for track in tracks:
            track_duration = 0.0
            for clip in track.get("clips", []):
                clip_end = clip.get("start", 0) + clip.get("duration", 0)
                track_duration = max(track_duration, clip_end)
            max_duration = max(max_duration, track_duration)

        return max_duration

    def validate_timeline(self, timeline_data: dict) -> list[str]:
        """
        Validate timeline data structure.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        if not isinstance(timeline_data, dict):
            errors.append("Timeline data must be a dictionary")
            return errors

        tracks = timeline_data.get("tracks", [])
        if not tracks:
            errors.append("Timeline must have at least one track")
            return errors

        video_tracks = [t for t in tracks if t.get("type") == "video"]
        if not video_tracks:
            errors.append("Timeline must have at least one video track")

        for i, track in enumerate(tracks):
            if not isinstance(track, dict):
                errors.append(f"Track {i} must be a dictionary")
                continue

            track_type = track.get("type")
            if track_type not in ["video", "audio", "caption"]:
                errors.append(f"Track {i} has invalid type: {track_type}")

            clips = track.get("clips", [])
            for j, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    errors.append(f"Track {i} clip {j} must be a dictionary")
                    continue

                if "clip_id" not in clip:
                    errors.append(f"Track {i} clip {j} missing clip_id")

                if "duration" not in clip and "start" not in clip:
                    errors.append(f"Track {i} clip {j} needs duration or start time")

        return errors


# Singleton instance
_editor_service = None


def get_editor_service() -> EditorService:
    """Get singleton editor service."""
    global _editor_service
    if _editor_service is None:
        _editor_service = EditorService()
    return _editor_service
