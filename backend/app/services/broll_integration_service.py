"""B-Roll integration service for inserting stock footage into video exports."""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BRollIntegrationService:
    """Service for integrating B-roll footage into video exports."""

    async def apply_broll(
        self,
        input_path: str,
        output_path: str,
        broll_insertions: list[dict],
        transition_duration: float = 0.5,
    ) -> str:
        """
        Apply B-roll insertions to a video.

        Args:
            input_path: Path to the main video
            output_path: Path for the output video
            broll_insertions: List of B-roll insertions with:
                - start_time: When to insert B-roll (in main video time)
                - duration: How long the B-roll should be
                - asset_path: Path to the B-roll video file
                - mode: 'full_replace' or 'pip_overlay'
                - transition: Transition type (crossfade, fade, cut)
            transition_duration: Duration of crossfade transitions

        Returns:
            Path to the output video
        """
        if not broll_insertions:
            # No B-roll to apply, just copy the file
            shutil.copy(input_path, output_path)
            return output_path

        # Sort insertions by start time
        insertions = sorted(broll_insertions, key=lambda x: x["start_time"])

        # Validate insertions don't overlap
        for i in range(1, len(insertions)):
            prev_end = insertions[i - 1]["start_time"] + insertions[i - 1]["duration"]
            curr_start = insertions[i]["start_time"]
            if curr_start < prev_end:
                logger.warning(f"Overlapping B-roll insertions at {curr_start}s, adjusting...")
                insertions[i]["start_time"] = prev_end

        # Check if all are full_replace or if we have any pip_overlay
        has_pip = any(ins.get("mode") == "pip_overlay" for ins in insertions)

        if has_pip:
            return await self._apply_broll_with_pip(
                input_path, output_path, insertions, transition_duration
            )
        else:
            return await self._apply_broll_full_replace(
                input_path, output_path, insertions, transition_duration
            )

    async def _apply_broll_full_replace(
        self,
        input_path: str,
        output_path: str,
        insertions: list[dict],
        transition_duration: float,
    ) -> str:
        """Apply B-roll with full replacement using segment-based approach."""
        loop = asyncio.get_event_loop()

        # Get main video info
        main_info = await self._get_video_info(input_path)
        main_duration = main_info.get("duration", 0)
        width = main_info.get("width", 1920)
        height = main_info.get("height", 1080)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            segments = []
            segment_files = []

            # Build list of segments (alternating main video and B-roll)
            current_time = 0.0

            for i, insertion in enumerate(insertions):
                broll_start = insertion["start_time"]
                broll_duration = insertion["duration"]
                broll_path = insertion["asset_path"]
                transition = insertion.get("transition", "crossfade")

                # Add main video segment before B-roll
                if broll_start > current_time:
                    segment_file = temp_path / f"main_seg_{i}.mp4"
                    await self._extract_segment(
                        input_path, str(segment_file),
                        current_time, broll_start - current_time,
                        width, height
                    )
                    segments.append({
                        "file": str(segment_file),
                        "type": "main",
                        "duration": broll_start - current_time,
                    })
                    segment_files.append(str(segment_file))

                # Add B-roll segment
                broll_segment_file = temp_path / f"broll_seg_{i}.mp4"
                await self._prepare_broll_segment(
                    broll_path, str(broll_segment_file),
                    broll_duration, width, height
                )
                segments.append({
                    "file": str(broll_segment_file),
                    "type": "broll",
                    "duration": broll_duration,
                    "transition": transition,
                })
                segment_files.append(str(broll_segment_file))

                current_time = broll_start + broll_duration

            # Add remaining main video
            if current_time < main_duration:
                segment_file = temp_path / f"main_seg_final.mp4"
                await self._extract_segment(
                    input_path, str(segment_file),
                    current_time, main_duration - current_time,
                    width, height
                )
                segments.append({
                    "file": str(segment_file),
                    "type": "main",
                    "duration": main_duration - current_time,
                })
                segment_files.append(str(segment_file))

            # Concatenate with crossfades
            video_output = temp_path / "video_merged.mp4"
            await self._concatenate_with_transitions(
                segments, str(video_output), transition_duration
            )

            # Mix audio (keep original audio throughout)
            await self._mix_audio(input_path, str(video_output), output_path)

        return output_path

    async def _apply_broll_with_pip(
        self,
        input_path: str,
        output_path: str,
        insertions: list[dict],
        transition_duration: float,
    ) -> str:
        """Apply B-roll with picture-in-picture overlays."""
        main_info = await self._get_video_info(input_path)
        width = main_info.get("width", 1920)
        height = main_info.get("height", 1080)

        # Build filter complex for PIP overlays
        filter_parts = []
        input_files = [input_path]

        # PIP settings
        pip_width = width // 4
        pip_x = width - pip_width - 20
        pip_y = 20

        for i, insertion in enumerate(insertions):
            broll_path = insertion["asset_path"]
            start_time = insertion["start_time"]
            duration = insertion["duration"]
            mode = insertion.get("mode", "full_replace")

            if mode == "pip_overlay":
                input_files.append(broll_path)
                input_idx = len(input_files) - 1

                # Scale B-roll for PIP
                filter_parts.append(
                    f"[{input_idx}:v]scale={pip_width}:-1,setpts=PTS-STARTPTS[pip{i}]"
                )

                # Overlay with enable expression
                if i == 0:
                    filter_parts.append(
                        f"[0:v][pip{i}]overlay=x={pip_x}:y={pip_y}:"
                        f"enable='between(t,{start_time},{start_time + duration})'[v{i}]"
                    )
                else:
                    filter_parts.append(
                        f"[v{i-1}][pip{i}]overlay=x={pip_x}:y={pip_y}:"
                        f"enable='between(t,{start_time},{start_time + duration})'[v{i}]"
                    )

        # Build FFmpeg command
        inputs = []
        for f in input_files:
            inputs.extend(["-i", f])

        if filter_parts:
            filter_complex = ";".join(filter_parts)
            last_output = f"[v{len(insertions) - 1}]"

            cmd = (
                ["ffmpeg", "-y"]
                + inputs
                + [
                    "-filter_complex", filter_complex,
                    "-map", last_output,
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-c:a", "aac",
                    output_path,
                ]
            )
        else:
            # No filters, just copy
            cmd = ["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path]

        await self._run_ffmpeg(cmd)
        return output_path

    async def _get_video_info(self, video_path: str) -> dict:
        """Get video duration, width, and height using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]

        loop = asyncio.get_event_loop()

        def _run():
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.stdout

        output = await loop.run_in_executor(None, _run)

        import json
        try:
            data = json.loads(output)
            streams = data.get("streams", [])
            stream = streams[0] if streams else {}
            format_info = data.get("format", {})

            return {
                "width": stream.get("width", 1920),
                "height": stream.get("height", 1080),
                "duration": float(stream.get("duration") or format_info.get("duration", 0)),
            }
        except (json.JSONDecodeError, ValueError, IndexError):
            return {"width": 1920, "height": 1080, "duration": 0}

    async def _extract_segment(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        duration: float,
        target_width: int,
        target_height: int,
    ):
        """Extract a segment from the main video."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-an",  # No audio for segments
            output_path,
        ]

        await self._run_ffmpeg(cmd)

    async def _prepare_broll_segment(
        self,
        input_path: str,
        output_path: str,
        target_duration: float,
        target_width: int,
        target_height: int,
    ):
        """Prepare B-roll segment with correct duration and dimensions."""
        # Get B-roll duration
        info = await self._get_video_info(input_path)
        broll_duration = info.get("duration", 0)

        # Build filter
        filters = [
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease",
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2",
        ]

        # Loop if B-roll is shorter than target
        loop_count = 1
        if broll_duration > 0 and broll_duration < target_duration:
            loop_count = int(target_duration / broll_duration) + 1

        # Build command
        input_args = ["-stream_loop", str(loop_count - 1)] if loop_count > 1 else []

        cmd = (
            ["ffmpeg", "-y"]
            + input_args
            + [
                "-i", input_path,
                "-t", str(target_duration),
                "-vf", ",".join(filters),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-an",  # No audio
                output_path,
            ]
        )

        await self._run_ffmpeg(cmd)

    async def _concatenate_with_transitions(
        self,
        segments: list[dict],
        output_path: str,
        transition_duration: float,
    ):
        """Concatenate video segments with crossfade transitions."""
        if len(segments) == 0:
            return

        if len(segments) == 1:
            shutil.copy(segments[0]["file"], output_path)
            return

        # Build xfade filter chain
        inputs = []
        for seg in segments:
            inputs.extend(["-i", seg["file"]])

        # Calculate xfade offsets
        filter_parts = []
        current_offset = 0.0

        for i in range(len(segments) - 1):
            current_duration = segments[i]["duration"]
            offset = current_offset + current_duration - transition_duration

            transition_type = segments[i + 1].get("transition", "fade")
            if transition_type == "cut":
                transition_type = "fade"
                trans_dur = 0.1
            else:
                trans_dur = transition_duration

            if i == 0:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition={transition_type}:duration={trans_dur}:offset={offset}[v{i}]"
                )
            else:
                filter_parts.append(
                    f"[v{i-1}][{i+1}:v]xfade=transition={transition_type}:duration={trans_dur}:offset={offset}[v{i}]"
                )

            current_offset = offset

        filter_complex = ";".join(filter_parts)
        last_output = f"[v{len(segments) - 2}]"

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex", filter_complex,
                "-map", last_output,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                output_path,
            ]
        )

        await self._run_ffmpeg(cmd)

    async def _mix_audio(
        self,
        original_video: str,
        video_without_audio: str,
        output_path: str,
    ):
        """Mix original audio back onto the processed video."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_without_audio,
            "-i", original_video,
            "-map", "0:v",
            "-map", "1:a?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]

        await self._run_ffmpeg(cmd)

    async def _run_ffmpeg(self, cmd: list[str]):
        """Run FFmpeg command asynchronously."""
        loop = asyncio.get_event_loop()

        def _run():
            logger.debug(f"Running FFmpeg: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            return result

        await loop.run_in_executor(None, _run)

    async def get_broll_preview_frame(
        self,
        main_video_path: str,
        broll_path: str,
        timestamp: float,
        mode: str = "full_replace",
    ) -> bytes:
        """
        Generate a preview frame showing B-roll insertion.

        Args:
            main_video_path: Path to main video
            broll_path: Path to B-roll video
            timestamp: Timestamp in main video
            mode: 'full_replace' or 'pip_overlay'

        Returns:
            PNG image bytes
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if mode == "pip_overlay":
                # Show main video with B-roll as PIP
                main_info = await self._get_video_info(main_video_path)
                width = main_info.get("width", 1920)
                pip_width = width // 4

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(timestamp),
                    "-i", main_video_path,
                    "-ss", "0",
                    "-i", broll_path,
                    "-filter_complex",
                    f"[1:v]scale={pip_width}:-1[pip];[0:v][pip]overlay=x=W-w-20:y=20",
                    "-vframes", "1",
                    tmp_path,
                ]
            else:
                # Show B-roll frame
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", "0",
                    "-i", broll_path,
                    "-vframes", "1",
                    tmp_path,
                ]

            await self._run_ffmpeg(cmd)

            with open(tmp_path, "rb") as f:
                return f.read()

        finally:
            Path(tmp_path).unlink(missing_ok=True)


# Singleton instance
_integration_service: Optional[BRollIntegrationService] = None


def get_broll_integration_service() -> BRollIntegrationService:
    """Get B-roll integration service instance."""
    global _integration_service
    if _integration_service is None:
        _integration_service = BRollIntegrationService()
    return _integration_service
