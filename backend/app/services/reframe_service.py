"""AI Reframe service for smart video cropping with subject tracking."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Aspect ratio dimensions mapping
ASPECT_RATIOS = {
    "9:16": (1080, 1920),   # TikTok, Reels, Shorts (portrait)
    "1:1": (1080, 1080),    # Instagram square
    "16:9": (1920, 1080),   # YouTube landscape
    "4:3": (1440, 1080),    # Standard
    "4:5": (1080, 1350),    # Instagram portrait
}


def _calculate_crop_dimensions(
    source_width: int,
    source_height: int,
    target_ratio: str,
) -> tuple[int, int]:
    """
    Calculate crop dimensions to achieve target aspect ratio.
    Returns (crop_width, crop_height) that fits within source dimensions.
    """
    if target_ratio not in ASPECT_RATIOS:
        target_ratio = "9:16"

    target_width, target_height = ASPECT_RATIOS[target_ratio]
    target_aspect = target_width / target_height
    source_aspect = source_width / source_height

    if target_aspect > source_aspect:
        # Target is wider than source, limit by width
        crop_width = source_width
        crop_height = int(source_width / target_aspect)
    else:
        # Target is taller than source, limit by height
        crop_height = source_height
        crop_width = int(source_height * target_aspect)

    # Ensure even dimensions for video encoding
    crop_width = (crop_width // 2) * 2
    crop_height = (crop_height // 2) * 2

    return crop_width, crop_height


def _smooth_positions(positions: list[dict], smooth_factor: float = 0.3) -> list[dict]:
    """
    Apply exponential smoothing to crop positions for smoother camera movement.

    Args:
        positions: List of {"time": float, "x": int, "y": int, "confidence": float}
        smooth_factor: 0 = no smoothing, 1 = maximum smoothing
    """
    if len(positions) <= 1:
        return positions

    smoothed = [positions[0].copy()]
    alpha = 1 - smooth_factor

    for i in range(1, len(positions)):
        prev = smoothed[-1]
        curr = positions[i]

        smoothed_x = int(alpha * curr["x"] + (1 - alpha) * prev["x"])
        smoothed_y = int(alpha * curr["y"] + (1 - alpha) * prev["y"])

        smoothed.append({
            "time": curr["time"],
            "x": smoothed_x,
            "y": smoothed_y,
            "confidence": curr.get("confidence", 1.0),
        })

    return smoothed


def _interpolate_keyframes(
    keyframes: list[dict],
    fps: float,
    total_duration: float,
) -> list[dict]:
    """
    Interpolate between keyframes to generate per-frame positions.
    Uses linear interpolation for smooth transitions.
    """
    if not keyframes:
        return []

    # Sort by time
    keyframes = sorted(keyframes, key=lambda k: k["time"])

    # Generate positions for each frame
    positions = []
    frame_duration = 1.0 / fps
    current_time = 0

    while current_time <= total_duration:
        # Find surrounding keyframes
        prev_kf = keyframes[0]
        next_kf = keyframes[-1]

        for i, kf in enumerate(keyframes):
            if kf["time"] <= current_time:
                prev_kf = kf
                if i + 1 < len(keyframes):
                    next_kf = keyframes[i + 1]
                else:
                    next_kf = kf

        # Linear interpolation
        if prev_kf["time"] == next_kf["time"]:
            t = 0
        else:
            t = (current_time - prev_kf["time"]) / (next_kf["time"] - prev_kf["time"])

        x = int(prev_kf["x"] + t * (next_kf["x"] - prev_kf["x"]))
        y = int(prev_kf["y"] + t * (next_kf["y"] - prev_kf["y"]))
        width = int(prev_kf.get("width", 608) + t * (next_kf.get("width", 608) - prev_kf.get("width", 608)))
        height = int(prev_kf.get("height", 1080) + t * (next_kf.get("height", 1080) - prev_kf.get("height", 1080)))

        positions.append({
            "time": current_time,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })

        current_time += frame_duration

    return positions


class ReframeService:
    """Service for AI-powered video reframing with subject tracking."""

    def __init__(self):
        self._cache_dir = settings.cache_path / "reframe"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._visual_service = None

    def _get_visual_service(self):
        """Lazy load visual analysis service."""
        if self._visual_service is None:
            from app.services.visual_analysis_service import get_visual_analysis_service
            self._visual_service = get_visual_analysis_service()
        return self._visual_service

    async def analyze_video_for_reframe(
        self,
        video_path: str,
        aspect_ratio: str = "9:16",
        tracking_mode: str = "speaker",
        sample_interval: float = 0.5,
    ) -> dict:
        """
        Analyze video to generate crop keyframes for reframing.

        Args:
            video_path: Path to source video
            aspect_ratio: Target aspect ratio (9:16, 1:1, 16:9, 4:3, 4:5)
            tracking_mode: speaker, action, center, or manual
            sample_interval: Seconds between frame samples

        Returns:
            Dictionary with keyframes and crop data
        """
        import cv2

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        logger.info(f"Analyzing video for reframe: {source_width}x{source_height}, "
                    f"{duration:.1f}s, target {aspect_ratio}")

        # Calculate crop dimensions
        crop_width, crop_height = _calculate_crop_dimensions(
            source_width, source_height, aspect_ratio
        )

        # Generate keyframes based on tracking mode
        if tracking_mode == "center":
            keyframes = self._generate_center_keyframes(
                source_width, source_height, crop_width, crop_height, duration
            )
        elif tracking_mode == "manual":
            # Manual mode returns empty keyframes for user to fill
            keyframes = []
        else:
            # speaker or action mode - use YOLO detection
            keyframes = await self._generate_tracking_keyframes(
                video_path, source_width, source_height,
                crop_width, crop_height, fps, duration,
                sample_interval, tracking_mode
            )

        # Calculate target dimensions for export
        target_width, target_height = ASPECT_RATIOS.get(aspect_ratio, (1080, 1920))

        return {
            "source_width": source_width,
            "source_height": source_height,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "target_width": target_width,
            "target_height": target_height,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "fps": fps,
            "keyframes": keyframes,
        }

    def _generate_center_keyframes(
        self,
        source_width: int,
        source_height: int,
        crop_width: int,
        crop_height: int,
        duration: float,
    ) -> list[dict]:
        """Generate keyframes for center crop (no tracking)."""
        center_x = (source_width - crop_width) // 2
        center_y = (source_height - crop_height) // 2

        return [
            {
                "time": 0,
                "x": center_x,
                "y": center_y,
                "width": crop_width,
                "height": crop_height,
                "confidence": 1.0,
            },
            {
                "time": duration,
                "x": center_x,
                "y": center_y,
                "width": crop_width,
                "height": crop_height,
                "confidence": 1.0,
            },
        ]

    async def _generate_tracking_keyframes(
        self,
        video_path: str,
        source_width: int,
        source_height: int,
        crop_width: int,
        crop_height: int,
        fps: float,
        duration: float,
        sample_interval: float,
        tracking_mode: str,
    ) -> list[dict]:
        """Generate keyframes by tracking subjects with YOLO."""
        import cv2
        import tempfile

        visual_service = self._get_visual_service()
        cap = cv2.VideoCapture(video_path)

        keyframes = []
        frame_interval = int(fps * sample_interval)
        frame_idx = 0

        # Default center position
        default_x = (source_width - crop_width) // 2
        default_y = (source_height - crop_height) // 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps

                # Save frame temporarily for YOLO analysis
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, frame)

                try:
                    # Analyze frame
                    result = await visual_service.analyze_frame(tmp_path)
                    detections = result.get("detections", [])

                    # Find target based on tracking mode
                    target_box = None
                    confidence = 0.0

                    if tracking_mode == "speaker":
                        # Look for persons (faces are detected as part of person)
                        persons = [d for d in detections if d["class"] == "person"]
                        if persons:
                            # Use the person with highest confidence
                            best = max(persons, key=lambda p: p["confidence"])
                            target_box = best["bbox"]
                            confidence = best["confidence"]

                    elif tracking_mode == "action":
                        # Look for any movement/action objects
                        action_objects = [d for d in detections if d["class"] in
                                          ["person", "sports ball", "skateboard",
                                           "bicycle", "car", "motorcycle"]]
                        if action_objects:
                            best = max(action_objects, key=lambda p: p["confidence"])
                            target_box = best["bbox"]
                            confidence = best["confidence"]

                    # Calculate crop position
                    if target_box:
                        # Center crop on detected target
                        target_center_x = (target_box["x1"] + target_box["x2"]) / 2
                        target_center_y = (target_box["y1"] + target_box["y2"]) / 2

                        # Calculate crop position (top-left corner)
                        crop_x = int(target_center_x - crop_width / 2)
                        crop_y = int(target_center_y - crop_height / 2)

                        # Clamp to valid range
                        crop_x = max(0, min(crop_x, source_width - crop_width))
                        crop_y = max(0, min(crop_y, source_height - crop_height))
                    else:
                        # No detection, use center
                        crop_x = default_x
                        crop_y = default_y
                        confidence = 0.5  # Lower confidence for default

                    keyframes.append({
                        "time": timestamp,
                        "x": crop_x,
                        "y": crop_y,
                        "width": crop_width,
                        "height": crop_height,
                        "confidence": confidence,
                    })

                finally:
                    # Cleanup temp file
                    Path(tmp_path).unlink(missing_ok=True)

            frame_idx += 1

        cap.release()

        # Add end keyframe if needed
        if keyframes and keyframes[-1]["time"] < duration:
            end_kf = keyframes[-1].copy()
            end_kf["time"] = duration
            keyframes.append(end_kf)

        logger.info(f"Generated {len(keyframes)} tracking keyframes")
        return keyframes

    async def generate_reframed_video(
        self,
        video_path: str,
        output_path: str,
        keyframes: list[dict],
        target_width: int,
        target_height: int,
        smooth_factor: float = 0.3,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """
        Generate reframed video with dynamic cropping.

        Args:
            video_path: Source video path
            output_path: Output video path
            keyframes: List of crop keyframes
            target_width: Target output width
            target_height: Target output height
            smooth_factor: Smoothing for crop movement
            start_time: Optional clip start time
            end_time: Optional clip end time

        Returns:
            Path to reframed video
        """
        from app.services.video_service import _run_subprocess

        if not keyframes:
            raise ValueError("No keyframes provided for reframing")

        # Sort keyframes by time
        keyframes = sorted(keyframes, key=lambda k: k["time"])

        # Apply offset if this is a clip
        if start_time is not None:
            keyframes = [
                {**kf, "time": kf["time"] - start_time}
                for kf in keyframes
                if kf["time"] >= start_time and (end_time is None or kf["time"] <= end_time)
            ]

        # For simple cases (1-2 keyframes), use static crop
        if len(keyframes) <= 2:
            kf = keyframes[0]
            crop_filter = (
                f"crop={kf['width']}:{kf['height']}:{kf['x']}:{kf['y']},"
                f"scale={target_width}:{target_height}"
            )
        else:
            # For dynamic tracking, generate zoompan filter
            # Note: This is a simplified approach - complex tracking might need
            # frame-by-frame processing for best results

            # Smooth positions
            smoothed = _smooth_positions(keyframes, smooth_factor)

            # Use the first keyframe for now (full dynamic requires sendcmd)
            # For advanced use, would generate sendcmd script
            kf = smoothed[0]
            crop_filter = (
                f"crop={kf['width']}:{kf['height']}:{kf['x']}:{kf['y']},"
                f"scale={target_width}:{target_height}"
            )

            # TODO: For truly dynamic cropping, implement sendcmd approach:
            # 1. Write keyframes to a file
            # 2. Use sendcmd filter to update crop parameters over time
            logger.warning("Dynamic tracking simplified to first keyframe - "
                           "advanced tracking requires sendcmd implementation")

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", video_path,
        ]

        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])

        if end_time is not None:
            duration = end_time - (start_time or 0)
            cmd.extend(["-t", str(duration)])

        cmd.extend([
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-y",
            output_path,
        ])

        loop = asyncio.get_event_loop()
        returncode, stdout, stderr = await loop.run_in_executor(
            None, lambda: _run_subprocess(cmd)
        )

        if returncode != 0:
            raise Exception(f"Reframe failed: {stderr.decode()}")

        logger.info(f"Generated reframed video: {output_path}")
        return output_path

    async def generate_preview(
        self,
        video_path: str,
        output_path: str,
        aspect_ratio: str = "9:16",
        tracking_mode: str = "speaker",
        duration: float = 5.0,
        start_time: float = 0,
    ) -> str:
        """
        Generate a short preview of the reframe effect.

        Args:
            video_path: Source video path
            output_path: Output preview path
            aspect_ratio: Target aspect ratio
            tracking_mode: Tracking mode
            duration: Preview duration
            start_time: Start time in source video

        Returns:
            Path to preview video
        """
        # Analyze the preview segment
        crop_data = await self.analyze_video_for_reframe(
            video_path=video_path,
            aspect_ratio=aspect_ratio,
            tracking_mode=tracking_mode,
            sample_interval=0.25,  # More frequent sampling for preview
        )

        # Generate preview
        await self.generate_reframed_video(
            video_path=video_path,
            output_path=output_path,
            keyframes=crop_data["keyframes"],
            target_width=crop_data["target_width"],
            target_height=crop_data["target_height"],
            start_time=start_time,
            end_time=start_time + duration,
        )

        return output_path


# Singleton instance
_reframe_service = None


def get_reframe_service() -> ReframeService:
    """Get singleton reframe service."""
    global _reframe_service
    if _reframe_service is None:
        _reframe_service = ReframeService()
    return _reframe_service
