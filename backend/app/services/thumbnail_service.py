"""Thumbnail generation service with AI scoring."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ThumbnailService:
    """Service for generating and scoring video thumbnails."""

    def __init__(self):
        self._cache_dir = settings.cache_path / "thumbnails"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._visual_service = None

    def _get_visual_service(self):
        """Lazy load visual analysis service."""
        if self._visual_service is None:
            from app.services.visual_analysis_service import get_visual_analysis_service
            self._visual_service = get_visual_analysis_service()
        return self._visual_service

    async def extract_frame(
        self,
        video_path: str,
        timestamp: float,
        output_path: str,
    ) -> str:
        """
        Extract a single frame from video at specified timestamp.

        Args:
            video_path: Source video path
            timestamp: Time in seconds
            output_path: Output image path

        Returns:
            Path to extracted frame
        """
        from app.services.video_service import _run_subprocess

        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",  # High quality JPEG
            "-y",
            output_path,
        ]

        loop = asyncio.get_event_loop()
        returncode, _, stderr = await loop.run_in_executor(
            None, lambda: _run_subprocess(cmd)
        )

        if returncode != 0:
            raise Exception(f"Frame extraction failed: {stderr.decode()}")

        return output_path

    async def analyze_frame(
        self,
        frame_path: str,
    ) -> dict:
        """
        Analyze a frame for thumbnail quality.

        Args:
            frame_path: Path to frame image

        Returns:
            Dictionary with analysis scores
        """
        visual_service = self._get_visual_service()

        # Get YOLO detections
        detection_result = await visual_service.analyze_frame(frame_path)
        detections = detection_result.get("detections", [])

        # Check for faces/persons
        has_person = any(d["class"] == "person" for d in detections)
        person_count = sum(1 for d in detections if d["class"] == "person")

        # Calculate composition score
        composition_score = await self._calculate_composition_score(frame_path, detections)

        # Calculate engagement score (heuristic based on content)
        engagement_score = self._calculate_engagement_score(detections, has_person)

        # Detect face emotion (simplified - would need face analysis model for real emotion)
        face_emotion = None
        if has_person:
            # For now, we just note person detected
            # Full emotion detection would require face-specific model
            face_emotion = "neutral"

        return {
            "has_face": has_person,
            "person_count": person_count,
            "face_emotion": face_emotion,
            "composition_score": composition_score,
            "engagement_score": engagement_score,
            "detections": detections,
        }

    async def _calculate_composition_score(
        self,
        frame_path: str,
        detections: list[dict],
    ) -> float:
        """
        Calculate composition score based on rule of thirds and subject placement.
        """
        try:
            from PIL import Image

            img = Image.open(frame_path)
            width, height = img.size

            # Rule of thirds points
            thirds_x = [width / 3, 2 * width / 3]
            thirds_y = [height / 3, 2 * height / 3]

            score = 50.0  # Base score

            # Check if subjects are near rule of thirds intersections
            for detection in detections:
                if detection["class"] != "person":
                    continue

                bbox = detection["bbox"]
                center_x = (bbox["x1"] + bbox["x2"]) / 2
                center_y = (bbox["y1"] + bbox["y2"]) / 2

                # Calculate distance to nearest third line
                min_dist_x = min(abs(center_x - tx) for tx in thirds_x) / width
                min_dist_y = min(abs(center_y - ty) for ty in thirds_y) / height

                # Closer to thirds = higher score
                thirds_bonus = max(0, (0.2 - min_dist_x) * 50 + (0.2 - min_dist_y) * 50)
                score += thirds_bonus

                # Check subject size (not too small, not too large)
                subject_width = bbox["x2"] - bbox["x1"]
                subject_height = bbox["y2"] - bbox["y1"]
                size_ratio = (subject_width * subject_height) / (width * height)

                # Optimal size is 10-40% of frame
                if 0.1 <= size_ratio <= 0.4:
                    score += 20
                elif 0.05 <= size_ratio <= 0.5:
                    score += 10

            # Clamp score
            return min(100, max(0, score))

        except Exception as e:
            logger.warning(f"Composition analysis failed: {e}")
            return 50.0

    def _calculate_engagement_score(
        self,
        detections: list[dict],
        has_person: bool,
    ) -> float:
        """
        Calculate engagement score based on visual content.
        """
        score = 30.0  # Base score

        # Person presence is highly engaging
        if has_person:
            score += 30

        # Multiple people can be more engaging
        person_count = sum(1 for d in detections if d["class"] == "person")
        if person_count > 1:
            score += min(person_count * 5, 15)

        # Interesting objects
        interesting_objects = ["dog", "cat", "car", "sports ball", "bicycle"]
        for detection in detections:
            if detection["class"] in interesting_objects:
                score += 5

        # High confidence detections suggest clearer image
        high_conf = sum(1 for d in detections if d["confidence"] > 0.8)
        score += min(high_conf * 2, 10)

        return min(100, max(0, score))

    async def generate_thumbnails(
        self,
        video_path: str,
        clip_start: float,
        clip_end: float,
        output_dir: str,
        count: int = 5,
        prefer_faces: bool = True,
        specific_timestamps: Optional[list[float]] = None,
    ) -> list[dict]:
        """
        Generate and score thumbnails for a video clip.

        Args:
            video_path: Source video path
            clip_start: Clip start time
            clip_end: Clip end time
            output_dir: Directory for output images
            count: Number of thumbnails to generate
            prefer_faces: Prioritize frames with faces
            specific_timestamps: Optional specific timestamps to extract

        Returns:
            List of thumbnail data with scores
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        duration = clip_end - clip_start

        # Determine timestamps to sample
        if specific_timestamps:
            timestamps = [t for t in specific_timestamps if clip_start <= t <= clip_end]
        else:
            # Evenly distribute timestamps, avoiding first/last second
            safe_start = clip_start + min(0.5, duration * 0.1)
            safe_end = clip_end - min(0.5, duration * 0.1)
            safe_duration = safe_end - safe_start

            if safe_duration <= 0:
                safe_start = clip_start
                safe_end = clip_end
                safe_duration = duration

            timestamps = [
                safe_start + (safe_duration * i / max(count - 1, 1))
                for i in range(count)
            ]

        # Extract and analyze frames
        thumbnails = []

        for i, timestamp in enumerate(timestamps):
            filename = f"thumb_{i:03d}_{timestamp:.2f}.jpg"
            output_path = str(Path(output_dir) / filename)

            try:
                # Extract frame
                await self.extract_frame(video_path, timestamp, output_path)

                # Analyze frame
                analysis = await self.analyze_frame(output_path)

                thumbnails.append({
                    "timestamp": timestamp,
                    "path": output_path,
                    "has_face": analysis["has_face"],
                    "face_emotion": analysis["face_emotion"],
                    "composition_score": analysis["composition_score"],
                    "engagement_score": analysis["engagement_score"],
                })

            except Exception as e:
                logger.warning(f"Failed to process frame at {timestamp}: {e}")
                continue

        # Sort and rank
        if prefer_faces:
            # Sort by: has_face (desc), engagement_score (desc)
            thumbnails.sort(key=lambda t: (t["has_face"], t["engagement_score"]), reverse=True)
        else:
            # Sort by engagement_score only
            thumbnails.sort(key=lambda t: t["engagement_score"], reverse=True)

        # Assign ranks
        for i, thumb in enumerate(thumbnails):
            thumb["rank"] = i + 1

        logger.info(f"Generated {len(thumbnails)} thumbnails for clip")
        return thumbnails

    async def add_text_overlay(
        self,
        input_path: str,
        output_path: str,
        text: str,
        position: str = "center",
        font_size: int = 48,
        color: str = "#FFFFFF",
        stroke_color: str = "#000000",
        stroke_width: int = 2,
    ) -> str:
        """
        Add text overlay to a thumbnail image.

        Args:
            input_path: Input image path
            output_path: Output image path
            text: Text to overlay
            position: Position (top, center, bottom)
            font_size: Font size
            color: Text color (hex)
            stroke_color: Stroke color (hex)
            stroke_width: Stroke width

        Returns:
            Path to output image
        """
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.open(input_path)
            draw = ImageDraw.Draw(img)

            # Try to load a font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except OSError:
                    font = ImageFont.load_default()

            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate position
            x = (img.width - text_width) // 2  # Center horizontally

            if position == "top":
                y = int(img.height * 0.1)
            elif position == "bottom":
                y = int(img.height * 0.85) - text_height
            else:  # center
                y = (img.height - text_height) // 2

            # Convert colors
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip("#")
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            text_rgb = hex_to_rgb(color)
            stroke_rgb = hex_to_rgb(stroke_color)

            # Draw stroke (text outline)
            if stroke_width > 0:
                for dx in range(-stroke_width, stroke_width + 1):
                    for dy in range(-stroke_width, stroke_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=font, fill=stroke_rgb)

            # Draw main text
            draw.text((x, y), text, font=font, fill=text_rgb)

            # Save
            img.save(output_path, quality=95)

            logger.info(f"Added text overlay to thumbnail: {output_path}")
            return output_path

        except ImportError:
            logger.warning("PIL not available, using ffmpeg for text overlay")
            return await self._add_text_overlay_ffmpeg(
                input_path, output_path, text, position, font_size, color
            )

    async def _add_text_overlay_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        text: str,
        position: str,
        font_size: int,
        color: str,
    ) -> str:
        """Fallback text overlay using ffmpeg."""
        from app.services.video_service import _run_subprocess

        # Calculate y position
        if position == "top":
            y_pos = "h*0.1"
        elif position == "bottom":
            y_pos = "h*0.85-th"
        else:  # center
            y_pos = "(h-th)/2"

        # Remove # from color
        color = color.lstrip("#")

        # Build drawtext filter
        drawtext = (
            f"drawtext=text='{text}':"
            f"fontsize={font_size}:"
            f"fontcolor=0x{color}:"
            f"x=(w-tw)/2:y={y_pos}:"
            f"borderw=2:bordercolor=black"
        )

        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", drawtext,
            "-y",
            output_path,
        ]

        loop = asyncio.get_event_loop()
        returncode, _, stderr = await loop.run_in_executor(
            None, lambda: _run_subprocess(cmd)
        )

        if returncode != 0:
            raise Exception(f"Text overlay failed: {stderr.decode()}")

        return output_path

    async def get_best_thumbnail_timestamp(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_count: int = 10,
    ) -> float:
        """
        Find the best timestamp for a thumbnail in a video segment.

        Args:
            video_path: Source video path
            start_time: Segment start
            end_time: Segment end
            sample_count: Number of frames to sample

        Returns:
            Best timestamp for thumbnail
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            thumbnails = await self.generate_thumbnails(
                video_path=video_path,
                clip_start=start_time,
                clip_end=end_time,
                output_dir=tmp_dir,
                count=sample_count,
                prefer_faces=True,
            )

            if thumbnails:
                return thumbnails[0]["timestamp"]

            # Fallback to middle of clip
            return (start_time + end_time) / 2


# Singleton instance
_thumbnail_service = None


def get_thumbnail_service() -> ThumbnailService:
    """Get singleton thumbnail service."""
    global _thumbnail_service
    if _thumbnail_service is None:
        _thumbnail_service = ThumbnailService()
    return _thumbnail_service
