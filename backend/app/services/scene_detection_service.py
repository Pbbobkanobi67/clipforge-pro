"""Scene detection service using OpenCV."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SceneDetectionService:
    """Service for detecting scene boundaries in video."""

    def __init__(self, threshold: float = 30.0):
        """
        Initialize scene detection service.

        Args:
            threshold: Threshold for scene change detection (0-100)
                       Higher = less sensitive, fewer scenes detected
        """
        self.threshold = threshold

    async def detect_scenes(
        self,
        video_path: str,
        min_scene_duration: float = 1.0,
        extract_keyframes: bool = True,
        keyframe_dir: Optional[str] = None,
    ) -> dict:
        """
        Detect scene boundaries in video.

        Args:
            video_path: Path to video file
            min_scene_duration: Minimum scene duration in seconds
            extract_keyframes: Extract representative frame for each scene
            keyframe_dir: Directory to save keyframes

        Returns:
            Dictionary with scene detection results
        """
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._detect_scenes_sync(
                video_path, min_scene_duration, extract_keyframes, keyframe_dir
            ),
        )
        return result

    def _detect_scenes_sync(
        self,
        video_path: str,
        min_scene_duration: float,
        extract_keyframes: bool,
        keyframe_dir: Optional[str],
    ) -> dict:
        """Synchronous scene detection."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        scenes = []
        scene_start = 0.0
        scene_start_frame = 0
        prev_hist = None
        frame_idx = 0

        # Motion tracking
        prev_gray = None
        motion_values = []
        scene_motion = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # Convert to grayscale and resize for efficiency
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (320, 180))

            # Calculate histogram
            hist = cv2.calcHist([small], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Detect scene change via histogram comparison
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                score = (1 - diff) * 100  # Convert to 0-100 scale

                if score > self.threshold:
                    scene_duration = timestamp - scene_start

                    if scene_duration >= min_scene_duration:
                        # Calculate average motion for the scene
                        avg_motion = (
                            np.mean(scene_motion) if scene_motion else 0.0
                        )

                        scene = {
                            "scene_index": len(scenes),
                            "start_time": scene_start,
                            "end_time": timestamp,
                            "duration": scene_duration,
                            "scene_type": self._classify_scene_change(score),
                            "motion_intensity": min(1.0, avg_motion / 50),  # Normalize
                        }

                        # Extract keyframe
                        if extract_keyframes and keyframe_dir:
                            keyframe_path = self._extract_keyframe(
                                cap, scene_start_frame, keyframe_dir, len(scenes)
                            )
                            scene["keyframe_path"] = keyframe_path

                            # Extract dominant colors
                            colors = self._extract_dominant_colors(frame)
                            scene["dominant_colors"] = colors

                        scenes.append(scene)
                        scene_motion = []

                    scene_start = timestamp
                    scene_start_frame = frame_idx

            # Calculate motion (optical flow approximation via frame diff)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, small)
                motion = np.mean(diff)
                motion_values.append(motion)
                scene_motion.append(motion)

            prev_hist = hist
            prev_gray = small
            frame_idx += 1

        # Add final scene
        final_timestamp = frame_idx / fps
        if final_timestamp - scene_start >= min_scene_duration:
            avg_motion = np.mean(scene_motion) if scene_motion else 0.0
            scenes.append({
                "scene_index": len(scenes),
                "start_time": scene_start,
                "end_time": final_timestamp,
                "duration": final_timestamp - scene_start,
                "scene_type": "end",
                "motion_intensity": min(1.0, avg_motion / 50),
            })

        cap.release()

        return {
            "scene_count": len(scenes),
            "total_duration": duration,
            "fps": fps,
            "scenes": scenes,
        }

    def _classify_scene_change(self, score: float) -> str:
        """Classify type of scene change based on score."""
        if score > 70:
            return "hard_cut"
        elif score > 50:
            return "fade"
        else:
            return "dissolve"

    def _extract_keyframe(
        self,
        cap,
        frame_idx: int,
        output_dir: str,
        scene_idx: int,
    ) -> str:
        """Extract and save keyframe for a scene."""
        import cv2

        # Save current position
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Seek to middle of scene
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        output_path = str(Path(output_dir) / f"scene_{scene_idx:04d}.jpg")

        if ret:
            cv2.imwrite(output_path, frame)

        # Restore position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        return output_path

    def _extract_dominant_colors(self, frame, n_colors: int = 3) -> list[str]:
        """Extract dominant colors from frame using k-means clustering."""
        import cv2

        # Resize for faster processing
        small = cv2.resize(frame, (100, 100))
        pixels = small.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Convert to hex colors
        colors = []
        for center in centers:
            b, g, r = center.astype(int)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(hex_color)

        return colors

    async def analyze_motion_intensity(self, video_path: str) -> list[dict]:
        """
        Analyze motion intensity throughout video.

        Returns motion intensity data at regular intervals.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._analyze_motion_sync(video_path)
        )

    def _analyze_motion_sync(self, video_path: str, interval: float = 1.0) -> list[dict]:
        """Synchronous motion analysis."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)

        motion_data = []
        prev_gray = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (320, 180))

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, small)
                    motion = np.mean(diff)
                    motion_data.append({
                        "timestamp": frame_idx / fps,
                        "motion_intensity": min(1.0, motion / 50),
                    })

                prev_gray = small

            frame_idx += 1

        cap.release()
        return motion_data


# Singleton instance
_scene_service = None


def get_scene_detection_service() -> SceneDetectionService:
    """Get singleton scene detection service."""
    global _scene_service
    if _scene_service is None:
        _scene_service = SceneDetectionService()
    return _scene_service
