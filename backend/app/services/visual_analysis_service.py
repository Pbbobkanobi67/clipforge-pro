"""Visual analysis service using YOLO."""

import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Check if ultralytics is available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed - visual analysis will be skipped")


class VisualAnalysisService:
    """Service for visual analysis using YOLO object detection."""

    def __init__(self):
        self._model = None
        self._available = ULTRALYTICS_AVAILABLE

    def _get_model(self):
        """Lazy load YOLO model."""
        if not self._available:
            return None

        if self._model is None:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {settings.yolo_model}")
            self._model = YOLO(settings.yolo_model)
            logger.info("YOLO model loaded")

        return self._model

    async def analyze_frame(
        self,
        frame_path: str,
        confidence_threshold: float = 0.5,
    ) -> dict:
        """
        Analyze a single frame for objects.

        Args:
            frame_path: Path to image file
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary with detected objects
        """
        import asyncio

        model = self._get_model()

        # Return empty results if YOLO is not available
        if model is None:
            return {
                "detections": [],
                "object_count": 0,
                "unique_objects": [],
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._analyze_frame_sync(model, frame_path, confidence_threshold),
        )
        return result

    def _analyze_frame_sync(
        self,
        model,
        frame_path: str,
        confidence_threshold: float,
    ) -> dict:
        """Synchronous frame analysis."""
        results = model(frame_path, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf >= confidence_threshold:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    })

        return {
            "detections": detections,
            "object_count": len(detections),
            "unique_objects": list(set(d["class"] for d in detections)),
        }

    async def analyze_video(
        self,
        video_path: str,
        sample_interval: float = 1.0,
        confidence_threshold: float = 0.5,
    ) -> dict:
        """
        Analyze video for objects at regular intervals.

        Args:
            video_path: Path to video file
            sample_interval: Seconds between frame samples
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dictionary with video-wide object analysis
        """
        import asyncio

        model = self._get_model()

        # Return empty results if YOLO is not available
        if model is None:
            logger.info("Visual analysis skipped - ultralytics not installed")
            return {
                "frames_analyzed": 0,
                "total_detections": 0,
                "unique_objects": [],
                "object_frequency": {},
                "key_visual_elements": [],
                "timeline": [],
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._analyze_video_sync(
                model, video_path, sample_interval, confidence_threshold
            ),
        )
        return result

    def _analyze_video_sync(
        self,
        model,
        video_path: str,
        sample_interval: float,
        confidence_threshold: float,
    ) -> dict:
        """Synchronous video analysis."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sample_interval)

        all_detections = []
        object_counts = {}
        frames_analyzed = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                results = model(frame, verbose=False)

                frame_detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = float(box.conf[0])
                        if conf >= confidence_threshold:
                            cls_id = int(box.cls[0])
                            cls_name = model.names[cls_id]

                            frame_detections.append({
                                "class": cls_name,
                                "confidence": conf,
                            })

                            # Count objects
                            object_counts[cls_name] = object_counts.get(cls_name, 0) + 1

                if frame_detections:
                    all_detections.append({
                        "timestamp": timestamp,
                        "detections": frame_detections,
                    })

                frames_analyzed += 1

            frame_idx += 1

        cap.release()

        # Sort objects by frequency
        sorted_objects = sorted(
            object_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Identify key visual elements
        key_elements = self._identify_key_elements(sorted_objects)

        return {
            "frames_analyzed": frames_analyzed,
            "total_detections": sum(object_counts.values()),
            "unique_objects": list(object_counts.keys()),
            "object_frequency": dict(sorted_objects),
            "key_visual_elements": key_elements,
            "timeline": all_detections,
        }

    def _identify_key_elements(self, object_counts: list[tuple]) -> list[dict]:
        """Identify key visual elements based on detection frequency."""
        # Categories of interest for video content
        people_keywords = ["person", "face", "man", "woman", "child"]
        action_keywords = ["sports ball", "skateboard", "surfboard", "bicycle"]
        setting_keywords = ["car", "building", "tree", "beach", "mountain"]

        key_elements = []

        for obj, count in object_counts[:10]:  # Top 10
            element = {"object": obj, "count": count}

            if any(kw in obj.lower() for kw in people_keywords):
                element["category"] = "people"
            elif any(kw in obj.lower() for kw in action_keywords):
                element["category"] = "action"
            elif any(kw in obj.lower() for kw in setting_keywords):
                element["category"] = "setting"
            else:
                element["category"] = "other"

            key_elements.append(element)

        return key_elements

    async def detect_faces(
        self,
        frame_path: str,
    ) -> list[dict]:
        """
        Detect faces in a frame (for speaker/presenter detection).

        Uses YOLO for general detection or can be extended with face-specific models.
        """
        result = await self.analyze_frame(frame_path)
        faces = [d for d in result["detections"] if d["class"] == "person"]
        return faces


# Singleton instance
_visual_service = None


def get_visual_analysis_service() -> VisualAnalysisService:
    """Get singleton visual analysis service."""
    global _visual_service
    if _visual_service is None:
        _visual_service = VisualAnalysisService()
    return _visual_service
