"""Speaker + Screen split layout service.

Detects PIP webcam regions in screen recordings and generates
portrait 9:16 videos with content on top and speaker on bottom.
Uses YOLO person detection with Haar cascade fallback.
"""

import asyncio
import logging
import statistics
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SplitLayoutService:
    """Analyze screen recordings for PIP webcam and generate split layouts."""

    def __init__(self):
        self._yolo_model = None
        self._face_cascade = None  # Haar fallback

    def _get_yolo_model(self):
        """Lazy-load YOLO model, reusing singleton from VisualAnalysisService."""
        if self._yolo_model is None:
            # Try reusing the existing singleton first
            try:
                from app.services.visual_analysis_service import get_visual_analysis_service
                model = get_visual_analysis_service()._get_model()
                if model is not None:
                    self._yolo_model = model
            except Exception:
                pass
            # Fall back to loading our own
            if self._yolo_model is None:
                try:
                    from ultralytics import YOLO
                    logger.info(f"SplitLayout loading YOLO model: {settings.yolo_model}")
                    self._yolo_model = YOLO(settings.yolo_model)
                except ImportError:
                    logger.warning("YOLO unavailable, falling back to Haar cascade")
        return self._yolo_model

    def _get_face_cascade(self):
        """Lazy-load Haar cascade as fallback when YOLO is unavailable."""
        if self._face_cascade is None:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        return self._face_cascade

    async def analyze_layout(
        self,
        video_path: str,
        sample_interval: float = 0.5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        # Legacy param kept for backward compat (generate_preview passes it)
        num_samples: Optional[int] = None,
    ) -> dict:
        """
        Sample frames and classify the video layout.

        Returns dict with:
            layout_type: "screen_speaker" | "talking_head" | "screen_only"
            source_width, source_height, fps, duration
            pip_region: {x, y, w, h, corner} or None
            content_region: {x, y, w, h} or None
            confidence: float 0-1
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._analyze_layout_sync(video_path, sample_interval, start_time, end_time),
        )

    def _analyze_layout_sync(
        self,
        video_path: str,
        sample_interval: float,
        start_time: Optional[float],
        end_time: Optional[float],
    ) -> dict:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Determine sampling range
        t_start = start_time or 0
        t_end = end_time or duration
        clip_duration = t_end - t_start
        if clip_duration <= 0:
            clip_duration = duration
            t_start = 0

        # Build sample timestamps at sample_interval spacing
        sample_times = []
        t = t_start
        while t < t_end:
            sample_times.append(t)
            t += sample_interval

        num_samples = len(sample_times)
        sample_frame_indices = [int(t * fps) for t in sample_times]
        sample_frame_indices = [min(f, total_frames - 1) for f in sample_frame_indices if f < total_frames]

        # Pick 3 evenly-spaced frames for content edge analysis
        edge_indices = set()
        if len(sample_frame_indices) >= 3:
            for i in [0, len(sample_frame_indices) // 2, len(sample_frame_indices) - 1]:
                edge_indices.add(sample_frame_indices[i])
        elif sample_frame_indices:
            edge_indices = set(sample_frame_indices[:3])

        yolo_model = self._get_yolo_model()
        use_yolo = yolo_model is not None

        all_persons = []  # list of (x1, y1, x2, y2)
        edge_frames = []  # frames for content region detection
        frames_with_detections = 0

        # Upscale low-res frames for better YOLO detection
        need_upscale = source_height < 720
        if need_upscale:
            scale_factor = 720.0 / source_height
            upscale_w = int(source_width * scale_factor)
            upscale_h = 720
            logger.info(f"SplitLayout: Low-res source ({source_width}x{source_height}), upscaling to {upscale_w}x{upscale_h} for detection")
        else:
            scale_factor = 1.0

        if use_yolo:
            logger.info(f"SplitLayout: YOLO person detection on {len(sample_frame_indices)} frames")
            for frame_idx in sample_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Collect edge analysis frames (at original resolution)
                if frame_idx in edge_indices:
                    edge_frames.append(frame.copy())

                # Upscale for YOLO if needed
                detect_frame = frame
                if need_upscale:
                    detect_frame = cv2.resize(frame, (upscale_w, upscale_h), interpolation=cv2.INTER_LINEAR)

                # Run YOLO
                results = yolo_model(detect_frame, verbose=False)
                frame_persons = []
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = yolo_model.names[cls_id]
                        if cls_name == "person" and conf >= 0.4:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            # Scale coords back to original resolution
                            if need_upscale:
                                x1 /= scale_factor
                                y1 /= scale_factor
                                x2 /= scale_factor
                                y2 /= scale_factor
                            frame_persons.append((x1, y1, x2, y2))

                if frame_persons:
                    frames_with_detections += 1
                    all_persons.extend(frame_persons)
        else:
            # Haar cascade fallback
            logger.info(f"SplitLayout: Haar cascade fallback on {len(sample_frame_indices)} frames")
            cascade = self._get_face_cascade()
            for frame_idx in sample_frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                if frame_idx in edge_indices:
                    edge_frames.append(frame.copy())

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(int(source_width * 0.02), int(source_height * 0.02)),
                )
                if len(faces) > 0:
                    frames_with_detections += 1
                    for (fx, fy, fw, fh) in faces:
                        # Convert (x, y, w, h) to (x1, y1, x2, y2) for unified format
                        all_persons.append((fx, fy, fx + fw, fy + fh))

        cap.release()

        # Detect content region (browser chrome removal)
        content_region = self._detect_content_region(edge_frames, source_width, source_height)

        # Classify the layout
        classification = self._classify_layout(
            all_persons, source_width, source_height, num_samples,
            frames_with_detections, content_region,
        )

        # Compute suggested split ratio based on layout type and PIP size
        suggested_ratio = self._suggest_split_ratio(
            classification["layout_type"],
            classification.get("pip_region"),
            source_width,
            source_height,
        )

        return {
            "layout_type": classification["layout_type"],
            "source_width": source_width,
            "source_height": source_height,
            "fps": fps,
            "duration": duration,
            "pip_region": classification.get("pip_region"),
            "content_region": classification.get("content_region"),
            "speaker_regions": classification.get("speaker_regions"),
            "confidence": classification["confidence"],
            "suggested_split_ratio": suggested_ratio,
        }

    def _detect_content_region(
        self,
        frames: list,
        frame_w: int,
        frame_h: int,
    ) -> dict:
        """
        Detect the content viewport by finding browser chrome and taskbar edges.

        Uses horizontal Sobel edge detection on sample frames to find:
        - Last strong horizontal edge in top 15% -> browser chrome bottom
        - First strong horizontal edge in bottom 10% -> taskbar top
        """
        import cv2

        if not frames:
            return {"x": 0, "y": 0, "w": frame_w, "h": frame_h}

        # Skip content region detection for low-res video — edges are unreliable
        if frame_h < 720:
            logger.info(f"Content region: skipping edge detection for low-res video ({frame_w}x{frame_h})")
            return {"x": 0, "y": 0, "w": frame_w, "h": frame_h}

        top_edges = []
        bottom_edges = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Horizontal Sobel (detects horizontal lines)
            sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_h = np.abs(sobel_h)

            # Compute row-wise edge strength (sum across width)
            row_strength = sobel_h.mean(axis=1)

            # Dynamic threshold: strong edges are > 2x the median
            threshold = np.median(row_strength) * 2.5
            if threshold < 10:
                threshold = 10

            # Search top 15% for browser chrome bottom edge
            top_limit = int(frame_h * 0.15)
            top_region = row_strength[:top_limit]
            strong_top = np.where(top_region > threshold)[0]
            if len(strong_top) > 0:
                # Last strong edge in top region = bottom of chrome
                top_edges.append(int(strong_top[-1]))

            # Search bottom 10% for taskbar top edge
            bottom_start = int(frame_h * 0.90)
            bottom_region = row_strength[bottom_start:]
            strong_bottom = np.where(bottom_region > threshold)[0]
            if len(strong_bottom) > 0:
                # First strong edge in bottom region = top of taskbar
                bottom_edges.append(bottom_start + int(strong_bottom[0]))

        # Use median across frames for stability
        crop_top = 0
        crop_bottom = frame_h

        if top_edges:
            chrome_bottom = int(statistics.median(top_edges))
            logger.info(f"Content region: detected chrome bottom edge at y={chrome_bottom}")
            # Safety: max 8% of frame height from top (scales with resolution)
            max_top_crop = max(120, int(frame_h * 0.08))
            chrome_bottom = min(chrome_bottom, max_top_crop)
            # Only crop if the edge is significant (> 15px from top)
            if chrome_bottom > 15:
                crop_top = chrome_bottom

        if bottom_edges:
            taskbar_top = int(statistics.median(bottom_edges))
            logger.info(f"Content region: detected taskbar top edge at y={taskbar_top}")
            # Safety: max 5% of frame height from bottom
            max_bottom_crop = max(60, int(frame_h * 0.05))
            taskbar_top = max(taskbar_top, frame_h - max_bottom_crop)
            # Only crop if the edge is significant (> 15px from bottom)
            if frame_h - taskbar_top > 15:
                crop_bottom = taskbar_top

        content_h = crop_bottom - crop_top
        if content_h < frame_h * 0.70:
            # Safety: if we'd crop more than 30%, something is wrong — use full frame
            logger.warning(f"Content region detection too aggressive ({content_h}px of {frame_h}px), using full frame")
            return {"x": 0, "y": 0, "w": frame_w, "h": frame_h}

        # Ensure even dimensions
        crop_top = (crop_top // 2) * 2
        content_h = ((crop_bottom - crop_top) // 2) * 2

        logger.info(f"Content region: crop_top={crop_top}, crop_bottom={crop_bottom}, content_h={content_h} (frame_h={frame_h})")

        return {
            "x": 0,
            "y": crop_top,
            "w": frame_w,
            "h": content_h,
        }

    def _classify_layout(
        self,
        persons: list[tuple],
        frame_w: int,
        frame_h: int,
        num_samples: int,
        frames_with_detections: int,
        content_region: dict,
    ) -> dict:
        """
        Classify video layout based on detected persons.

        Persons are (x1, y1, x2, y2) tuples.

        Decision logic:
        - Small person region in a corner (3-30% of frame width), consistent -> screen_speaker
        - Large person (>35% width), centered -> talking_head
        - No detections -> screen_only
        """
        if not persons:
            return {"layout_type": "screen_only", "confidence": 0.5, "content_region": content_region}

        # Compute median person metrics
        person_widths = [x2 - x1 for (x1, _, x2, _) in persons]
        person_centers_x = [(x1 + x2) / 2 for (x1, _, x2, _) in persons]
        person_centers_y = [(y1 + y2) / 2 for (_, y1, _, y2) in persons]

        median_width = statistics.median(person_widths)
        median_cx = statistics.median(person_centers_x)
        median_cy = statistics.median(person_centers_y)

        width_ratio = median_width / frame_w
        detection_rate = frames_with_detections / max(num_samples, 1)

        # Determine corner
        corner = self._detect_corner(median_cx, median_cy, frame_w, frame_h)

        logger.info(
            f"Layout classify: {len(persons)} detections, median_width={median_width:.0f}, "
            f"width_ratio={width_ratio:.3f}, detection_rate={detection_rate:.2f}, corner={corner}"
        )

        # Decision tree
        if width_ratio < 0.03:
            return {"layout_type": "screen_only", "confidence": 0.4, "content_region": content_region}

        if 0.03 <= width_ratio <= 0.30 and corner != "center" and detection_rate >= 0.15:
            # Small person in a corner/edge = PIP webcam.
            # Low threshold (0.15) because PIP overlays may not be visible in every frame
            # and low-res upscaling can miss detections.
            pip_region = self._estimate_pip_region_from_persons(persons, frame_w, frame_h, corner)
            return {
                "layout_type": "screen_speaker",
                "pip_region": pip_region,
                "content_region": content_region,
                "confidence": min(detection_rate, 1.0),
            }

        if width_ratio > 0.35:
            speaker_regions = self._cluster_speakers_for_talking_head(persons, frame_w, frame_h)
            return {
                "layout_type": "talking_head",
                "confidence": min(detection_rate, 1.0),
                "speaker_regions": speaker_regions,
            }

        # Medium size or centered — ambiguous
        if detection_rate >= 0.15:
            pip_region = self._estimate_pip_region_from_persons(persons, frame_w, frame_h, corner)
            return {
                "layout_type": "screen_speaker",
                "pip_region": pip_region,
                "content_region": content_region,
                "confidence": min(detection_rate * 0.7, 1.0),
            }

        return {"layout_type": "screen_only", "confidence": 0.3, "content_region": content_region}

    def _detect_corner(self, cx: float, cy: float, frame_w: int, frame_h: int) -> str:
        """Detect which corner/edge the center point is in."""
        margin = 0.40  # must be in outer 40% to count as an edge

        in_left = cx < frame_w * margin
        in_right = cx > frame_w * (1 - margin)
        in_top = cy < frame_h * margin
        in_bottom = cy > frame_h * (1 - margin)

        if in_top and in_left:
            return "top_left"
        if in_top and in_right:
            return "top_right"
        if in_bottom and in_left:
            return "bottom_left"
        if in_bottom and in_right:
            return "bottom_right"
        # Allow edge detection even if not in a strict corner
        if in_left:
            return "bottom_left"  # default to bottom-left for left-edge PIP
        if in_right:
            return "bottom_right"  # default to bottom-right for right-edge PIP
        if in_bottom:
            return "bottom_left"  # bottom center → snap left
        if in_top:
            return "top_left"  # top center → snap left
        return "center"

    def _estimate_pip_region_from_persons(
        self,
        persons: list[tuple],
        frame_w: int,
        frame_h: int,
        corner: str,
    ) -> dict:
        """
        Estimate the PIP webcam region from person detections.

        Multi-person aware: computes union bounding box of all detections,
        uses percentile filtering for outlier robustness.
        """
        if not persons:
            return {"x": 0, "y": 0, "w": frame_w, "h": frame_h, "corner": corner}

        x1s = [p[0] for p in persons]
        y1s = [p[1] for p in persons]
        x2s = [p[2] for p in persons]
        y2s = [p[3] for p in persons]

        # Use 10th/90th percentile to filter outlier detections
        p10 = 10
        p90 = 90
        union_x1 = float(np.percentile(x1s, p10))
        union_y1 = float(np.percentile(y1s, p10))
        union_x2 = float(np.percentile(x2s, p90))
        union_y2 = float(np.percentile(y2s, p90))

        # Compute the raw person region
        person_w = union_x2 - union_x1
        person_h = union_y2 - union_y1
        person_cx = (union_x1 + union_x2) / 2
        person_cy = (union_y1 + union_y2) / 2

        # Add padding: 15% headroom above, 10% sides, 5% bottom
        pad_top = person_h * 0.15
        pad_side = person_w * 0.10
        pad_bottom = person_h * 0.05

        pip_x1 = person_cx - person_w / 2 - pad_side
        pip_y1 = person_cy - person_h / 2 - pad_top
        pip_x2 = person_cx + person_w / 2 + pad_side
        pip_y2 = person_cy + person_h / 2 + pad_bottom

        pip_w = pip_x2 - pip_x1
        pip_h = pip_y2 - pip_y1

        # Enforce minimum 15% of frame size
        min_w = frame_w * 0.15
        min_h = frame_h * 0.15
        if pip_w < min_w:
            expand = (min_w - pip_w) / 2
            pip_x1 -= expand
            pip_x2 += expand
            pip_w = min_w
        if pip_h < min_h:
            expand = (min_h - pip_h) / 2
            pip_y1 -= expand
            pip_y2 += expand
            pip_h = min_h

        pip_x = int(pip_x1)
        pip_y = int(pip_y1)
        pip_w = int(pip_w)
        pip_h = int(pip_h)

        # Snap to edge ONLY if the person center is actually near that edge.
        # This prevents snapping detections that happen to be classified
        # into a corner but are actually in the mid-frame.
        edge_margin = int(frame_w * 0.02)
        near_threshold = 0.35  # person center must be within outer 35% to snap

        if "left" in corner and person_cx < frame_w * near_threshold:
            pip_x = max(0, min(pip_x, edge_margin))
        elif "right" in corner and person_cx > frame_w * (1 - near_threshold):
            pip_x = max(frame_w - pip_w - edge_margin, pip_x)

        if "top" in corner and person_cy < frame_h * near_threshold:
            pip_y = max(0, min(pip_y, edge_margin))
        elif "bottom" in corner and person_cy > frame_h * (1 - near_threshold):
            pip_y = max(frame_h - pip_h - edge_margin, pip_y)

        # Clamp to frame
        pip_x = max(0, min(pip_x, frame_w - pip_w))
        pip_y = max(0, min(pip_y, frame_h - pip_h))
        pip_w = min(pip_w, frame_w - pip_x)
        pip_h = min(pip_h, frame_h - pip_y)

        logger.info(
            f"PIP region: persons={len(persons)}, union=({union_x1:.0f},{union_y1:.0f})-({union_x2:.0f},{union_y2:.0f}), "
            f"pip=({pip_x},{pip_y},{pip_w},{pip_h}), corner={corner}, "
            f"frame=({frame_w},{frame_h}), ratio=({pip_w/frame_w:.2f},{pip_h/frame_h:.2f})"
        )

        return {
            "x": pip_x,
            "y": pip_y,
            "w": pip_w,
            "h": pip_h,
            "corner": corner,
        }

    def _cluster_speakers_for_talking_head(
        self,
        persons: list[tuple],
        frame_w: int,
        frame_h: int,
    ) -> list[dict]:
        """
        Cluster person detections into 1 or 2 speaker regions for talking_head layouts.

        Splits persons into left/right groups by X position using the frame midpoint.
        Returns a list of speaker regions sorted left-to-right:
            [{"x", "y", "w", "h", "cx"}, ...]

        If only one cluster has data (single speaker), returns a 1-element list.
        Each region is padded (15% headroom, 10% sides) and clamped to the frame.
        """
        if not persons:
            return []

        mid_x = frame_w / 2
        left_group = []
        right_group = []
        for p in persons:
            x1, y1, x2, y2 = p
            cx = (x1 + x2) / 2
            if cx < mid_x:
                left_group.append(p)
            else:
                right_group.append(p)

        def _region_from_group(group: list[tuple]) -> Optional[dict]:
            if len(group) < 3:  # need at least a few detections for stability
                return None
            x1s = [p[0] for p in group]
            y1s = [p[1] for p in group]
            x2s = [p[2] for p in group]
            y2s = [p[3] for p in group]
            # 10th/90th percentile filtering for outlier robustness
            rx1 = float(np.percentile(x1s, 10))
            ry1 = float(np.percentile(y1s, 10))
            rx2 = float(np.percentile(x2s, 90))
            ry2 = float(np.percentile(y2s, 90))
            raw_w = rx2 - rx1
            raw_h = ry2 - ry1
            if raw_w <= 0 or raw_h <= 0:
                return None
            # Padding: 18% headroom above, 8% sides, 5% bottom
            pad_top = raw_h * 0.18
            pad_side = raw_w * 0.08
            pad_bottom = raw_h * 0.05
            rx1 -= pad_side
            ry1 -= pad_top
            rx2 += pad_side
            ry2 += pad_bottom
            # Clamp to frame
            rx1 = max(0, rx1)
            ry1 = max(0, ry1)
            rx2 = min(frame_w, rx2)
            ry2 = min(frame_h, ry2)
            x = int(rx1)
            y = int(ry1)
            w = int(rx2 - rx1)
            h = int(ry2 - ry1)
            # Ensure even dimensions for FFmpeg
            w = (w // 2) * 2
            h = (h // 2) * 2
            if w < 16 or h < 16:
                return None
            return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2}

        regions = []
        left_region = _region_from_group(left_group)
        right_region = _region_from_group(right_group)
        if left_region:
            regions.append(left_region)
        if right_region:
            regions.append(right_region)

        # If clustering failed (e.g., all persons on one side), fall back to
        # a single union region from all detections.
        if not regions and persons:
            fallback = _region_from_group(persons)
            if fallback:
                regions.append(fallback)

        # Sort left-to-right
        regions.sort(key=lambda r: r["cx"])

        logger.info(
            f"Speaker cluster: {len(persons)} detections → {len(regions)} speaker(s): "
            + ", ".join(f"({r['x']},{r['y']},{r['w']}x{r['h']})" for r in regions)
        )
        return regions

    def _suggest_split_ratio(
        self,
        layout_type: str,
        pip_region: Optional[dict],
        frame_w: int,
        frame_h: int,
    ) -> float:
        """
        Suggest an optimal split ratio based on layout type and PIP size.

        Returns a float between 0.5 and 0.8:
        - screen_speaker with small PIP: 0.58 (more content)
        - screen_speaker with large PIP: 0.50 (balanced, more speaker)
        - talking_head: 0.35 (speaker is the focus, top panel is blurred BG)
        - screen_only: 0.72 (all content, bottom is blurred filler)
        """
        if layout_type == "talking_head":
            return 0.35

        if layout_type == "screen_only":
            return 0.72

        # screen_speaker: adjust based on PIP size relative to frame
        if pip_region:
            pip_area = pip_region.get("w", 0) * pip_region.get("h", 0)
            frame_area = frame_w * frame_h
            pip_ratio = pip_area / max(frame_area, 1)

            if pip_ratio < 0.04:
                # Tiny PIP (< 4% of frame) — content-heavy
                return 0.60
            elif pip_ratio < 0.10:
                # Small PIP — slightly more content
                return 0.55
            elif pip_ratio < 0.20:
                # Medium PIP — balanced
                return 0.52
            else:
                # Large PIP — give speaker more space
                return 0.50

        return 0.55  # default

    async def generate_split_video(
        self,
        video_path: str,
        output_path: str,
        layout_analysis: dict,
        split_ratio: float = 0.55,
        separator_height: int = 4,
        separator_color: str = "#333333",
        target_width: int = 1080,
        target_height: int = 1920,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        ass_subtitle_path: Optional[str] = None,
        dual_speaker_mode: bool = False,
        custom_top_crop: Optional[dict] = None,
        custom_bottom_crop: Optional[dict] = None,
    ) -> str:
        """
        Generate a portrait split video: content on top, speaker on bottom.

        Args:
            video_path: Source video
            output_path: Output path
            layout_analysis: Result from analyze_layout()
            split_ratio: Fraction for content panel (0.5-0.8)
            separator_height: Height of separator line in pixels
            separator_color: Hex color of separator
            target_width: Output width (default 1080)
            target_height: Output height (default 1920)
            start_time: Clip start
            end_time: Clip end
            ass_subtitle_path: Optional path to ASS subtitle file to burn in

        Returns:
            Output path
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_split_video_sync(
                video_path, output_path, layout_analysis,
                split_ratio, separator_height, separator_color,
                target_width, target_height, start_time, end_time,
                ass_subtitle_path, dual_speaker_mode,
                custom_top_crop, custom_bottom_crop,
            ),
        )

    def _generate_split_video_sync(
        self,
        video_path: str,
        output_path: str,
        layout_analysis: dict,
        split_ratio: float,
        separator_height: int,
        separator_color: str,
        target_width: int,
        target_height: int,
        start_time: Optional[float],
        end_time: Optional[float],
        ass_subtitle_path: Optional[str] = None,
        dual_speaker_mode: bool = False,
        custom_top_crop: Optional[dict] = None,
        custom_bottom_crop: Optional[dict] = None,
    ) -> str:
        from app.services.video_service import _run_subprocess

        layout_type = layout_analysis.get("layout_type", "screen_only")
        source_w = layout_analysis["source_width"]
        source_h = layout_analysis["source_height"]
        fps = layout_analysis.get("fps", 30)

        # Calculate panel heights
        content_h = int((target_height - separator_height) * split_ratio)
        speaker_h = target_height - separator_height - content_h

        # Ensure even dimensions
        content_h = (content_h // 2) * 2
        speaker_h = (speaker_h // 2) * 2

        # Calculate duration for color source
        dur = (end_time or layout_analysis.get("duration", 60)) - (start_time or 0)

        # Convert hex color to FFmpeg format (0xRRGGBB)
        sep_color = separator_color.lstrip("#")
        ffmpeg_color = f"0x{sep_color}"

        # Content region from analysis (may crop browser chrome/taskbar)
        content_region = layout_analysis.get("content_region")

        # Create modified ASS file for split layout dimensions if needed
        effective_ass_path = ass_subtitle_path
        if ass_subtitle_path and Path(ass_subtitle_path).exists():
            split_ass_path = ass_subtitle_path.replace(".ass", "_split_adjusted.ass")
            try:
                effective_ass_path = self._create_split_ass(
                    ass_subtitle_path, split_ass_path, target_width, target_height,
                    content_panel_h=content_h,
                )
            except Exception as e:
                logger.warning(f"Failed to create split ASS, using original: {e}")
                effective_ass_path = ass_subtitle_path

        # Custom user-positioned crops take precedence over all auto layouts.
        # Both must be provided to engage manual mode.
        def _validate_crop(c):
            if not c:
                return None
            try:
                x = max(0, int(c.get("x", 0)))
                y = max(0, int(c.get("y", 0)))
                w = int(c.get("w", 0))
                h = int(c.get("h", 0))
                if w < 16 or h < 16:
                    return None
                # Clamp to source
                w = min(w, source_w - x)
                h = min(h, source_h - y)
                w = (w // 2) * 2
                h = (h // 2) * 2
                if w < 16 or h < 16:
                    return None
                return {"x": x, "y": y, "w": w, "h": h}
            except (TypeError, ValueError):
                return None

        valid_top = _validate_crop(custom_top_crop)
        valid_bot = _validate_crop(custom_bottom_crop)

        if valid_top and valid_bot:
            logger.info(
                f"Custom split: top_crop={valid_top}, bottom_crop={valid_bot}"
            )
            # Reuse the dual-speaker filter path - it accepts arbitrary crop boxes.
            speaker_regions = [
                {**valid_top, "cx": valid_top["x"] + valid_top["w"] / 2},
                {**valid_bot, "cx": valid_bot["x"] + valid_bot["w"] / 2},
            ]
            filter_complex = self._build_talking_head_filter(
                source_w, source_h,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps,
                speaker_regions=speaker_regions,
                ass_subtitle_path=effective_ass_path,
            )
        elif layout_type == "screen_speaker" and layout_analysis.get("pip_region"):
            pip = layout_analysis["pip_region"]
            filter_complex = self._build_screen_speaker_filter(
                source_w, source_h, pip,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps, content_region,
                ass_subtitle_path=effective_ass_path,
            )
        elif layout_type == "talking_head":
            # speaker_regions is only used when user opts into dual_speaker_mode.
            # Default behavior keeps the full frame fit-with-blur in both panels.
            speaker_regions = layout_analysis.get("speaker_regions") if dual_speaker_mode else None
            filter_complex = self._build_talking_head_filter(
                source_w, source_h,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps,
                speaker_regions=speaker_regions,
                ass_subtitle_path=effective_ass_path,
            )
        else:
            filter_complex = self._build_screen_only_filter(
                source_w, source_h,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps, content_region,
                ass_subtitle_path=effective_ass_path,
            )

        # Build ffmpeg command
        cmd = ["ffmpeg"]

        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])

        cmd.extend(["-i", video_path])

        if end_time is not None:
            duration = end_time - (start_time or 0)
            cmd.extend(["-t", str(duration)])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-shortest",
            "-y",
            output_path,
        ])

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        returncode, stdout, stderr = _run_subprocess(cmd)
        if returncode != 0:
            error_msg = stderr.decode(errors="replace")
            logger.error(f"Split layout FFmpeg failed: {error_msg}")
            raise RuntimeError(f"Split layout generation failed: {error_msg[-500:]}")

        logger.info(f"Generated split layout video: {output_path}")
        return output_path

    @staticmethod
    def _create_split_ass(
        original_ass_path: str,
        output_ass_path: str,
        target_width: int = 1080,
        target_height: int = 1920,
        content_panel_h: int = 0,
    ) -> str:
        """
        Create a modified ASS subtitle file for split layout output.

        The original ASS is authored at PlayResX:1920, PlayResY:1080.
        The split layout output is 1080x1920. We adjust:
        1. PlayRes to match output dimensions
        2. Font size scaled down for narrower output
        3. MarginV set so captions appear at the bottom of the CONTENT panel
           (above the separator), not at the bottom of the full frame
        """
        import re

        with open(original_ass_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update PlayRes to match split output dimensions
        content = re.sub(r"PlayResX:\s*\d+", f"PlayResX: {target_width}", content)
        content = re.sub(r"PlayResY:\s*\d+", f"PlayResY: {target_height}", content)

        # Scale font size down: original designed for 1920px wide, now 1080px wide
        # Scale factor: 1080/1920 = 0.5625, but we want readable text so use ~0.70
        scale = 0.70

        # Scale font sizes in Style definitions
        content = re.sub(
            r"(Style:\s*\w+),([^,]+),(\d+),",
            lambda m: f"{m.group(1)},{m.group(2)},{max(24, int(int(m.group(3)) * scale))},",
            content,
        )

        # Position captions at the bottom of the CONTENT panel.
        # In ASS with Alignment=2 (bottom center), MarginV = distance from bottom edge.
        # We want captions ~30px above the separator line.
        # Separator is at y = content_panel_h, so distance from bottom =
        #   target_height - content_panel_h + 30
        if content_panel_h > 0:
            margin_v = target_height - content_panel_h + 30
        else:
            # Fallback: place at roughly the middle of the frame
            margin_v = target_height // 2

        content = re.sub(
            r"(Style:[^\n]*,)(\d+)(,\d+\s*$)",
            lambda m: f"{m.group(1)}{margin_v}{m.group(3)}",
            content,
            flags=re.MULTILINE,
        )

        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(
            f"Created split layout ASS: {output_ass_path} "
            f"(PlayRes: {target_width}x{target_height}, MarginV: {margin_v})"
        )
        return output_ass_path

    @staticmethod
    def _escape_ass_path_for_ffmpeg(ass_path: str) -> str:
        """Escape an ASS subtitle file path for use in ffmpeg filter chains.

        On Windows, ffmpeg's ass filter requires:
        - Forward slashes instead of backslashes
        - Colons escaped with backslash (e.g. C\\:/path)
        - Single quotes around the path in the filter expression
        """
        escaped = ass_path.replace("\\", "/")
        escaped = escaped.replace(":", "\\:")
        return escaped

    def _build_screen_speaker_filter(
        self,
        source_w: int, source_h: int,
        pip: dict,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
        content_region: Optional[dict] = None,
        ass_subtitle_path: Optional[str] = None,
    ) -> str:
        """Build filter for screen + speaker PIP layout.

        Content panel: FIT the full screen content (scale to width, blur-fill gaps).
        Speaker panel: TIGHT crop around PIP webcam, scaled up to fill.
        """
        pip_x = pip["x"]
        pip_y = pip["y"]
        pip_w = pip["w"]
        pip_h = pip["h"]

        # Content region (may exclude browser chrome/taskbar)
        if content_region and (content_region.get("y", 0) > 0 or content_region.get("h", source_h) < source_h):
            cr_x = content_region["x"]
            cr_y = content_region["y"]
            cr_w = content_region["w"]
            cr_h = content_region["h"]
        else:
            cr_x, cr_y, cr_w, cr_h = 0, 0, source_w, source_h

        # --- Content panel: FIT full content (no cropping!), blur-fill background ---
        # Scale content to fit panel width, preserving aspect ratio
        content_scale = target_width / cr_w
        scaled_ch = int(cr_h * content_scale)
        # Ensure even
        scaled_ch = (scaled_ch // 2) * 2

        # Does the scaled content fit the panel height?
        if scaled_ch <= content_h:
            # Content is shorter than panel → scale to width + blur-fill background
            # Vertical padding needed
            pad_y = (content_h - scaled_ch) // 2
            use_fit_mode = True
            logger.info(
                f"Content FIT: source=({cr_w}x{cr_h}), scaled=({target_width}x{scaled_ch}), "
                f"panel=({target_width}x{content_h}), pad_y={pad_y}"
            )
        else:
            # Content is taller than panel → scale to height instead
            use_fit_mode = False
            content_scale_h = content_h / cr_h
            scaled_cw = int(cr_w * content_scale_h)
            scaled_cw = (scaled_cw // 2) * 2
            logger.info(
                f"Content FILL-height: source=({cr_w}x{cr_h}), scaled=({scaled_cw}x{content_h}), "
                f"panel=({target_width}x{content_h})"
            )

        # --- Speaker panel: TIGHT crop around PIP webcam, FIT with blur bg ---
        # Goal: show just the webcam faces WITHOUT bleeding screen content.
        # Crop EXACTLY the PIP (+ small pad), then fit into panel with blur background.
        # This avoids the widen-to-match-aspect hack that captured content next to the PIP.
        speaker_aspect = target_width / speaker_h

        # Tight crop: PIP box + 10% padding on each side (small breathing room)
        pad_factor = 0.10
        crop_sw = int(pip_w * (1 + pad_factor * 2))
        crop_sh = int(pip_h * (1 + pad_factor * 2))

        # Enforce minimum crop (avoid pixelated upscale of tiny PIPs)
        min_crop = int(min(source_w, source_h) * 0.15)
        crop_sw = max(crop_sw, min_crop)
        crop_sh = max(crop_sh, min_crop)

        # Ensure even dimensions
        crop_sw = max((crop_sw // 2) * 2, 2)
        crop_sh = max((crop_sh // 2) * 2, 2)

        # Center crop on PIP region
        pip_cx = pip_x + pip_w // 2
        pip_cy = pip_y + pip_h // 2
        crop_sx = pip_cx - crop_sw // 2
        crop_sy = pip_cy - crop_sh // 2

        # Clamp to frame bounds
        crop_sx = max(0, min(crop_sx, source_w - crop_sw))
        crop_sy = max(0, min(crop_sy, source_h - crop_sh))

        # FIT crop into speaker panel preserving aspect ratio
        crop_aspect = crop_sw / max(crop_sh, 1)
        if crop_aspect >= speaker_aspect:
            # Crop is wider → fit to panel width, pad top/bottom
            speaker_scaled_w = target_width
            speaker_scaled_h = int(crop_sh * (target_width / crop_sw))
            speaker_scaled_h = (speaker_scaled_h // 2) * 2
            speaker_pad_x = 0
            speaker_pad_y = (speaker_h - speaker_scaled_h) // 2
        else:
            # Crop is taller → fit to panel height, pad left/right
            speaker_scaled_h = speaker_h
            speaker_scaled_w = int(crop_sw * (speaker_h / crop_sh))
            speaker_scaled_w = (speaker_scaled_w // 2) * 2
            speaker_pad_x = (target_width - speaker_scaled_w) // 2
            speaker_pad_y = 0

        logger.info(
            f"Speaker FIT: pip=({pip_x},{pip_y},{pip_w},{pip_h}), "
            f"crop=({crop_sw}x{crop_sh}) at ({crop_sx},{crop_sy}), "
            f"scaled=({speaker_scaled_w}x{speaker_scaled_h}), pad=({speaker_pad_x},{speaker_pad_y})"
        )

        # --- Build filter chain ---
        # Both panels use the same technique: crop → scale → overlay on blurred background.
        # Each panel needs a foreground+background pair, so we split the source 4 ways:
        #   [cfr] = content foreground (sharp)
        #   [cbr] = content background (blurred)
        #   [sfr] = speaker foreground (sharp)
        #   [sbr] = speaker background (blurred)
        if use_fit_mode:
            content_fg_filter = (
                f"[cfr]crop={cr_w}:{cr_h}:{cr_x}:{cr_y},"
                f"scale={target_width}:{scaled_ch}:flags=lanczos[content_fg]"
            )
            content_overlay = f"[content_bg][content_fg]overlay=0:{pad_y}[content]"
        else:
            pad_x_content = max(0, (target_width - scaled_cw) // 2)
            content_fg_filter = (
                f"[cfr]crop={cr_w}:{cr_h}:{cr_x}:{cr_y},"
                f"scale={scaled_cw}:{content_h}:flags=lanczos[content_fg]"
            )
            content_overlay = f"[content_bg][content_fg]overlay={pad_x_content}:0[content]"

        filter_parts = [
            f"[0:v]split=4[cfr][cbr][sfr][sbr]",
            # Content BG: blurred full screen scaled to panel
            f"[cbr]crop={cr_w}:{cr_h}:{cr_x}:{cr_y},"
            f"scale={target_width}:{content_h}:flags=fast_bilinear,"
            f"boxblur=30:15[content_bg]",
            # Content FG: sharp content
            content_fg_filter,
            # Composite content panel
            content_overlay,
            # Speaker BG: blurred TIGHT crop of PIP area, scaled to fill panel
            # (Same crop as foreground, just blurred and filling the whole panel)
            f"[sbr]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},"
            f"scale={target_width}:{speaker_h}:flags=fast_bilinear,"
            f"boxblur=25:12[speaker_bg]",
            # Speaker FG: sharp tight crop of PIP, fit with letterbox
            f"[sfr]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},"
            f"scale={speaker_scaled_w}:{speaker_scaled_h}:flags=lanczos[speaker_fg]",
            # Composite speaker panel
            f"[speaker_bg][speaker_fg]overlay={speaker_pad_x}:{speaker_pad_y}[speaker]",
        ]

        # Stack panels with separator
        vstack_label = "[stacked]" if ass_subtitle_path else "[outv]"

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3{vstack_label}")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2{vstack_label}")

        # Burn in ASS subtitles on top of the composited split layout
        if ass_subtitle_path:
            escaped_path = self._escape_ass_path_for_ffmpeg(ass_subtitle_path)
            filter_parts.append(f"[stacked]ass='{escaped_path}'[outv]")

        return ";".join(filter_parts)

    def _build_talking_head_filter(
        self,
        source_w: int, source_h: int,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
        speaker_regions: Optional[list] = None,
        ass_subtitle_path: Optional[str] = None,
    ) -> str:
        """Build filter for talking head (no separate screen content).

        Two modes:
        1. DUAL-SPEAKER (len(speaker_regions) == 2): crop the left speaker into
           the top panel and the right speaker into the bottom panel, each fit
           with a blurred background of the full frame. Produces the classic
           OpusClip-style vertical interview layout.
        2. FALLBACK: both panels show the full source frame fit with blur
           (used when only 1 speaker or clustering failed).
        """
        # --- Helper: compute fit parameters for a crop region into a panel ---
        def fit_crop_to_panel(crop_w: int, crop_h: int, panel_w: int, panel_h: int) -> tuple:
            if crop_w <= 0 or crop_h <= 0:
                return panel_w, panel_h, 0, 0
            scale = panel_w / crop_w
            sh = int(crop_h * scale)
            sh = (sh // 2) * 2
            if sh <= panel_h:
                sw = panel_w
                py = (panel_h - sh) // 2
            else:
                scale_h = panel_h / crop_h
                sw = int(crop_w * scale_h)
                sw = (sw // 2) * 2
                sh = panel_h
                py = 0
            px = max(0, (panel_w - sw) // 2)
            return sw, sh, px, py

        dual_speaker = speaker_regions is not None and len(speaker_regions) == 2
        if dual_speaker:
            left = speaker_regions[0]
            right = speaker_regions[1]
            top_sw, top_sh, top_px, top_py = fit_crop_to_panel(left["w"], left["h"], target_width, content_h)
            bot_sw, bot_sh, bot_px, bot_py = fit_crop_to_panel(right["w"], right["h"], target_width, speaker_h)

            logger.info(
                f"TalkingHead DUAL: source=({source_w}x{source_h}), "
                f"top_crop=({left['w']}x{left['h']})@({left['x']},{left['y']}) → "
                f"({top_sw}x{top_sh})+pad({top_px},{top_py}), "
                f"bot_crop=({right['w']}x{right['h']})@({right['x']},{right['y']}) → "
                f"({bot_sw}x{bot_sh})+pad({bot_px},{bot_py})"
            )

            filter_parts = [
                # Split input into 4 streams: content bg/fg, speaker bg/fg
                f"[0:v]split=4[cfr][cbr][sfr][sbr]",
                # Content background: heavily blurred FULL frame (for letterboxing)
                f"[cbr]scale={target_width}:{content_h}:flags=fast_bilinear,boxblur=30:15[content_bg]",
                # Content foreground: LEFT speaker crop, sharp fit
                f"[cfr]crop={left['w']}:{left['h']}:{left['x']}:{left['y']},scale={top_sw}:{top_sh}:flags=lanczos[content_fg]",
                f"[content_bg][content_fg]overlay={top_px}:{top_py}[content]",
                # Speaker background: lightly blurred FULL frame
                f"[sbr]scale={target_width}:{speaker_h}:flags=fast_bilinear,boxblur=25:12[speaker_bg]",
                # Speaker foreground: RIGHT speaker crop, sharp fit
                f"[sfr]crop={right['w']}:{right['h']}:{right['x']}:{right['y']},scale={bot_sw}:{bot_sh}:flags=lanczos[speaker_fg]",
                f"[speaker_bg][speaker_fg]overlay={bot_px}:{bot_py}[speaker]",
            ]
        else:
            # Fallback: full-frame fit with blur in both panels
            top_sw, top_sh, top_px, top_py = fit_crop_to_panel(source_w, source_h, target_width, content_h)
            bot_sw, bot_sh, bot_px, bot_py = fit_crop_to_panel(source_w, source_h, target_width, speaker_h)

            logger.info(
                f"TalkingHead FIT: source=({source_w}x{source_h}), "
                f"top=({top_sw}x{top_sh})+pad({top_px},{top_py}), "
                f"bottom=({bot_sw}x{bot_sh})+pad({bot_px},{bot_py}) "
                f"[speaker_regions={len(speaker_regions) if speaker_regions else 0}]"
            )

            filter_parts = [
                f"[0:v]split=4[cfr][cbr][sfr][sbr]",
                f"[cbr]scale={target_width}:{content_h}:flags=fast_bilinear,boxblur=30:15[content_bg]",
                f"[cfr]scale={top_sw}:{top_sh}:flags=lanczos[content_fg]",
                f"[content_bg][content_fg]overlay={top_px}:{top_py}[content]",
                f"[sbr]scale={target_width}:{speaker_h}:flags=fast_bilinear,boxblur=25:12[speaker_bg]",
                f"[sfr]scale={bot_sw}:{bot_sh}:flags=lanczos[speaker_fg]",
                f"[speaker_bg][speaker_fg]overlay={bot_px}:{bot_py}[speaker]",
            ]

        # Use intermediate label if ASS subtitles will be applied after vstack
        vstack_label = "[stacked]" if ass_subtitle_path else "[outv]"

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3{vstack_label}")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2{vstack_label}")

        # Burn in ASS subtitles on top of the composited split layout
        if ass_subtitle_path:
            escaped_path = self._escape_ass_path_for_ffmpeg(ass_subtitle_path)
            filter_parts.append(f"[stacked]ass='{escaped_path}'[outv]")

        return ";".join(filter_parts)

    def _build_screen_only_filter(
        self,
        source_w: int, source_h: int,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
        content_region: Optional[dict] = None,
        ass_subtitle_path: Optional[str] = None,
    ) -> str:
        """Build filter for screen-only (no face detected). Bottom panel is blurred."""
        # Use content region if available
        if content_region and (content_region.get("y", 0) > 0 or content_region.get("h", source_h) < source_h):
            cr_x = content_region["x"]
            cr_y = content_region["y"]
            cr_w = content_region["w"]
            cr_h = content_region["h"]
        else:
            cr_x, cr_y, cr_w, cr_h = 0, 0, source_w, source_h

        content_aspect = target_width / content_h
        if cr_w / cr_h > content_aspect:
            crop_ch = cr_h
            crop_cw = int(cr_h * content_aspect)
        else:
            crop_cw = cr_w
            crop_ch = int(cr_w / content_aspect)
        crop_cw = (crop_cw // 2) * 2
        crop_ch = (crop_ch // 2) * 2
        crop_cx = cr_x + (cr_w - crop_cw) // 2
        crop_cy = cr_y + (cr_h - crop_ch) // 2

        # Bottom panel: blurred center-bottom of frame
        speaker_aspect = target_width / speaker_h
        if source_w / source_h > speaker_aspect:
            crop_sh = source_h
            crop_sw = int(source_h * speaker_aspect)
        else:
            crop_sw = source_w
            crop_sh = int(source_w / speaker_aspect)
        crop_sw = (crop_sw // 2) * 2
        crop_sh = (crop_sh // 2) * 2
        crop_sx = (source_w - crop_sw) // 2
        crop_sy = max(0, source_h - crop_sh)

        filter_parts = [
            f"[0:v]split=2[cs][ss]",
            f"[cs]crop={crop_cw}:{crop_ch}:{crop_cx}:{crop_cy},scale={target_width}:{content_h}:flags=lanczos[content]",
            f"[ss]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},scale={target_width}:{speaker_h}:flags=lanczos,boxblur=10:5[speaker]",
        ]

        # Use intermediate label if ASS subtitles will be applied after vstack
        vstack_label = "[stacked]" if ass_subtitle_path else "[outv]"

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3{vstack_label}")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2{vstack_label}")

        # Burn in ASS subtitles on top of the composited split layout
        if ass_subtitle_path:
            escaped_path = self._escape_ass_path_for_ffmpeg(ass_subtitle_path)
            filter_parts.append(f"[stacked]ass='{escaped_path}'[outv]")

        return ";".join(filter_parts)

    async def generate_preview(
        self,
        video_path: str,
        output_path: str,
        split_ratio: float = 0.55,
        separator_color: str = "#333333",
        duration: float = 5.0,
        start_time: float = 0,
    ) -> dict:
        """Generate a short preview of the split layout."""
        analysis = await self.analyze_layout(
            video_path,
            sample_interval=0.5,
            start_time=start_time,
            end_time=start_time + duration,
        )

        await self.generate_split_video(
            video_path=video_path,
            output_path=output_path,
            layout_analysis=analysis,
            split_ratio=split_ratio,
            separator_color=separator_color,
            start_time=start_time,
            end_time=start_time + duration,
        )

        return {
            "preview_path": output_path,
            "layout_type": analysis["layout_type"],
            "confidence": analysis["confidence"],
            "duration": duration,
        }


# Singleton
_split_layout_service = None


def get_split_layout_service() -> SplitLayoutService:
    """Get singleton split layout service."""
    global _split_layout_service
    if _split_layout_service is None:
        _split_layout_service = SplitLayoutService()
    return _split_layout_service
