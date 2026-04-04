"""Speaker + Screen split layout service.

Detects PIP webcam regions in screen recordings and generates
portrait 9:16 videos with content on top and speaker on bottom.
Uses OpenCV Haar cascade for face detection (zero extra deps).
"""

import asyncio
import logging
import statistics
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SplitLayoutService:
    """Analyze screen recordings for PIP webcam and generate split layouts."""

    def __init__(self):
        self._face_cascade = None

    def _get_face_cascade(self):
        """Lazy-load the Haar cascade for frontal face detection."""
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
        num_samples: int = 8,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
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
            lambda: self._analyze_layout_sync(video_path, num_samples, start_time, end_time),
        )

    def _analyze_layout_sync(
        self,
        video_path: str,
        num_samples: int,
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
        frame_start = int((start_time or 0) * fps)
        frame_end = int((end_time or duration) * fps)
        frame_end = min(frame_end, total_frames)

        if frame_end <= frame_start:
            frame_end = total_frames

        # Sample frames evenly
        sample_range = frame_end - frame_start
        if sample_range <= 0:
            sample_range = total_frames
            frame_start = 0

        step = max(1, sample_range // num_samples)
        sample_frames = list(range(frame_start, frame_end, step))[:num_samples]

        cascade = self._get_face_cascade()
        all_faces = []  # list of (x, y, w, h) per sample

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(int(source_width * 0.02), int(source_height * 0.02)),
            )

            if len(faces) > 0:
                for (fx, fy, fw, fh) in faces:
                    all_faces.append((fx, fy, fw, fh))

        cap.release()

        # Classify the layout
        classification = self._classify_layout(
            all_faces, source_width, source_height, num_samples
        )

        return {
            "layout_type": classification["layout_type"],
            "source_width": source_width,
            "source_height": source_height,
            "fps": fps,
            "duration": duration,
            "pip_region": classification.get("pip_region"),
            "content_region": classification.get("content_region"),
            "confidence": classification["confidence"],
        }

    def _classify_layout(
        self,
        faces: list[tuple],
        frame_w: int,
        frame_h: int,
        num_samples: int,
    ) -> dict:
        """
        Classify video layout based on detected faces.

        Decision logic:
        - Small face in a corner (3-30% of frame width), consistent detection -> screen_speaker
        - Large face (>35% width), centered -> talking_head
        - No faces -> screen_only
        """
        if not faces:
            return {"layout_type": "screen_only", "confidence": 0.5}

        # Compute median face metrics
        face_widths = [fw for (_, _, fw, _) in faces]
        face_centers_x = [fx + fw / 2 for (fx, _, fw, _) in faces]
        face_centers_y = [fy + fh / 2 for (_, fy, _, fh) in faces]

        median_width = statistics.median(face_widths)
        median_cx = statistics.median(face_centers_x)
        median_cy = statistics.median(face_centers_y)

        face_width_ratio = median_width / frame_w
        detection_rate = len(faces) / max(num_samples, 1)

        # Determine corner
        corner = self._detect_corner(median_cx, median_cy, frame_w, frame_h)

        # Decision tree
        if face_width_ratio < 0.03:
            # Too small, likely noise
            return {"layout_type": "screen_only", "confidence": 0.4}

        if 0.03 <= face_width_ratio <= 0.30 and corner != "center" and detection_rate >= 0.4:
            # Small face in a corner = PIP webcam
            pip_region = self._estimate_pip_region(faces, frame_w, frame_h, corner)
            content_region = self._estimate_content_region(pip_region, frame_w, frame_h)
            return {
                "layout_type": "screen_speaker",
                "pip_region": pip_region,
                "content_region": content_region,
                "confidence": min(detection_rate, 1.0),
            }

        if face_width_ratio > 0.35:
            # Large face, likely talking head
            return {"layout_type": "talking_head", "confidence": min(detection_rate, 1.0)}

        # Medium face or centered - ambiguous, default to screen_only
        if detection_rate >= 0.4:
            # Decent detection but medium size - could be screen_speaker with larger PIP
            pip_region = self._estimate_pip_region(faces, frame_w, frame_h, corner)
            content_region = self._estimate_content_region(pip_region, frame_w, frame_h)
            return {
                "layout_type": "screen_speaker",
                "pip_region": pip_region,
                "content_region": content_region,
                "confidence": min(detection_rate * 0.7, 1.0),
            }

        return {"layout_type": "screen_only", "confidence": 0.3}

    def _detect_corner(self, cx: float, cy: float, frame_w: int, frame_h: int) -> str:
        """Detect which corner the face center is in."""
        half_w = frame_w / 2
        half_h = frame_h / 2
        margin = 0.35  # must be in outer 35% to count as a corner

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
        return "center"

    def _estimate_pip_region(
        self,
        faces: list[tuple],
        frame_w: int,
        frame_h: int,
        corner: str,
    ) -> dict:
        """Estimate the PIP webcam region from detected faces."""
        face_xs = [fx for (fx, _, _, _) in faces]
        face_ys = [fy for (_, fy, _, _) in faces]
        face_ws = [fw for (_, _, fw, _) in faces]
        face_hs = [fh for (_, _, _, fh) in faces]

        median_x = statistics.median(face_xs)
        median_y = statistics.median(face_ys)
        median_w = statistics.median(face_ws)
        median_h = statistics.median(face_hs)

        # Expand face bbox to estimate PIP region (~2.2x height, ~1.8x width)
        pip_w = int(median_w * 1.8)
        pip_h = int(median_h * 2.2)

        # Center the PIP region on the face
        pip_x = int(median_x + median_w / 2 - pip_w / 2)
        pip_y = int(median_y + median_h / 2 - pip_h / 2)

        # Snap to corner edge
        if "left" in corner:
            pip_x = max(0, min(pip_x, int(frame_w * 0.05)))
        elif "right" in corner:
            pip_x = max(frame_w - pip_w - int(frame_w * 0.05), pip_x)

        if "top" in corner:
            pip_y = max(0, min(pip_y, int(frame_h * 0.05)))
        elif "bottom" in corner:
            pip_y = max(frame_h - pip_h - int(frame_h * 0.05), pip_y)

        # Clamp
        pip_x = max(0, min(pip_x, frame_w - pip_w))
        pip_y = max(0, min(pip_y, frame_h - pip_h))
        pip_w = min(pip_w, frame_w - pip_x)
        pip_h = min(pip_h, frame_h - pip_y)

        return {
            "x": pip_x,
            "y": pip_y,
            "w": pip_w,
            "h": pip_h,
            "corner": corner,
        }

    def _estimate_content_region(
        self,
        pip_region: dict,
        frame_w: int,
        frame_h: int,
    ) -> dict:
        """Estimate the screen content region (everything except PIP)."""
        # For most screen recordings, content fills the entire frame
        # The PIP overlays on top of content, so content = full frame
        return {
            "x": 0,
            "y": 0,
            "w": frame_w,
            "h": frame_h,
        }

    async def generate_split_video(
        self,
        video_path: str,
        output_path: str,
        layout_analysis: dict,
        split_ratio: float = 0.65,
        separator_height: int = 4,
        separator_color: str = "#333333",
        target_width: int = 1080,
        target_height: int = 1920,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
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

        # Build filter complex based on layout type
        if layout_type == "screen_speaker" and layout_analysis.get("pip_region"):
            pip = layout_analysis["pip_region"]
            filter_complex = self._build_screen_speaker_filter(
                source_w, source_h, pip,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps,
            )
        elif layout_type == "talking_head":
            # For talking head, use full frame for speaker and blurred for content
            filter_complex = self._build_talking_head_filter(
                source_w, source_h,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps,
            )
        else:
            # screen_only fallback: content on top, blurred bottom
            filter_complex = self._build_screen_only_filter(
                source_w, source_h,
                target_width, content_h, speaker_h, separator_height,
                ffmpeg_color, dur, fps,
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

    def _build_screen_speaker_filter(
        self,
        source_w: int, source_h: int,
        pip: dict,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
    ) -> str:
        """Build filter for screen + speaker PIP layout."""
        pip_x = pip["x"]
        pip_y = pip["y"]
        pip_w = pip["w"]
        pip_h = pip["h"]

        # Content panel: crop from center of full frame, aspect-matched to top panel
        content_aspect = target_width / content_h
        # Calculate crop that matches the content panel aspect ratio
        if source_w / source_h > content_aspect:
            # Source is wider, limit by height
            crop_ch = source_h
            crop_cw = int(source_h * content_aspect)
        else:
            # Source is taller, limit by width
            crop_cw = source_w
            crop_ch = int(source_w / content_aspect)

        crop_cw = (crop_cw // 2) * 2
        crop_ch = (crop_ch // 2) * 2
        crop_cx = (source_w - crop_cw) // 2
        crop_cy = (source_h - crop_ch) // 2

        # Speaker panel: crop from PIP region, aspect-matched to bottom panel
        speaker_aspect = target_width / speaker_h
        if pip_w / pip_h > speaker_aspect:
            crop_sh = pip_h
            crop_sw = int(pip_h * speaker_aspect)
        else:
            crop_sw = pip_w
            crop_sh = int(pip_w / speaker_aspect)

        crop_sw = max((crop_sw // 2) * 2, 2)
        crop_sh = max((crop_sh // 2) * 2, 2)
        # Center the crop within the PIP region
        crop_sx = pip_x + (pip_w - crop_sw) // 2
        crop_sy = pip_y + (pip_h - crop_sh) // 2
        # Clamp
        crop_sx = max(0, min(crop_sx, source_w - crop_sw))
        crop_sy = max(0, min(crop_sy, source_h - crop_sh))

        filter_parts = [
            f"[0:v]split=2[cs][ss]",
            f"[cs]crop={crop_cw}:{crop_ch}:{crop_cx}:{crop_cy},scale={target_width}:{content_h}:flags=lanczos[content]",
            f"[ss]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},scale={target_width}:{speaker_h}:flags=lanczos[speaker]",
        ]

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3[outv]")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2[outv]")

        return ";".join(filter_parts)

    def _build_talking_head_filter(
        self,
        source_w: int, source_h: int,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
    ) -> str:
        """Build filter for talking head (speaker fills bottom, blurred top)."""
        # Content panel: blurred version of full frame
        content_aspect = target_width / content_h
        if source_w / source_h > content_aspect:
            crop_ch = source_h
            crop_cw = int(source_h * content_aspect)
        else:
            crop_cw = source_w
            crop_ch = int(source_w / content_aspect)
        crop_cw = (crop_cw // 2) * 2
        crop_ch = (crop_ch // 2) * 2
        crop_cx = (source_w - crop_cw) // 2
        crop_cy = (source_h - crop_ch) // 2

        # Speaker panel: center crop
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
        crop_sy = (source_h - crop_sh) // 2

        filter_parts = [
            f"[0:v]split=2[cs][ss]",
            f"[cs]crop={crop_cw}:{crop_ch}:{crop_cx}:{crop_cy},scale={target_width}:{content_h}:flags=lanczos,boxblur=8:4[content]",
            f"[ss]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},scale={target_width}:{speaker_h}:flags=lanczos[speaker]",
        ]

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3[outv]")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2[outv]")

        return ";".join(filter_parts)

    def _build_screen_only_filter(
        self,
        source_w: int, source_h: int,
        target_width: int, content_h: int, speaker_h: int,
        separator_height: int, sep_color: str, dur: float, fps: float,
    ) -> str:
        """Build filter for screen-only (no face detected). Bottom panel is blurred."""
        content_aspect = target_width / content_h
        if source_w / source_h > content_aspect:
            crop_ch = source_h
            crop_cw = int(source_h * content_aspect)
        else:
            crop_cw = source_w
            crop_ch = int(source_w / content_aspect)
        crop_cw = (crop_cw // 2) * 2
        crop_ch = (crop_ch // 2) * 2
        crop_cx = (source_w - crop_cw) // 2
        crop_cy = (source_h - crop_ch) // 2

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
        # Crop from bottom portion
        crop_sy = max(0, source_h - crop_sh)

        filter_parts = [
            f"[0:v]split=2[cs][ss]",
            f"[cs]crop={crop_cw}:{crop_ch}:{crop_cx}:{crop_cy},scale={target_width}:{content_h}:flags=lanczos[content]",
            f"[ss]crop={crop_sw}:{crop_sh}:{crop_sx}:{crop_sy},scale={target_width}:{speaker_h}:flags=lanczos,boxblur=10:5[speaker]",
        ]

        if separator_height > 0:
            filter_parts.append(
                f"color=c={sep_color}:s={target_width}x{separator_height}:d={dur}:r={int(fps)}[sep]"
            )
            filter_parts.append(f"[content][sep][speaker]vstack=inputs=3[outv]")
        else:
            filter_parts.append(f"[content][speaker]vstack=inputs=2[outv]")

        return ";".join(filter_parts)

    async def generate_preview(
        self,
        video_path: str,
        output_path: str,
        split_ratio: float = 0.65,
        separator_color: str = "#333333",
        duration: float = 5.0,
        start_time: float = 0,
    ) -> dict:
        """Generate a short preview of the split layout."""
        analysis = await self.analyze_layout(
            video_path,
            num_samples=4,
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
