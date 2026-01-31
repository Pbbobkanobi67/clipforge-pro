"""Animated caption service using ASS subtitles."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _hex_to_ass_color(hex_color: str) -> str:
    """
    Convert hex color (#RRGGBB) to ASS color format (&HBBGGRR&).
    ASS uses BGR order and different format.
    """
    # Remove # prefix
    hex_color = hex_color.lstrip("#")

    # Parse RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Return in ASS BGR format
    return f"&H00{b:02X}{g:02X}{r:02X}&"


def _hex_to_ass_color_alpha(hex_color: str, alpha: int = 0) -> str:
    """
    Convert hex color to ASS color with alpha.
    Alpha: 0 = opaque, 255 = transparent
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _format_ass_time(seconds: float) -> str:
    """Format seconds as ASS timestamp (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


class CaptionService:
    """Service for generating animated captions in ASS format."""

    def __init__(self):
        self._cache_dir = settings.cache_path / "captions"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_style_definition(
        self,
        style_name: str,
        font_family: str,
        font_size: int,
        text_color: str,
        stroke_color: str,
        stroke_width: int,
        background_color: Optional[str],
        font_weight: str,
        position: str,
        margin_bottom: int,
        margin_horizontal: int,
    ) -> str:
        """Generate ASS style definition."""
        # Convert colors to ASS format
        primary_color = _hex_to_ass_color(text_color)
        outline_color = _hex_to_ass_color(stroke_color)

        # Background color (shadow in ASS)
        if background_color:
            back_color = _hex_to_ass_color_alpha(background_color, 128)
        else:
            back_color = "&H80000000"  # Semi-transparent black

        # Bold flag
        bold = -1 if font_weight.lower() == "bold" else 0

        # Alignment based on position
        # ASS numpad-style alignment: 1=bottom-left, 2=bottom-center, 5=center, 8=top-center
        alignment_map = {
            "top": 8,
            "center": 5,
            "bottom": 2,
        }
        alignment = alignment_map.get(position, 2)

        # Vertical margin
        margin_v = margin_bottom

        return (
            f"Style: {style_name},{font_family},{font_size},"
            f"{primary_color},{primary_color},{outline_color},{back_color},"
            f"{bold},0,0,0,100,100,0,0,1,{stroke_width},0,"
            f"{alignment},{margin_horizontal},{margin_horizontal},{margin_v},1"
        )

    def _create_ass_header(
        self,
        video_width: int,
        video_height: int,
        style_config: dict,
    ) -> str:
        """Create ASS file header with styles."""
        # Main style
        main_style = self._get_style_definition(
            style_name="Default",
            font_family=style_config.get("font_family", "Arial"),
            font_size=style_config.get("font_size", 48),
            text_color=style_config.get("text_color", "#FFFFFF"),
            stroke_color=style_config.get("stroke_color", "#000000"),
            stroke_width=style_config.get("stroke_width", 2),
            background_color=style_config.get("background_color"),
            font_weight=style_config.get("font_weight", "bold"),
            position=style_config.get("position", "bottom"),
            margin_bottom=style_config.get("margin_bottom", 50),
            margin_horizontal=style_config.get("margin_horizontal", 40),
        )

        # Highlight style for karaoke
        highlight_style = self._get_style_definition(
            style_name="Highlight",
            font_family=style_config.get("font_family", "Arial"),
            font_size=style_config.get("font_size", 48),
            text_color=style_config.get("highlight_color", "#FFFF00"),
            stroke_color=style_config.get("stroke_color", "#000000"),
            stroke_width=style_config.get("stroke_width", 2),
            background_color=style_config.get("background_color"),
            font_weight=style_config.get("font_weight", "bold"),
            position=style_config.get("position", "bottom"),
            margin_bottom=style_config.get("margin_bottom", 50),
            margin_horizontal=style_config.get("margin_horizontal", 40),
        )

        return f"""[Script Info]
Title: Animated Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{main_style}
{highlight_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def _generate_karaoke_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate karaoke-style events with word-by-word highlighting.
        Uses ASS \\k tag for karaoke timing.
        """
        events = []
        highlight_color = _hex_to_ass_color(style_config.get("highlight_color", "#FFFF00"))
        words_per_line = style_config.get("words_per_line", 6)

        for segment in segments:
            words = segment.get("words", [])
            if not words:
                # Fallback to segment text if no word-level timing
                start = segment["start_time"] - clip_start_offset
                end = segment["end_time"] - clip_start_offset
                if start < 0:
                    continue
                events.append(
                    f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                    f"Default,,0,0,0,,{segment['text']}"
                )
                continue

            # Group words into lines
            word_groups = []
            current_group = []

            for word in words:
                current_group.append(word)
                if len(current_group) >= words_per_line:
                    word_groups.append(current_group)
                    current_group = []

            if current_group:
                word_groups.append(current_group)

            # Generate karaoke events for each line
            for group in word_groups:
                if not group:
                    continue

                line_start = group[0]["start"] - clip_start_offset
                line_end = group[-1]["end"] - clip_start_offset

                if line_start < 0:
                    continue

                # Build karaoke text with timing tags
                # \k<duration> marks syllable duration in centiseconds
                karaoke_parts = []
                for i, word in enumerate(group):
                    word_text = word["word"].strip()
                    if not word_text:
                        continue

                    # Duration in centiseconds
                    word_duration = int((word["end"] - word["start"]) * 100)

                    # Use \kf for smooth fill effect
                    karaoke_parts.append(f"{{\\kf{word_duration}}}{word_text}")

                if karaoke_parts:
                    karaoke_text = " ".join(karaoke_parts)
                    # Add highlight color override for the karaoke effect
                    karaoke_text = f"{{\\1c{highlight_color}}}" + karaoke_text

                    events.append(
                        f"Dialogue: 0,{_format_ass_time(line_start)},{_format_ass_time(line_end)},"
                        f"Default,,0,0,0,,{karaoke_text}"
                    )

        return events

    def _generate_bounce_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate bounce animation - words bounce in from below.
        Uses ASS \\move and \\t for animation.
        """
        events = []
        words_per_line = style_config.get("words_per_line", 6)
        anim_duration = int(style_config.get("animation_duration", 0.1) * 1000)

        for segment in segments:
            words = segment.get("words", [])
            if not words:
                start = segment["start_time"] - clip_start_offset
                end = segment["end_time"] - clip_start_offset
                if start < 0:
                    continue
                events.append(
                    f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                    f"Default,,0,0,0,,{segment['text']}"
                )
                continue

            # Group words into lines
            word_groups = []
            current_group = []

            for word in words:
                current_group.append(word)
                if len(current_group) >= words_per_line:
                    word_groups.append(current_group)
                    current_group = []

            if current_group:
                word_groups.append(current_group)

            for group in word_groups:
                if not group:
                    continue

                line_start = group[0]["start"] - clip_start_offset
                line_end = group[-1]["end"] - clip_start_offset

                if line_start < 0:
                    continue

                # Build text with per-word animation
                text_parts = []
                for i, word in enumerate(group):
                    word_text = word["word"].strip()
                    if not word_text:
                        continue

                    # Calculate word appearance time relative to line start
                    word_delay = int((word["start"] - clip_start_offset - line_start) * 1000)

                    # Bounce effect: start below, move up with overshoot
                    # \fad for fade, \t for transform
                    bounce_tag = (
                        f"{{\\fad({anim_duration},0)"
                        f"\\t({word_delay},{word_delay + anim_duration},\\fry0\\frx0)}}"
                    )
                    text_parts.append(bounce_tag + word_text)

                if text_parts:
                    text = " ".join(text_parts)
                    events.append(
                        f"Dialogue: 0,{_format_ass_time(line_start)},{_format_ass_time(line_end)},"
                        f"Default,,0,0,0,,{text}"
                    )

        return events

    def _generate_typewriter_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate typewriter effect - text appears character by character.
        """
        events = []
        char_duration = style_config.get("animation_duration", 0.05)

        for segment in segments:
            text = segment.get("text", "").strip()
            start = segment["start_time"] - clip_start_offset
            end = segment["end_time"] - clip_start_offset

            if start < 0 or not text:
                continue

            # Calculate timing for each character
            total_chars = len(text)
            available_time = end - start
            actual_char_duration = min(char_duration, available_time / max(total_chars, 1))

            # Create progressive reveal using clip tags
            current_text = ""
            char_start = start

            for i, char in enumerate(text):
                current_text += char
                char_end = min(char_start + actual_char_duration, end)

                # For the last character, extend to segment end
                if i == total_chars - 1:
                    char_end = end

                events.append(
                    f"Dialogue: 0,{_format_ass_time(char_start)},{_format_ass_time(char_end)},"
                    f"Default,,0,0,0,,{current_text}"
                )

                char_start = char_end
                if char_start >= end:
                    break

        return events

    def _generate_fade_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate fade-in effect for words.
        """
        events = []
        words_per_line = style_config.get("words_per_line", 6)
        fade_duration = int(style_config.get("animation_duration", 0.2) * 1000)

        for segment in segments:
            words = segment.get("words", [])
            if not words:
                start = segment["start_time"] - clip_start_offset
                end = segment["end_time"] - clip_start_offset
                if start < 0:
                    continue
                events.append(
                    f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                    f"Default,,0,0,0,,{{\\fad({fade_duration},0)}}{segment['text']}"
                )
                continue

            # Group words
            word_groups = []
            current_group = []

            for word in words:
                current_group.append(word)
                if len(current_group) >= words_per_line:
                    word_groups.append(current_group)
                    current_group = []

            if current_group:
                word_groups.append(current_group)

            for group in word_groups:
                if not group:
                    continue

                line_start = group[0]["start"] - clip_start_offset
                line_end = group[-1]["end"] - clip_start_offset

                if line_start < 0:
                    continue

                text = " ".join(w["word"].strip() for w in group if w["word"].strip())
                events.append(
                    f"Dialogue: 0,{_format_ass_time(line_start)},{_format_ass_time(line_end)},"
                    f"Default,,0,0,0,,{{\\fad({fade_duration},0)}}{text}"
                )

        return events

    def _generate_static_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate static captions with no animation.
        """
        events = []
        words_per_line = style_config.get("words_per_line", 6)

        for segment in segments:
            words = segment.get("words", [])
            if not words:
                start = segment["start_time"] - clip_start_offset
                end = segment["end_time"] - clip_start_offset
                if start < 0:
                    continue
                events.append(
                    f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},"
                    f"Default,,0,0,0,,{segment['text']}"
                )
                continue

            # Group words
            word_groups = []
            current_group = []

            for word in words:
                current_group.append(word)
                if len(current_group) >= words_per_line:
                    word_groups.append(current_group)
                    current_group = []

            if current_group:
                word_groups.append(current_group)

            for group in word_groups:
                if not group:
                    continue

                line_start = group[0]["start"] - clip_start_offset
                line_end = group[-1]["end"] - clip_start_offset

                if line_start < 0:
                    continue

                text = " ".join(w["word"].strip() for w in group if w["word"].strip())
                events.append(
                    f"Dialogue: 0,{_format_ass_time(line_start)},{_format_ass_time(line_end)},"
                    f"Default,,0,0,0,,{text}"
                )

        return events

    async def generate_ass_subtitles(
        self,
        segments: list[dict],
        output_path: str,
        video_width: int = 1920,
        video_height: int = 1080,
        style_config: Optional[dict] = None,
        clip_start_offset: float = 0,
    ) -> str:
        """
        Generate ASS subtitle file with animated captions.

        Args:
            segments: List of transcription segments with word-level timing
            output_path: Path for output ASS file
            video_width: Video width for positioning
            video_height: Video height for positioning
            style_config: Caption style configuration
            clip_start_offset: Offset to subtract from timestamps (for clips)

        Returns:
            Path to generated ASS file
        """
        if style_config is None:
            style_config = {}

        # Generate header
        header = self._create_ass_header(video_width, video_height, style_config)

        # Generate events based on style type
        style_type = style_config.get("style_type", "karaoke")

        if style_type == "karaoke":
            events = self._generate_karaoke_events(segments, style_config, clip_start_offset)
        elif style_type == "bounce":
            events = self._generate_bounce_events(segments, style_config, clip_start_offset)
        elif style_type == "typewriter":
            events = self._generate_typewriter_events(segments, style_config, clip_start_offset)
        elif style_type == "fade":
            events = self._generate_fade_events(segments, style_config, clip_start_offset)
        else:  # static or unknown
            events = self._generate_static_events(segments, style_config, clip_start_offset)

        # Combine and write file
        content = header + "\n".join(events)

        # Write file
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Path(output_path).write_text(content, encoding="utf-8")
        )

        logger.info(f"Generated ASS subtitles: {output_path} ({len(events)} events)")
        return output_path

    async def generate_preview(
        self,
        segments: list[dict],
        video_path: str,
        output_path: str,
        style_config: Optional[dict] = None,
        duration: float = 5.0,
        clip_start_offset: float = 0,
    ) -> str:
        """
        Generate a short preview video with animated captions.

        Args:
            segments: Transcription segments
            video_path: Source video path
            output_path: Output preview path
            style_config: Caption style configuration
            duration: Preview duration in seconds
            clip_start_offset: Start offset for clip

        Returns:
            Path to preview video
        """
        from app.services.video_service import VideoService

        video_service = VideoService()

        # Get video metadata
        metadata = await video_service.get_video_metadata(video_path)
        width = metadata.get("width", 1920)
        height = metadata.get("height", 1080)

        # Generate ASS file
        ass_path = str(Path(output_path).with_suffix(".ass"))
        await self.generate_ass_subtitles(
            segments=segments,
            output_path=ass_path,
            video_width=width,
            video_height=height,
            style_config=style_config,
            clip_start_offset=clip_start_offset,
        )

        # Export preview with subtitles using ffmpeg
        from app.services.video_service import _run_subprocess

        # Escape path for ffmpeg filter (handle Windows paths)
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")

        cmd = [
            "ffmpeg",
            "-ss", str(clip_start_offset),
            "-i", video_path,
            "-t", str(duration),
            "-vf", f"ass='{escaped_ass_path}'",
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-y",
            output_path,
        ]

        loop = asyncio.get_event_loop()
        returncode, _, stderr = await loop.run_in_executor(None, lambda: _run_subprocess(cmd))

        if returncode != 0:
            raise Exception(f"Preview generation failed: {stderr.decode()}")

        # Cleanup ASS file
        Path(ass_path).unlink(missing_ok=True)

        logger.info(f"Generated caption preview: {output_path}")
        return output_path


# Singleton instance
_caption_service = None


def get_caption_service() -> CaptionService:
    """Get singleton caption service."""
    global _caption_service
    if _caption_service is None:
        _caption_service = CaptionService()
    return _caption_service
