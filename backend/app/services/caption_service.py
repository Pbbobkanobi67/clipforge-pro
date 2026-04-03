"""Animated caption service using ASS subtitles."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Filler words to detect and optionally remove
FILLER_WORDS = {
    "um", "uh", "uhm", "umm", "hmm", "hm", "ah", "er", "erm",
    "like",  # Only when standalone filler, not "I like this"
    "you know", "i mean", "basically", "literally", "actually",
    "sort of", "kind of", "right",
}

# Single-word fillers (always removable)
SINGLE_FILLERS = {"um", "uh", "uhm", "umm", "hmm", "hm", "ah", "er", "erm"}

# Emoji mappings: keyword -> emoji
EMOJI_MAP = {
    # Positive emotions
    "amazing": "\U0001f525", "awesome": "\U0001f525", "incredible": "\U0001f525",
    "fire": "\U0001f525", "lit": "\U0001f525", "insane": "\U0001f525",
    "love": "\u2764\ufe0f", "heart": "\u2764\ufe0f",
    "happy": "\U0001f60a", "excited": "\U0001f389", "wow": "\U0001f62e",
    "beautiful": "\u2728", "perfect": "\U0001f44c", "great": "\U0001f44d",
    "cool": "\U0001f60e", "fantastic": "\U0001f31f", "brilliant": "\U0001f4a1",
    # Negative emotions
    "sad": "\U0001f622", "cry": "\U0001f622", "angry": "\U0001f621",
    "scary": "\U0001f631", "terrible": "\U0001f62c", "horrible": "\U0001f62c",
    # Money/business
    "money": "\U0001f4b0", "dollar": "\U0001f4b5", "rich": "\U0001f4b0",
    "crypto": "\U0001f4b0", "bitcoin": "\u20bf", "blockchain": "\u26d3\ufe0f",
    "invest": "\U0001f4c8", "profit": "\U0001f4c8", "stock": "\U0001f4c8",
    "million": "\U0001f4b0", "billion": "\U0001f4b0",
    # Tech
    "ai": "\U0001f916", "robot": "\U0001f916", "computer": "\U0001f4bb",
    "phone": "\U0001f4f1", "app": "\U0001f4f1", "website": "\U0001f310",
    "code": "\U0001f4bb", "software": "\U0001f4bb",
    # Actions
    "subscribe": "\U0001f514", "like": "\U0001f44d", "share": "\U0001f4e4",
    "follow": "\U0001f44b", "comment": "\U0001f4ac", "click": "\U0001f449",
    # People/social
    "team": "\U0001f91d", "community": "\U0001f465", "friend": "\U0001f46b",
    # Food/drink
    "food": "\U0001f354", "eat": "\U0001f37d\ufe0f", "coffee": "\u2615",
    # Nature
    "sun": "\u2600\ufe0f", "rain": "\U0001f327\ufe0f", "earth": "\U0001f30d",
    # Misc
    "question": "\u2753", "idea": "\U0001f4a1", "warning": "\u26a0\ufe0f",
    "check": "\u2705", "star": "\u2b50", "music": "\U0001f3b5",
    "game": "\U0001f3ae", "win": "\U0001f3c6", "trophy": "\U0001f3c6",
    "rocket": "\U0001f680", "growth": "\U0001f680", "launch": "\U0001f680",
    "new": "\U0001f195", "free": "\U0001f381", "secret": "\U0001f92b",
    "breaking": "\U0001f6a8", "alert": "\U0001f6a8",
}

# Keywords to highlight (important/impactful words)
HIGHLIGHT_KEYWORDS = {
    # Numbers and money
    "million", "billion", "thousand", "hundred", "percent",
    # Impact words
    "never", "always", "every", "only", "first", "last", "best", "worst",
    "biggest", "smallest", "most", "least", "must", "need", "critical",
    "important", "essential", "breaking", "exclusive", "secret", "free",
    "new", "just", "now", "today", "finally", "amazing", "incredible",
    "insane", "crazy", "huge", "massive", "revolutionary", "game-changer",
}


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

    @staticmethod
    def remove_filler_words(segments: list[dict]) -> list[dict]:
        """Remove filler words from segments and their word-level data.

        Returns a new list of segments with filler words stripped out.
        Preserves timing continuity by adjusting gaps.
        """
        cleaned = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                # Segment-level only: strip filler phrases from text
                text = seg.get("text", "")
                for filler in sorted(FILLER_WORDS, key=len, reverse=True):
                    pattern = r'\b' + re.escape(filler) + r'\b'
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                text = re.sub(r'\s{2,}', ' ', text).strip()
                if text:
                    cleaned.append({**seg, "text": text})
                continue

            # Word-level: remove single-filler words
            filtered_words = []
            for w in words:
                word_lower = w.get("word", "").strip().lower().rstrip(".,!?;:")
                if word_lower in SINGLE_FILLERS:
                    continue
                filtered_words.append(w)

            if filtered_words:
                new_text = " ".join(w["word"].strip() for w in filtered_words)
                cleaned.append({
                    **seg,
                    "text": new_text,
                    "words": filtered_words,
                    "start_time": filtered_words[0].get("start", seg.get("start_time", 0)),
                    "end_time": filtered_words[-1].get("end", seg.get("end_time", 0)),
                })
        return cleaned

    @staticmethod
    def insert_emojis(segments: list[dict]) -> list[dict]:
        """Insert relevant emojis into segment text based on keyword matching.

        Adds emoji after matching keywords in the word list.
        """
        result = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                # Segment-level: add emojis to text
                text = seg.get("text", "")
                for keyword, emoji in EMOJI_MAP.items():
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        text = re.sub(
                            pattern,
                            lambda m: m.group(0) + " " + emoji,
                            text,
                            count=1,
                            flags=re.IGNORECASE,
                        )
                result.append({**seg, "text": text})
                continue

            # Word-level: add emoji after matching words
            new_words = []
            for w in words:
                word_clean = w.get("word", "").strip().lower().rstrip(".,!?;:")
                new_words.append(w)
                if word_clean in EMOJI_MAP:
                    # Append emoji to the word text
                    emoji = EMOJI_MAP[word_clean]
                    new_w = dict(w)
                    new_w["word"] = w["word"].rstrip() + " " + emoji
                    new_words[-1] = new_w

            new_text = " ".join(w["word"].strip() for w in new_words)
            result.append({**seg, "text": new_text, "words": new_words})
        return result

    @staticmethod
    def detect_keywords(words: list[dict]) -> set[int]:
        """Return indices of words that should be highlighted as keywords.

        Detects: numbers, proper nouns (capitalized), and impact words.
        """
        keyword_indices = set()
        for i, w in enumerate(words):
            word_text = w.get("word", "").strip()
            word_lower = word_text.lower().rstrip(".,!?;:")

            # Highlight if it's a known impact keyword
            if word_lower in HIGHLIGHT_KEYWORDS:
                keyword_indices.add(i)
                continue

            # Highlight numbers
            if re.match(r'^\$?\d[\d,.%]*$', word_text.rstrip(".,!?;:")):
                keyword_indices.add(i)
                continue

            # Highlight capitalized words (proper nouns) - skip first word
            if i > 0 and word_text[0:1].isupper() and len(word_text) > 2:
                keyword_indices.add(i)
                continue

        return keyword_indices

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

        # Alignment based on position (case-insensitive)
        # ASS numpad-style alignment: 1=bottom-left, 2=bottom-center, 5=center, 8=top-center
        position = position.lower() if position else "bottom"
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

    @staticmethod
    def _add_transition_tags(events: list[str], fade_in_ms: int = 80, fade_out_ms: int = 60) -> list[str]:
        """Add smooth fade-in/fade-out transitions between caption groups.

        Inserts \\fad(in,out) tags into existing Dialogue lines so each caption
        group fades in when appearing and fades out when disappearing.
        """
        if not events or fade_in_ms <= 0 and fade_out_ms <= 0:
            return events

        result = []
        fad_tag = f"\\fad({fade_in_ms},{fade_out_ms})"
        for line in events:
            if not line.startswith("Dialogue:"):
                result.append(line)
                continue
            # Find the text portion (after the 9th comma in ASS format)
            parts = line.split(",", 9)
            if len(parts) < 10:
                result.append(line)
                continue
            text = parts[9]
            # Insert fad tag: if text starts with an override block, inject inside it
            if text.startswith("{"):
                close = text.index("}")
                text = text[:close] + fad_tag + text[close:]
            else:
                text = "{" + fad_tag + "}" + text
            parts[9] = text
            result.append(",".join(parts))
        return result

    def _generate_karaoke_events(
        self,
        segments: list[dict],
        style_config: dict,
        clip_start_offset: float = 0,
    ) -> list[str]:
        """
        Generate karaoke-style events with word-by-word highlighting.
        Uses ASS \\k tag for karaoke timing.
        Supports keyword highlighting with scale/color emphasis.
        """
        events = []
        highlight_color = _hex_to_ass_color(style_config.get("highlight_color", "#FFFF00"))
        keyword_color = _hex_to_ass_color(style_config.get("keyword_color", style_config.get("highlight_color", "#FFFF00")))
        words_per_line = style_config.get("words_per_line", 6)
        enable_keywords = style_config.get("keyword_highlight", True)

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

            # Detect keywords for highlighting
            keyword_indices = self.detect_keywords(words) if enable_keywords else set()

            # Group words into lines
            word_groups = []
            keyword_groups = []  # Track which indices are keywords per group
            current_group = []
            current_kw = []
            global_idx = 0

            for word in words:
                current_group.append(word)
                current_kw.append(global_idx in keyword_indices)
                if len(current_group) >= words_per_line:
                    word_groups.append(current_group)
                    keyword_groups.append(current_kw)
                    current_group = []
                    current_kw = []
                global_idx += 1

            if current_group:
                word_groups.append(current_group)
                keyword_groups.append(current_kw)

            # Generate karaoke events for each line
            for group, kw_flags in zip(word_groups, keyword_groups):
                if not group:
                    continue

                line_start = group[0]["start"] - clip_start_offset
                line_end = group[-1]["end"] - clip_start_offset

                if line_start < 0:
                    continue

                # Build karaoke text with timing tags
                karaoke_parts = []
                for i, word in enumerate(group):
                    word_text = word["word"].strip()
                    if not word_text:
                        continue

                    # Duration in centiseconds
                    word_duration = int((word["end"] - word["start"]) * 100)

                    is_keyword = kw_flags[i] if i < len(kw_flags) else False
                    if is_keyword:
                        # Keyword: scale up + bold color override
                        karaoke_parts.append(
                            f"{{\\kf{word_duration}\\fscx115\\fscy115\\1c{keyword_color}}}"
                            f"{word_text.upper()}"
                            f"{{\\fscx100\\fscy100}}"
                        )
                    else:
                        karaoke_parts.append(f"{{\\kf{word_duration}}}{word_text}")

                if karaoke_parts:
                    karaoke_text = " ".join(karaoke_parts)
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
        remove_fillers: bool = True,
        add_emojis: bool = False,
        smooth_transitions: bool = True,
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
            remove_fillers: Remove filler words (um, uh, etc.)
            add_emojis: Auto-insert relevant emojis

        Returns:
            Path to generated ASS file
        """
        if style_config is None:
            style_config = {}

        # Pre-process segments
        processed = segments
        if remove_fillers:
            processed = self.remove_filler_words(processed)
            logger.info(f"Filler removal: {len(segments)} -> {len(processed)} segments")
        if add_emojis:
            processed = self.insert_emojis(processed)

        # Generate header
        header = self._create_ass_header(video_width, video_height, style_config)

        # Generate events based on style type (case-insensitive)
        style_type = style_config.get("style_type", "karaoke").lower()

        if style_type == "karaoke":
            events = self._generate_karaoke_events(processed, style_config, clip_start_offset)
        elif style_type == "bounce":
            events = self._generate_bounce_events(processed, style_config, clip_start_offset)
        elif style_type == "typewriter":
            events = self._generate_typewriter_events(processed, style_config, clip_start_offset)
        elif style_type == "fade":
            events = self._generate_fade_events(processed, style_config, clip_start_offset)
        else:  # static or unknown
            events = self._generate_static_events(processed, style_config, clip_start_offset)

        # Apply smooth transitions between caption groups
        if smooth_transitions:
            events = self._add_transition_tags(events, fade_in_ms=80, fade_out_ms=60)

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
