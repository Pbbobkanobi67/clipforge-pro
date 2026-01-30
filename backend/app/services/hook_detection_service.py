"""Hook detection service for identifying engaging video openers."""

import logging
import re
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HookDetectionService:
    """Service for detecting and scoring potential hooks in video content."""

    # Patterns that indicate engaging hooks
    QUESTION_PATTERNS = [
        r"^(what|how|why|when|where|who|which|can|do|does|did|is|are|was|were|will|would|should|could)\b",
        r"\?$",
    ]

    BOLD_STATEMENT_PATTERNS = [
        r"\b(never|always|everyone|no one|nobody|the (only|best|worst|first|last))\b",
        r"\b(secret|truth|revealed|shocking|surprising|unexpected)\b",
        r"\b(you (need|must|have) to|you won't believe)\b",
    ]

    STATISTIC_PATTERNS = [
        r"\b\d+%\b",
        r"\b\d+ (out of|in) \d+\b",
        r"\b(million|billion|thousand)\b",
        r"\b\d+x\b",
    ]

    def __init__(self):
        self.question_re = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self.bold_re = [re.compile(p, re.IGNORECASE) for p in self.BOLD_STATEMENT_PATTERNS]
        self.stat_re = [re.compile(p, re.IGNORECASE) for p in self.STATISTIC_PATTERNS]

    async def detect_hooks(
        self,
        transcription_segments: list[dict],
        visual_analysis: Optional[dict] = None,
        analyze_first_seconds: float = 30.0,
        max_hooks: int = 5,
    ) -> list[dict]:
        """
        Detect potential hooks in video content.

        Args:
            transcription_segments: List of transcription segments with timing
            visual_analysis: Optional visual analysis results
            analyze_first_seconds: Focus on first N seconds for main hook
            max_hooks: Maximum number of hooks to return

        Returns:
            List of detected hooks with scores
        """
        hooks = []

        # Analyze video start (primary hook)
        start_hooks = self._find_hooks_in_range(
            transcription_segments,
            0,
            analyze_first_seconds,
            is_video_start=True,
        )
        hooks.extend(start_hooks)

        # Find hooks throughout video (potential clip openers)
        all_hooks = self._find_hooks_in_range(
            transcription_segments,
            analyze_first_seconds,
            float("inf"),
            is_video_start=False,
        )
        hooks.extend(all_hooks)

        # Score hooks
        scored_hooks = []
        for hook in hooks:
            score = self._score_hook(hook, visual_analysis)
            hook.update(score)
            scored_hooks.append(hook)

        # Sort by score and limit
        scored_hooks.sort(key=lambda x: x["hook_score"], reverse=True)
        scored_hooks = scored_hooks[:max_hooks]

        # Assign ranks
        for i, hook in enumerate(scored_hooks):
            hook["rank"] = i + 1

        return scored_hooks

    def _find_hooks_in_range(
        self,
        segments: list[dict],
        start_time: float,
        end_time: float,
        is_video_start: bool,
    ) -> list[dict]:
        """Find potential hooks within a time range."""
        hooks = []

        for i, segment in enumerate(segments):
            seg_start = segment["start_time"]
            seg_end = segment["end_time"]

            if seg_start < start_time or seg_start > end_time:
                continue

            text = segment["text"]

            # Check for hook patterns
            hook_type = self._classify_hook_type(text)
            if hook_type:
                hooks.append({
                    "start_time": seg_start,
                    "end_time": seg_end,
                    "text": text,
                    "hook_type": hook_type,
                    "is_video_start": is_video_start and seg_start < 5.0,
                })

            # Look for sentence openers (first segment after silence)
            if i > 0:
                prev_end = segments[i - 1]["end_time"]
                gap = seg_start - prev_end
                if gap > 1.0:  # Significant pause
                    hook_type = self._classify_hook_type(text)
                    if hook_type:
                        hooks.append({
                            "start_time": seg_start,
                            "end_time": seg_end,
                            "text": text,
                            "hook_type": hook_type or "statement",
                            "is_video_start": False,
                        })

        return hooks

    def _classify_hook_type(self, text: str) -> Optional[str]:
        """Classify the type of hook based on text patterns."""
        # Check for questions
        for pattern in self.question_re:
            if pattern.search(text):
                return "question"

        # Check for statistics
        for pattern in self.stat_re:
            if pattern.search(text):
                return "statistic"

        # Check for bold statements
        for pattern in self.bold_re:
            if pattern.search(text):
                return "statement"

        # Check for short punchy statements (good hooks are often concise)
        words = text.split()
        if 3 <= len(words) <= 15:
            # Short statements can be hooks if they're engaging
            if any(w.isupper() for w in words) or "!" in text:
                return "statement"

        return None

    def _score_hook(
        self,
        hook: dict,
        visual_analysis: Optional[dict] = None,
    ) -> dict:
        """
        Score a potential hook on multiple dimensions.

        Returns scores for:
        - curiosity_gap: Does it create intrigue?
        - emotional_trigger: Does it evoke emotion?
        - clarity: Is it clear and understandable?
        - visual_interest: Is there visual engagement? (if visual data available)
        """
        text = hook["text"]
        hook_type = hook["hook_type"]

        scores = {
            "curiosity_gap": 0.0,
            "emotional_trigger": 0.0,
            "clarity": 0.0,
            "visual_interest": 0.0,
        }

        # Curiosity gap scoring
        if hook_type == "question":
            scores["curiosity_gap"] = 18.0
        elif hook_type == "statistic":
            scores["curiosity_gap"] = 15.0
        elif hook_type == "statement":
            scores["curiosity_gap"] = 12.0

        # Bonus for "you" language (personal relevance)
        if re.search(r"\byou\b", text, re.IGNORECASE):
            scores["curiosity_gap"] += 2.0

        # Emotional trigger scoring
        emotional_words = [
            "amazing", "incredible", "shocking", "surprising", "beautiful",
            "terrible", "worst", "best", "love", "hate", "fear", "excited",
            "angry", "happy", "sad", "funny", "crazy", "insane", "mind-blowing"
        ]
        emotion_count = sum(1 for w in emotional_words if w in text.lower())
        scores["emotional_trigger"] = min(20.0, 10.0 + emotion_count * 3)

        # Clarity scoring
        word_count = len(text.split())
        if 5 <= word_count <= 20:
            scores["clarity"] = 18.0
        elif word_count < 5:
            scores["clarity"] = 12.0
        else:
            scores["clarity"] = max(5.0, 18.0 - (word_count - 20) * 0.5)

        # Visual interest (from visual analysis if available)
        if visual_analysis:
            # Check if there's action/movement at hook time
            scores["visual_interest"] = 15.0  # Base score
            key_elements = visual_analysis.get("key_visual_elements", [])
            if any(e.get("category") == "people" for e in key_elements):
                scores["visual_interest"] += 3.0
            if any(e.get("category") == "action" for e in key_elements):
                scores["visual_interest"] += 2.0
        else:
            scores["visual_interest"] = 10.0  # Default middle score

        # Calculate total hook score (0-100)
        total = sum(scores.values())
        # Bonus for video start hooks
        if hook.get("is_video_start"):
            total = min(100, total + 5)

        scores["hook_score"] = min(100.0, total)

        return scores

    async def suggest_alternative_hooks(
        self,
        original_hook: dict,
        full_transcript: str,
        llm_service=None,
    ) -> list[str]:
        """
        Use LLM to suggest alternative hook phrasings.

        Args:
            original_hook: The detected hook
            full_transcript: Full video transcript for context
            llm_service: LLM service instance

        Returns:
            List of alternative hook suggestions
        """
        if not llm_service:
            return []

        prompt = f"""Analyze this video hook and suggest 3 alternative, more engaging versions.

Original hook: "{original_hook['text']}"
Hook type: {original_hook['hook_type']}

Context from video:
{full_transcript[:500]}...

Provide 3 alternative hooks that:
1. Maintain the same core message
2. Create more curiosity or emotional engagement
3. Are concise (under 15 words)

Format: Return only the 3 alternatives, one per line."""

        try:
            response = await llm_service.generate(prompt, max_tokens=200)
            alternatives = [
                line.strip().strip('"').strip("'")
                for line in response.split("\n")
                if line.strip() and not line.startswith("#")
            ]
            return alternatives[:3]
        except Exception as e:
            logger.warning(f"Failed to generate alternative hooks: {e}")
            return []


# Singleton instance
_hook_service = None


def get_hook_detection_service() -> HookDetectionService:
    """Get singleton hook detection service."""
    global _hook_service
    if _hook_service is None:
        _hook_service = HookDetectionService()
    return _hook_service
