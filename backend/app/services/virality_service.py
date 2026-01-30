"""Virality scoring service for video clips."""

import logging
import re
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ViralityService:
    """Service for scoring viral potential of video clips."""

    # Score weights (out of 20 each, total 100)
    WEIGHTS = {
        "emotional_resonance": 20,
        "shareability": 20,
        "uniqueness": 20,
        "hook_strength": 20,
        "production_quality": 20,
    }

    def __init__(self):
        pass

    async def score_clip(
        self,
        clip_data: dict,
        transcription_data: Optional[dict] = None,
        visual_data: Optional[dict] = None,
        hook_data: Optional[dict] = None,
        llm_service=None,
    ) -> dict:
        """
        Calculate virality score for a video clip.

        Args:
            clip_data: Clip timing and metadata
            transcription_data: Transcription for the clip
            visual_data: Visual analysis results
            hook_data: Hook detection results
            llm_service: Optional LLM for enhanced analysis

        Returns:
            Dictionary with score breakdown (0-100 total, 0-20 per category)
        """
        scores = {
            "emotional_resonance": 0.0,
            "shareability": 0.0,
            "uniqueness": 0.0,
            "hook_strength": 0.0,
            "production_quality": 0.0,
        }

        # Get transcript text
        transcript = ""
        if transcription_data:
            transcript = transcription_data.get("text", "")

        # Score emotional resonance
        scores["emotional_resonance"] = self._score_emotional_resonance(
            transcript, visual_data
        )

        # Score shareability
        scores["shareability"] = self._score_shareability(
            transcript, clip_data.get("duration", 0)
        )

        # Score uniqueness
        scores["uniqueness"] = self._score_uniqueness(transcript, visual_data)

        # Score hook strength
        scores["hook_strength"] = self._score_hook_strength(
            transcript, hook_data, clip_data.get("start_time", 0)
        )

        # Score production quality
        scores["production_quality"] = self._score_production_quality(
            clip_data, visual_data, transcription_data
        )

        # LLM-enhanced scoring if available
        if llm_service and transcript:
            try:
                llm_scores = await self._get_llm_scores(
                    transcript, llm_service
                )
                # Blend LLM scores with rule-based (60% LLM, 40% rules)
                for key in scores:
                    if key in llm_scores:
                        scores[key] = scores[key] * 0.4 + llm_scores[key] * 0.6
            except Exception as e:
                logger.warning(f"LLM scoring failed, using rule-based: {e}")

        # Calculate total
        total = sum(scores.values())

        return {
            "virality_score": min(100.0, total),
            "emotional_resonance": min(20.0, scores["emotional_resonance"]),
            "shareability": min(20.0, scores["shareability"]),
            "uniqueness": min(20.0, scores["uniqueness"]),
            "hook_strength": min(20.0, scores["hook_strength"]),
            "production_quality": min(20.0, scores["production_quality"]),
        }

    def _score_emotional_resonance(
        self,
        transcript: str,
        visual_data: Optional[dict],
    ) -> float:
        """Score emotional resonance (0-20)."""
        score = 10.0  # Base score

        # Check for emotional language
        emotional_words = {
            "high": ["amazing", "incredible", "shocking", "devastating", "hilarious",
                     "heartbreaking", "inspiring", "terrifying", "beautiful", "mind-blowing"],
            "medium": ["surprising", "exciting", "sad", "happy", "angry", "funny",
                      "weird", "crazy", "awesome", "terrible"],
            "low": ["good", "bad", "nice", "interesting", "cool", "okay", "fine"],
        }

        transcript_lower = transcript.lower()

        high_count = sum(1 for w in emotional_words["high"] if w in transcript_lower)
        medium_count = sum(1 for w in emotional_words["medium"] if w in transcript_lower)

        score += high_count * 2.0
        score += medium_count * 1.0

        # Exclamation marks indicate emotional delivery
        exclamation_count = transcript.count("!")
        score += min(2.0, exclamation_count * 0.5)

        # Check for people in visuals (emotional connection)
        if visual_data:
            key_elements = visual_data.get("key_visual_elements", [])
            if any(e.get("category") == "people" for e in key_elements):
                score += 2.0

        return min(20.0, score)

    def _score_shareability(self, transcript: str, duration: float) -> float:
        """Score shareability potential (0-20)."""
        score = 10.0  # Base score

        # Optimal duration for sharing (15-60 seconds)
        if 15 <= duration <= 60:
            score += 4.0
        elif 60 < duration <= 90:
            score += 2.0
        elif duration < 15:
            score += 1.0
        # Longer clips get no bonus

        # Check for quotable content
        sentences = transcript.split(". ")
        short_punchy = sum(1 for s in sentences if 3 <= len(s.split()) <= 12)
        score += min(3.0, short_punchy * 0.5)

        # Check for universally relatable content
        relatable_patterns = [
            r"\beveryone\b", r"\bwe all\b", r"\byou know\b",
            r"\blife\b", r"\blove\b", r"\bwork\b", r"\bmoney\b",
        ]
        for pattern in relatable_patterns:
            if re.search(pattern, transcript, re.IGNORECASE):
                score += 0.5

        return min(20.0, score)

    def _score_uniqueness(
        self,
        transcript: str,
        visual_data: Optional[dict],
    ) -> float:
        """Score uniqueness/novelty (0-20)."""
        score = 10.0  # Base score

        # Check for unique insights or facts
        unique_indicators = [
            r"\d+%",  # Statistics
            r"\bonly\b",
            r"\bfirst\b",
            r"\bsecret\b",
            r"\bnobody\b",
            r"\beveryone thinks\b",
            r"\bcontrary to\b",
            r"\bactually\b",
        ]

        for pattern in unique_indicators:
            if re.search(pattern, transcript, re.IGNORECASE):
                score += 1.5

        # Check for specific numbers (signals specificity)
        numbers = re.findall(r"\b\d+\b", transcript)
        if len(numbers) >= 1:
            score += 1.0
        if len(numbers) >= 3:
            score += 1.0

        # Visual uniqueness (variety of objects)
        if visual_data:
            unique_objects = visual_data.get("unique_objects", [])
            score += min(2.0, len(unique_objects) * 0.3)

        return min(20.0, score)

    def _score_hook_strength(
        self,
        transcript: str,
        hook_data: Optional[dict],
        start_time: float,
    ) -> float:
        """Score hook/opening strength (0-20)."""
        score = 10.0  # Base score

        # Use existing hook analysis if available
        if hook_data:
            hook_score = hook_data.get("hook_score", 50)
            score = hook_score / 5  # Scale from 0-100 to 0-20

        # Check first sentence for hook elements
        first_sentence = transcript.split(".")[0] if transcript else ""
        first_words = first_sentence.split()[:10] if first_sentence else []
        first_text = " ".join(first_words).lower()

        # Question opening
        if first_text.endswith("?") or any(
            first_text.startswith(w) for w in ["what", "how", "why", "did you", "have you"]
        ):
            score += 3.0

        # Bold claim
        if any(w in first_text for w in ["never", "always", "everyone", "secret"]):
            score += 2.0

        # Direct address
        if "you" in first_text:
            score += 1.5

        return min(20.0, score)

    def _score_production_quality(
        self,
        clip_data: dict,
        visual_data: Optional[dict],
        transcription_data: Optional[dict],
    ) -> float:
        """Score production quality (0-20)."""
        score = 12.0  # Base score (assume decent quality)

        # Video resolution
        width = clip_data.get("width", 0)
        height = clip_data.get("height", 0)
        if height >= 1080:
            score += 3.0
        elif height >= 720:
            score += 2.0
        elif height >= 480:
            score += 1.0

        # Audio quality (clear transcription = good audio)
        if transcription_data:
            segments = transcription_data.get("segments", [])
            if segments:
                avg_prob = sum(
                    s.get("avg_log_prob", -1) for s in segments
                ) / len(segments)
                # avg_log_prob closer to 0 = better quality
                if avg_prob > -0.3:
                    score += 2.0
                elif avg_prob > -0.5:
                    score += 1.0

        # Visual stability (low variance in motion = stable)
        if visual_data:
            timeline = visual_data.get("timeline", [])
            if timeline:
                # Consistent object detection suggests good framing
                detection_counts = [len(t.get("detections", [])) for t in timeline]
                if detection_counts:
                    variance = sum((c - sum(detection_counts)/len(detection_counts))**2
                                   for c in detection_counts) / len(detection_counts)
                    if variance < 5:
                        score += 1.0

        return min(20.0, score)

    async def _get_llm_scores(
        self,
        transcript: str,
        llm_service,
    ) -> dict:
        """Get LLM-based virality scores."""
        prompt = f"""Analyze this video clip transcript for viral potential. Score each category from 0-20:

Transcript:
"{transcript[:1000]}"

Score these aspects:
1. Emotional Resonance (0-20): Does it evoke strong emotions?
2. Shareability (0-20): Would people want to share this?
3. Uniqueness (0-20): Is the content novel or surprising?
4. Hook Strength (0-20): Does it grab attention immediately?
5. Production Quality (0-20): Based on content clarity and structure

Return ONLY a JSON object with these exact keys:
{{"emotional_resonance": X, "shareability": X, "uniqueness": X, "hook_strength": X, "production_quality": X}}"""

        try:
            response = await llm_service.generate(prompt, max_tokens=100)

            # Parse JSON from response
            import json
            # Find JSON in response
            match = re.search(r'\{[^}]+\}', response)
            if match:
                scores = json.loads(match.group())
                return {
                    "emotional_resonance": float(scores.get("emotional_resonance", 10)),
                    "shareability": float(scores.get("shareability", 10)),
                    "uniqueness": float(scores.get("uniqueness", 10)),
                    "hook_strength": float(scores.get("hook_strength", 10)),
                    "production_quality": float(scores.get("production_quality", 10)),
                }
        except Exception as e:
            logger.warning(f"Failed to parse LLM scores: {e}")

        return {}


# Singleton instance
_virality_service = None


def get_virality_service() -> ViralityService:
    """Get singleton virality service."""
    global _virality_service
    if _virality_service is None:
        _virality_service = ViralityService()
    return _virality_service
