"""B-Roll detection service for identifying optimal insertion points."""

import logging
import re
from typing import Optional
from uuid import UUID

from app.config import get_settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
settings = get_settings()

# Keywords that often indicate abstract concepts not visually present
ABSTRACT_CONCEPT_KEYWORDS = {
    # Data and statistics
    "data", "statistics", "numbers", "percent", "percentage", "growth", "decline",
    "increase", "decrease", "trend", "forecast", "prediction", "analysis",
    # Technology
    "technology", "digital", "algorithm", "software", "internet", "cloud",
    "artificial intelligence", "ai", "machine learning", "automation",
    # Emotions and concepts
    "success", "failure", "happiness", "stress", "anxiety", "motivation",
    "inspiration", "creativity", "innovation", "productivity", "efficiency",
    # Business concepts
    "strategy", "marketing", "branding", "roi", "revenue", "profit", "market",
    "competition", "industry", "economy", "investment", "capital",
    # Abstract ideas
    "future", "past", "history", "philosophy", "science", "research",
    "education", "learning", "knowledge", "wisdom", "experience",
}

# Common visual objects that YOLO can detect
VISUAL_OBJECTS = {
    "person", "face", "car", "bicycle", "motorcycle", "bus", "truck",
    "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear",
    "chair", "couch", "table", "tv", "laptop", "phone", "book",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
}


class BRollDetectionService:
    """Service for detecting optimal B-roll insertion points in video clips."""

    def __init__(self):
        self.llm_service: Optional[LLMService] = None

    async def detect_broll_opportunities(
        self,
        clip_id: UUID,
        transcription_segments: list[dict],
        visual_analysis: Optional[dict] = None,
        llm_service: Optional[LLMService] = None,
        max_suggestions: int = 5,
        min_duration: float = 2.0,
        max_duration: float = 10.0,
    ) -> list[dict]:
        """
        Detect optimal B-roll insertion points in a clip.

        Args:
            clip_id: UUID of the clip
            transcription_segments: List of transcript segments with start/end times
            visual_analysis: Optional YOLO visual analysis data
            llm_service: Optional LLM service for query generation
            max_suggestions: Maximum suggestions to return
            min_duration: Minimum B-roll duration
            max_duration: Maximum B-roll duration

        Returns:
            List of B-roll opportunities with metadata
        """
        self.llm_service = llm_service
        opportunities = []

        # Strategy 1: Detect abstract concepts
        abstract_opportunities = await self._detect_abstract_concepts(
            transcription_segments, visual_analysis, min_duration, max_duration
        )
        opportunities.extend(abstract_opportunities)

        # Strategy 2: Detect topic changes
        topic_opportunities = self._detect_topic_changes(
            transcription_segments, min_duration, max_duration
        )
        opportunities.extend(topic_opportunities)

        # Strategy 3: Detect visual gaps
        if visual_analysis:
            gap_opportunities = self._detect_visual_gaps(
                transcription_segments, visual_analysis, min_duration, max_duration
            )
            opportunities.extend(gap_opportunities)

        # Strategy 4: Detect natural transitions (pauses)
        transition_opportunities = self._detect_transitions(
            transcription_segments, min_duration, max_duration
        )
        opportunities.extend(transition_opportunities)

        # Sort by relevance and deduplicate overlapping windows
        opportunities = self._deduplicate_and_rank(opportunities, max_suggestions)

        # Generate search queries for each opportunity
        for opp in opportunities:
            if self.llm_service and opp.get("transcript_context"):
                queries = await self._generate_search_queries(opp["transcript_context"])
                opp["search_queries"] = queries
            else:
                # Fallback to keyword-based queries
                opp["search_queries"] = self._generate_keyword_queries(opp.get("keywords", []))

        return opportunities

    async def _detect_abstract_concepts(
        self,
        segments: list[dict],
        visual_analysis: Optional[dict],
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Detect segments mentioning abstract concepts not visually present."""
        opportunities = []
        detected_objects = set()

        # Extract detected objects from visual analysis
        if visual_analysis and "timeline" in visual_analysis:
            for frame in visual_analysis.get("timeline", []):
                for obj in frame.get("objects", []):
                    detected_objects.add(obj.get("label", "").lower())

        for i, seg in enumerate(segments):
            text = seg.get("text", "").lower()
            words = set(re.findall(r'\b\w+\b', text))

            # Find abstract keywords in transcript
            abstract_found = words & ABSTRACT_CONCEPT_KEYWORDS

            # Check if these concepts are NOT visually represented
            visual_in_text = words & VISUAL_OBJECTS & detected_objects

            if abstract_found and not visual_in_text:
                start_time = seg.get("start_time", seg.get("start", 0))
                end_time = seg.get("end_time", seg.get("end", 0))
                duration = end_time - start_time

                # Adjust duration to fit constraints
                if duration < min_duration:
                    # Extend the window
                    mid = (start_time + end_time) / 2
                    start_time = max(0, mid - min_duration / 2)
                    end_time = start_time + min_duration
                    duration = min_duration
                elif duration > max_duration:
                    duration = max_duration
                    end_time = start_time + duration

                # Calculate relevance based on abstract keyword density
                relevance = min(1.0, len(abstract_found) / 3)

                opportunities.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "detection_reason": "abstract_concept",
                    "transcript_context": seg.get("text", ""),
                    "keywords": list(abstract_found),
                    "relevance_score": relevance,
                    "confidence": 0.7 + (0.3 * relevance),
                })

        return opportunities

    def _detect_topic_changes(
        self,
        segments: list[dict],
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Detect topic changes through vocabulary shifts."""
        opportunities = []

        if len(segments) < 3:
            return opportunities

        # Use sliding window to detect vocabulary shifts
        window_size = 3

        for i in range(window_size, len(segments)):
            # Get words from previous window
            prev_words = set()
            for j in range(i - window_size, i):
                text = segments[j].get("text", "").lower()
                prev_words.update(re.findall(r'\b\w{4,}\b', text))

            # Get words from current segment
            curr_text = segments[i].get("text", "").lower()
            curr_words = set(re.findall(r'\b\w{4,}\b', curr_text))

            # Calculate vocabulary overlap
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))

                # Low overlap indicates topic change
                if overlap < 0.2:
                    start_time = segments[i].get("start_time", segments[i].get("start", 0))
                    end_time = segments[i].get("end_time", segments[i].get("end", 0))
                    duration = end_time - start_time

                    # Adjust duration
                    if duration < min_duration:
                        end_time = start_time + min_duration
                        duration = min_duration
                    elif duration > max_duration:
                        duration = max_duration
                        end_time = start_time + duration

                    # Extract key new words as keywords
                    new_words = curr_words - prev_words
                    keywords = list(new_words)[:5]

                    opportunities.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "detection_reason": "topic_change",
                        "transcript_context": segments[i].get("text", ""),
                        "keywords": keywords,
                        "relevance_score": 1.0 - overlap,
                        "confidence": 0.6,
                    })

        return opportunities

    def _detect_visual_gaps(
        self,
        segments: list[dict],
        visual_analysis: dict,
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Detect periods of low visual interest."""
        opportunities = []

        timeline = visual_analysis.get("timeline", [])
        if not timeline:
            return opportunities

        # Find periods with few detected objects
        sparse_regions = []
        current_sparse_start = None

        for frame in timeline:
            time = frame.get("time", 0)
            objects = frame.get("objects", [])

            if len(objects) <= 1:  # Low visual interest
                if current_sparse_start is None:
                    current_sparse_start = time
            else:
                if current_sparse_start is not None:
                    sparse_regions.append((current_sparse_start, time))
                    current_sparse_start = None

        # Close any open region
        if current_sparse_start is not None and timeline:
            sparse_regions.append((current_sparse_start, timeline[-1].get("time", 0)))

        # Convert sparse regions to opportunities
        for start, end in sparse_regions:
            duration = end - start
            if duration < min_duration:
                continue
            if duration > max_duration:
                end = start + max_duration
                duration = max_duration

            # Find corresponding transcript
            context = ""
            keywords = []
            for seg in segments:
                seg_start = seg.get("start_time", seg.get("start", 0))
                seg_end = seg.get("end_time", seg.get("end", 0))
                if seg_start <= start <= seg_end or seg_start <= end <= seg_end:
                    context = seg.get("text", "")
                    keywords = re.findall(r'\b\w{4,}\b', context.lower())[:5]
                    break

            opportunities.append({
                "start_time": start,
                "end_time": end,
                "duration": duration,
                "detection_reason": "visual_gap",
                "transcript_context": context,
                "keywords": keywords,
                "relevance_score": 0.7,
                "confidence": 0.5,
            })

        return opportunities

    def _detect_transitions(
        self,
        segments: list[dict],
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Detect natural transition points (pauses between segments)."""
        opportunities = []

        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end_time", segments[i - 1].get("end", 0))
            curr_start = segments[i].get("start_time", segments[i].get("start", 0))

            gap = curr_start - prev_end

            # Significant pause indicates natural transition
            if gap >= 0.5:
                # Use the next segment's content for context
                context = segments[i].get("text", "")
                keywords = re.findall(r'\b\w{4,}\b', context.lower())[:5]

                duration = min(max_duration, max(min_duration, gap + 2.0))
                start_time = max(0, prev_end - 0.5)
                end_time = start_time + duration

                opportunities.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "detection_reason": "transition",
                    "transcript_context": context,
                    "keywords": keywords,
                    "relevance_score": min(1.0, gap / 2.0),
                    "confidence": 0.6,
                })

        return opportunities

    def _deduplicate_and_rank(
        self,
        opportunities: list[dict],
        max_suggestions: int,
    ) -> list[dict]:
        """Remove overlapping opportunities and rank by relevance."""
        if not opportunities:
            return []

        # Sort by relevance score descending
        opportunities.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Remove overlapping windows
        selected = []
        for opp in opportunities:
            is_overlapping = False
            for selected_opp in selected:
                # Check for overlap
                if (opp["start_time"] < selected_opp["end_time"] and
                    opp["end_time"] > selected_opp["start_time"]):
                    is_overlapping = True
                    break

            if not is_overlapping:
                selected.append(opp)
                if len(selected) >= max_suggestions:
                    break

        # Assign ranks
        for i, opp in enumerate(selected):
            opp["rank"] = i + 1

        return selected

    async def _generate_search_queries(self, transcript_context: str) -> list[str]:
        """Use LLM to generate relevant stock footage search queries."""
        if not self.llm_service:
            return []

        try:
            prompt = f"""Given this transcript excerpt from a video, suggest 3 stock footage search queries that would provide good visual B-roll.

The B-roll should:
- Illustrate the concepts being discussed
- Be generic enough to find good stock footage
- Not be too specific or niche

Transcript: "{transcript_context[:500]}"

Return only the 3 search queries, one per line, without numbering or quotes."""

            response = await self.llm_service.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
            )

            # Parse response
            queries = [
                line.strip()
                for line in response.split("\n")
                if line.strip() and len(line.strip()) > 2
            ]

            return queries[:3]

        except Exception as e:
            logger.warning(f"Failed to generate search queries with LLM: {e}")
            return []

    def _generate_keyword_queries(self, keywords: list[str]) -> list[str]:
        """Generate search queries from extracted keywords."""
        if not keywords:
            return ["business technology", "abstract background", "professional footage"]

        queries = []

        # Use top keywords as queries
        for keyword in keywords[:3]:
            if keyword in ABSTRACT_CONCEPT_KEYWORDS:
                # Map abstract concepts to visual queries
                concept_mappings = {
                    "data": "data visualization graphs",
                    "technology": "technology innovation",
                    "success": "success celebration",
                    "growth": "business growth charts",
                    "strategy": "business strategy meeting",
                    "future": "futuristic technology",
                    "innovation": "innovation technology",
                    "productivity": "productive work office",
                }
                query = concept_mappings.get(keyword, f"{keyword} concept footage")
            else:
                query = keyword

            queries.append(query)

        return queries if queries else ["professional b-roll footage"]


# Singleton instance
_detection_service: Optional[BRollDetectionService] = None


def get_broll_detection_service() -> BRollDetectionService:
    """Get B-roll detection service instance."""
    global _detection_service
    if _detection_service is None:
        _detection_service = BRollDetectionService()
    return _detection_service
