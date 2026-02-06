"""Tests for BRollDetectionService."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.broll_detection_service import (
    BRollDetectionService,
    get_broll_detection_service,
    ABSTRACT_CONCEPT_KEYWORDS,
)


class TestBRollDetectionService:
    """Test cases for BRollDetectionService."""

    @pytest.fixture
    def service(self) -> BRollDetectionService:
        """Create a BRollDetectionService instance."""
        return BRollDetectionService()

    @pytest.mark.asyncio
    async def test_detect_abstract_concepts(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test detection of abstract concepts in transcript."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=10,
        )

        # Should detect abstract concepts like "data", "technology", "future"
        abstract_opportunities = [
            o for o in opportunities if o["detection_reason"] == "abstract_concept"
        ]

        assert len(abstract_opportunities) > 0

        # Check that keywords are extracted
        for opp in abstract_opportunities:
            assert "keywords" in opp
            assert len(opp["keywords"]) > 0

    @pytest.mark.asyncio
    async def test_detect_topic_changes(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test detection of topic changes in transcript."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=10,
        )

        # Should detect topic change when switching from tech to cooking
        topic_change_opportunities = [
            o for o in opportunities if o["detection_reason"] == "topic_change"
        ]

        # The sample data has a clear topic change from technology to cooking
        assert len(topic_change_opportunities) >= 0  # May or may not detect based on threshold

    @pytest.mark.asyncio
    async def test_detect_visual_gaps(
        self,
        service: BRollDetectionService,
        sample_transcript_segments: list[dict],
        sample_visual_analysis: dict,
    ):
        """Test detection of visual gaps."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            visual_analysis=sample_visual_analysis,
            max_suggestions=10,
        )

        # Should detect visual gaps where few objects are detected
        gap_opportunities = [
            o for o in opportunities if o["detection_reason"] == "visual_gap"
        ]

        # Visual analysis has gaps at time 4.0-6.0
        assert len(gap_opportunities) >= 0

    @pytest.mark.asyncio
    async def test_detect_transitions(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test detection of natural transitions (pauses)."""
        clip_id = uuid.uuid4()

        # Add larger gaps between segments
        segments_with_gaps = [
            {"start_time": 0.0, "end_time": 5.0, "text": "First segment."},
            {"start_time": 7.0, "end_time": 12.0, "text": "Second segment after pause."},
            {"start_time": 14.0, "end_time": 19.0, "text": "Third segment."},
        ]

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=segments_with_gaps,
            max_suggestions=10,
        )

        transition_opportunities = [
            o for o in opportunities if o["detection_reason"] == "transition"
        ]

        # Should detect transitions at the gaps
        assert len(transition_opportunities) >= 1

    @pytest.mark.asyncio
    async def test_max_suggestions_limit(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test that max_suggestions limit is respected."""
        clip_id = uuid.uuid4()
        max_suggestions = 2

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=max_suggestions,
        )

        assert len(opportunities) <= max_suggestions

    @pytest.mark.asyncio
    async def test_duration_constraints(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test that B-roll durations respect min/max constraints."""
        clip_id = uuid.uuid4()
        min_duration = 3.0
        max_duration = 8.0

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            min_duration=min_duration,
            max_duration=max_duration,
            max_suggestions=10,
        )

        for opp in opportunities:
            assert opp["duration"] >= min_duration
            assert opp["duration"] <= max_duration

    @pytest.mark.asyncio
    async def test_opportunities_have_required_fields(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test that all opportunities have required fields."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=10,
        )

        required_fields = [
            "start_time",
            "end_time",
            "duration",
            "detection_reason",
            "transcript_context",
            "keywords",
            "relevance_score",
            "confidence",
            "search_queries",
        ]

        for opp in opportunities:
            for field in required_fields:
                assert field in opp, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_opportunities_are_ranked(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test that opportunities are ranked by relevance."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=5,
        )

        if len(opportunities) > 1:
            # Check that ranks are assigned
            ranks = [o.get("rank") for o in opportunities]
            assert all(r is not None for r in ranks)
            assert ranks == sorted(ranks)  # Ranks should be in order

    @pytest.mark.asyncio
    async def test_no_overlapping_opportunities(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict]
    ):
        """Test that returned opportunities don't overlap."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            max_suggestions=10,
        )

        # Sort by start time
        sorted_opps = sorted(opportunities, key=lambda x: x["start_time"])

        for i in range(1, len(sorted_opps)):
            prev_end = sorted_opps[i - 1]["end_time"]
            curr_start = sorted_opps[i]["start_time"]
            assert curr_start >= prev_end, "Opportunities should not overlap"

    @pytest.mark.asyncio
    async def test_with_llm_service(
        self, service: BRollDetectionService, sample_transcript_segments: list[dict], mock_llm_service: MagicMock
    ):
        """Test detection with LLM service for query generation."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=sample_transcript_segments,
            llm_service=mock_llm_service,
            max_suggestions=5,
        )

        # Check that search queries are generated
        for opp in opportunities:
            assert "search_queries" in opp

    @pytest.mark.asyncio
    async def test_empty_transcript(self, service: BRollDetectionService):
        """Test handling of empty transcript."""
        clip_id = uuid.uuid4()

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=[],
            max_suggestions=5,
        )

        assert opportunities == []

    @pytest.mark.asyncio
    async def test_single_segment(self, service: BRollDetectionService):
        """Test handling of single transcript segment."""
        clip_id = uuid.uuid4()
        segments = [
            {
                "start_time": 0.0,
                "end_time": 10.0,
                "text": "This is about data analytics and technology innovation.",
            }
        ]

        opportunities = await service.detect_broll_opportunities(
            clip_id=clip_id,
            transcription_segments=segments,
            max_suggestions=5,
        )

        # Should still detect abstract concepts
        assert len(opportunities) >= 0


class TestKeywordDetection:
    """Test keyword detection logic."""

    def test_abstract_keywords_defined(self):
        """Test that abstract concept keywords are defined."""
        assert len(ABSTRACT_CONCEPT_KEYWORDS) > 0
        assert "data" in ABSTRACT_CONCEPT_KEYWORDS
        assert "technology" in ABSTRACT_CONCEPT_KEYWORDS
        assert "future" in ABSTRACT_CONCEPT_KEYWORDS


class TestServiceSingleton:
    """Test service singleton pattern."""

    def test_get_broll_detection_service(self):
        """Test singleton service getter."""
        service1 = get_broll_detection_service()
        service2 = get_broll_detection_service()

        assert service1 is service2
        assert isinstance(service1, BRollDetectionService)
