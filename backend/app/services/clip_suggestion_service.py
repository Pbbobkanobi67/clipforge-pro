"""Clip suggestion service for finding optimal video segments."""

import logging
from typing import Optional

from app.config import get_settings
from app.services.virality_service import ViralityService, get_virality_service

logger = logging.getLogger(__name__)
settings = get_settings()


class ClipSuggestionService:
    """Service for suggesting optimal video clips."""

    def __init__(self, virality_service: Optional[ViralityService] = None):
        self.virality_service = virality_service or get_virality_service()

    async def suggest_clips(
        self,
        transcription_data: dict,
        scene_data: Optional[dict] = None,
        hook_data: Optional[list[dict]] = None,
        visual_data: Optional[dict] = None,
        diarization_data: Optional[dict] = None,
        video_metadata: Optional[dict] = None,
        min_duration: float = 15.0,
        max_duration: float = 90.0,
        target_count: int = 5,
        llm_service=None,
    ) -> list[dict]:
        """
        Suggest optimal video clips for extraction.

        Args:
            transcription_data: Full transcription with segments
            scene_data: Scene detection results
            hook_data: Detected hooks
            visual_data: Visual analysis results
            diarization_data: Speaker diarization results
            video_metadata: Video dimensions, duration, etc.
            min_duration: Minimum clip duration in seconds
            max_duration: Maximum clip duration in seconds
            target_count: Number of clips to suggest
            llm_service: Optional LLM for title generation

        Returns:
            List of suggested clips with scores and metadata
        """
        segments = transcription_data.get("segments", [])
        if not segments:
            return []

        # Find candidate clip boundaries
        candidates = self._find_candidate_clips(
            segments=segments,
            scene_data=scene_data,
            hook_data=hook_data,
            diarization_data=diarization_data,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        # Score each candidate
        scored_clips = []
        for candidate in candidates:
            # Get relevant transcription for this clip
            clip_transcript = self._get_clip_transcript(
                segments, candidate["start_time"], candidate["end_time"]
            )

            # Get relevant hook data
            clip_hook = self._get_clip_hook(
                hook_data, candidate["start_time"]
            )

            # Score the clip
            scores = await self.virality_service.score_clip(
                clip_data={
                    **candidate,
                    **(video_metadata or {}),
                },
                transcription_data={"text": clip_transcript},
                visual_data=visual_data,
                hook_data=clip_hook,
                llm_service=llm_service,
            )

            clip = {
                **candidate,
                **scores,
                "transcript_excerpt": clip_transcript[:500],
            }

            scored_clips.append(clip)

        # Sort by virality score
        scored_clips.sort(key=lambda x: x["virality_score"], reverse=True)

        # Remove overlapping clips
        final_clips = self._remove_overlaps(scored_clips, target_count)

        # Generate titles with LLM if available
        if llm_service:
            final_clips = await self._generate_titles(final_clips, llm_service)
        else:
            # Generate basic titles
            for clip in final_clips:
                clip["title"] = self._generate_basic_title(clip)

        # Assign ranks
        for i, clip in enumerate(final_clips):
            clip["rank"] = i + 1

        return final_clips

    def _find_candidate_clips(
        self,
        segments: list[dict],
        scene_data: Optional[dict],
        hook_data: Optional[list[dict]],
        diarization_data: Optional[dict],
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Find candidate clip boundaries."""
        candidates = []
        scenes = scene_data.get("scenes", []) if scene_data else []
        hooks = hook_data or []

        # Strategy 1: Use hooks as starting points
        for hook in hooks:
            start = hook["start_time"]

            # Find good ending point
            end = self._find_clip_end(
                start, segments, scenes, min_duration, max_duration
            )

            if end and (end - start) >= min_duration:
                candidates.append({
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "source": "hook",
                })

        # Strategy 2: Use scene boundaries
        for i, scene in enumerate(scenes):
            start = scene["start_time"]

            # Try to include 2-4 scenes for natural grouping
            for j in range(i + 1, min(i + 5, len(scenes))):
                end = scenes[j]["end_time"]
                duration = end - start

                if min_duration <= duration <= max_duration:
                    candidates.append({
                        "start_time": start,
                        "end_time": end,
                        "duration": duration,
                        "source": "scene",
                    })

        # Strategy 3: Use speaker turns (complete thoughts)
        if diarization_data:
            diar_segments = diarization_data.get("segments", [])
            speaker_groups = self._group_by_speaker(diar_segments)

            for group in speaker_groups:
                if group["duration"] >= min_duration:
                    # Extend to complete sentence
                    end = self._extend_to_sentence_end(
                        group["end_time"], segments, max_duration - group["duration"]
                    )

                    candidates.append({
                        "start_time": group["start_time"],
                        "end_time": min(end, group["start_time"] + max_duration),
                        "duration": min(end - group["start_time"], max_duration),
                        "source": "speaker",
                    })

        # Strategy 4: Sliding window for content without clear boundaries
        if not candidates:
            candidates = self._sliding_window_clips(
                segments, min_duration, max_duration
            )

        return candidates

    def _find_clip_end(
        self,
        start_time: float,
        segments: list[dict],
        scenes: list[dict],
        min_duration: float,
        max_duration: float,
    ) -> Optional[float]:
        """Find optimal ending point for a clip starting at given time."""
        min_end = start_time + min_duration
        max_end = start_time + max_duration

        # Find segment that contains potential endings
        candidate_ends = []

        # Option 1: End at scene boundary
        for scene in scenes:
            if min_end <= scene["end_time"] <= max_end:
                candidate_ends.append({
                    "time": scene["end_time"],
                    "score": 10,  # Good break point
                })

        # Option 2: End at sentence boundary
        for seg in segments:
            if min_end <= seg["end_time"] <= max_end:
                text = seg["text"].strip()
                if text.endswith((".", "!", "?")):
                    candidate_ends.append({
                        "time": seg["end_time"],
                        "score": 8,  # Sentence end
                    })

        # Option 3: End at pause
        for i, seg in enumerate(segments[:-1]):
            next_seg = segments[i + 1]
            gap = next_seg["start_time"] - seg["end_time"]
            if gap > 0.5 and min_end <= seg["end_time"] <= max_end:
                candidate_ends.append({
                    "time": seg["end_time"],
                    "score": 5 + gap,  # Longer pause = better break
                })

        if not candidate_ends:
            # Fallback: just use max duration
            return max_end

        # Choose best ending
        candidate_ends.sort(key=lambda x: x["score"], reverse=True)
        return candidate_ends[0]["time"]

    def _get_clip_transcript(
        self,
        segments: list[dict],
        start_time: float,
        end_time: float,
    ) -> str:
        """Extract transcript text for a clip."""
        texts = []
        for seg in segments:
            if seg["end_time"] >= start_time and seg["start_time"] <= end_time:
                texts.append(seg["text"])
        return " ".join(texts)

    def _get_clip_hook(
        self,
        hooks: Optional[list[dict]],
        start_time: float,
        tolerance: float = 2.0,
    ) -> Optional[dict]:
        """Get hook data near clip start."""
        if not hooks:
            return None

        for hook in hooks:
            if abs(hook["start_time"] - start_time) <= tolerance:
                return hook

        return None

    def _group_by_speaker(
        self,
        diar_segments: list[dict],
    ) -> list[dict]:
        """Group consecutive segments by speaker."""
        if not diar_segments:
            return []

        groups = []
        current = None

        for seg in diar_segments:
            if current is None:
                current = {
                    "speaker_id": seg["speaker_id"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                }
            elif seg["speaker_id"] == current["speaker_id"]:
                current["end_time"] = seg["end_time"]
            else:
                current["duration"] = current["end_time"] - current["start_time"]
                groups.append(current)
                current = {
                    "speaker_id": seg["speaker_id"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                }

        if current:
            current["duration"] = current["end_time"] - current["start_time"]
            groups.append(current)

        return groups

    def _extend_to_sentence_end(
        self,
        current_end: float,
        segments: list[dict],
        max_extension: float,
    ) -> float:
        """Extend clip end to the next sentence boundary."""
        max_end = current_end + max_extension

        for seg in segments:
            if seg["start_time"] > current_end and seg["end_time"] <= max_end:
                if seg["text"].strip().endswith((".", "!", "?")):
                    return seg["end_time"]

        return current_end

    def _sliding_window_clips(
        self,
        segments: list[dict],
        min_duration: float,
        max_duration: float,
    ) -> list[dict]:
        """Generate clips using sliding window approach."""
        if not segments:
            return []

        clips = []
        target_duration = (min_duration + max_duration) / 2
        step = target_duration / 2

        total_duration = segments[-1]["end_time"]
        start = 0

        while start < total_duration - min_duration:
            end = min(start + target_duration, total_duration)

            # Adjust to sentence boundary
            for seg in segments:
                if start + min_duration <= seg["end_time"] <= end:
                    if seg["text"].strip().endswith((".", "!", "?")):
                        end = seg["end_time"]
                        break

            if end - start >= min_duration:
                clips.append({
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "source": "window",
                })

            start += step

        return clips

    def _remove_overlaps(
        self,
        clips: list[dict],
        target_count: int,
    ) -> list[dict]:
        """Remove overlapping clips, keeping highest scored."""
        if len(clips) <= target_count:
            return clips

        selected = []

        for clip in clips:
            # Check overlap with already selected
            overlaps = False
            for sel in selected:
                if (clip["start_time"] < sel["end_time"] and
                    clip["end_time"] > sel["start_time"]):
                    # More than 50% overlap
                    overlap = min(clip["end_time"], sel["end_time"]) - max(clip["start_time"], sel["start_time"])
                    if overlap > clip["duration"] * 0.5:
                        overlaps = True
                        break

            if not overlaps:
                selected.append(clip)

            if len(selected) >= target_count:
                break

        return selected

    async def _generate_titles(
        self,
        clips: list[dict],
        llm_service,
    ) -> list[dict]:
        """Generate engaging titles for clips using LLM."""
        for clip in clips:
            transcript = clip.get("transcript_excerpt", "")[:300]

            prompt = f"""Generate a short, engaging title (max 10 words) for this video clip.
The title should be catchy and make people want to watch.

Clip transcript:
"{transcript}"

Return ONLY the title, nothing else."""

            try:
                title = await llm_service.generate(prompt, max_tokens=30)
                clip["title"] = title.strip().strip('"').strip("'")
            except Exception as e:
                logger.warning(f"Failed to generate title: {e}")
                clip["title"] = self._generate_basic_title(clip)

        return clips

    def _generate_basic_title(self, clip: dict) -> str:
        """Generate a basic title without LLM."""
        transcript = clip.get("transcript_excerpt", "")

        # Use first sentence or first few words
        sentences = transcript.split(". ")
        if sentences:
            first = sentences[0].strip()
            words = first.split()[:8]
            if len(words) >= 3:
                return " ".join(words) + "..."

        return f"Clip {clip.get('rank', 1)}: {clip['duration']:.0f}s"


# Singleton instance
_clip_service = None


def get_clip_suggestion_service() -> ClipSuggestionService:
    """Get singleton clip suggestion service."""
    global _clip_service
    if _clip_service is None:
        _clip_service = ClipSuggestionService()
    return _clip_service
